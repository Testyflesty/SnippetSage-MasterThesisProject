from tqdm import tqdm
import json
import os
import sys
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from elasticsearch import Elasticsearch
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

class JsonlCollectionIterator:
    """
    adapted from https://github.com/castorini/pyserini/blob/master/pyserini/encode/_base.py
    """
    def __init__(self, collection_path: str, fields=None, delimiter="\n"):
        if fields:
            self.fields = fields
        else:
            self.fields = ['text']
        self.delimiter = delimiter
        self.all_info = self._load(collection_path)
        self.size = len(self.all_info['sha'])
        self.batch_size = 1
        self.shard_id = 0
        self.shard_num = 1

    def __call__(self, batch_size=1, shard_id=0, shard_num=1):
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.shard_num = shard_num
        return self

    def __iter__(self):
        total_len = self.size
        shard_size = int(total_len / self.shard_num)
        start_idx = self.shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_len)
        if self.shard_id == self.shard_num - 1:
            end_idx = total_len
        to_yield = {}
        for idx in tqdm(range(start_idx, end_idx, self.batch_size)):
            for key in self.all_info:
                to_yield[key] = self.all_info[key][idx: min(idx + self.batch_size, end_idx)]
            yield to_yield

    def _parse_fields_from_info(self, info):
        """
        :params info: dict, containing all fields as speicifed in self.fields either under 
        the key of the field name or under the key of 'contents'.  If under `contents`, this 
        function will parse the input contents into each fields based the self.delimiter
        return: List, each corresponds to the value of self.fields
        """
        n_fields = len(self.fields)

        # if all fields are under the key of info, read these rather than 'contents' 
        if all([field in info for field in self.fields]):
            return [info[field].strip() for field in self.fields]

        assert "contents" in info, f"contents not found in info: {info}"
        contents = info['contents']
        # whether to remove the final self.delimiter (especially \n)
        # in CACM, a \n is always there at the end of contents, which we want to remove;
        # but in SciFact, Fiqa, and more, there are documents that only have title but not text (e.g. "This is title\n")
        # where the trailing \n indicates empty fields
        if contents.count(self.delimiter) == n_fields:
            # the user appends one more delimiter to the end, we remove it
            if contents.endswith(self.delimiter):
                # not using .rstrip() as there might be more than one delimiters at the end
                contents = contents[:-len(self.delimiter)]
        return [field.strip(" ") for field in contents.split(self.delimiter)]

    def _load(self, collection_path):
        filenames = []
        if os.path.isfile(collection_path):
            filenames.append(collection_path)
        else:
            for filename in os.listdir(collection_path):
                filenames.append(os.path.join(collection_path, filename))
        all_info = {field: [] for field in self.fields}
        all_info['sha'] = []
        for filename in filenames:
            with open(filename) as f:
                for line_i, line in tqdm(enumerate(f)):
                    info = json.loads(line)
                    _id = info.get('sha', info.get('docid', None))
                    if _id is None:
                        raise ValueError(f"Cannot find 'id' or 'docid' from {filename}.")
                    all_info['sha'].append(str(_id))
                    fields_info = self._parse_fields_from_info(info)
                    if len(fields_info) != len(self.fields):
                        raise ValueError(
                            f"{len(fields_info)} fields are found at Line#{line_i} in file {filename}." \
                            f"{len(self.fields)} fields expected." \
                            f"Line content: {info['contents']}"
                        )

                    for i in range(len(fields_info)):
                        all_info[self.fields[i]].append(fields_info[i])
        return all_info


class Encoder:
    def __init__(self, model_name: str, use_cuda: bool=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = self.model.to(self.device)


    def encode(self, text:str, max_length: int):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            model_output = self.model(**inputs, return_dict=True)
        
        # Perform pooling
        embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



def create_indices():
    es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False, )
    es_client.ping()
    es_client.indices.delete(index='intentify', ignore=[400, 404])
    config = {
        "mappings": {
            "properties": {
                "repo": {"type": "text"},
                "func_name": {"type": "text"},
                "language": {"type": "text"},
                "code": {"type": "text"},
                "docstring": {"type": "text"},
                "url": {"type": "text"},
                "embeddings": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
        },
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1
        }
    }

    es_client.indices.create(
        index="intentify",
        body=config,
    )
    collection_path = 'msmarco'
    index_name = "intentify"
    collection_iterator = JsonlCollectionIterator(collection_path, fields=['repo','func_name', "language", "code", "docstring", "url"])
    encoder = Encoder('hamzab/codebert_code_search')
    for batch_info in collection_iterator(batch_size=64, shard_id=0, shard_num=1):
            embeddings = encoder.encode(batch_info['code'], 512)
            batch_info["dense_vectors"] = embeddings

            actions = []
            for i in range(len(batch_info['sha'])):
                action = {"index": {"_index": index_name, "_id": batch_info['sha'][i]}}
                print(batch_info['repo'][i])
                doc = {
                        "repo": batch_info['repo'][i],
                        "func_name": batch_info['func_name'][i],
                        "language": batch_info['language'][i],
                        "code": batch_info['code'][i],
                        "docstring": batch_info['docstring'][i],
                        "url": batch_info['url'][i],
                        "embeddings": batch_info['dense_vectors'][i].tolist()
                    }
                actions.append(action)
                actions.append(doc)
            es_client.bulk(index=index_name, body=actions)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, label, idx):
        self.code_tokens = code_tokens
        self.code_ids = code_ids        # code tokenized idxs
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids            # nl tokenized idxs
        self.label = label
        self.idx = idx



class InputFeaturesTrip(InputFeatures):
    """A single training/test features for a example. Add docstring seperately. """
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, ds_tokens, ds_ids, label, idx):
        super(InputFeaturesTrip, self).__init__(code_tokens, code_ids, nl_tokens, nl_ids, label, idx)
        self.ds_tokens = ds_tokens
        self.ds_ids = ds_ids



def convert_examples_to_features(js, tokenizer):
    label = js['label'] if js.get('label', None) else 0
    
    max_seq_length = tokenizer.max_len_single_sentence 

    code = js['code']
    code = js['code_tokens']
    code_tokens = tokenizer.tokenize(code)[:max_seq_length-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = max_seq_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    nl = js['doc']
    nl_tokens = tokenizer.tokenize(nl)[:max_seq_length-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = max_seq_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, label, js['idx'])

def casqoIndex():
    es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False, )
    es_client.ping()    
    # es_client.indices.delete(index='questions', ignore=[400, 404])
    # es_client.indices.delete(index='code_snippets', ignore=[400, 404])


    question_mapping = {
        "mappings": {
            "properties": {
                "question_id": {"type": "integer"},
                "question": {"type": "text"},
                "question_emb": {"type": "dense_vector", "dims": 768, "index": True,
                "similarity": "cosine"},      
                }
        }
    }
    
    code_mapping = {
        "mappings": {
            "properties": {
                "question_id": {"type": "integer"},
                "code_snippet": {"type": "text"},
                "code_emb": {"type": "dense_vector", "dims": 768, "index": True,
                "similarity": "cosine"},       
                }
        }
    }

    if not es_client.indices.exists(index="questions"):
        es_client.indices.create(index="questions", body=question_mapping)    
    if not es_client.indices.exists(index="code_snippets"):
        es_client.indices.create(index="code_snippets", body=code_mapping)
    # Load CodeBERT pre-trained model and tokenizer
    model_name = "hamzab/codebert_code_search"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_questions = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained(model_name)
    model_questions = AutoModel.from_pretrained('bert-base-uncased')

    
    with open('./cosqa-retrieval-train-19604.json', 'r') as f:
            examples = []
            data = json.load(f)
            for js in data:
                examples.append(convert_examples_to_features(js, tokenizer))

    # # Convert text and code snippets to embeddings
    # for element in tqdm(examples):
    #     element.idx = int(element.idx.split('-')[2])
    #     inputs = tokenizer_questions(' '.join(element.nl_tokens), return_tensors="pt", padding=True, truncation=True)
    #     with torch.no_grad():
    #         outputs = model_questions(**inputs)
    #     embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    #     body = {"question_id": element.idx, "question": ' '.join(element.nl_tokens), "question_emb": embedding.numpy()}
    #     # if es_client.exists(index="questions", id=element.idx):
    #     #     print(f"Question {element.idx} already indexed, skipping")
    #     #     continue

    #     es_client.index(index='questions', id=element.idx, body=body)

    for element in tqdm(examples):
        element.idx = int(element.idx.split('-')[2])
        inputs = tokenizer_questions(' '.join(element.code_tokens), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        body = {"question_id": element.idx, "code_snippet": ' '.join(element.code_tokens), "code_emb": embedding.numpy()}
        # if es_client.exists(index="questions", id=element.idx):
        #     print(f"Question {element.idx} already indexed, skipping")
        #     continue
        
        es_client.index(index='code_snippets', id=element.idx, body=body)



def testsearch():
    es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False)
    index_name = "intentify"

    search("database connect", es_client, "hamzab/codebert_code_search", index_name)
    

def stacqsearch():
    es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False)
    index_name = "questions"

    searchstacq("How to make an array?", es_client, "hamzab/codebert_code_search", index_name)
   

def search(query: str, es_client: Elasticsearch, model: str, index: str, top_k: int = 10):

    encoder = Encoder(model)
    query_vector = encoder.encode(query, max_length=64)
    query_dict = {
        "knn":{
        "field": "embeddings",
        "query_vector": query_vector[0].tolist(),
        "k": 10,
        "num_candidates": top_k
        },
        "fields":["repo","func_name",
                        "language",
                        "code",
                        "docstring",
                        "url",
                        "embeddings"]
    }
    res = es_client.search(index=index, body=query_dict)

    for hit in res["hits"]["hits"]:
        print(f"Document ID: {hit['_id']}")
        print(f"Document Score: {hit['_score']}")
        print(f"Document Title: {hit['_source']['repo']}")
        print(f"Document Language: {hit['_source']['language']}")
        print(f"Document Text: {hit['_source']['code']}")
        print(f"Document Embedding: {hit['_source']['embeddings']}")

        print("=====================================================================\n")
        
def searchstacq(query: str, es_client: Elasticsearch, model: str, index: str, top_k: int = 10):

    encoder = Encoder(model)
    query_vector = encoder.encode(query, max_length=64)
    query_dict = {
        "knn":{
        "field": "question_emb",
        "query_vector": query_vector[0].tolist(),
        "k": 10,
        "num_candidates": top_k
        },
        "fields":["question_id","question",
                        "question_emb"]
    }
    res = es_client.search(index=index, body=query_dict)

    for hit in res["hits"]["hits"]:
        print(f"Document ID: {hit['_source']['question_id']}")
        print(f"Document Score: {hit['_score']}")
        print(f"Document Title: {hit['_source']['question']}")
        search_result = es_client.search(index="code_snippets", q=f"question_id:{hit['_source']['question_id']}")
        for code_hit in search_result["hits"]["hits"]:
            print(f"Code: {code_hit['_source']['code_snippet']}")
            print(f"Code Embeddings: {code_hit['_source']['code_emb']}")

        # print(f"Document Embedding: {hit['_source']['question_emb']}")
     
        
def plotclusters():

    import plotly.express as px
    from scipy.spatial import distance
    from elasticsearch.helpers import scan
    
    es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False)
    index_name = "questions"
    scroll_size = 1000

    # get the initial search results and scroll ID
    query_dict = {
        "query": {
            "match_all": {}
        },
        "_source": ["question", "question_id", "question_emb"],
        "size": scroll_size
    }
    embeddings = es_client.search(index=index_name, body=query_dict, scroll="1m")
    scroll_id = embeddings["_scroll_id"]
    hits = embeddings["hits"]["hits"]
    numpembedding = []
    # use the scroll ID to retrieve subsequent batches of results
    while hits:
        for hit in hits:
            numpembedding.append(hit['_source']['question_emb'])
            pass

        embeddings = es_client.scroll(scroll_id=scroll_id, scroll="1m")
        hits = embeddings["hits"]["hits"]
    
        
    numpembedding = np.array(numpembedding)
    print(numpembedding)
    # Set the number of clusters and target dimensions
    num_clusters = 3
    target_dimensions = 25  # Choose a suitable target dimension value

    # Perform PCA to reduce the dimensionality of the embeddings
    # pca = PCA(n_components=target_dimensions)
    # reduced_embeddings = pca.fit_transform(numpembedding)

    # # Create a KMeans instance
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)


    # Fit the model to the embeddings
    kmeans.fit(numpembedding)

    # Get the cluster assignments for each embedding
    cluster_assignments = kmeans.predict(numpembedding)

    # Store the cluster assignments in a file
    np.save('cluster_assignments.npy', cluster_assignments)
    # # Fit the KMeans model to the reduced embeddings
    
    
    # kmeans.fit(numpembedding)

    # # Get the cluster assignments for each embedding
    # cluster_assignments = kmeans.labels_

    # # Perform dimensionality reduction using t-SNE
    # tsne = TSNE(n_components=3, random_state=0)
    # embeddings_3d = tsne.fit_transform(numpembedding)


    # # Define markers and colors for each cluster
    # markers = ['o', 's', 'v', '^', '<', '>', '1', '2', '3', '4']
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

    # # Visualize the clusters using matplotlib
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # for cluster in range(num_clusters):
    #     x = embeddings_3d[cluster_assignments == cluster, 0]
    #     y = embeddings_3d[cluster_assignments == cluster, 1]
    #     z = embeddings_3d[cluster_assignments == cluster, 2]
    #     marker = markers[cluster % len(markers)]
    #     color = colors[cluster % len(colors)]
    #     ax.scatter(x, y, z, marker=marker, color=color, label=f'Cluster {cluster}', alpha=0.7)

    # ax.legend()
    # ax.set_title('Clusters of code embeddings')
    # ax.set_xlabel('t-SNE 1')
    # ax.set_ylabel('t-SNE 2')
    # ax.set_zlabel('t-SNE 3')
    # plt.show()
    
    # kmeans.fit(numpembedding)

    # # Get the cluster assignments for each embedding
    # cluster_assignments = kmeans.labels_

    # tsne = TSNE(n_components=2, random_state=0)
    # embeddings_2d = tsne.fit_transform(numpembedding)

    # # Visualize the clusters using matplotlib
    # plt.figure(figsize=(10, 8))
    # for cluster in range(num_clusters):
    #     plt.scatter(embeddings_2d[cluster_assignments == cluster, 0],
    #                 embeddings_2d[cluster_assignments == cluster, 1],
    #                 label=f'Cluster {cluster}', alpha=0.7)

    # plt.legend()
    # plt.title('Clusters of code embeddings')
    # plt.xlabel('t-SNE 1')
    # plt.ylabel('t-SNE 2')
    # plt.show()
    
    # Reduce the embeddings to 2D using PCA
    # pca = PCA(n_components=2)
    # embeddings_2d = pca.fit_transform(numpembedding)

    # # Plot the embeddings with different colors for each cluster
    # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_assignments)
    # plt.show()
    
    # # Reduce the embeddings to 3D using PCA
    # pca = PCA(n_components=3)
    # embeddings_3d = pca.fit_transform(numpembedding)

    # # Create a 3D scatter plot using Plotly
    # fig = px.scatter_3d(x=embeddings_3d[:, 0], y=embeddings_3d[:, 1], z=embeddings_3d[:, 2],
    #                     color=cluster_assignments)

    # fig.show()
    
    centers = kmeans.cluster_centers_

    # calculate the distance matrix between cluster centers
    dist_matrix = np.zeros((3,3))
    for i in range(3):
        for j in range(i+1, 3):
            dist_matrix[i,j] = np.linalg.norm(centers[i]-centers[j])
            dist_matrix[j,i] = dist_matrix[i,j]


    # sort the clusters by their distance from each other
    cluster_order = np.argsort(np.sum(dist_matrix, axis=1))

    # get the indices of the farthest apart questions for each cluster
    indices = []
    for i in cluster_order:
        mask = kmeans.labels_ == i
        dist = np.linalg.norm(embeddings - centers[i], axis=1)
        farthest = np.argsort(dist)[-5:]
        indices.append(farthest[mask])

    # print the questions for each cluster
    for i, idx in enumerate(indices):
        print(f"Cluster {i}:")
        for j in idx:
            print(embeddings["hits"]["hits"][j]["_source"]["question"])


    # # Retrieve the original text for each cluster
    # cluster_indices = [np.where(cluster_assignments == i)[0] for i in range(k)]

    # for i, indices in enumerate(cluster_indices):
    #     print(f"Cluster {i}:")
    #     for index in indices:
    #         # Retrieve the original text using the index
    #         if index < len(question_embeddings):
    #             # This is a question
    #             text = get_question_text(index)
    #         else:
    #             # This is a code snippet
    #             text = get_code_text(index - len(question_embeddings))
            
    #         print(f"- {text}")

if __name__ == '__main__':
    globals()[sys.argv[1]]()