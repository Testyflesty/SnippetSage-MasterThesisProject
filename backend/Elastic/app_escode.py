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
# Connect to the Elastic instance and create a document store.
# document_store = ElasticsearchDocumentStore(
#     host="localhost",
#     username="",
#     password="",
#     index="document",
#     create_index=True,
#     similarity="dot_product"
# )

# retriever = EmbeddingRetriever(
#     document_store=document_store,
#     embedding_model="hamzab/codebert_code_search",
#     model_format="hamzab/codebert_code_search"
# )
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
    def __init__(self, model_name: str, use_cuda: bool=True):
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


# Update the embeddings in the document store to use the retriever.
# document_store.update_embeddings(retriever)





# from haystack.pipelines import DocumentSearchPipeline
# from haystack.utils import print_documents

# # Build and execute the query pipeline.
# pipeline = DocumentSearchPipeline(retriever)
# query = "what is the Red Wedding?"
# result = pipeline.run(query, params={"Retriever": {"top_k": 2}})

# # View the query results.
# # You should see a document for "The Rains of Castamere", which is the
# # episode the Red Wedding occurred in, so a very relevant response.
# print_documents(result, max_text_len=100, print_name=True, print_meta=True)
def stacqIndex():
    es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False, )
    es_client.ping()    


    question_mapping = {
        "mappings": {
            "properties": {
                "question_id": {"type": "integer"},
                "question_text": {"type": "text"},
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
    model = AutoModel.from_pretrained(model_name)

    # Load text and code snippets from pickle files
    with open("./python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle", "rb") as f:
        text = pickle.load(f)

    with open("./python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle", "rb") as f:
        code = pickle.load(f)

    # Convert text and code snippets to embeddings
    for question_id, question in tqdm(list(text.items())[:100]):
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        body = {"question_id": question_id, "question": question, "question_emb": embedding.numpy()}
        if es_client.exists(index="questions", id=question_id):
            print(f"Question {question_id} already indexed, skipping")
            continue
        
        es_client.index(index='questions', id=question_id, body=body)

    for question_id, code in tqdm(list(code.items())[:100]):
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        body = {"question_id": question_id, "code_snippet": code, "code_emb": embedding.numpy()}
        if es_client.exists(index="code_snippets", id=question_id):
            print(f"Answer to question {question_id} already indexed, skipping")
            continue
        es_client.index(index='code_snippets',id=question_id, body=body)



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
        



if __name__ == '__main__':
    globals()[sys.argv[1]]()