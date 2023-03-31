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
import annoy
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

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
        for hit in tqdm(hits):
            numpembedding.append(hit['_source']['question_emb'])
            pass

        embeddings = es_client.scroll(scroll_id=scroll_id, scroll="1m")
        hits = embeddings["hits"]["hits"]
    
        
    numpembedding = np.array(numpembedding)
    # Set the number of clusters and target dimensions
    num_clusters = 3
    target_dimensions = 25  # Choose a suitable target dimension value

    # # Create a KMeans instance
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)


    # Fit the model to the embeddings
    kmeans.fit(numpembedding)


    # Get the cluster assignments for each embedding
    cluster_assignments = kmeans.labels_

    pca = PCA(n_components=25)
    numpembedding = pca.fit_transform(numpembedding)
  # Build an Annoy index with the embeddings
    annoy_index = annoy.AnnoyIndex(target_dimensions, metric="angular")
    for i, embedding in enumerate(numpembedding):
        annoy_index.add_item(i, embedding)
    annoy_index.build(n_trees=50)

    for cluster in range(num_clusters):
        cluster_indices = np.where(cluster_assignments == cluster)[0]
        cluster_embeddings = numpembedding[cluster_indices]
        cluster_questions = [hits[i]["_source"]["question"] for i in cluster_indices]
        print(f"Cluster {cluster+1}:")
        for i, embedding in enumerate(cluster_embeddings):
            distances = [annoy_index.get_distance(i, j) for j in cluster_indices]
            farthest_indices = np.argsort(distances)[-6:-1]
            farthest_questions = [cluster_questions[j] for j in farthest_indices[::-1] if j != i]
            print(f"Question {i+1}: {cluster_questions[i]}")
            print("Farthest questions:")
            for j, question in enumerate(farthest_questions[:5]):
                print(f"{j+1}. {question}")
    # ...

    # # Reduce the embeddings to 3D using PCA
    # pca = PCA(n_components=3)
    # embeddings_3d = pca.fit_transform(numpembedding)

    # # Create a 3D scatter plot using Plotly
    # fig = px.scatter_3d(x=embeddings_3d[:, 0], y=embeddings_3d[:, 1], z=embeddings_3d[:, 2],
    #                     color=cluster_assignments)

    # fig.show()
    

