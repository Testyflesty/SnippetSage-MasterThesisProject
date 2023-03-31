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

    # Print 5 questions for each cluster
    for cluster in range(num_clusters):
        # Get indices and questions for the current cluster
 # Get indices and questions for the current cluster
        cluster_indices = np.where(kmeans.labels_ == cluster)[0]
        cluster_questions = [hits[i]['_source']['question'] for i in cluster_indices if i < len(hits)]

        # Print header for current cluster
        print(f"Cluster {cluster+1}:")
        # Print the first 5 questions for the current cluster
        for i in range(5):
            if i < len(cluster_questions):
                print(f"Question {i+1}: {cluster_questions[i]}")
            else:
                break
        print()