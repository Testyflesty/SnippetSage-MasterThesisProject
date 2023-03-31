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
        # "_source": ["question", "question_id", "question_emb"],
        "size": scroll_size
    }
    embeddings = es_client.search(index=index_name, body=query_dict, scroll="1m")
    scroll_id = embeddings["_scroll_id"]

    hits = embeddings["hits"]["hits"]

    numpembedding = []
    
    while hits:
        for hit in hits:
            numpembedding.append(hit['_source']['question_emb'])
            pass

        embeddings = es_client.scroll(scroll_id=scroll_id, scroll="1m")
        hits = embeddings["hits"]["hits"]
    
        
    numpembedding = np.array(numpembedding)
    print(len(numpembedding))
    # Set the number of clusters and target dimensions
    num_clusters = 3
    target_dimensions = 25  # Choose a suitable target dimension value

    # Perform PCA to reduce the dimensionality of the embeddings
    # pca = PCA(n_components=target_dimensions)
    # reduced_embeddings = pca.fit_transform(numpembedding)

    # # Create a KMeans instance
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)


    # # Fit the model to the embeddings
    # kmeans.fit(numpembedding)

    # # Get the cluster assignments for each embedding
    # cluster_assignments = kmeans.predict(numpembedding)

    # # Store the cluster assignments in a file
    # np.save('cluster_assignments.npy', cluster_assignments)
    # # # Fit the KMeans model to the reduced embeddings
 
    
    # # centers = kmeans.cluster_centers_

    # # # calculate the distance matrix between cluster centers
    # # dist_matrix = np.zeros((3,3))
    # # for i in range(3):
    # #     for j in range(i+1, 3):
    # #         dist_matrix[i,j] = np.linalg.norm(centers[i]-centers[j])
    # #         dist_matrix[j,i] = dist_matrix[i,j]


    # # sort the clusters by their distance from each other
    # cluster_order = np.argsort(np.sum(dist_matrix, axis=1))

    # # get the indices of the farthest apart questions for each cluster
    # indices = []
    # for i in cluster_order:
    #     mask = kmeans.labels_ == i
    #     dist = np.linalg.norm(embeddings - centers[i], axis=1)
    #     farthest = np.argsort(dist)[-5:]
    #     indices.append(farthest[mask])

    # # print the questions for each cluster
    # for i, idx in enumerate(indices):
    #     print(f"Cluster {i}:")
    #     for j in idx:
    #         print(embeddings["hits"]["hits"][j]["_source"]["question"])

    kmeans.fit(numpembedding)

    # Get the cluster assignments for each embedding
    cluster_assignments = kmeans.labels_

    # Perform dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(numpembedding)

    # Visualize the clusters using matplotlib
    plt.figure(figsize=(10, 8))
    for cluster in range(num_clusters):
        plt.scatter(embeddings_2d[cluster_assignments == cluster, 0],
                    embeddings_2d[cluster_assignments == cluster, 1],
                    label=f'Cluster {cluster}', alpha=0.7)

    plt.legend()
    plt.title('Clusters of code embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

