from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from elasticsearch import Elasticsearch

if __name__ == '__main__':

    import plotly.express as px
    from scipy.spatial import distance
    from elasticsearch.helpers import scan
    
    es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False)
    index_name = "questions"
    scroll_size = 1000

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
    questions_and_embeddings= []
    
    while hits:
        for hit in tqdm(hits):
            question = hit['_source']['question']
            embedding = hit['_source']['question_emb']
            questions_and_embeddings.append((question, embedding))
            pass

        embeddings = es_client.scroll(scroll_id=scroll_id, scroll="1m")
        hits = embeddings["hits"]["hits"]

    numpembedding = np.array([embedding for _, embedding in questions_and_embeddings]).reshape(-1, 1)
    max_clusters = 10
    inertias = []

    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(numpembedding)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, max_clusters+1), inertias)
    plt.title('Elbow Graph')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

    optimal_num_clusters = int(input("Enter the optimal number of clusters: "))
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=0)
    kmeans.fit(numpembedding)

            
