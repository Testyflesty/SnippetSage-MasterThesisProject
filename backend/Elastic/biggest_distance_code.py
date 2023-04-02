import plotly.express as px
from scipy.spatial import distance
from elasticsearch.helpers import scan
from sklearn.cluster import KMeans
from gensim import corpora, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from elasticsearch import Elasticsearch
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in text.split() if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens



if __name__ == '__main__':

    import plotly.express as px
    from scipy.spatial import distance
    from elasticsearch.helpers import scan
    
    es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False)
    index_name = "code_snippets"
    scroll_size = 1000

    query_dict = {
        "query": {
            "match_all": {}
        },
        "_source": ["code_snippet", "question_id", "code_emb"],
        "size": scroll_size
    }
    embeddings = es_client.search(index=index_name, body=query_dict, scroll="1m")
    scroll_id = embeddings["_scroll_id"]
    hits = embeddings["hits"]["hits"]
    questions_and_embeddings= []
    
    while hits:
        for hit in tqdm(hits):
            code = hit['_source']['code_snippet']
            embedding = hit['_source']['code_emb']
            questions_and_embeddings.append((code, embedding))
            pass

        embeddings = es_client.scroll(scroll_id=scroll_id, scroll="1m")
        hits = embeddings["hits"]["hits"]

    
        

    numpembedding = np.array([embedding for _, embedding in questions_and_embeddings]).reshape(-1, 1)
    max_clusters = 10

    optimal_num_clusters = 5 
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=0)
    kmeans.fit(numpembedding)

    num_topics = optimal_num_clusters
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    num_topics = 5  
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)

    for cluster_id in range(optimal_num_clusters):
        cluster_questions = [q for q, label in zip([q for q, _ in questions_and_embeddings], kmeans.labels_) if label == cluster_id]
        
        preprocessed_questions = [preprocess_text(q) for q in cluster_questions]
        
        vectorizer = CountVectorizer(stop_words='english')
        bag_of_words = vectorizer.fit_transform([' '.join(q) for q in preprocessed_questions])
        
        lda.fit(bag_of_words)
        topics = lda.components_.argsort()[:, ::-1]
        feature_names = np.array(vectorizer.get_feature_names())
        top_words = [feature_names[topics[i, :5]].tolist() for i in range(num_topics)]
        
        print(f'Cluster {cluster_id + 1}:')
        for i, (question, words) in enumerate(zip(cluster_questions, top_words)):
            print(f'  {i+1}. {question} --> {" ".join(words)}')