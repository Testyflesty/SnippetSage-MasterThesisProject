import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from elasticsearch import Elasticsearch
from tqdm import tqdm

# Download the NLTK stopwords corpus
nltk.download("stopwords")



if __name__ == '__main__':
    
    
    from sklearn.cluster import KMeans
    
       
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


    optimal_num_clusters = 10 
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    kmeans.fit(numpembedding)

    cluster_labels = kmeans.fit_predict(numpembedding)
    # Define the stopword list
    stopwords_list = stopwords.words("english")

    questions = [hit["_source"]["question"] for hit in hits]
    # For each cluster, extract the questions and tokenize them
    for i in range(optimal_num_clusters):
        cluster_questions = [q for q, label in zip(questions, cluster_labels) if label == i]
        tokenized_questions = [word_tokenize(q.lower()) for q in cluster_questions]
        
        # Flatten the tokenized questions and remove stopwords
        flat_questions = [word for q in tokenized_questions for word in q if word not in stopwords_list]
        
        # Get the most frequent words and their counts
        word_counts = Counter(flat_questions)
        most_common_words = word_counts.most_common(10)
        
        # Print the most common words and examples for the cluster
        print(f"Cluster {i}")
        print("Most common words:", most_common_words)
        print("Example questions:")
        for q in cluster_questions[:5]:
            print("- ", q)
        print()
