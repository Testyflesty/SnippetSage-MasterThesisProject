import pickle
from typing import List
from uuid import uuid4

from app_escode import Encoder
from elasticsearch import Elasticsearch
from tqdm import tqdm


class MultiCodeEmbeddingMap:
    def __init__(
        self,
        url: str = "https://localhost:9200",
        username: str = "elastic",
        password: str = "-LwMtIIcEYpWG7rx_zSC",
    ):
        self.es_client = Elasticsearch(
            url, http_auth=(username, password), verify_certs=False
        )

    def map_from_pickle(self, path: str, question_titles: str) -> None:
        # Load the embedding map from a pickle file
        with open(path, "rb") as f:
            code_snippets = pickle.load(f)
        with open(question_titles, "rb") as f:
            questions = pickle.load(f)

        encoder = Encoder("hamzab/codebert_code_search")
        for question_id, code_snippet_index in tqdm(code_snippets):
            code = code_snippets[(question_id, code_snippet_index)]
            question = questions.get(question_id, None)
            if question is None:
                continue
            embedding = encoder.encode(question, 512)
            self.insert(embedding[0], question_id, code_snippet_index, code)

    def insert(
        self,
        embedding: List[float],
        question_id: int,
        code_snippet_index: int,
        code_snippet: str,
    ) -> None:
        # Map the embedding to the question and code snippet in ElasticSearch
        doc = {
            "embeddings": embedding,
            "question_id": question_id,
            "code_snippet_index": code_snippet_index,
            "code_snippet": code_snippet,
        }
        self.es_client.index(index="embedding_map", document=doc, id=str(uuid4()))

    def setup_index(self):
        self.es_client.indices.delete(index="embedding_map", ignore=[400, 404])
        config = {
            "mappings": {
                "properties": {
                    "embeddings": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "question_id": {"type": "integer"},
                    "code_snippet_index": {"type": "integer"},
                    "code_snippet": {"type": "text"},
                },
            },
            "settings": {"number_of_shards": 2, "number_of_replicas": 1},
        }

        self.es_client.indices.create(
            index="embedding_map",
            body=config,
        )


if __name__ == "__main__":
    # Create the embedding map in ElasticSearch
    map = MultiCodeEmbeddingMap()
    map.setup_index()
    staqd = "python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle"
    question_titles = "python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle"
    map.map_from_pickle(staqd, question_titles)
