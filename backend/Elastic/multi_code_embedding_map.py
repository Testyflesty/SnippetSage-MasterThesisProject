import pickle
from typing import List
from uuid import uuid4

from backend.Elastic.app_escode import Encoder
from elasticsearch import Elasticsearch


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

    def map_from_pickle(self, path: str) -> None:
        # Load the embedding map from a pickle file
        with open(path, "rb") as f:
            code_snippets = pickle.load(f)

        encoder = Encoder("hamzab/codebert_code_search")
        for question_id, code_snippet_index in code_snippets:
            embedding = encoder.encode(
                code_snippets[(question_id, code_snippet_index)], 512
            )
            self.insert(embedding, question_id, code_snippet_index)

    def insert(
        self, embedding: List[float], question_id: int, code_snippet_index: int
    ) -> None:
        # Map the embedding to the question and code snippet in ElasticSearch
        doc = {
            "embeddings": embedding,
            "question_id": question_id,
            "code_snippet_index": code_snippet_index,
        }
        self.es_client.index(index="embedding_map", body=doc, id=str(uuid4()))

    def setup_index(self):
        config = {
            "mappings": {
                "properties": {
                    "embeddings": {
                        "type": "dense_vector",
                        "dims": 768,
                    },
                    "question_id": {"type": "integer"},
                    "code_snippet_index": {"type": "integer"},
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
    staqd = "StackOverflow-Question-Code-Dataset/annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle"
    map.map_from_pickle(staqd)
