# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from elasticsearch import Elasticsearch
from typing import Any, Text, Dict, List
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests


class ActionElastic(Action):

    def name(self) -> Text:
        return "call_haystack"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False)
        index_name = "intentify"

        response = search(tracker.latest_message["text"], es_client, "hamzab/codebert_code_search", index_name)

        answer = response

        dispatcher.utter_message(text=answer)

        return []
    
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
        
        code = f"Document Text: {hit['_source']['code']}"
    return code
        
class Encoder:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)


    def encode(self, text:str, max_length: int):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

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

