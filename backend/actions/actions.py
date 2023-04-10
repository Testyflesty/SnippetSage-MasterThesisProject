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
import json
import numpy as np


class ActionElastic(Action):

    def name(self) -> Text:
        return "call_haystack"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False)
        index_name = "questions"

        # response = search(tracker.latest_message["text"], es_client, "hamzab/codebert_code_search", index_name, 2500)

        # Get the detected intent and entities
        intent = tracker.latest_message['intent'].get('name')
        entities = tracker.latest_message.get('entities')

        # Create a dictionary with the metadata
        metadata = {
            "intent": intent,
            "entities": entities
        }
        response = searchstacq(tracker.latest_message["text"], es_client, "bert-base-uncased", index_name, 2500, metadata)
        
        answer = json.dumps(response)
        print(answer)
        dispatcher.utter_message(text=answer)

        return []
    
def searchstacq(query: str, es_client: Elasticsearch, model: str, index: str, top_k: int = 40, metadata: dict = {}):

    encoder = Encoder(model)
    query_vector = encoder.encode(query, max_length=768)

    intent = metadata.get("intent")
    if intent:
        intent_vector = encoder.encode(intent, max_length=768)
    else:
        intent_vector = None

    entities = metadata.get("entities", [])
    entity_vectors = []
    for entity in entities:
        entity_vector = encoder.encode(entity["value"], max_length=768)
        entity_vectors.append(entity_vector)
        print("entity_vector")
    if entity_vectors:
        entity_vectors = np.mean(entity_vectors, axis=0)
    else:
        entity_vectors = None

    if intent_vector is not None and entity_vectors is not None:
        combined_vector = 0.6 * query_vector[0] + 0.1 * intent_vector + 0.3 * entity_vectors
        print("both intend and entity")

    if(intent_vector is not None and entity_vectors is None):
        combined_vector = 0.7 * query_vector[0] + 0.3 * intent_vector
        print("only intent")

    if(intent_vector is None and entity_vectors is not None):
        combined_vector = 0.7 * query_vector[0] + 0.3 * entity_vectors
        print("only entity")

    if(combined_vector is None):
        combined_vector = query_vector[0]
        print("else only query")

    
    query_dict = {
        "knn":{
        "field": "question_emb",
        "query_vector": combined_vector[0].tolist(),
        "k": 10,
        "num_candidates": 1000
        }
    }
    res = es_client.search(index=index, body=query_dict)
    results = res["hits"]["hits"]

    for id, hit in enumerate(res["hits"]["hits"]):
        search_result = es_client.search(index="code_snippets", q=f"question_id:{hit['_source']['question_id']}")
        hit["_source"].pop("question_emb", None)
        for code_hit in search_result["hits"]["hits"]:

            code = '```' + code_hit['_source']['code_snippet'] + '```'
            hit['_source']['code'] = code
    
    
    response = {
        "results": results,
        "intent": metadata["intent"],
        "entities": metadata["entities"]
    }
    return response
    
def search(query: str, es_client: Elasticsearch, model: str, index: str, top_k: int = 10):

    encoder = Encoder(model)
    query_vector = encoder.encode(query, max_length=64)
    query_dict = {
        "knn":{
        "field": "embeddings",
        "query_vector": query_vector[0].tolist(),
        "k": 3,
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

    # for hit in res["hits"]["hits"]:
    #     print(f"Document ID: {hit['_id']}")
    #     print(f"Document Score: {hit['_score']}")
    #     print(f"Document Title: {hit['_source']['repo']}")
    #     print(f"Document Language: {hit['_source']['language']}")
    #     print(f"Document Text: {hit['_source']['code']}")
    #     print(f"Document Embedding: {hit['_source']['embeddings']}")

    #     print("=====================================================================\n")
    
    # def get_best_document(res):
    #     best_score = float('-inf')
    #     best_doc = None

    #     for hit in res["hits"]["hits"]:
    #         if '_score' not in hit:
    #             continue  # skip this hit if it doesn't have a _score key
    #         score = hit['_score']
    #         if score > best_score:
    #             best_score = score
    #             best_doc = hit

    #     return best_doc



    # print(res)
    # best_doc = get_best_document(res)
    # print(best_doc)

    # code = best_doc['_source']['code']
    # repo = best_doc['_source']['repo']
    # score = best_doc['_score']
    
    resultstring = "I found the following code snippets for you: <br/>"

    for index, hit in enumerate(res["hits"]["hits"]):

        code = hit['_source']['code']
        repo = hit['_source']['repo']
        score = hit['_score']
        resultstring += str(index +1)
        resultstring += ": <br/><p> <pre class='language-python'><code class='language-python hljs'> " + str(code) + "</code></pre> It is used in this repository: " + str(repo) + "<br/> And this is the score I assigned it: " + str(score) + "</p><br/>"
    
    return resultstring
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

