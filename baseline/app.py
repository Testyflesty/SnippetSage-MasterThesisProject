from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch 
import torch.nn.functional as F


from flask import Flask, render_template, request, jsonify
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import json
import torch
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/save-messages', methods=['POST'])
@app.route('/save_messages', methods=['POST'])
def save_messages():
    try:
        # Load existing data from file
        with open('messages.json', 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        # If file doesn't exist, start with empty data
        existing_data = {}

    # Add new data to existing data with current timestamp as key
    timestamp = datetime.utcnow().isoformat()
    new_data = {timestamp: request.get_json()['messages']}
    existing_data.update(new_data)

    # Write updated data back to file
    with open('messages.json', 'w') as f:
        json.dump(existing_data, f)

    return 'Messages saved successfully'

es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "jCJ2SMeF5mDqXMPlvs92"),  verify_certs=False)
index_name = "questions"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        encoder = Encoder("bert-base-uncased")
        query_vector = encoder.encode(query, max_length=64)
        query_dict = {
            "knn":{
            "field": "question_emb",
            "query_vector": query_vector[0].tolist(),
            "k": 10,
            "num_candidates": 1000
            },
            "fields":["question_id","question",
                            "question_emb"]
        }
        response = es_client.search(index='questions', body=query_dict)
        results = response["hits"]["hits"]
        for id, hit in enumerate(response["hits"]["hits"][:10]):

            search_result = es_client.search(index="code_snippets", q=f"question_id:{hit['_source']['question_id']}")
            for code_hit in search_result["hits"]["hits"]:
                hit['_source']['code'] = code_hit['_source']['code_snippet']
                print(str(id+1))

        print(results)
        flaskresponse = jsonify(results)
        return flaskresponse

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)