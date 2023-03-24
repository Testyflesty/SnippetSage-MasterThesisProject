from elasticsearch import Elasticsearch
es_client = Elasticsearch("https://localhost:9200", http_auth=("elastic", "GyR7vXXYWxRzAWGjN+c_"),  verify_certs=False, )

es_client.indices.delete(index='msmarco-demo', ignore=[400, 404])