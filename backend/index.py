from haystack.document_stores import PineconeDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack import Document
from tqdm.auto import tqdm  # progress bar
import json

def read_jsonlines(file_path):
    json_files = []
    with open(file_path, "r") as f:
        for line in f:
            json_files.append(json.loads(line))
    return json_files

document_store = PineconeDocumentStore(
    api_key='72471adc-04a0-4633-8e60-f84d1a6be0e5',
    index='haystack-lfqa',
    similarity="dot_product",
    environment="us-east1-gcp",
    embedding_dim=768
)


retriever = EmbeddingRetriever(
   document_store=document_store,
   embedding_model="hamzab/codebert_code_search"
   )

# batch_size = 50
# json_linesdata = read_jsonlines('./data/test.jsonl')
# counter = 0
# docs = []
# for d in tqdm(json_linesdata):
#     # create haystack document object with text content and doc metadata
#     doc = Document(
#         content=d["code"],
#         meta={
#             "repo": d["repo"],
#             'func_name': d['func_name'],
#             'language': d['language'],
#             'url': d['url'],
#         }
#     )
#     docs.append(doc)
#     counter += 1
#     if counter % batch_size == 0:
#         # writing docs everytime 10k docs are reached
#         document_store.write_documents(docs)
#         docs.clear()
#     if counter == 100000:
#         break

print(document_store.get_document_count())


from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

search_pipe = DocumentSearchPipeline(retriever)
result = search_pipe.run(
    query="What is an array?",
    params={"Retriever": {"top_k": 2}}
)

print_documents(result)