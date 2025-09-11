import os
import json
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import torch
from sentence_transformers import SentenceTransformer
import json_repair

model = SentenceTransformer("/data4/students/zhangguangyin/LLMs/gte-qwen2-7b/a8d08b36ada9cacfe34c4d6f80957772a025daf2", trust_remote_code=True,cache_folder="/data/zhangguangyin/LLMs",model_kwargs={"torch_dtype":torch.bfloat16})
# In case you want to reduce the maximum length:
with open("/data4/students/zhangguangyin/chatNum/table-item_2phrase.json","r")as f:
    docdict=json.load(f)
documents=[]
document_source=[]
for key,value in docdict.items():
    for item in value:
        documents.append(item)
        document_source.append(key)
document_embeddings = model.encode(documents)
with open("/data4/students/zhangguangyin/chatNum/leaderboards.json","r")as f:
    leaddoc=json.load(f)
querylist=[]
for item in leaddoc:
    query=f"retrieval all the model's result on  {item['Task']} task on  {item['Dataset']} dataset using metric {item['Metric']}"
    querylist.append(query)
query_embeddings = model.encode(querylist, prompt_name="query")

savedict={}
for queryid,query in enumerate(querylist):
    print("-----------query----------")
    print(query)
    scores = (query_embeddings[queryid] @ document_embeddings.T) * 100
    indexed_scores = [(value, index) for index, value in enumerate(scores)]
    
    templist=[]
    # 按元素值排序
    indexed_scores.sort(key=lambda x: x[0], reverse=True)
    for x in indexed_scores[:15]:
        print(document_source[x[1]])
        print(documents[x[1]])
        templist.append([document_source[x[1]],documents[x[1]]])
    savedict[query]=templist

with open("embedding_save_dict.json","w")as f:
    json.dump(savedict,f,ensure_ascii=False,indent=1)