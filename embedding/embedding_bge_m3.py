import os
import json
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import torch
from sentence_transformers import SentenceTransformer
import json_repair
import pickle
import logging
from flask import Flask, request, jsonify
import yaml
import numpy as np

logger = logging.getLogger("myapp")

app = Flask(__name__)
embedder = None

class bge_m3_embedder:
    def __init__(self,config):
        # BAAI/bge-m3
        self.model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True,device='cpu',cache_folder="/data4/students/zhangguangyin/LLMs",model_kwargs={"torch_dtype":torch.bfloat16})
        # In case you want to reduce the maximum length:
        self.config = config
   
        self.gpu_id = config["gpu_id"]  # 指定你想使用的GPU编号
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.model=self.model.to(self.device)
        with open("/data4/students/zhangguangyin/chatNum/embedding_pkl/bge_m3_10000.pickle","rb")as f:#这个存的是向量
            self.doc_embeddings=pickle.load(f)#ndarray
        self.doc_embeddings=torch.tensor(self.doc_embeddings,dtype=torch.float32).to(self.device)
        print(self.doc_embeddings.shape)


    def embed_document(self,doclist):
        document_embeddings = self.model.encode(doclist,batch_size=self.config["embedding_batch_size"],show_progress_bar=True)
        return document_embeddings

    
    def embed_query(self,query_list):
        query_embeddings = self.model.encode(query_list)
        # print(query_embeddings.shape)
        return query_embeddings

    def calculate_similarity(self,query_embeddings,document_embeddings=None,return_itemid=None):
        if document_embeddings is not None:
            query_embeddings = torch.tensor(query_embeddings,dtype=torch.float32).to(self.device)
            document_embeddings = torch.tensor(document_embeddings,dtype=torch.float32).to(self.device)
            scores = (query_embeddings @ document_embeddings.T) * 100
        elif return_itemid is not None:
            query_embeddings = torch.tensor(query_embeddings,dtype=torch.float32).to(self.device)
            selected_doc_embeddings = self.doc_embeddings[return_itemid]
            scores = (query_embeddings @ selected_doc_embeddings.T) * 100
        else:
            query_embeddings = torch.tensor(query_embeddings,dtype=torch.float32).to(self.device)
            scores = (query_embeddings @ self.doc_embeddings.T) * 100
        return scores.tolist()
    

@app.route('/embedding', methods=['POST'])
def embedding_api():
    try:
        data = request.get_json()
        if not data or 'method' not in data:
            return jsonify({'error': 'No method provided'}), 400
        method = data['method']
        if method == 'embed_document':
            documents = data.get('documents')
            if not isinstance(documents, list):
                return jsonify({'error': 'documents must be a list'}), 400
            embeddings = embedder.embed_document(documents)
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            return jsonify({'embeddings': embeddings})
        elif method == 'embed_query':
            queries = data.get('queries')
            # if not isinstance(queries, list):
            #     return jsonify({'error': 'queries must be a list'}), 400
            embeddings = embedder.embed_query(queries)
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            return jsonify({'embeddings': embeddings})
        elif method == 'calculate_similarity':
            query_embeddings = data.get('query_embeddings')
            query_embeddings = np.asarray(query_embeddings)
            document_embeddings=None
            return_itemid=None
            if "document_embedding" in data:
                document_embeddings = data.get('document_embedding')
                document_embeddings = np.asarray(document_embeddings)
            elif "return_itemid" in data:
                return_itemid = data.get('return_itemid')
            else:
                document_embeddings = None
            # if not (isinstance(query_embeddings, list) and isinstance(document_embeddings, list)):
            #     return jsonify({'error': 'query_embeddings and document_embeddings must be lists'}), 400
            
            
            scores = embedder.calculate_similarity(query_embeddings, document_embeddings=document_embeddings,return_itemid=return_itemid)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            return jsonify({'scores': scores})
        else:
            return jsonify({'error': f'Unknown method: {method}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__=="__main__":
    # config={
    #     "gpu_id":7,
    #     "embedding_batch_size":10,
    #     "embedding_model":"bge_m3"
    # }
    with open("/data4/students/zhangguangyin/chatNum/config/config.yaml","r")as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
    embedder=bge_m3_embedder(config)
    # Run Flask app
    app.run(host='0.0.0.0', port=config.get("bge_m3_embedding_port", 5439))



