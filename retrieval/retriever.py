# retriever.py
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
from typing import List, Dict
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
from pymilvus import connections, Collection,MilvusClient
import yaml
import mysql.connector
from mysql.connector import Error
from sentence_transformers import SentenceTransformer
import time
from embedding.embedding_jina_v3 import embedder_jina
from embedding.embedding_qwen2_7b import qwen_embedder
from collections import defaultdict
from retrieval.reranker import Reranker
import pickle
import json
import logging
from retrieval.bm25 import engishbm25indexer
from embedding.embedding_bge_m3 import bge_m3_embedder
import requests
logger = logging.getLogger("myapp")



class Retriever:
    def __init__(self, config: Dict):
        self.config = config
        # self.method = config.get('method', 'faiss')
        # self.text_model_name = config.get("text_encode_model","None")
        # self.image_model_name = config.get("image_model","None")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize text model
        # self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        # self.text_model = AutoModel.from_pretrained(self.text_model_name).to(self.device)
        if self.config["embedding_model"]=="jina":
            self.embedding_port = self.config.get("jina_v3_embedding_port", 5438)
        elif self.config["embedding_model"]=="qwen2_7b":
            self.embedding_port = self.config.get("qwen2_7b_embedding_port", 5437)
        elif self.config["embedding_model"]=="bge_m3":
            self.embedding_port = self.config.get("bge_m3_embedding_port", 5439)
        else:
            raise ValueError("Unknown embedding model: {}".format(self.config["embedding_model"]))
        self.embedding_url = f"http://localhost:{self.embedding_port}/embedding"
        
        with open(config["doc_embedding_name"],"rb")as f:#这个存的是向量
            self.doc_embeddings=pickle.load(f)#ndarray
         
        # print(type(self.doc_embeddings))
        # print(self.doc_embeddings.shape)
        # assert 1==2
        with open(config["doc_embedding_idlistname"],"r")as f:#这个存的是doc_id和向量下标的对应关系
            self.doc_embeddings_docid=json.load(f)#list
        print("len doc_embeddings_docid",len(self.doc_embeddings_docid))

        if self.config["use_rerank"]:
            self.rerankmodel = Reranker(config)
        self.bm25=engishbm25indexer(config)




    def encode_query(self, texts) -> np.ndarray:
        """Encodes texts into vectors, now via embedding API"""
        payload = {"method": "embed_query", "queries": texts}
        response = requests.post(self.embedding_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            return np.array(result["embeddings"])
        else:
            raise RuntimeError(f"Embedding API error: {response.text}")

    def encode_images(self, image_urls: List[str]) -> np.ndarray:
        """Encodes images into vectors"""
        # Implement image encoding logic based on the image model
        # For now, return random vectors as a placeholder
        return np.random.random((len(image_urls), self.dimension)).astype('float32')
    
    

    
    def retrieve_texts(self, query_list, top_k: int = 5) -> List[Dict]:
        """Retrieve texts using the specified method"""
        self.embedding_model



    

    def retrieve_images(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve images based on query"""
        # Implement image retrieval logic if needed
        # For now, return empty list as a placeholder
        return []



    def retrieve(self,question,doc_list=[],blacklist=[], offset: int = 0, limit: int = None) -> List[Dict]:
        """
        Retrieve documents based on query, including texts and images.
        Supports pagination via offset and limit.
        - question: 问题
        - doc_list: 指定的文档id，返回的chunk只能从这些doc_id中选取
        - blacklist: 黑名单，返回的chunk不能从这些doc_id中选取
        - offset: start index in the ranked list (default 0)
        - limit: number of results to return (default: config["retrieval_top_k"]).
        """
        if self.config["simple_retrieval"]:
            new_documentid,new_return_itemid=self.simple_retrieval(question,doc_list,blacklist, offset=offset, limit=limit)
            return new_documentid,new_return_itemid
        if self.config["simple_bm25"]:
            new_documentid,new_return_itemid=self.simple_bm25(question,doc_list,blacklist, offset=offset, limit=limit)
            return new_documentid,new_return_itemid

        #下面这个是完整的流程，既会使用bm25,也会使用dense retrieval,也会使用到reranker
        return_itemid=[]
        if self.config["use_bm25"]:
            bm25_score_list=self.bm25.search(question,top_k=self.config["bm25_retrieval_top_k"])#这个返回的是按照相似度分数排序的id列表，不是分数
        
        #对问题进行编码 
        query_embedding=self.encode_query(question)
        
        #判断是否存在sql筛选
        if len(doc_list)>0:
            #存在sql筛选，挑出限定的doc id list,这里默认的情况是如果存在sql筛选，就不开启bm25检索
            resized_embeddings=[]
            resized_doc_embeddings_docid=[]
            if self.config["use_bm25"]:
                for item in bm25_score_list:
                    if self.doc_embeddings_docid[item][1][0] in doc_list and len(resized_doc_embeddings_docid)<self.config["bm25_retrieval_top_k"] and item not in blacklist:
                        resized_doc_embeddings_docid.append(self.doc_embeddings_docid[item])
                        resized_embeddings.append(self.doc_embeddings[item])
                        return_itemid.append(item)
                if len(resized_embeddings)>0:
                    resized_embeddings=np.stack(resized_embeddings)
                else:
                    resized_embeddings=self.doc_embeddings[:self.config["bm25_retrieval_top_k"]]
                    resized_doc_embeddings_docid=self.doc_embeddings_docid[:self.config["bm25_retrieval_top_k"]]
                    return_itemid=list(range(len(resized_embeddings)))
                assert resized_embeddings.shape[0]==len(resized_doc_embeddings_docid)
            else:
                for itemid,item in enumerate(self.doc_embeddings_docid):
                    if item[1][0] in doc_list and len(resized_doc_embeddings_docid)<self.config["bm25_retrieval_top_k"] and itemid not in blacklist:
                        resized_doc_embeddings_docid.append(item)
                        resized_embeddings.append(self.doc_embeddings[itemid])
                        return_itemid.append(itemid)
                if len(resized_embeddings)>0:
                    resized_embeddings=np.stack(resized_embeddings)
                else:
                    resized_embeddings=self.doc_embeddings[:self.config["bm25_retrieval_top_k"]]
                    resized_doc_embeddings_docid=self.doc_embeddings_docid[:self.config["bm25_retrieval_top_k"]]
                    return_itemid=list(range(len(resized_embeddings)))
                assert resized_embeddings.shape[0]==len(resized_doc_embeddings_docid)

        else:
            #不存在sql筛选
            if self.config["use_bm25"]:
                resized_embeddings=[]
                resized_doc_embeddings_docid=[]
                for item in bm25_score_list:
                    if self.config["use_blacklist"]:
                        if item in blacklist:
                             continue
                    if len(resized_doc_embeddings_docid)<self.config["bm25_retrieval_top_k"]:
                        resized_embeddings.append(self.doc_embeddings[item])
                        resized_doc_embeddings_docid.append(self.doc_embeddings_docid[item])
                        return_itemid.append(item)
                try:
                    resized_embeddings=np.stack(resized_embeddings)
                except:
                    print(bm25_score_list)
                    print(len(resized_embeddings))
                    assert 1==2
                assert resized_embeddings.shape[0]==len(resized_doc_embeddings_docid)

            else:
                resized_embeddings=[]
                resized_doc_embeddings_docid=[]
                for itemid,item in enumerate(self.doc_embeddings_docid):
                    if self.config["use_blacklist"] and itemid in blacklist:
                        continue
                    resized_embeddings.append(self.doc_embeddings[itemid])
                    resized_doc_embeddings_docid.append(item)
                    return_itemid.append(itemid)
                resized_embeddings=np.stack(resized_embeddings)
                assert resized_embeddings.shape[0]==len(resized_doc_embeddings_docid)

            
        #计算相似度,这里的query_embedding和resized_embeddings都要先被改为list，才能传输
        #这里更改为只传query_embedding，document_embedding更改为只传选取的子列表id
        payload = {"method": "calculate_similarity", "query_embeddings": query_embedding.tolist(), "return_itemid": return_itemid}
        response = requests.post(self.embedding_url, json=payload,timeout=1000)
        if response.status_code == 200:
            similarity_score_list = response.json()["scores"]
        else:
            raise RuntimeError(f"Embedding API error: {response.text}")
        
        #分页参数
        if limit is None:
            k = int(self.config["retrieval_top_k"])  # page size
        else:
            k = int(limit)
        start = max(int(offset), 0)
        
        # 排序
        try:
                top_k_indices = sorted(
                    range(len(similarity_score_list)),
                    key=lambda i: similarity_score_list[i],
                    reverse=True
                )
                indexed_scores_ = sorted(enumerate(similarity_score_list), key=lambda x: x[1], reverse=True)
                return_itemid = [return_itemid[i] for i, score in indexed_scores_]
        except:
            print(type(sorted(
                range(len(similarity_score_list)),
                key=lambda i: similarity_score_list[i],
                reverse=True
            )))
            assert 1==2
        
        new_return_itemid=[]

        #判断是否使用rerank
        if self.config["use_rerank"]:
            pool_size = max(4*(start + k), 4*k)
            top_k_indices=top_k_indices[:pool_size]
            new_documentid=[]

            for i in top_k_indices:
                new_documentid.append(resized_doc_embeddings_docid[i])
                new_return_itemid.append(return_itemid[i])
            rerankscores=self.rerankmodel.rerank([[question,j[0]] for j in new_documentid])
            sorted_by_rerank = sorted(range(len(rerankscores)), key=lambda i: rerankscores[i], reverse=True)
            page_indices = sorted_by_rerank[start:start+k]
            new_documentid2=[]
            new_return_itemid2=[]
            for idx in page_indices:
                new_documentid2.append({"doc_id":new_documentid[idx][1][0],"content":new_documentid[idx][0],"match_score":rerankscores[idx],"chunk_type":new_documentid[idx][1][1],})
                new_return_itemid2.append(new_return_itemid[idx])
            new_documentid=new_documentid2
            new_return_itemid=new_return_itemid2
        else:
            page_indices = top_k_indices[start:start+k]
            new_documentid=[]

            for i in page_indices:
                new_documentid.append({"doc_id":resized_doc_embeddings_docid[i][1][0],"content":resized_doc_embeddings_docid[i][0],"chunk_type":resized_doc_embeddings_docid[i][1][1],"match_score":similarity_score_list[i]})
                new_return_itemid.append(return_itemid[i])
        
        return new_documentid, new_return_itemid



    def simple_retrieval(self,question,doc_list=[],blacklist=[], offset: int = 0, limit: int = None):
        """
        这个函数是用来进行简单的检索的，只使用dense retrieval，不使用bm25和rerank
        输入：
            question: 问题
            doc_list: 文档列表
            blacklist: 黑名单
            offset: 偏移量
            limit: 限制数量
        """

        #这里面虽然那传入了doc_list和blacklist，但是没有用到
        query_embedding=self.encode_query(question)
        payload = {"method": "calculate_similarity", "query_embeddings": query_embedding.tolist()}
        response = requests.post(self.embedding_url, json=payload,timeout=3000)
        if response.status_code == 200:
            similarity_score_list = response.json()["scores"]
        else:
            raise RuntimeError(f"Embedding API error: {response.text}")
        
        if limit is None:
            k = int(self.config["retrieval_top_k"])  # page size
        else:
            k = int(limit)
        start = max(int(offset), 0)
        
        ranked_indices = sorted(range(len(similarity_score_list)), key=lambda i: similarity_score_list[i], reverse=True)
        page_indices = ranked_indices[start:start+k]
        new_documentid=[]
        new_return_itemid=[]
        for i in page_indices:
            new_documentid.append({"doc_id":self.doc_embeddings_docid[i][1][0],"content":self.doc_embeddings_docid[i][0],"chunk_type":self.doc_embeddings_docid[i][1][1],"match_score":similarity_score_list[i]})
            new_return_itemid.append(i)
        return new_documentid,new_return_itemid

    def simple_bm25(self,question,doc_list=[],blacklist=[], offset: int = 0, limit: int = None):
        """
        这个函数是用来进行简单的bm25检索的，只使用bm25，不使用dense retrieval和rerank
        输入：
            question: 问题
            doc_list: 文档列表
            blacklist: 黑名单
            offset: 偏移量
            limit: 限制数量
        """
        #这里面虽然那传入了doc_list和blacklist，但是没有用到
        bm25_score_list=self.bm25.search(question,top_k=self.config["bm25_retrieval_top_k"])
        
        if limit is None:
            k = int(self.config["retrieval_top_k"])  # page size
        else:
            k = int(limit)
        start = max(int(offset), 0)
        
        ranked_indices = sorted(range(len(bm25_score_list)), key=lambda i: bm25_score_list[i], reverse=True)
        page_indices = ranked_indices[start:start+k]
        new_documentid=[]
        new_return_itemid=[]
        for i in page_indices:
            new_documentid.append({"doc_id":self.doc_embeddings_docid[i][1][0],"content":self.doc_embeddings_docid[i][0],"chunk_type":self.doc_embeddings_docid[i][1][1],"match_score":bm25_score_list[i]})
            new_return_itemid.append(i)
        return new_documentid,new_return_itemid


import re

def clean_illegal_chars(text):
    # 允许的控制字符：\x09 (tab), \x0A (LF), \x0D (CR)
    ILLEGAL_CHARACTERS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
    if isinstance(text, str):
        return ILLEGAL_CHARACTERS_RE.sub("", text)
    return text



if __name__ == "__main__":
    import pandas as pd
    # Load configuration
    # with open('config/config.yaml', 'r', encoding='utf-8') as f:
    #     config = yaml.safe_load(f)
    retrieval_result={}
    for embedding_model in ["qwen2_7b","bge_m3","jina"]:
        config={
            "embedding_model":embedding_model,
            "jina_v3_embedding_port":5438,
            "qwen2_7b_embedding_port":5437,
            "bge_m3_embedding_port":5439,
            "doc_embedding_name":f"/data4/students/zhangguangyin/chatNum/{embedding_model}_10000.pickle",
            "doc_embedding_idlistname":f"/data4/students/zhangguangyin/chatNum/{embedding_model}_10000_docid.json",
            "use_rerank":True,
            "use_bm25":True,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "retrieval_top_k": 10,
            "bm25_retrieval_top_k": 300,
            "bm25_index_name": "/data4/students/zhangguangyin/chatNum/bm25_10000_index.pkl",
            "use_blacklist": False,
            "small_trieval_range": False,
            "gpu_id":2
        }
        
        # Initialize Retriever
        retriever = Retriever(config)
        
        # Example query
        # user_query = "帮我查找关于基因表达调控的文章"
        # retrieved_docs = retriever.get_doc_list()
        
        # print(type(retrieved_docs["doc_list"]))
        # print(retrieved_docs["doc_list"][0])
        # print(retrieved_docs["total_num"])
        # print("检索到的文档:")
        query=""""In the paper where the proposed method achieves a value of 0.898 for metric SSIM on the Video Prediction task of dataset Moving MNIST (Moving MNIST), What is the MSE value when the gradient highway unit (GHU) is injected at the bottom layer (k1=1, k2=2) in the 4-layer causal LSTM network?\n\n  ",
            """
        retrieved_docs,_ = retriever.retrieve(query)
        retrieval_list=[]
        for doc in retrieved_docs:
            retrieval_list.append(clean_illegal_chars(doc["content"]))
        retrieval_result[embedding_model]=retrieval_list

    pd.DataFrame(retrieval_result).to_excel("retrieval_result.xlsx",index=False)






