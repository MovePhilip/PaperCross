import os
import json
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"]="5"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import torch
from sentence_transformers import SentenceTransformer
import json_repair
import pickle
from typing import cast
import pickle
import torch
import numpy as np
from PIL import Image


import logging

logger = logging.getLogger("myapp")





class embedder_gme:
    def __init__(self,config):
        # BAAI/bge-m3
        self.config = config

        self.model = SentenceTransformer("Alibaba-NLP/gme-Qwen2-VL-7B-Instruct", trust_remote_code=True,device='cpu',cache_folder="/data4/students/zhangguangyin/LLMs")

        self.gpu_id = config["gpu_id"]  # 指定你想使用的GPU编号
        self.device = torch.device(f"cuda:{self.gpu_id}")

        # with open(self.config["gme_image_embedding_name"],"rb")as f:
        #     self.image_embedding_list=pickle.load(f)
        # print(f"image_embedding_list.shape:{len(self.image_embedding_list)}")
        # self.image_embedding_list=torch.tensor(self.image_embedding_list,dtype=torch.float32).to(self.device)
        with open(self.config["image_embedding_name"],"rb")as f:
            self.image_embedding_list=pickle.load(f)
        print(f"image_embedding_list.shape:{len(self.image_embedding_list)}")
        self.image_embedding_list=torch.tensor(self.image_embedding_list,dtype=torch.float32).to(self.device)

        self.model=self.model.to(self.device)

    def embed_document(self,images):
        # Process the inputs
        #将images分批处理，每200张图片处理一次
        subimages=[]
        for i in range(0,len(images),20):
            subimages.append(images[i:i+20])
        image_embeddings_list=[]
        for idx,subimages in enumerate(subimages):
            image_embeddings = self.model.encode([dict(image=i) for i in subimages], convert_to_tensor=True)
            image_embeddings_list.append(image_embeddings)
            with open(f"/data4/students/zhangguangyin/chatNum/embedding/gme_embedding/{idx}_7B.pickle","wb")as f:    
                pickle.dump(image_embeddings,f)
            print(f"idx:{idx}")
        
        image_embedding_total = torch.cat(image_embeddings_list, dim=0).cpu()
        print(f"len(image_embeddings_list):{len(image_embedding_total)}")
        return image_embedding_total

    


    def embed_query(self,query):
        query=[query,]
        t2i_prompt = "Find an document image that is relevant to the given requests"
        query_embeddings = self.model.encode([dict(text=t, prompt=t2i_prompt) for t in query], convert_to_tensor=True)

        # query_embeddings = self.model.encode(query, convert_to_tensor=True)   
        logger.info(f"query_embeddings:{type(query_embeddings)}")
        logger.info(f"query_embeddings.shape:{query_embeddings.shape}")     
        return query_embeddings#[0]


    def calculate_similarity(self,query_embeddings,image_embeddings=None):
        if image_embeddings is None:
            logger.info(f"query_embeddings.shape:{query_embeddings.shape}")
            query_embeddings=torch.tensor(query_embeddings,dtype=torch.float32).to(self.device)
            scores = (query_embeddings@self.image_embedding_list.T).tolist()
        else:
            query_embeddings=torch.tensor(query_embeddings,dtype=torch.float32).to(self.device)
            image_embeddings=torch.tensor(image_embeddings,dtype=torch.float32).to(self.device)
            scores = (query_embeddings@image_embeddings.T).tolist()
        return scores[0]#.tolist()[0]



def craft():
    #
    with open("/data4/students/zhangguangyin/chatNum/colpali_v1.3.pickle","rb")as f:
        image_embedding_list=pickle.load(f)
    #合并image_embedding_list，维度n*(60,1031,128)(list of torch.tensor)变为(n*60,1031,128)
    image_embedding_list = [t.cpu() for t in image_embedding_list]
    image_embedding_list = torch.cat(image_embedding_list, dim=0)
    print(f"image_embedding_list.shape:{image_embedding_list.shape}")
    with open("/data4/students/zhangguangyin/chatNum/colpali_v1.3_craft.pickle","wb")as f:
        pickle.dump(image_embedding_list,f)
    return image_embedding_list


if __name__ == "__main__":
    config={
        "gpu_id":6,
        "gme_image_embedding_name":"/data4/students/zhangguangyin/chatNum/embedding_pkl/gme_image_embedding.pickle"
    }

    embedding_model=embedder_gme(config)
    with open("/data4/students/zhangguangyin/chatNum/embedding_pkl/gme_image_embedding_docid.json","r")as f:
        image_embedding_idlist=json.load(f)
    print(f"len(image_embedding_idlist):{len(image_embedding_idlist)}")


    query="Please help me find the performance of methods proposed in different papers on the FID-8 metric for the Text-to-Image Generation task on the COCO (COCO) dataset, list the top three metric result. \n        Requirements:\n        1.for each paper, you should only give one metric result (the highest one) of its own proposed method, since most papers will compare the performance of methods proposed in other papers and some variants of the same method in abalation study. \n        2. The top three metric result means that you need to find at least three papers that have reported the metric result .\n        \n                            "
    query_embed=embedding_model.embed_query(query)
    scores=embedding_model.calculate_similarity(query_embed)
    # print(f"scores:{scores}")
    #获取scores中最大的10个值的索引和值
    argsort=np.argsort(scores[0])[::-1][:15]
    top_scores = np.array(scores[0])[argsort]
    print(f"argsort:{argsort}")
    print(f"top_scores:{top_scores}")

    for i in range(len(argsort)):
        print(f"index:{argsort[i]}, score:{top_scores[i]:.4f}, image_embedding_idlist:{image_embedding_idlist[argsort[i]]}")







