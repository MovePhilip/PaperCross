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

from colpali_engine.models import ColPali, ColPaliProcessor

import logging

logger = logging.getLogger("myapp")





class embedder_colpali:
    def __init__(self,config):
        # BAAI/bge-m3
        self.config = config
        model_name = "vidore/colpali-v1.2-merged"

        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir="/data4/students/zhangguangyin/LLMs"
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name,cache_dir="/data4/students/zhangguangyin/LLMs")        
        self.gpu_id = config["gpu_id"]  # 指定你想使用的GPU编号
        self.device = torch.device(f"cuda:{self.gpu_id}")

        with open(self.config["image_embedding_name"],"rb")as f:
            self.image_embedding_list=pickle.load(f)
        print(f"image_embedding_list.shape:{len(self.image_embedding_list)}")
        self.image_embedding_list=torch.tensor(self.image_embedding_list,dtype=torch.float32).to(self.device)
        

        self.model=self.model.to(self.device)

    def embed_document(self,images):
        # Process the inputs
        #将images分批处理，每200张图片处理一次
        print(f"len(images):{len(images)}")
        batch_size=65
        image_embeddings_list=[]
        for i in range(0,len(images),batch_size):
            batch_images=images[i:i+batch_size]
            imagelist=[]
            for image in batch_images:
                imagelist.append(Image.open(image))#.convert('RGB')
            batch_images = self.processor.process_images(imagelist).to(self.model.device)
            with torch.no_grad():
                image_embeddings = self.model(**batch_images)
            logger.info(f"i:{i}")
            print(image_embeddings.shape)
            image_embeddings_list.append(image_embeddings)
            # with open(self.config["image_embedding_name"],"wb")as f:
            #     pickle.dump(image_embeddings_list,f)
        image_embeddings_list = torch.cat(image_embeddings_list, dim=0).cpu()
        print(f"len(image_embeddings_list):{len(image_embeddings_list)}")
        return image_embeddings_list

    


    def embed_query(self,query_list):
        batch_queries = self.processor.process_queries([query_list,]).to(self.model.device)

        # Forward pass
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)   
        logger.info(f"query_embeddings:{type(query_embeddings)}")
        logger.info(f"query_embeddings.shape:{query_embeddings.shape}")     
        return query_embeddings#[0]


    def calculate_similarity(self,query_embeddings,image_embeddings=None):
        if image_embeddings is None:
            logger.info(f"query_embeddings.shape:{query_embeddings.shape}")
            query_embeddings=torch.tensor(query_embeddings,dtype=torch.float32).to(self.device)
            scores = self.processor.score_multi_vector(query_embeddings, self.image_embedding_list)
        else:
            query_embeddings=torch.tensor(query_embeddings,dtype=torch.float32).to(self.device)
            image_embeddings=torch.tensor(image_embeddings,dtype=torch.float32).to(self.device)
            scores = self.processor.score_multi_vector(query_embeddings, image_embeddings)
        return scores.tolist()[0]
    
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
        "gpu_id":0,
        "image_embedding_name":"/data4/students/zhangguangyin/chatNum/colpali_v1.3.pickle"
    }

    embedding_model=embedder_colpali(config)
    with open("/data4/students/zhangguangyin/chatNum/colpali_v1.3_docid.json","r")as f:
        image_embedding_idlist=json.load(f)
    print(f"len(image_embedding_idlist):{len(image_embedding_idlist)}")
    # image_path="/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1000/pdf_images/page_1_dpi600.png"
    # image=Image.open(image_path)
    query="""Please help me find the document images that has report the performance of methods  on the FID-8 metric for the Text-to-Image Generation task on the COCO (COCO) dataset"""
    query_embed=embedding_model.embed_query(query)
    scores=embedding_model.calculate_similarity(query_embed)
    # print(f"scores:{scores}")
    argsort=np.argsort(scores)[:15]#[::-1]
    top_scores = np.array(scores)[argsort]
    print(f"argsort:{argsort}")
    print(f"top_scores:{top_scores}")

    for i in range(len(argsort)):
        print(f"index:{argsort[i]}, score:{top_scores[i]:.4f}, image_embedding_idlist:{image_embedding_idlist[argsort[i]]}")






