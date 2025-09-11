#这个agent是用来回答关于论文的问题的,它需要先检索出相关的论文图片,然后输入论文的全文内容回答问题

import os
import json
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import openai
from sentence_transformers import SentenceTransformer
import pickle
from retrieval.retriever import Retriever
from generation.model_generation import generationmodel
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import concurrent.futures  # 新增并行处理
from embedding.embedding_colpali import embedder_colpali
from embedding.embedding_gme import embedder_gme
import logging  

logger=logging.getLogger("myapp")







class visual_agent:
    def __init__(self,config):
        self.config=config
        self.config["generation_model"]="deepseek_zijie"
        self.text_generation_model=generationmodel(config)
        self.config["generation_model"]="qwen_vl"
        self.visual_generation_model=generationmodel(config)
        self.config["generation_model"]="deepseek_zijie"


        # self.retriever=Retriever(config)
        if config["embedding_model"]=="colpali":
            self.embedder=embedder_colpali(config)
        elif config["embedding_model"]=="gme":
            self.embedder=embedder_gme(config)
        self.fulltext_list=json.load(open(config["fulltext_list_name"],"r"))
        with open(config["image_embedding_name"],"rb")as f:
            self.image_embedding_list=pickle.load(f)
        with open(config["image_embedding_idlistname"],"r")as f:
            self.image_embedding_idlist=json.load(f)
        self.visual_extraction_prompt="""Now i give you some PDF document images and a question. You need to find all the relevant information to the question in these PDF images and make a summary of these relavant information. The summary should be a segment of free text. You don't need to answer the question in here, you just need to summarize the relevant information in these images."""
        self.text_prompt="""Now i give you a question and some summary segments of different PDF documents, each summary segment is corresponding to a PDF document. You need to give a final answer to the question based on these summary segments. Remember to follow the answer format."""

    


    def run(self,query,retrieval_list=None):
        #获取相关的10张图片，然后输入到视觉模型中，最后再然后输入到文本模型中，然后输出答案
        #1.获取相关的10张图片
        logger.info(f"[visual_agent] query:{query}")
        try:
            query_embedding=self.embedder.embed_query(query)

            print(f"type(query_embedding):{type(query_embedding)}")
            score_list=self.embedder.calculate_similarity(query_embedding)
        
            #将score_list按照从大到小排序，然后取前10个
            score_list_ranked=np.argsort(score_list)[::-1][:self.config["retrieval_top_k"]]
        except Exception as e:
            logger.exception(f"[visual_agent] Error calculating similarity: {e}")
            return "Error: Failed to calculate similarity"
        
        print(f"score_list_ranked:{score_list_ranked}")
        logger.info(f"[visual_agent] score_list_ranked:{score_list_ranked}")
    
        try:
            doc_id_list={}
            just_image_list=[]
            for i in score_list_ranked:
                if self.image_embedding_idlist[i][0] not in doc_id_list:
                    doc_id_list[self.image_embedding_idlist[i][0]]=[]
                doc_id_list[self.image_embedding_idlist[i][0]].append([self.image_embedding_idlist[i][1],f"/data4/students/zhangguangyin/chatNum/rank200_pdf_images/{self.image_embedding_idlist[i][0]}/pdf_images/page_{self.image_embedding_idlist[i][1]}_dpi600.png"])
                just_image_list.append(self.image_embedding_idlist[i])            
            logger.info(f"[visual_agent] doc_id_list:{just_image_list}")
        except Exception as e:
            logger.exception(f"[visual_agent] Error getting doc_id_list: {e}")
            return "Error: Failed to get doc_id_list"
        
        #对doc_id_list每个value进行排序，按照value中每个元素的第一个值进行排序，升序排列
        for key,value in doc_id_list.items():
            doc_id_list[key]=sorted(value,key=lambda x:int(x[0]))


        #按照paper id进行分组，每组图片输入到视觉模型中，然后输入到文本模型中，然后输出答案
        #将下面这个改为并行处理，发送多个请求，然后等待所有请求返回，然后合并结果
        summary_dict = {}

        # 构造并行执行的函数
        def build_and_generate(args):
            key, value = args
            messages = []
            for i in value:
                messages.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(i[1])}"}})
            messages.append({"type": "text", "text": self.visual_extraction_prompt + f"\n\n The paper id of this image is: {key}" + f"\n\n The question is: {query}"})

            messagestotal=[
            {
                "role": "system",
                "content": [{"type":"text","text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": messages
            }
            ]
            logger.info(f"[visual_agent] Sending visual generation request for paper {key} with {len(value)} images")
            visual_answer = self.visual_generation_model.generate(messagestotal)
            return key, visual_answer
        


        # 使用线程池并行发送请求
        max_workers = min(8, len(doc_id_list)) or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(build_and_generate, item) for item in doc_id_list.items()]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    key, answer = fut.result()
                    summary_dict[key] = answer
                except Exception as e:
                    logger.exception(f"Error processing visual generation for a paper: {e}")

        logger.info(f"[visual_agent] summary_dict:{summary_dict}")


        #根据获得到的summary_dict，生成最终的答案
        prompt=self.text_prompt
        prompt+=f"\n\n The question is: {query}"
        for key,value in summary_dict.items():
            prompt+=f"\n\n The useful summary segment from Paper id {key} is: {value}"
        prompt_list=[
            {
                "role":"system",
                "content":[{"type":"text","text":"You are a helpful assistant."}]
            },
            {
                "role":"user",
                "content":prompt
            }
        ]
        
        text_answer=self.text_generation_model.generate(prompt_list)


        return text_answer,{'just_image_list':just_image_list,'summary_dict':summary_dict}

    
    
def encode_image(image_path, max_side: int = 1000):
    """
    1400输入10张时没问题的，但是幻觉严重到不行，没法用

    读取图片，若最长边超过 max_side（默认 1600px）则按比例缩放，
    再返回 base64 编码后的 PNG 字节串。最终图片 DPI 设为 300。
    """
    with Image.open(image_path) as img:
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG", dpi=(300, 300))
        return base64.b64encode(buf.getvalue()).decode("utf-8")


    
    
    
    
if __name__ == "__main__":
    config={
        "model":"qwen2_7b",
        "tokenizer":"qwen2_7b",
        "device":"cuda",
        "max_length":1024,
        "temperature":0.7,
        "top_p":0.95,
        "top_k":50
    }




