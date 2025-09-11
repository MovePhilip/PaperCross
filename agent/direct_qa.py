import torch
import os
import json
import yaml
from generation.model_generation import generationmodel
from retrieval.retriever import Retriever
from embedding.split_doc import embeddoc
import logging

logger = logging.getLogger("myapp")

class DirectQA:
    def __init__(self, config, retriever=None, generation_model=None):
        self.config = config
        num_gpus = torch.cuda.device_count()
        logger.info("当前可用GPU数量:", num_gpus)
        
        if not os.path.exists(self.config["doc_embedding_name"]):
            doc_embeddings, new_embedding_docid = embeddoc(self.config)
        
        if generation_model is None:
            self.generationmodeler = generationmodel(self.config)
        else:
            self.generationmodeler = generation_model


    def run(self, question,docidlist=None,retrieval_list=[]):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            # {"role": "assistant", "content": "Thought:","prefix": True}
        ]

        outputtext = self._call_llm(messages)
        # logger.info(outputtext)
        return outputtext,retrieval_list
    
    def _call_llm(self, messages):
        """调用大语言模型"""
        try:
            response = self.generationmodeler.generate(messages)
            logger.info(f"model generation content  : {response}")
            return response
            
        except Exception as e:
            return f"调用LLM时出错: {str(e)}"



