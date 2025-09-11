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
    def __init__(self, config, retriever=None, generationmodel=None):
        self.config = config
        num_gpus = torch.cuda.device_count()
        logger.info("当前可用GPU数量:", num_gpus)
        
        if not os.path.exists(self.config["doc_embedding_name"]):
            doc_embeddings, new_embedding_docid = embeddoc(self.config)
        
        if generationmodel is None:
            self.generationmodeler = generationmodel(self.config)
        else:
            self.generationmodeler = generationmodel
        if retriever is None:
            self.retirevalmodel = Retriever(self.config)
        else:
            self.retirevalmodel = retriever

    def run(self, question):
        retrieval_list = self.retirevalmodel.retrieve(question)
        prompt = ""
        for key, value in retrieval_list.items():
            prompt += f"\nPaper id:{key}\n"
            for itemid, item in enumerate(value):
                prompt += f"segment{itemid} {item[1]}\n\n"

        prompt = question + "\nThe following are relavant context you can refer to:\n" + prompt
        outputtext = self.generationmodeler.generate(prompt)
        logger.info(outputtext)
        return outputtext