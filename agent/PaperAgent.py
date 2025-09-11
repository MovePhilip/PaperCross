#这个agent是用来回答关于论文的问题的,它需要先检索出相关的论文,然后输入论文的全文内容回答问题

import os
import json
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import openai
from sentence_transformers import SentenceTransformer

from retrieval.retriever import Retriever
from generation.model_generation import generationmodel
import logging  

logger=logging.getLogger("myapp")

class PaperAgent:
    """PaperQA主代理"""
    
    def __init__(self, config):
        # 初始化不同的LLM实例
        self.generation_model = generationmodel(config)
        self.fulltext_list = json.load(open(config["fulltext_list_name"],"r"))
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用大语言模型"""
        try:
            response = self.generation_model.generate(messages)
            return response
        except Exception as e:
            return f"调用LLM时出错: {str(e)}"
    
    def run(self, question: str, docidlist: List[str]) -> str:
        """运行PaperQA代理"""
        print(f"🤖 开始处理问题: {question}")
        
        # 重置状态
        self.context_library = []
        self.papers = []
        # retrieval_query=self._call_llm([{"role": "user", "content": f"Please generate a query for retrieving chunks of computer science research papers based on the following question. The query can be a series of keywords or a sentence. Do not generate any other additional characters: {question}"}])
        # # 初始搜索
        # retrieval_docs=self.retriever.retrieve(retrieval_query)
        # docidlist=set()
        # for item in retrieval_docs:
        #     docidlist.add(item["doc_id"])
        responsedict={}

        for docid in docidlist:
            fulltext=self.fulltext_list[docid][:180000]
            try:
                prompt=f"""
                You are given a question and a full text of computer science research papers.
                You need to firstly identify which information is relevant to the question, and then extract and summarize all the relevant information from the full text . You don't need to answer this question, just extract and summarize all the useful information for the question. 
                Question: {question}
                Paper id: {docid}
                Full text: {fulltext}

                """
                response=self._call_llm([{"role": "user", "content": prompt}])
                responsedict[docid]=response
                logger.info(f"Doc id: {docid} Relevant information: {response}")
            except Exception as e:
                logger.info(f"Doc id: {docid} Error: {e}")
                continue
        
        prompt=""
        for docid,response in responsedict.items():
            prompt+=f"""
            Doc id: {docid}
            Relevant information: {response}
            """
        prompt=f"""I give you a question and a list of relevant information from different papers. You need to answer the question based on the relevant information.
        Question: {question}
        """+prompt
        response=self._call_llm([{"role": "user", "content": prompt}])
        return response














