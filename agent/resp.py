import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from retrieval.retriever import Retriever
from generation.model_generation import generationmodel
import logging

logger = logging.getLogger("myapp")


@dataclass
class Document:
    """文档数据结构"""
    content: str
    title: str = ""
    doc_id: str = ""

class ReSPFramework:
    """ReSP (Retrieve, Summarize, Plan) 框架实现"""
    
    def __init__(self, 
                 config: Dict,
                 llm_model: str = "gpt-3.5-turbo",
                 retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_iterations: int = 3,
                 top_k: int = 5,
                 max_input_length: int = 12000,
                 max_output_length: int = 2048):
        """
        初始化ReSP框架
        
        Args:
            llm_model: 大语言模型名称
            retriever_model: 检索模型名称  
            max_iterations: 最大迭代次数
            top_k: 每次检索的文档数量
            max_input_length: 最大输入长度
            max_output_length: 最大输出长度
        """
        self.generation_model = generationmodel(config)
        self.max_iterations = config["max_iterations"]
        self.top_k = top_k
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # 初始化检索器
        self.retriever = Retriever(config) #这个地方也再更换下
        
        # 记忆队列
        self.global_evidence_memory = []  # 全局证据记忆
        self.local_pathway_memory = []    # 局部路径记忆
        self.retrieved_subquestions = []  # 已检索的子问题
        
        # 文档库
        self.document_corpus = []
        self.document_embeddings = None
        self.visited_itemid=set()

        self.retrieval_list=[]
        self.config=config

    def _call_llm(self, prompt: str) -> str:

        messages=[{"role": "user", "content": prompt}]
        """调用大语言模型"""
        try:
            response = self.generation_model.generate(messages)
            return response
        except Exception as e:
            return f"调用LLM时出错: {str(e)}"


    def retrieve_documents(self, query: str):
        """检索相关文档
        
        Args:
            query: 查询问题
        
        Returns:
            List[Dict]: 
            {
                "doc_id": [list of str],
                "match_score": float,
                "content": str,
                "chunk_type": str
            }
        
        
        """

        
        # 编码查询
        if self.config["use_blacklist"]:
            retrieved_docs,return_itemid = self.retriever.retrieve(query,doc_list=self.retrieval_list,blacklist=self.visited_itemid)
        else:
            retrieved_docs,return_itemid = self.retriever.retrieve(query,doc_list=self.retrieval_list)
        self.visited_itemid.update(return_itemid)
        
        return retrieved_docs

    def reasoner_judge(self, overarching_question: str) -> bool:
        """推理器判断模块：判断当前信息是否足够回答问题"""
        combined_memory = self._combine_memory_queues()
        
        prompt = f"""Judging based solely on the current known information and without allowing for inference, are you able to completely and accurately respond to the question {overarching_question}? 
Known information: {combined_memory}
If you can, please reply with 'Yes' directly; if you cannot and need more information, please reply with 'No' directly."""
        
        response = self._call_llm(prompt)
        return response.lower().startswith('yes')

    def reasoner_plan(self, overarching_question: str) -> str:
        """推理器规划模块：生成下一个子问题"""
        combined_memory = self._combine_memory_queues()
        retrieved_questions_str = "; ".join(self.retrieved_subquestions)
        
        prompt = f"""You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. Please understand the information gap between the currently known information and the target problem. Your task is to generate one thought in the form of question for next retrieval step directly.
DON'T generate the whole thoughts at once!
DON'T generate thought which has been retrieved.
[Known information]: {combined_memory}
[Target question]: {overarching_question}
[Previously retrieved questions]: {retrieved_questions_str}
[Your Thought]:"""
        
        sub_question = self._call_llm(prompt)
        
        # 清理输出，提取问题
        sub_question = sub_question.strip()
        if sub_question and sub_question not in self.retrieved_subquestions:
            self.retrieved_subquestions.append(sub_question)
        
        return sub_question

    def summarizer_global_evidence(self, docs: List[Document], overarching_question: str) -> str:
        """摘要器全局证据模块：为总体问题生成支持性摘要"""
        docs_text = "\n\n".join([f"Paper id: {doc["doc_id"]}:  {doc["content"]}" for i, doc in enumerate(docs)])
        logger.info(f"docs_text: {docs_text}")
        prompt = f"""Passages: {docs_text}
Your job is to act as a professional writer. You will write a good-quality passage that can support the given prediction about the question only based on the information in the provided supporting passages. Now, let's start. After you write, please write [DONE] to indicate you are done. Do not write a prefix (e.g., 'Response:') while writing a passage.
Question: {overarching_question}
Passage:"""
        
        response = self._call_llm(prompt)
        # 移除[DONE]标记
        response = response.replace('[DONE]', '').strip()
        return response

    def summarizer_local_pathway(self, sub_question: str) -> str:
        """摘要器局部路径模块：为当前子问题生成回答"""
        combined_memory = self._combine_memory_queues()
        
        prompt = f"""Judging based solely on the current known information and without allowing for inference, are you able to respond completely and accurately to the question {sub_question}? 
Known information: {combined_memory}
If yes, please reply with 'Yes', followed by an accurate response to the question {sub_question}, without restating the question; if no, please reply with 'No' directly."""
        
        response = self._call_llm(prompt)
        return response

    def generator(self, overarching_question: str) -> str:
        """生成器：基于记忆队列生成最终答案"""
        combined_memory = self._combine_memory_queues()
        
        prompt = f"""Answer the question based on the given reference.
Only give me the answer and do not output any other words.
The following are given reference: {combined_memory}
Question: {overarching_question}"""
        
        answer = self._call_llm(prompt)
        return answer

    def _combine_memory_queues(self) -> str:
        """合并记忆队列，就是把所有的字符串直接合并"""
        combined = []
        
        if self.global_evidence_memory:
            combined.append("Global Evidence:")
            combined.extend(self.global_evidence_memory)
        
        if self.local_pathway_memory:
            combined.append("Local Pathway:")
            combined.extend(self.local_pathway_memory)
        
        return "\n".join(combined)

    def reset_memory(self):
        """重置记忆队列"""
        self.global_evidence_memory = []
        self.local_pathway_memory = []
        self.retrieved_subquestions = []

    def run(self, question: str,retrieval_list=[]) -> Dict:
        """
        回答多跳问题的主函数
        
        Args:
            question: 输入的问题
            
        Returns:
            包含答案和过程信息的字典
        """
        self.retrieval_list=retrieval_list
        self.reset_memory()
        self.visited_itemid=set()
        
        result = {
            'question': question,
            'answer': '',
            'iterations': [],
            'total_iterations': 0
        }
        
        current_question = question
        self.docidlist_iteration={}
        for iteration in range(self.max_iterations):
            logger.info(f"\n=== 迭代 {iteration + 1} ===")
            logger.info(f"当前问题: {current_question}")
            
            # 步骤1: 检索文档
            retrieved_docs = self.retrieve_documents(current_question)
            logger.info(f"检索到 {len(retrieved_docs)} 个文档")
            
            # 步骤2: 摘要器处理
            if retrieved_docs:
                templist=set()
                for doc in retrieved_docs:
                    templist.add(doc["doc_id"])
                self.docidlist_iteration[iteration]=list(templist)

                # 全局证据摘要
                global_summary = self.summarizer_global_evidence(retrieved_docs, question)
                logger.info(f"全局证据摘要: {global_summary}")
                if global_summary:
                    self.global_evidence_memory.append(global_summary)
                
                # 局部路径摘要（第一次迭代跳过，因为子问题就是原问题）
                if iteration > 0:
                    local_response = self.summarizer_local_pathway(current_question)
                    pathway_entry = f"Q: {current_question} | A: {local_response}"
                    self.local_pathway_memory.append(pathway_entry)
            
            # 步骤3: 推理器判断
            is_sufficient = self.reasoner_judge(question)
            print(f"信息是否充分: {is_sufficient}")
            
            iteration_info = {
                'iteration': iteration + 1,
                'sub_question': current_question,
                'retrieved_docs_count': len(retrieved_docs),
                'is_sufficient': is_sufficient,
                'global_evidence': global_summary if retrieved_docs else "",
                'local_pathway': pathway_entry if iteration > 0 and retrieved_docs else ""
            }
            
            if is_sufficient:
                # 生成最终答案
                final_answer = self.generator(question)
                logger.info(f"最终答案: {final_answer}")
                
                result['answer'] = final_answer
                result['total_iterations'] = iteration + 1
                result['iterations'].append(iteration_info)
                break
            else:
                # 生成下一个子问题
                next_question = self.reasoner_plan(question)
                logger.info(f"下一个子问题: {next_question}")
                
                iteration_info['next_sub_question'] = next_question
                result['iterations'].append(iteration_info)
                
                if next_question:
                    current_question = next_question
                else:
                    # 无法生成新的子问题，强制结束
                    result['answer'] = self.generator(question)
                    result['total_iterations'] = iteration + 1
                    break
        
        # 如果达到最大迭代次数，强制生成答案
        if not result['answer']:
            result['answer'] = self.generator(question)
            result['total_iterations'] = self.max_iterations
        
        return result['answer'],self.docidlist_iteration


if __name__ == "__main__":
    main()