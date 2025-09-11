import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM  # 需要安装transformers
from retrieval.retriever import Retriever
from generation.model_generation import generationmodel
import logging

logger = logging.getLogger("myapp")



@dataclass
class Paragraph:
    """表示一个检索到的段落"""
    title: str
    content: str
    score: float = 0.0

class BaseRetriever(ABC):
    """基础检索器抽象类"""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Paragraph]:
        """检索相关段落"""
        pass

class BM25Retriever(BaseRetriever):
    """BM25检索器的简化实现"""
    
    def __init__(self, corpus: List[Dict[str, str]]):
        """
        初始化BM25检索器
        corpus: [{"title": "标题", "content": "内容"}, ...]
        """
        self.corpus = corpus
        self.documents = [doc["content"] for doc in corpus]
        self.titles = [doc["title"] for doc in corpus]
        
        # 使用TF-IDF作为BM25的简化实现
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)
    
    def retrieve(self, query: str, k: int = 5) -> List[Paragraph]:
        """使用TF-IDF相似度检索文档"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 只返回有相关性的结果
                results.append(Paragraph(
                    title=self.titles[idx],
                    content=self.documents[idx],
                    score=similarities[idx]
                ))
        
        return results

class LanguageModel(ABC):
    """语言模型抽象类"""
    
    @abstractmethod
    def generate_cot_sentence(self, prompt: str) -> str:
        """生成chain-of-thought句子"""
        pass
    
    @abstractmethod
    def answer_question(self, prompt: str) -> str:
        """回答问题"""
        pass

class OpenAIModel(LanguageModel):
    """OpenAI GPT模型封装"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key
    
    def generate_cot_sentence(self, prompt: str) -> str:
        """生成CoT句子"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "I need to think about this step by step."
    
    def answer_question(self, prompt: str) -> str:
        """回答问题"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "Unable to generate answer."

class FlanT5Model(LanguageModel):
    """Flan-T5模型封装"""
    
    def __init__(self, model_name: str = "google/flan-t5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            do_sample=False
        )
    
    def generate_cot_sentence(self, prompt: str) -> str:
        """生成CoT句子"""
        try:
            # 为Flan-T5格式化提示
            formatted_prompt = f"Answer +the following question by reasoning step-by-step. {prompt}"
            result = self.generator(formatted_prompt, max_length=150, num_return_sequences=1)
            generated_text = result[0]['generated_text'].strip()
            
            # 只取第一句话
            sentences = generated_text.split('.')
            if sentences:
                return sentences[0].strip() + '.'
            return generated_text
        except Exception as e:
            print(f"Flan-T5 generation error: {e}")
            return "I need to think about this step by step."
    
    def answer_question(self, prompt: str) -> str:
        """回答问题"""
        try:
            result = self.generator(prompt, max_length=200, num_return_sequences=1)
            return result[0]['generated_text'].strip()
        except Exception as e:
            print(f"Flan-T5 generation error: {e}")
            return "Unable to generate answer."

class IRCoTQA:
    """IRCoT问答系统 - 集成了检索和推理的完整流程"""
    
    def __init__(self, 
                 config,
                 retriever: BaseRetriever,
                 language_model: LanguageModel,
                 k: int = 4,  # 每步检索的段落数
                 max_steps: int = 8,  # 最大推理步数
                 max_paragraphs: int = 15,  # 最大段落数
                 use_cot_reader: bool = True):

        self.k = k
        self.max_steps = max_steps
        self.max_paragraphs = max_paragraphs
        self.use_cot_reader = use_cot_reader
        
        # 示例演示，实际应用中应该从训练数据中获取
        self.demonstrations = self._get_demonstrations()
        self.generation_model = generationmodel(config)
        self.retriever = Retriever(config) #这个地方也再更换下
    
    def _get_demonstrations(self) -> str:
        """获取in-context学习的演示例子"""
        demos = """
            Wikipedia Title: Mack Rides
            Mack Rides GmbH & Co KG, also known as Mack Rides, is a German amusement ride manufacturer.

            Q: In what country was Lost Gravity manufactured?
            A: The Lost Gravity was manufactured by Mack Rides. Mack Rides is a company from Germany. The answer is: Germany.

            Wikipedia Title: Murray Head
            Murray Seafield St George Head is an English actor and singer.

            Q: Who wrote the 1970 international hit song that Murray Head is most recognized for?
            A: The 1970 international hit song that Murray Head is most recognized for is "Superstar". "Superstar" was written by Andrew Lloyd Webber and Tim Rice. So the answer is: Andrew Lloyd Webber and Tim Rice.
            """
        return demos.strip()
    
    def _format_paragraphs(self, paragraphs: List[Paragraph]) -> str:
        """格式化段落为prompt格式"""
        formatted = []
        for para in paragraphs:
            formatted.append(f"Wikipedia Title: {para.title}\n{para.content}")
        return "\n\n".join(formatted)
    
    def _create_reason_prompt(self, 
                            question: str, 
                            paragraphs: List[Paragraph], 
                            cot_sentences: List[str]) -> str:
        """创建推理步骤的prompt"""
        para_text = self._format_paragraphs(paragraphs)
        cot_text = " ".join(cot_sentences) if cot_sentences else ""
        
        prompt = f"""{self.demonstrations}

                    {para_text}

                    Q: {question}
                    A: {cot_text}"""
        return prompt
    
    def _extract_first_sentence(self, text: str) -> str:
        """提取生成文本的第一句话"""
        # 移除常见的前缀
        text = text.strip()
        if text.startswith("A: "):
            text = text[3:].strip()
        
        # 分割句子
        sentences = re.split(r'[.!?]+', text)
        if sentences and sentences[0].strip():
            return sentences[0].strip() + "."
        return text.split('.')[0].strip() + "." if text else ""
    
    def _is_final_answer(self, sentence: str) -> bool:
        """检查是否是最终答案"""
        answer_patterns = [
            r"answer is[:：]\s*",
            r"so the answer is[:：]\s*",
            r"therefore[,，]\s*the answer is[:：]\s*"
        ]
        
        sentence_lower = sentence.lower()
        for pattern in answer_patterns:
            if re.search(pattern, sentence_lower):
                return True
        return False
    
    def _ircot_retrieve(self, question: str) -> Tuple[List[Paragraph], List[str]]:
        """
        IRCoT检索过程 - 交替进行检索和推理
        返回: (检索到的段落列表, CoT推理句子列表)
        """
        # 步骤1: 使用问题进行初始检索
        collected_paragraphs = self.retriever.retrieve(question)
        cot_sentences = []
        
        logger.info(f"Initial retrieval: {len(collected_paragraphs)} paragraphs")
        
        # 步骤2: 交替进行推理和检索
        for step in range(self.max_steps):
            logger.info(f"\n--- Step {step + 1} ---")
            
            # 推理步骤: 生成下一个CoT句子
            reason_prompt = self._create_reason_prompt(question, collected_paragraphs, cot_sentences)
            generated_text = self.language_model.generate_cot_sentence(reason_prompt)
            
            # 提取第一句话作为当前推理步骤
            cot_sentence = self._extract_first_sentence(generated_text)
            cot_sentences.append(cot_sentence)
            
            logger.info(f"Generated CoT: {cot_sentence}")
            
            # 检查是否到达最终答案
            if self._is_final_answer(cot_sentence):
                logger.info("Reached final answer, stopping.")
                break
            
            # 检索步骤: 使用最新的CoT句子检索更多段落
            if len(collected_paragraphs) < self.max_paragraphs:
                new_paragraphs = self.retriever.retrieve(cot_sentence)
                
                # 避免重复段落
                existing_contents = {para.content for para in collected_paragraphs}
                unique_new_paragraphs = [
                    para for para in new_paragraphs 
                    if para.content not in existing_contents
                ]
                
                # 添加新段落，但不超过最大限制
                space_left = self.max_paragraphs - len(collected_paragraphs)
                collected_paragraphs.extend(unique_new_paragraphs[:space_left])
                
                logger.info(f"Retrieved {len(unique_new_paragraphs[:space_left])} new paragraphs")
                logger.info(f"Total paragraphs: {len(collected_paragraphs)}")
        
        return collected_paragraphs, cot_sentences
    
    def _create_qa_prompt(self, 
                         question: str, 
                         paragraphs: List[Paragraph],
                         use_cot: bool = True) -> str:
        """创建问答prompt"""
        para_text = self._format_paragraphs(paragraphs)
        
        if use_cot:
            prompt = f"""{self.demonstrations}

                    {para_text}

                    Q: {question}
                    A: Let me think step by step."""
        else:
            prompt = f"""{para_text}

                    Q: {question}
                    A:"""
        
        return prompt
    
    def _extract_final_answer(self, answer_text: str) -> str:
        """从生成的文本中提取最终答案"""
        # 寻找"answer is:"模式
        answer_patterns = [
            r"answer is[:：]\s*([^.!?]+)",
            r"so the answer is[:：]\s*([^.!?]+)",
            r"therefore[,，]\s*the answer is[:：]\s*([^.!?]+)"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, answer_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 如果没找到模式，返回最后一句话
        sentences = answer_text.split('.')
        return sentences[-2].strip() if len(sentences) > 1 else answer_text.strip()
    
    def run(self, question: str, retrieval_list=[]) -> Dict[str, any]:
        """
        使用IRCoT方法回答问题
        
        Args:
            question: 要回答的问题
            verbose: 是否打印详细过程
            
        Returns:
            Dict包含:
            - answer: 最终答案
            - paragraphs: 检索到的段落
            - cot_reasoning: 推理步骤
            - raw_answer: 原始答案文本
            - confidence: 置信度
        """

        
        # 使用IRCoT进行检索和推理
        paragraphs, cot_sentences = self._ircot_retrieve(question)
        
        
        logger.info(f"\nTotal retrieved paragraphs: {len(paragraphs)}")
        logger.info(f"CoT reasoning steps: {len(cot_sentences)}")
        
        # 使用检索到的段落回答问题
        qa_prompt = self._create_qa_prompt(question, paragraphs, self.use_cot_reader)
        answer_text = self.language_model.answer_question(qa_prompt)
        
        # 提取最终答案
        final_answer = self._extract_final_answer(answer_text)
        
        logger.info(f"raw_answer: {answer_text}")
        logger.info(f"Final answer: {final_answer}")

        return {
            "answer": final_answer,
            "paragraphs": paragraphs,
            "cot_reasoning": cot_sentences,
            "raw_answer": answer_text,
            "confidence": len(paragraphs) / self.max_paragraphs  # 简单的置信度计算
        }
    

    
    def simple_retrieve(self, query: str, k: int = None) -> List[Paragraph]:
        """简单检索功能，直接调用基础检索器"""
        if k is None:
            k = self.k
        return self.retriever.retrieve(query, k)

# 示例使用


if __name__ == "__main__":
    print("test")
    