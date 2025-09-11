import openai
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import json
from retrieval.retriever import Retriever
from generation.model_generation import generationmodel
import logging

logger = logging.getLogger("myapp")



class DocumentRetriever:
    """文档检索器，使用dense retrieval"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.doc_embeddings = None
        self.index = None
        
    def build_index(self, documents: List[Dict[str, str]]):
        """构建文档索引
        Args:
            documents: 文档列表，每个文档包含 {"title": "", "text": ""}
        """
        self.documents = documents
        
        # 为每个文档创建检索文本（标题+内容）
        doc_texts = [f"Title: {doc['title']} Context: {doc['text']}" for doc in documents]
        
        # 编码文档
        print("Encoding documents...")
        self.doc_embeddings = self.encoder.encode(doc_texts, show_progress_bar=True)
        
        # 构建FAISS索引
        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
        
        # 归一化嵌入向量
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings)
        
        print(f"Built index with {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """检索相关文档
        Args:
            query: 查询文本
            top_k: 返回的文档数量
        Returns:
            检索到的文档列表
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # 编码查询
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 检索
        scores, indices = self.index.search(query_embedding, top_k)
        
        retrieved_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # 有效索引
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                doc['rank'] = i + 1
                retrieved_docs.append(doc)
        
        return retrieved_docs

class LLMGenerator:
    """大语言模型生成器"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
        """生成回答
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 生成温度
        Returns:
            生成的文本
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return "Error in generation"

class IterRetGen:
    """ITER-RETGEN主类"""
    
    def __init__(self, config, retriever=None, generation_model=None):
        if retriever is None:
            self.retriever = Retriever(config)
        else:
            self.retriever = retriever
        if generation_model is None:
            self.generation_model = generationmodel(config)
        else:
            self.generation_model = generation_model
        
        self.config=config
        
        # Chain-of-Thought提示模板
        self.cot_template = """Based on the following knowledge, answer the question step by step.

                                Knowledge:
                                {knowledge}

                                Question: {question}

                                Let's think step by step.
                                {previous_reasoning}"""

        self.simple_template = """Based on the following knowledge, answer the question step by step.

                                Knowledge:
                                {knowledge}

                                Question: {question}

                                Let's think step by step."""
    
    def format_knowledge(self, docs: List[Dict[str, str]]) -> str:
        """格式化检索到的知识"""
        if not docs:
            return "No relevant documents found."
        
        knowledge_parts = []
        for i, doc in enumerate(docs, 1):
            knowledge_parts.append(f"Paper id: {doc['doc_id']} Context: {doc['content']}")
        
        return "\n".join(knowledge_parts)
    
    def run(self, question: str, max_iterations: int = 3, 
                       top_k: int = 5, verbose: bool = True) -> Dict:
        """使用ITER-RETGEN生成答案
        Args:
            question: 输入问题
            max_iterations: 最大迭代次数
            top_k: 每次检索的文档数
            verbose: 是否打印详细信息
        Returns:
            包含最终答案和中间结果的字典
        """
        results = {
            "question": question,
            "iterations": [],
            "final_answer": ""
        }
        
        current_query = question
        previous_generation = ""
        self.docidlist_iteration={}
        max_iterations=self.config["max_iterations"]
        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n--- Iteration {iteration} ---")
                print(f"Query: {current_query}")
            
            # 1. 检索相关文档
            retrieved_docs = self.retriever.retrieve(current_query, top_k)
            templist=set()
            for doc in retrieved_docs:
                templist.add(doc["doc_id"])
            self.docidlist_iteration[iteration]=list(templist)
            if verbose:
                print(f"Retrieved {len(retrieved_docs)} documents")
                for i, doc in enumerate(retrieved_docs[:3]):  # 只显示前3个
                    print(f"  Paper id {doc['doc_id']}: {doc['title'][:50]}... (score: {doc['score']:.3f})")
            
            # 2. 格式化知识
            knowledge = self.format_knowledge(retrieved_docs)
            
            # 3. 构建提示
            if iteration == 1:
                prompt = self.simple_template.format(
                    knowledge=knowledge,
                    question=question
                )
            else:
                prompt = self.cot_template.format(
                    knowledge=knowledge,
                    question=question,
                    previous_reasoning=f"Previous reasoning:\n{previous_generation}\n\nNow, let me reconsider with new information:"
                )
            
            # 4. 生成回答
            generation = self.generator.generate([{"role": "user", "content": prompt}])
            
            if verbose:
                print(f"Generation: {generation[:200]}...")
            
            # 5. 记录本轮结果
            iteration_result = {
                "iteration": iteration,
                "query": current_query,
                "retrieved_docs": retrieved_docs,
                "generation": generation,
                "prompt_length": len(prompt)
            }
            results["iterations"].append(iteration_result)
            
            # 6. 为下一轮准备查询（生成增强的检索）
            # 将当前生成结果与原问题结合作为下一轮的查询
            previous_generation = generation
            current_query = f"{question} {generation}"
            
            # 更新最终答案
            results["final_answer"] = generation
        
        return results["final_answer"],self.docidlist_iteration
    
    def extract_final_answer(self, generation: str) -> str:
        """从生成文本中提取最终答案"""
        # 寻找 "So the answer is" 模式
        patterns = [
            "So the answer is",
            "Therefore, the answer is",
            "The answer is",
            "Answer:"
        ]
        
        generation_lower = generation.lower()
        for pattern in patterns:
            pattern_lower = pattern.lower()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            if pattern_lower in generation_lower:
                idx = generation_lower.find(pattern_lower)
                answer_part = generation[idx + len(pattern):].strip()
                # 取第一句话作为答案
                answer = answer_part.split('.')[0].strip()
                return answer
        
        # 如果没找到模式，返回最后一句话
        sentences = generation.split('.')
        return sentences[-2].strip() if len(sentences) > 1 else generation.strip()

# 示例使用代码
def example_usage():
    """示例使用方法"""
    
    # 1. 准备示例文档数据
    sample_documents = [
        {
            "title": "Little Richard",
            "text": "Little Richard, born Richard Wayne Penniman on December 5, 1932, was an American musician, singer, actor, comedian, and songwriter. He worked with Modern Records in the 1950s."
        },
        {
            "title": "Modern Records",
            "text": "Modern Records was an American record label founded in 1945. Artists who worked with Modern Records include Etta James, Little Richard, Joe Houston, Ike and Tina Turner."
        },
        {
            "title": "Etta James",
            "text": "Etta James was born on January 25, 1938. She was an American singer who worked with Modern Records and other labels during her career."
        },
        {
            "title": "Jesse Hogan",
            "text": "Jesse Hogan is a professional Australian rules footballer playing for the Melbourne Football Club. A key forward, Hogan is 1.95 m tall and made his AFL debut in the 2015 season and won the Ron Evans Medal as the AFL Rising Star."
        },
        {
            "title": "2015 AFL Rising Star",
            "text": "The NAB AFL Rising Star award is given annually to a stand out young player in the Australian Football League. The award was won by Jesse Hogan of Melbourne in 2015."
        }
    ]
    
    # 2. 初始化组件
    print("Initializing components...")
    retriever = DocumentRetriever()
    retriever.build_index(sample_documents)
    
    # 注意：您需要设置OpenAI API密钥
    generator = LLMGenerator(api_key="your-openai-api-key")
    
    # 3. 创建ITER-RETGEN实例
    iter_retgen = IterRetGen(retriever, generator)
    
    # 4. 测试问题
    questions = [
        "What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?",
        "What is the height of the player who won the 2015 AFL Rising Star award?"
    ]
    
    # 5. 对每个问题进行推理
    for question in questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print('='*80)
        
        results = iter_retgen.generate_answer(
            question=question,
            max_iterations=3,
            top_k=5,
            verbose=True
        )
        
        print(f"\nFinal Answer: {results['final_answer']}")
        
        # 提取简洁答案
        final_answer = iter_retgen.extract_final_answer(results['final_answer'])
        print(f"Extracted Answer: {final_answer}")



# 运行示例（需要先设置OpenAI API密钥）
if __name__ == "__main__":
    # example_usage()
    print("ITER-RETGEN implementation ready!")
    print("Please set your OpenAI API key and uncomment the example_usage() call to test.")