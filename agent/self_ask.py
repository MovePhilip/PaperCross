import openai
import re
import requests
from typing import List, Dict, Optional
import json

class SelfAskWithSearch:
    def __init__(self, openai_api_key: str, search_api_key: str):
        """
        初始化Self-Ask + Search Engine系统
        
        Args:
            openai_api_key: OpenAI API密钥
            search_api_key: 搜索引擎API密钥（SerpAPI）
        """
        openai.api_key = openai_api_key
        self.search_api_key = search_api_key
        self.model = "gpt-3.5-turbo"  # 可以改为gpt-4
        
        # Self-Ask的few-shot示例
        self.examples = [
            {
                "question": "Who lived longer, Theodor Haecker or Harry Vaughan Watkins?",
                "response": """Are follow up questions needed here: Yes.
Follow up: How old was Theodor Haecker when he died?
Intermediate answer: Theodor Haecker was 65 years old when he died.
Follow up: How old was Harry Vaughan Watkins when he died?
Intermediate answer: Harry Vaughan Watkins was 69 years old when he died.
So the final answer is: Harry Vaughan Watkins."""
            },
            {
                "question": "Who was president of the U.S. when superconductivity was discovered?",
                "response": """Are follow up questions needed here: Yes.
Follow up: When was superconductivity discovered?
Intermediate answer: Superconductivity was discovered in 1911.
Follow up: Who was president of the U.S. in 1911?
Intermediate answer: William Howard Taft.
So the final answer is: William Howard Taft."""
            }
        ]

    def create_prompt(self, question: str) -> str:
        """创建Self-Ask的prompt"""
        prompt = ""
        
        # 添加few-shot示例
        for example in self.examples:
            prompt += f"Question: {example['question']}\n"
            prompt += f"{example['response']}\n\n"
        
        # 添加当前问题
        prompt += f"Question: {question}\n"
        prompt += "Are follow up questions needed here:"
        
        return prompt

    def search_web(self, query: str) -> str:
        """使用SerpAPI搜索答案"""
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": self.search_api_key,
                "engine": "google"
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            # 尝试获取featured snippet或第一个搜索结果
            if "answer_box" in data and "answer" in data["answer_box"]:
                return data["answer_box"]["answer"]
            elif "featured_snippet" in data and "snippet" in data["featured_snippet"]:
                return data["featured_snippet"]["snippet"]
            elif "organic_results" in data and len(data["organic_results"]) > 0:
                return data["organic_results"][0]["snippet"]
            else:
                return "No search results found"
                
        except Exception as e:
            return f"Search error: {str(e)}"

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



    def call_llm(self, prompt: str, max_tokens: int = 500) -> str:
            """调用语言模型"""
            messages=[{"role": "user", "content": prompt}]
            """调用大语言模型"""
            try:
                response = self.generation_model.generate(messages)
                return response
            except Exception as e:
                return f"调用LLM时出错: {str(e)}"

    def parse_follow_up_question(self, text: str) -> Optional[str]:
        """从文本中提取follow-up问题"""
        match = re.search(r"Follow up:\s*(.+?)(?:\n|$)", text)
        if match:
            return match.group(1).strip()
        return None

    def run(self, question: str, max_iterations: int = 5) -> Dict:
        """
        使用Self-Ask + Search Engine回答问题
        
        Args:
            question: 输入问题
            max_iterations: 最大迭代次数
            
        Returns:
            包含推理过程和最终答案的字典
        """
        prompt = self.create_prompt(question)
        full_response = ""
        reasoning_steps = []
        iteration = 0
        
        while iteration < max_iterations:
            # 调用LLM生成响应
            response = self.call_llm(prompt)
            full_response += response
            
            # 检查是否需要follow-up问题
            if "Follow up:" in response:
                follow_up_q = self.parse_follow_up_question(response)
                if follow_up_q:
                    reasoning_steps.append({
                        "step": len(reasoning_steps) + 1,
                        "question": follow_up_q,
                        "type": "follow_up"
                    })
                    
                    # 使用搜索引擎获取答案
                    search_answer = self.search_web(follow_up_q)
                    reasoning_steps.append({
                        "step": len(reasoning_steps) + 1,
                        "answer": search_answer,
                        "type": "search_result"
                    })
                    
                    # 更新prompt，将搜索结果作为intermediate answer插入
                    prompt += f" {response}\nIntermediate answer: {search_answer}\n"
                    full_response += f"\nIntermediate answer: {search_answer}\n"
            
            # 检查是否有最终答案
            if "So the final answer is:" in response:
                final_match = re.search(r"So the final answer is:\s*(.+?)(?:\n|$)", response)
                if final_match:
                    final_answer = final_match.group(1).strip()
                    return {
                        "question": question,
                        "final_answer": final_answer,
                        "reasoning_steps": reasoning_steps,
                        "full_reasoning": full_response,
                        "success": True
                    }
                break
            
            # 如果没有找到follow-up问题和最终答案，继续生成
            if "Follow up:" not in response and "So the final answer is:" not in response:
                prompt += f" {response}\n"
            
            iteration += 1
        
        # 如果没有找到明确的最终答案，返回失败状态
        return {
            "question": question,
            "final_answer": "Unable to determine final answer",
            "reasoning_steps": reasoning_steps,
            "full_reasoning": full_response,
            "success": False
        }

    def print_reasoning(self, result: Dict):
        """格式化打印推理过程"""
        print(f"Question: {result['question']}")
        print("-" * 60)
        
        print("Reasoning Process:")
        current_question = None
        for step in result['reasoning_steps']:
            if step['type'] == 'follow_up':
                current_question = step['question']
                print(f"  └─ Follow up: {step['question']}")
            elif step['type'] == 'search_result':
                print(f"     └─ Search result: {step['answer']}")
        
        print(f"\nFinal Answer: {result['final_answer']}")
        print(f"Success: {result['success']}")

def main():
    """示例使用"""
    
    # 初始化系统（需要提供有效的API密钥）
    qa_system = SelfAskWithSearch(
        openai_api_key="your-openai-api-key",  # 替换为你的OpenAI API密钥
        search_api_key="your-serpapi-key"      # 替换为你的SerpAPI密钥
    )
    
    # 测试问题（来自论文中的示例）
    test_questions = [
        "Who won the Masters Tournament in the year that Justin Bieber was born?",
        "What is the capital of the birthplace of Frida Kahlo?",
        "What rocket was the first spacecraft that ever approached Uranus launched on?",
        "Who was the head of NASA during Apollo 11?",
        "What is the calling code of the birthplace of Plato?"
    ]
    
    print("Self-Ask + Search Engine Demo")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nExample {i}:")
        result = qa_system.answer(question)
        qa_system.print_reasoning(result)
        print("=" * 80)

if __name__ == "__main__":
    main()