import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import openai
from pathlib import Path
import re
import tiktoken
from retrieval.retriever import Retriever
from generation.model_generation import generationmodel
import logging
import yaml
import traceback
import hashlib
import concurrent


logger = logging.getLogger("myapp")






class ActionType(Enum):
    """定义Agent可以执行的动作类型"""
    THOUGHT = "thought"
    SEARCH = "search"
    ANSWER = "answer"


@dataclass
class Action:
    """动作数据类"""
    type: ActionType
    content: str


@dataclass
class Observation:
    """观察结果数据类"""
    content: str
    source: Optional[str] = None
    docidlist: Optional[List[str]] = None
    sql: Optional[List[str]] = None
    list_format_result: Optional[List[Dict[str, Any]]] = None

#定义一个conversation_history的类，存储Question,Thought,Search,Observation,Answer,他应该是个list结构，每个元素包括一个type和一个content
@dataclass
class ConversationHistory:
    """对话历史数据类"""
    history: List[Dict[str, str]]

@dataclass
class RunContext:
    """单个 Query 的运行上下文"""
    conversation_history: ConversationHistory
    retrieval_list: List[str]
    visited_itemid: set
    log_buffer: List[str]
    actionlist: List[Action]
    docidlist_iteration: Dict[int, Any]
    performance_results: List[Any]


class ReActAgent:
    """ReAct Agent 实现"""
    
    def __init__(self, config: Dict, retriever=None, generation_model=None):
        if retriever is None:
            self.retriever = Retriever(config)
        else:
            self.retriever = retriever
        if generation_model is None:
            self.generation_model = generationmodel(config)
        else:
            self.generation_model = generation_model
        self.max_iterations = config["max_iterations"]
        self.action_pattern = re.compile(r'<(Thought|Search|Answer)>\s*(.*?)\s*</\1>', re.DOTALL)
        self.conversation_history = ConversationHistory(history=[])
        self.config=config
        self.retrieval_list=[]

        self.avado_docid=set()
        with open("/data4/students/zhangguangyin/chatNum/qa_selected4_200_updated-ranked-filtered.json","r")as f:
                rankedquestion=json.load(f)
        for key,value in rankedquestion.items():
            for docid in value["src_docs"]:
                 self.avado_docid.add(docid)
        
        self.stop_token=["</Answer>","</Search>"]#这个关键就是看后端支不支持了
        
        # 添加日志缓冲区
        self.log_buffer = []
        self.current_run_logs = {}
        
        
        print("初始化完毕")

    def _add_to_log_buffer(self, ctx: RunContext, level: str, message: str):
        #添加日志消息到缓冲区
        ctx.log_buffer.append(f"[{level}] {message}")


    def _new_context(self, retrieval_list=None) -> RunContext:
        """创建一个新的独立运行上下文"""
        return RunContext(
            conversation_history=ConversationHistory(history=[]),
            retrieval_list=retrieval_list or [],
            visited_itemid=set(),
            log_buffer=[],
            actionlist=[],
            docidlist_iteration={},
            performance_results=[]
        )




    def _convert_for_json(self, obj):
        """递归转换对象中的numpy类型为Python原生类型，以便JSON序列化"""
        import numpy as np
        
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj


    def _save_complete_log(self, ctx: RunContext, user_question: str, final_answer: str):
        """保存完整的运行日志"""
        save_list_observation=[]
        for item in ctx.conversation_history.history:
            if item["type"]=="Observation":
                converted_result = self._convert_for_json(item["list_format_result"])
                save_list_observation.append({"type": "Observation", "content": converted_result})
            else:
                save_list_observation.append(item)

        return {
            "user_question": user_question,
            "final_answer": final_answer,
            "conversation_history": save_list_observation,
            "docid_iterations": ctx.docidlist_iteration,
            "detailed_logs": ctx.log_buffer.copy()
        }


    def _clear_log_buffer(self):
        """清空日志缓冲区"""
        self.log_buffer = []


    def _call_llm(self, ctx: RunContext,  messages: List[Dict[str, str]],stop_token_addtional=None) -> str:
        """调用大语言模型"""
        
        print("开始调用大语言模型")
        try:
            # 处理额外的停止token
            stop_tokens = self.stop_token.copy()
            if stop_token_addtional is not None:
                stop_tokens.extend(stop_token_addtional)
                
            response = self.generation_model.generate(messages, stop_token=stop_tokens)
            print(response)
            self._add_to_log_buffer(ctx,"INFO", f"model generation content: {response}")
            return response
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"调用LLM时出错: {str(e)}"
            self._add_to_log_buffer(ctx,"ERROR", error_msg)
            print(f"LLM调用出错: {error_msg}")
            return error_msg

    
    def parse_search_input(self,s):
        # 匹配 <Query> 字段
        query_match = re.search(r'<Query>\s*(.*?)\s*</Query>', s, re.DOTALL)
        query = query_match.group(1).strip() if query_match else ''

        # 匹配 <SQL> 字段
        sql_match = re.search(r'<PaperScope>\s*\[([^\]]*)\]\s*</PaperScope>', s, re.DOTALL)
        if sql_match:
            sql_list = [doc_id.strip() for doc_id in sql_match.group(1).split(',') if doc_id.strip()]
        else:
            sql_list = []

        return {"Query": query, "PaperScope": sql_list}
    


    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """
            You are an **enhanced ReAct (Reasoning and Acting) agent** with access to a **document database**.  
            The document database contains text chunks extracted from papers in the field of **machine learning**.  
            There are three types of chunks:  
            - A plain text segment
            - A HTML-format table  
            - A JSON-format chart  

            The search results are always a list of chunks retrieved from different papers, accompanied by their **paper IDs**.  
            Your role is to answer the user’s question by combining reasoning with iterative document retrieval. Remember to understand the each detained requirements in the user’s question.

            ### Agent Available Actions

            1. Thought  
            - Internal reasoning step.  
            - Use it to analyze the user’s question, interpret observations, decide whether to search again or finalize the answer.  
            
            2. Search  
            - Perform a retrieval from the database.  
            - Must include a <Query> field that specifies what chunks to retrieve. The query can be a short natural language description (e.g., “performance results on CIFAR-10 in 2022”) or simply a list of key keywords (e.g., “CIFAR-10, accuracy, 2022”).  
            - Optionally include a <PaperScope> filter to restrict the chunk search range to either a specific paper or a defined set of papers.
            - Format(xml):
                ```
                <Query> … </Query>  
                <PaperScope>[DOC_ID1, DOC_ID2, …]</PaperScope>   (optional)
                ```  
            
            3. Answer  
            - Provide the user with a clear, well-structured final answer.  
            - This is your **last action** once enough evidence is gathered.  

            When taking any action, you should use one of these three special tokens: <Thought></Thought>, <Search></Search>, <Answer></Answer>. Wrap the content of the action with these tokens. You must strictly follow this format.


            ### Standard Action Flow

            1. Upon receiving a question:
            Always begin with a **Thought** action:  
            - Analyze user’s intent.  
            - Identify key concepts, entities, and needed evidence.  
            - Decide what to do next. If you need to retrieve, think about how to formulate the query.

            2. During retrieval:
            - Issue a **Search** action with queries.  
            - After a Search, **STOP generation** and wait for returned observations.  

            3. Upon receiving Observations: 
            - Continue with a **Thought** step.  
            - First, **summarize the useful information** from the observations.  
            - Decide if the current evidence is sufficient.  
            - If not sufficient → issue another **Search** with refined queries.  

            4. Iterative process:
            - You may alternate between **Thought** and **Search** multiple times.  
            - When confident and with sufficient evidence, end with **Answer**.  


            ### Example Workflow

            ```
            <Question> A user’s question about recent methods and their results. </Question>

            <Thought> Analyze the question: identify what type of information is needed (e.g., performance results, method details, comparisons). Decide whether I need to search the database and how to phrase the query. </Thought>

            <Search> <Query> Relevant methods and their reported results </Query> </Search>

            <Observation> Retrieved text chunks containing information about several methods and results. </Observation>

            <Thought> Summarize the important findings from the retrieved chunks. Check if the information is sufficient. If not, refine the query or narrow the scope using <PaperScope>. </Thought>

            <Answer> Provide a clear and concise answer based on the retrieved evidence. </Answer>
            ```

            *Note: The **Search** and **Thought** actions can repeat in loops — you may alternate them multiple times until enough evidence is collected for the final **Answer**.*

            ### Additional Tips

            - **Paper-specific follow-ups**:  
            If a useful table is found but the paper’s own proposed method is unclear, narrow the next search to that specific paper’s chunks using the `<PaperScope>` filter.  

            - **Systematic data analysis**:  
            When encountering complex tables (e.g., benchmark comparisons), use the **Thought** step to carefully interpret and organize the data before answering.  

            - **Final answers must be evidence-based**:  
            Avoid speculation. Base your conclusions strictly on retrieved chunks.  


        """


    def _parse_action(self, output: str) -> List[Action]:
        """解析LLM输出，提取动作
          检查是否遵守格式，是否存在遗漏token的情况
          如果存在左token，而不存在右token，则自动补全右token,
          如果存在<Search>，而不存在</Search>，则自动在生成内容最后补全</Search>
          如果存在<Answer>，而不存在</Answer>，则自动在生成内容最后补全</Answer>
          如果存在<Thought>，而不存在</Thought>，则自动在生成内容最后补全</Thought>
        
        """
        # 先做token补全：扫描 <Thought>/<Search>/<Answer> 开闭标签，
        # 若出现未闭合的左标签，则在“下一个token”前补上相应的右标签；若到结尾仍未闭合则在末尾补上右标签。
        tag_names = ["Thought", "Search", "Answer"]
        tag_pattern = re.compile(r"</?(Thought|Search|Answer)>")

        result_parts = []
        cursor = 0
        open_stack = []  # 存放未闭合的标签名，栈顶为最后一个未闭合的标签

        for m in tag_pattern.finditer(output):
            start, end = m.span()
            tag = m.group(0)
            name = m.group(1)
            is_closing = tag.startswith("</")

            if not is_closing:
                # 遇到新的左标签，若已有未闭合标签，则在本标签出现之前补上栈顶的右标签
                if open_stack:
                    result_parts.append(output[cursor:start])
                    result_parts.append(f"</{open_stack.pop()}>")
                    cursor = start
                # 记录当前打开的标签
                open_stack.append(name)
            else:
                # 遇到右标签，若与栈顶不匹配，则依次为栈顶补齐右标签直到匹配或栈空
                while open_stack and open_stack[-1] != name:
                    result_parts.append(output[cursor:start])
                    result_parts.append(f"</{open_stack.pop()}>")
                    cursor = start
                # 若匹配，弹出栈顶，保留原右标签
                if open_stack and open_stack[-1] == name:
                    open_stack.pop()

            # 追加原始内容到当前标签为止（包含标签本身）
            if cursor < end:
                result_parts.append(output[cursor:end])
                cursor = end

        # 追加剩余文本
        if cursor < len(output):
            result_parts.append(output[cursor:])

        # 文末为所有仍未闭合的标签补齐右标签
        while open_stack:
            result_parts.append(f"</{open_stack.pop()}>")

        normalized_output = "".join(result_parts)

        # 使用补全后的字符串解析动作
        actions = []
        matches = self.action_pattern.findall(normalized_output)
        for action_type, content in matches:
            action_type = action_type.lower()
            content = content.strip()
            if action_type in [t.value for t in ActionType]:
                actions.append(Action(type=ActionType(action_type), content=content))

        return actions


    def _execute_action(self,ctx, action: Action) -> Optional[Observation]:
        """执行动作并返回观察结果"""
        if action.type == ActionType.THOUGHT:
            # 思考动作不产生外部观察
            self._add_to_log_buffer(ctx, "INFO", f"Thinking: {action.content}")
            return None
            
            
            
        elif action.type == ActionType.SEARCH:
            # 执行搜索
            self._add_to_log_buffer(ctx, "INFO", f"Searching: {action.content}")

            search_input = self.parse_search_input(action.content)
            # temp_sql=search_input["SQL"]
            temp=[item for item in search_input["PaperScope"] if item in self.avado_docid]

            search_input["PaperScope"]=list(set(temp+self.retrieval_list))

            print("开始检索")
            if self.config["use_blacklist"]:
                results,return_itemid = self.retriever.retrieve(search_input["Query"],search_input["PaperScope"],blacklist=self.visited_itemid)#document retriecal result: {list of dict}
            else:
                results,return_itemid = self.retriever.retrieve(search_input["Query"],search_input["PaperScope"])#document retriecal result: {list of dict}
            print("结束检索")
            docidlist=set()
            if results:
                # 格式化搜索结果
                formatted_results = []
                for i, result in enumerate(results):
                    docidlist.add(result["doc_id"])
                    formatted_results.append(
                        f"[Paper id {result['doc_id']}] (match_score: {result['match_score']:.3f})\n{result['content']}"
                    )
                    ctx.visited_itemid.add(return_itemid[i])
                    self._add_to_log_buffer(ctx, "INFO", f"Observation: [Paper id {result['doc_id']}] (match_score: {result['match_score']:.3f})Chunk type: {result['chunk_type']}\n{result['content'][:100]}...")
                observation_content = "\n\n".join(formatted_results)
                
                observation = Observation(content=observation_content, source="document_search",docidlist=list(docidlist),sql=search_input["PaperScope"], list_format_result=results)

                
                return observation
            else:
                return Observation(content="No relevant documents found.", source="document_search",docidlist=[],list_format_result=[])
                
        elif action.type == ActionType.ANSWER:
            # 答案动作标志着完成
            self._add_to_log_buffer(ctx, "INFO", f"Final Answer: {action.content}")
            return Observation(content=action.content, source="final_answer")
            
        return None


    def format_conversation_history(self, ctx: RunContext) -> str:
        """格式化对话历史
          遇到type是Observation的，只将最近一次的Observation放入输入中，先前的Observation只保留500的长度
        """
        formatted_history = ""
        for itemid,item in enumerate(ctx.conversation_history.history):
            #遍历获取最近一次的Observation
            if item["type"]=="Observation":
                latestid=itemid
                break

        for itemid,item in enumerate(ctx.conversation_history.history):
            if item["type"]=="Observation" and itemid<latestid:
                # formatted_history += f"{item['type']}: {item['content'][:100]+"......"}\n\n"
                continue
            else:
                formatted_history += f"<{item['type']}>{item['content']}</{item['type']}>\n\n"
        return formatted_history



    def hash_function(self,user_question:str):
        """使用hashlib库对user_question进行哈希"""
        return hashlib.sha256(user_question.encode()).hexdigest()

    def run(self, user_question: str,retrieval_list=[]) -> str:

        """运行单个 Query"""
        ctx = self._new_context(retrieval_list)
        self._add_to_log_buffer(ctx,"INFO", f"Starting run for: {user_question}")
        
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_question},
            # {"role": "assistant", "content": "Thought:","prefix": True}
        ]
        self.retrieval_list=retrieval_list
        ctx.visited_itemid=set()
        ctx.conversation_history = ConversationHistory(history=[])
        ctx.actionlist = []
        ctx.docidlist_iteration={}
        self.max_failed_times=5
        failed_times=0
        for iteration in range(self.max_iterations):
            self._add_to_log_buffer(ctx, "INFO", f"\n=== iteration {iteration + 1} ===")
            if len(ctx.conversation_history.history)>0:
                previous_generation_content=self.format_conversation_history(ctx)
                messages = [
                                    {"role": "system", "content": self._get_system_prompt()},
                                    {"role": "user", "content": user_question},
                                    {"role": "assistant", "content": previous_generation_content,"prefix": True}
                                ]
            else:
                messages = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": user_question},
                ]
            self._add_to_log_buffer(ctx, "INFO", f"messages length: {len(messages[-1]['content'])}")
            success=False
            # 调用LLM，直到找到明确的动作
            while success==False:
                if failed_times>=self.max_failed_times:
                    self._add_to_log_buffer(ctx, "WARNING", "生成动作次数达到最大限制")
                    break
                ## 如果没有找到明确的动作，则让模型重新生成一次
                response = self._call_llm(ctx,messages)     
                
                # 解析动作
                self.actionlist = self._parse_action(response)
                print(f"解析到 {len(self.actionlist)} 个动作")
                if len(self.actionlist)==0:
                    self._add_to_log_buffer(ctx, "WARNING", f"actionlist长度为0，未生成动作，生成失败, 正在重新生成")
                    self._add_to_log_buffer(ctx, "DEBUG", f"messages: {str(messages)[:200]}...")
                    self._add_to_log_buffer(ctx, "DEBUG", f"response: {response}")
                    failed_times+=1
                    success=False
                    continue
                else:
                    success=True
                for action in self.actionlist:
                    if action.type == ActionType.SEARCH:
                        search_input=self.parse_search_input(action.content)
                        if search_input["Query"]=="":
                            self._add_to_log_buffer(ctx, "WARNING", f"Search: {action.content} 搜索字符串内容为空，准备重新生成")
                            success=False
                            continue
                        else:
                            success=True
                    if success==False:
                        break
                failed_times+=1
            if failed_times==self.max_failed_times:
                self._add_to_log_buffer(ctx, "WARNING", "生成下一轮动作失败，次数达到最大限制")
                break
                

            for action in self.actionlist:
                if action.type == ActionType.ANSWER:
                    final_answer = action.content
                    self._add_to_log_buffer(ctx, "INFO", f"Predicted Answer: {final_answer}")
                    ctx.conversation_history.history.append({"type": "Answer", "content": action.content})
                    # 保存完整日志并返回
                    return_content=self._save_complete_log(ctx, user_question, final_answer)
                    return return_content
                elif action.type == ActionType.SEARCH:
                    # 执行检索
                    observation = self._execute_action(ctx, action)
                    
                    ctx.conversation_history.history.append({"type": "Search", "content": action.content})
                    ctx.conversation_history.history.append({"type": "Observation", "content": observation.content,"list_format_result":observation.list_format_result})                  #这个content已经被组合成字符串了
                    ctx.docidlist_iteration[iteration]={"docidlist":observation.docidlist,"sql_query":observation.sql}
                elif action.type == ActionType.THOUGHT:
                    ctx.conversation_history.history.append({"type": "Thought", "content": action.content})
                    self._execute_action(ctx, action)

            ctx.actionlist=[]
            iteration+=1
            if iteration>=self.max_iterations:
                break


        ctx.conversation_history.history.append({"type": "Thought", "content": "The max iteration limit is reached, I need to give the final answer based on all the information I've got. \n\n <Answer> "})
        previous_generation_content=self.format_conversation_history(ctx)
        messages = [
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": previous_generation_content,"prefix": True}
                        ]
        response = self._call_llm(ctx, messages)
        self.actionlist = self._parse_action(response)
        for action in self.actionlist:
            if action.type == ActionType.ANSWER:
                final_answer = action.content
                self._add_to_log_buffer(ctx, "INFO", f"Predicted Answer: {final_answer}")
                ctx.conversation_history.history.append({"type": "Answer", "content": final_answer})
                # 保存完整日志并返回
                return_content=self._save_complete_log(ctx, user_question, final_answer)
                return return_content

        # 保存完整日志并返回
        return_content=self._save_complete_log(ctx, user_question, response)
        return return_content

    def parallel_run(
        self, questions: List[str], retrieval_lists: Optional[List[List[str]]] = None, save_paths: Optional[List[str]] = None,  max_workers: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        并行运行多个 Query
        - questions: 多个问题
        - retrieval_lists: 每个 query 对应的检索列表（可选）
        - max_workers: 并发线程数量
        返回: list, 保持输入顺序
        [
        {"question": q1, "result": {...}},
        {"question": q2, "result": {...}},
        ...
        ]
        """
        if retrieval_lists is None:
            retrieval_lists = [[] for _ in range(len(questions))]

        if save_paths is None:
            save_paths = [None] * len(questions)

        if len(save_paths) != len(questions):
            raise ValueError("save_paths 的长度必须与 questions 相同")

        results: List[Dict[str, Any]] = [None] * len(questions)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.run, q, rlist): idx
                for idx, (q, rlist) in enumerate(zip(questions, retrieval_lists))
             }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                question = questions[idx]
                save_path = save_paths[idx]

                try:
                    res = future.result()
                    results[idx] = {"question": question, "result": res}

                    if save_path:
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(res, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    error_res = {"error": str(e)}
                    results[idx] = {"question": question, "result": error_res}

                    if save_path:
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(error_res, f, ensure_ascii=False, indent=2)

        return results


if __name__ == "__main__":
    #定义logger的输出文件
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("react_agent_v2.log"))
    logger.addHandler(logging.StreamHandler())

    with open("/data4/students/zhangguangyin/chatNum/config/config.yaml","r")as f:
        config=yaml.safe_load(f)
    agent=ReActAgent(config)
    question1=f"""<Question> Please help me find the performance of methods proposed in different papers on the Accuracy metric for the Graph Classification task on the MNIST (MNIST) dataset, list the top three metric result. 
        Requirements:
        1. For each entry, you need to complete three fields: metric value, model name and paper id. Do not miss any field or randomly generate a answer for any field. Your answer should based on the retrieved chunks from the document database.
        2. For each paper, you should give only one metric result, namely the highest one. Pay attention to identify which method is proposed by the paper and which method is proposed by other papers for comparison. Try to Avoid use the "ours" expression in the "method" field except the paper did not give a formal name of their proposed method.
        3. The article ID must correspond to the method name, meaning the article ID should refer to the paper in which the method was originally proposed.
        """
    question1=question1+"""
                            You need to give the result in JSON format:
                            [
                                {"rank_id":1
                                "method":"model name",
                                "value": "metric value"
                                "paper_id":"paper id"
                                },
                                {"rank_id":2
                                "method":"model name",
                                "value": "metric value"
                                "paper_id":"paper id"
                                },
                                {"rank_id":3
                                "method":"model name",
                                "value": "metric value"
                                "paper_id":"paper id"
                                }
                            ]
                            </Question>
                            
                            """
    
    question2="""<Question> Please help me find the performance of methods proposed in different papers on the Accuracy metric for the Visual Question Answering (VQA) task on the OK-VQA (OK-VQA) dataset, list the top three metric result. 
        Requirements:
        1. For each entry, you need to complete three fields: metric value, model name and paper id. Do not miss any field or randomly generate a answer for any field. Your answer should based on the retrieved chunks from the document database.
        2. For each paper, you should give only one metric result, namely the highest one. Pay attention to identify which method is proposed by the paper and which method is proposed by other papers for comparison. Try to Avoid use the "ours" expression in the "method" field except the paper did not give a formal name of their proposed method.
        3. The article ID must correspond to the method name, meaning the article ID should refer to the paper in which the method was originally proposed.
         """

    question2=question2+"""
                            You need to give the result in JSON format:
                            [
                                {"rank_id":1
                                "method":"model name",
                                "value": "metric value"
                                "paper_id":"paper id"
                                },
                                {"rank_id":2
                                "method":"model name",
                                "value": "metric value"
                                "paper_id":"paper id"
                                },
                                {"rank_id":3
                                "method":"model name",
                                "value": "metric value"
                                "paper_id":"paper id"
                                }
                            ]
                            </Question>
                            
                            """
    save_paths = [
        "/data4/students/zhangguangyin/chatNum/second_version_result/mnist_result.json",
        "/data4/students/zhangguangyin/chatNum/second_version_result/cifar10_result.json"
    ]

    result= agent.parallel_run([question1, question2],save_paths=save_paths)
    print("Result:")
    print(result)
    #保存到logger
    with open("complete_log.json","w")as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    # print("Document IDs:", docidlist)
    
    print("\n" + "="*50 + "\n")
    
    # 也可以使用标准方法进行比较
    # print("=== Using Standard ReAct Method ===")
    # result_standard, docidlist_standard = agent.run(question)
    # print("Standard Result:")
    # print(result_standard)
    # print("Document IDs:", docidlist_standard)


    # print(agent._parse_action("""<Search> <Query> What is the main idea of the paper?</Query> <SQL>[1,2,3]</SQL> </Search>
    #                              """))
    # print(agent.parse_search_input("""<Query> What is the main idea of the paper?</Query> <SQL>[1,2,3]</SQL>"""))






# curl -X POST "https://uni-api.cstcloud.cn/v1/chat/completions" \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer e36b5397eee2bb61205e0939d143a00292cec18a0d1b14bce1077c502285d660" \
#     -d '{
#         "model": "qwen3:235b",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Hello!"
#             }
#         ]
#     }'





