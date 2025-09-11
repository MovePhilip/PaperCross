import os
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

#定义一个conversation_history的类，存储Question,Thought,Search,Observation,Answer,他应该是个list结构，每个元素包括一个type和一个content
@dataclass
class ConversationHistory:
    """对话历史数据类"""
    history: List[Dict[str, str]]

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
        self.action_pattern = re.compile(r'(Thought|Search|Answer):\s*(.*?)(?=(?:Thought|Search|Answer):|$)', re.DOTALL)
        self.conversation_history = ConversationHistory(history=[])
        self.config=config
        self.retrieval_list=[]

        self.avado_docid=set()
        with open("/data4/students/zhangguangyin/chatNum/qa_selected4_200_updated-ranked-filtered.json","r")as f:
                rankedquestion=json.load(f)
        for key,value in rankedquestion.items():
            for docid in value["src_docs"]:
                 self.avado_docid.add(docid)

        
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用大语言模型"""
        try:
            response = self.generation_model.generate(messages)
            logger.info(f"model generation content  : {response}")
            return response
            
        except Exception as e:
            return f"调用LLM时出错: {str(e)}"
    

    
    def parse_search_input(self,s):
        # 匹配 Query 字段
        query_match = re.search(r'Query:\s*(.*?);', s)
        query = query_match.group(1).strip() if query_match else ''

        # 匹配 SQL 字段
        sql_match = re.search(r'SQL:\s*\[([^\]]*)\]', s)
        if sql_match:
            sql_list = [doc_id.strip() for doc_id in sql_match.group(1).split(',') if doc_id.strip()]
        else:
            sql_list = []

        return {"Query": query, "SQL": sql_list}
    


    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """
            You are a ReAct (Reasoning and Acting) agent with access to a document database. The document database is a collection of text chunks from papers in the field of machine learning, each chunk can be a text segment, a html format table or a json format chart. The search results are a list of chunks from different papers with their paper id.
            Your goal is to answer user's questions by combining reasoning with document retrieval.

            Available actions:
            - Thought: Reason about what you need to do next, further retrieval or generate the final answer
            - Search: Search the document database with a query and sql condition. The query is a sentences describing the targeted chunk you are looking for. The SQL statement restricts the list of Doc IDs to search. A "Search" action must have a query statement, while the SQL statement is optional. The format of this action should be "Query: XXXXX; SQL: [DOCID1, DOCID2,...](optional)"
            - Answer: Provide the final answer to the user


            Action Flow Rules:
            - After receiving the user's question, you should firstly generate a "Though" action to analyze the user's question and think about what you need to do next, further retrieval or generate the final answer. You should devise a retrieval process to acquire all required fragments.
            - After "Search" action, you MUST STOP generation and wait for search results.
            - After receiving "Observation" with search results, you should continue with "Thought". In the thought action, you should firstly summarize all the useful information from the "Observation", and then you need to think about what to do next, if the current Observation is not enough to answer the user's question, you should continue with "Search" and formulate a new query.
            - The two actions of Thought and Search can be alternated multiple times.
            - "Answer" should be your final action when you have sufficient information.


            some tips:
            -If you find a particular table very useful but don't know which method in that table was proposed by the paper to which the table belongs, you could limit your Search range to only the chunks of that specific paper in the next round of retrieval to find the name of the method proposed by that paper.




            A typical action flow to process the question is as follows:
            Question: [user's question]

            Thought: [analyze the user's question and determine whether need to retrieve and how to formulate the query text]
            Search: [Query: the search query which should be no more than 100 words;]

            [STOP Generation HERE - Wait for search results]

            Observation: [the retrieved text chunk from different papers]
            Thought: [summarize the information related to the question from the search results and identify whether the retrieval info is adequate to answer the user's question, if the Obervation do not contain the answer, try to reformulate a new query]
            Search: [Query: the search query; SQL:[DOC_id,...]]

            [STOP Generation HERE - Wait for search results]

            Observation: [the retrieved text chunk from different papers]
            Thought: [analyze the search results and identify whether the retrieval info is adequate to answer the user's question, if not enough, keep search, else, prepare the final answer]
            Answer: [generate your final answer]


        """


    def _parse_action(self, output: str) -> List[Action]:
        """解析LLM输出，提取动作"""
        actions = []
        matches = self.action_pattern.findall(output)
        
        for action_type, content in matches:
            action_type = action_type.lower()
            content = content.strip()
            
            if action_type in [t.value for t in ActionType]:
                actions.append(Action(
                    type=ActionType(action_type),
                    content=content
                ))
                
        return actions


    def _execute_action(self, action: Action) -> Optional[Observation]:
        """执行动作并返回观察结果"""
        if action.type == ActionType.THOUGHT:
            # 思考动作不产生外部观察
            logger.info(f"Thinking: {action.content}")
            return None
            
        elif action.type == ActionType.SEARCH:
            # 执行搜索
            logger.info(f"Searching: {action.content}")

            search_input = self.parse_search_input(action.content)
            # temp_sql=search_input["SQL"]
            temp=[item for item in search_input["SQL"] if item in self.avado_docid]

            search_input["SQL"]=list(set(temp+self.retrieval_list))


            if self.config["use_blacklist"]:
                results,return_itemid = self.retriever.retrieve(search_input["Query"],search_input["SQL"],blacklist=self.visited_itemid)#document retriecal result: {list of dict}
            else:
                results,return_itemid = self.retriever.retrieve(search_input["Query"],search_input["SQL"])#document retriecal result: {list of dict}
            docidlist=set()
            if results:
                # 格式化搜索结果
                formatted_results = []
                # logger.info(f"Observation: ")
                for i, result in enumerate(results):
                    docidlist.add(result["doc_id"])
                    formatted_results.append(
                        f"[Paper id {result['doc_id']}] (match_score: {result['match_score']:.3f})\n{result['content']}"
                    )
                    self.visited_itemid.add(return_itemid[i])
                    logger.info(f"Observation: [Paper id {result['doc_id']}] (match_score: {result['match_score']:.3f})Chunk type: {result["chunk_type"]}\n{result['content']}...")
                observation_content = "\n\n".join(formatted_results)
                
                return Observation(content=observation_content, source="document_search",docidlist=list(docidlist),sql=search_input["SQL"])
            else:
                return Observation(content="No relevant documents found.", source="document_search",docidlist=[])
                
        elif action.type == ActionType.ANSWER:
            # 答案动作标志着完成
            logger.info(f"Final Answer: {action.content}")
            return Observation(content=action.content, source="final_answer")
            
        return None


    def format_conversation_history(self) -> str:
        """格式化对话历史
          遇到type是Observation的，只将最近一次的Observation放入输入中，先前的Observation只保留500的长度
        """
        formatted_history = ""
        for itemid,item in enumerate(self.conversation_history.history):
            #遍历获取最近一次的Observation
            if item["type"]=="Observation":
                latestid=itemid
                break

        for itemid,item in enumerate(self.conversation_history.history):
            if item["type"]=="Observation" and itemid<latestid:
                # formatted_history += f"{item['type']}: {item['content'][:100]+"......"}\n\n"
                continue
            else:
                formatted_history += f"{item['type']}: {item['content']}\n\n"
        return formatted_history



    def run(self, user_question: str,retrieval_list=[]) -> str:
        """运行ReAct Agent"""
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_question},
            # {"role": "assistant", "content": "Thought:","prefix": True}
        ]
        self.retrieval_list=retrieval_list
        self.visited_itemid=set()
        self.conversation_history = ConversationHistory(history=[])
        self.actionlist = []
        self.docidlist_iteration={}
        self.max_failed_times=5
        failed_times=0
        for iteration in range(self.max_iterations):
            logger.info(f"\n=== iteration {iteration + 1} ===")
            if len(self.conversation_history.history)>0:
                previous_generation_content=self.format_conversation_history()
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
            logger.info(f"messages length: {len(messages[-1]['content'])}")
            success=False
            # 调用LLM，直到找到明确的动作
            while success==False:
                if failed_times>=self.max_failed_times:
                    logger.info(f"LLM响应: {"生成动作次数达到最大限制"}")
                    break
                ## 如果没有找到明确的动作，则让模型重新生成一次
                response = self._call_llm(messages)     
                
                # 解析动作
                self.actionlist = self._parse_action(response)
                if len(self.actionlist)==0:
                    logger.info(f"LLM响应: {"actionlist长度为0，未生成动作，生成失败, 正在重新生成"}")
                    logger.info(f"messages: {messages}")
                    logger.info(f"response: {response}")
                    success=False
                    continue
                else:
                    success=True
                for action in self.actionlist:
                    if action.type == ActionType.SEARCH:
                        search_input=self.parse_search_input(action.content)
                        if search_input["Query"]=="":
                            logger.info(f"Search: {action.content} 搜索字符串内容为空，准备重新生成")
                            success=False
                            continue
                        else:
                            success=True
                    if success==False:
                        break
                failed_times+=1
            if failed_times==self.max_failed_times:
                break
                

            for action in self.actionlist:
                if action.type == ActionType.ANSWER:
                    logger.info(f"Predicted Answer: {action.content}")
                    return action.content,self.docidlist_iteration
                elif action.type == ActionType.SEARCH:
                    # 执行检索
                    observation = self._execute_action(action)
                    
                    self.conversation_history.history.append({"type": "Search", "content": action.content})
                    self.conversation_history.history.append({"type": "Observation", "content": observation.content})                  #这个content已经被组合成字符串了
                    self.docidlist_iteration[iteration]={"docidlist":observation.docidlist,"sql_query":observation.sql}
                elif action.type == ActionType.THOUGHT:
                    self.conversation_history.history.append({"type": "Thought", "content": action.content})
                    self._execute_action(action)

            self.actionlist=[]
            iteration+=1
            if iteration>=self.max_iterations:
                break


        self.conversation_history.history.append({"type": "Thought", "content": "The max iteration limit is reached, I need to give the final answer based on all the information I've got. \n\n Answer: "})
        previous_generation_content=self.format_conversation_history()
        messages = [
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": previous_generation_content,"prefix": True}
                        ]
        response = self._call_llm(messages)
        return response,self.docidlist_iteration





if __name__ == "__main__":
    with open("/data4/students/zhangguangyin/chatNum/config/config.yaml","r")as f:
        config=yaml.safe_load(f)
    agent=ReActAgent(config)
    print(agent._parse_action("""Search: Query: What is the main idea of the paper?; SQL: [1,2,3] 
                              [STOP HERE - Wait for search results]
                                 """))
    print(agent.parse_search_input("""Query: What is the main idea of the paper?; SQL: [1,2,3] \n                              [STOP HERE - Wait for search results]"""))






