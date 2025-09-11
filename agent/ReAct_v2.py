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
logger = logging.getLogger("myapp")






class ActionType(Enum):
    """定义Agent可以执行的动作类型"""
    THOUGHT = "thought"
    SEARCH = "search"
    ANSWER = "answer"
    PLAN = "plan"  # 新增：制定搜索计划
    ANALYZE = "analyze"  # 新增：分析收集的数据

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
class PerformanceResult:
    """性能结果数据类"""
    method_name: str
    metric_value: float
    paper_id: str
    dataset: Optional[str] = None  # 动态设置
    task: Optional[str] = None     # 动态设置
    metric_type: Optional[str] = None  # 动态设置
    source_chunk: str = ""
    confidence: float = 1.0

@dataclass 
class SearchPlan:
    """搜索计划数据类"""
    queries: List[str]
    target_papers: List[str]
    search_strategy: str
    expected_results: int

class ChunkRelevance(Enum):
    """Chunk相关性评估等级"""
    IRRELEVANT = "irrelevant"  # 无用，丢弃
    PARTIALLY_USEFUL = "partially_useful"  # 部分有用，需要更多上下文
    DIRECTLY_USEFUL = "directly_useful"  # 直接有用，包含完整答案

@dataclass
class ChunkEvaluation:
    """Chunk评估结果"""
    chunk_id: str
    doc_id: str
    relevance: ChunkRelevance
    content: str
    key_info: str  # 提取的关键信息
    needs_context: bool  # 是否需要上下文
    context_direction: Optional[str] = None  # "before", "after", "both", "full"
    confidence: float = 0.0

@dataclass
class DocumentCursor:
    """文档游标数据类"""
    doc_id: str
    current_chunk_id: str
    current_position: int  # 当前chunk在文档中的位置索引
    total_chunks: int      # 文档总chunk数
    context_window_size: int = 3  # 上下文窗口大小（chunk数量）
    document_structure: Optional[Dict[str, Any]] = None  # 文档结构信息

@dataclass
class ContextWindow:
    """上下文窗口数据类"""
    chunks: List[Dict[str, Any]]
    start_position: int
    end_position: int
    cursor_position: int
    document_structure: Optional[Dict[str, Any]] = None
    navigation_info: Optional[str] = None

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
        self.action_pattern = re.compile(r'<(Thought|Search|Answer|Plan|Analyze)>\s*(.*?)\s*</\1>', re.DOTALL)
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
        
        # 新增：性能数据管理
        self.performance_results: List[PerformanceResult] = []
        self.search_plans: List[SearchPlan] = []
        
        # 新增：Chunk评估管理
        self.chunk_evaluations: List[ChunkEvaluation] = []
        self.useful_chunks: List[Dict[str, Any]] = []  # 存储有用的chunks
        self.docs_to_expand: set = set()  # 需要扩展检索的文档ID
        
        # 新增：文档游标管理
        self.document_cursors: Dict[str, DocumentCursor] = {}  # 每个文档的游标
        
        print("初始化完毕")

    def _add_to_log_buffer(self, level: str, message: str):
        """添加日志消息到缓冲区"""
        self.log_buffer.append(f"[{level}] {message}")

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

    def _save_complete_log(self, user_question: str, final_answer: str):
        """保存完整的运行日志"""
        #删除conversation_history中的Observation中content键值对
        save_list_observation=[]
        for item in self.conversation_history.history:
            if item["type"]=="Observation":
                # 转换list_format_result中的numpy类型
                converted_result = self._convert_for_json(item["list_format_result"])
                save_list_observation.append({"type": "Observation", "content": converted_result})
            else:
                save_list_observation.append(item)

        complete_log = {
            "user_question": user_question,
            "final_answer": final_answer,
            "conversation_history": save_list_observation,
            "docid_iterations": getattr(self, 'docidlist_iteration', {}),
            "detailed_logs": self.log_buffer.copy()
        }
        
        # 保存到logger
        with open("complete_log.json","w")as f:
            json.dump(complete_log,f,ensure_ascii=False,indent=2)
        
        return complete_log

    def _clear_log_buffer(self):
        """清空日志缓冲区"""
        self.log_buffer = []
        
    def _call_llm(self, messages: List[Dict[str, str]],stop_token_addtional=None) -> str:
        """调用大语言模型"""
        
        print("开始调用大语言模型")
        try:
            # 处理额外的停止token
            stop_tokens = self.stop_token.copy()
            if stop_token_addtional is not None:
                stop_tokens.extend(stop_token_addtional)
                
            response = self.generation_model.generate(messages, stop_token=stop_tokens)
            print(response)
            self._add_to_log_buffer("INFO", f"model generation content: {response}")
            return response
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"调用LLM时出错: {str(e)}"
            self._add_to_log_buffer("ERROR", error_msg)
            print(f"LLM调用出错: {error_msg}")
            return error_msg


    def parse_search_input(self,s):
        """
        "<Query> ... </Query> <SQL> [DOC_ID1, DOC_ID2, ...] </SQL> <Page> 1 </Page> <PageSize> 10 </PageSize> <Offset> 0 </Offset>"
        
        
        """

        # 匹配 <Query> 字段
        query_match = re.search(r'<Query>\s*(.*?)\s*</Query>', s, re.DOTALL)
        query = query_match.group(1).strip() if query_match else ''

        # 匹配 <SQL> 字段
        sql_match = re.search(r'<SQL>\s*\[([^\]]*)\]\s*</SQL>', s, re.DOTALL)
        if sql_match:
            sql_list = [doc_id.strip() for doc_id in sql_match.group(1).split(',') if doc_id.strip()]
        else:
            sql_list = []
        
        # 匹配分页: <Page> 和 <PageSize>（可选）
        page_match = re.search(r'<Page>\s*(\d+)\s*</Page>', s, re.DOTALL)
        page = int(page_match.group(1)) if page_match else None
        page_size_match = re.search(r'<PageSize>\s*(\d+)\s*</PageSize>', s, re.DOTALL)
        page_size = int(page_size_match.group(1)) if page_size_match else None
        
        # 直接偏移量（可选）：<Offset>
        offset_match = re.search(r'<Offset>\s*(\d+)\s*</Offset>', s, re.DOTALL)
        offset = int(offset_match.group(1)) if offset_match else None

        return {"Query": query, "SQL": sql_list, "Page": page, "PageSize": page_size, "Offset": offset}
    


    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """
            You are a ReAct (Reasoning and Acting) agent with access to a document database. The database contains text chunks (plain text, HTML tables, JSON figures) extracted from machine learning papers. 
            Available actions (use as needed based on your reasoning):
            - Thought: Reflect on the question, the current evidence, and decide the next step.
            - Search: Retrieve relevant chunks from the database. Format: "<Query> ... </Query> <SQL> [DOC_ID1, DOC_ID2, ...] </SQL>" where SQL is optional and limits search to given paper IDs. Each search returns a list of chunks with their paper IDs.
            - Next: Ask the retriever to return the next page of previous search results.
            - Answer: Produce the final answer in the user-requested format.

            Context Retrieval System:
            When you search and find chunks that are useful but incomplete, the system will automatically:
            1. Use a document cursor to locate the chunk's position within its source paper
            2. Provide you with a context window (typically 3 chunks) around that position
            3. Show you the document structure (Abstract, Introduction, Method, Experiments, etc.)
            4. Allow intelligent navigation to different sections if current context is insufficient
            5. Extract and synthesize information from multiple related locations in the same document
            
            The system will show you navigation information like:
            "Current Section: Method, Current Window: chunks 15-17 (cursor at 16), Available Sections: Abstract (0-2), Introduction (3-8), Method (9-20), Experiments (21-35)"

            General guidance:
            - Begin with a Thought to decide whether to retrieve more evidence or answer directly.
            - After emitting a Search, stop generation and wait for the Observation.
            - When an Observation arrives, you'll receive intelligently filtered and contextually enriched results.
            - The system automatically handles context expansion, so focus on analyzing the provided information.
            - Use Next when you think the next page may contain useful information.
            - Use Answer only when you have sufficient evidence and can respond in the requested format.

            Notes about search results you will receive:
            - Irrelevant chunks are automatically filtered out.
            - Partially useful chunks trigger intelligent context expansion using the cursor system.
            - You may see enriched observations with document structure and navigation information.
            - All results are pre-processed for maximum relevance and completeness.
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
            self._add_to_log_buffer("INFO", f"Thinking: {action.content}")
            return None
            
            
        elif action.type == ActionType.SEARCH:
            # 执行搜索
            self._add_to_log_buffer("INFO", f"Searching: {action.content}")

            search_input = self.parse_search_input(action.content)
            # temp_sql=search_input["SQL"]
            temp=[item for item in search_input["SQL"] if item in self.avado_docid]

            search_input["SQL"]=list(set(temp+self.retrieval_list))

            print("开始检索")
            
            # 计算分页参数
            offset = 0
            limit = None
            if search_input.get("Offset") is not None:
                offset = search_input["Offset"]
            elif search_input.get("Page") is not None:
                page = max(search_input["Page"], 1)
                page_size = search_input.get("PageSize") or self.config.get("retrieval_top_k", 10)
                offset = (page - 1) * int(page_size)
                limit = int(page_size)
            
            if self.config["use_blacklist"]:
                results,return_itemid = self.retriever.retrieve(search_input["Query"],search_input["SQL"],blacklist=self.visited_itemid, offset=offset, limit=limit)#document retriecal result: {list of dict}
            else:
                results,return_itemid = self.retriever.retrieve(search_input["Query"],search_input["SQL"], offset=offset, limit=limit)#document retriecal result: {list of dict}
            print("结束检索")
            
            # 新增：使用智能评估处理搜索结果

            # 使用保存的用户问题进行评估
            user_question = getattr(self, 'user_question', search_input["Query"])
            processed_chunks, processing_summary = self._process_search_results_with_evaluation(results, user_question)
            
            self._add_to_log_buffer("INFO", processing_summary)
            
            # 格式化处理后的结果
            if processed_chunks:
                formatted_results = []
                docidlist = set()
                for chunk in processed_chunks:
                    docidlist.add(chunk["doc_id"])
                    formatted_results.append(
                        f"[Paper id {chunk['doc_id']}] (Relevance: {chunk['relevance']})\n"
                        f"Key Info: {chunk['key_info']}\n"
                        f"{chunk['content'][:500]}..."
                    )
                observation_content = "\n\n".join(formatted_results)
                
                # 记录访问的item_id
                for i in range(min(len(results), len(return_itemid))):
                    self.visited_itemid.add(return_itemid[i])
                
                observation = Observation(
                    content=observation_content, 
                    source="document_search",
                    docidlist=list(docidlist),
                    sql=search_input["SQL"], 
                    list_format_result=processed_chunks
                )
            else:
                observation = Observation(
                    content="No relevant documents found after evaluation.", 
                    source="document_search",
                    docidlist=[],
                    list_format_result=[]
                )

            docidlist=set()
            if results:
                # 格式化搜索结果
                formatted_results = []
                for i, result in enumerate(results):
                    docidlist.add(result["doc_id"])
                    formatted_results.append(
                        f"[Paper id {result['doc_id']}] (match_score: {result['match_score']:.3f})\n{result['content']}"
                    )
                    self.visited_itemid.add(return_itemid[i])
                    self._add_to_log_buffer("INFO", f"Observation: [Paper id {result['doc_id']}] (match_score: {result['match_score']:.3f})Chunk type: {result['chunk_type']}\n{result['content'][:100]}...")
                    observation_content = "\n\n".join(formatted_results)
                
                    observation = Observation(content=observation_content, source="document_search",docidlist=list(docidlist),sql=search_input["SQL"], list_format_result=results)
                else:
                    observation = Observation(content="No relevant documents found.", source="document_search",docidlist=[],list_format_result=[])
            
            
            return observation
                
        elif action.type == ActionType.ANSWER:
            # 答案动作标志着完成
            self._add_to_log_buffer("INFO", f"Final Answer: {action.content}")
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
                formatted_history += f"<{item['type']}>{item['content']}</{item['type']}>\n\n"
        return formatted_history



    def run(self, user_question: str,retrieval_list=[]) -> str:
        """运行ReAct Agent"""
        # 清空日志缓冲区，开始新的运行
        self._clear_log_buffer()
        self._add_to_log_buffer("INFO", f"Starting ReAct Agent run for question: {user_question}")
        

        
        # 初始化chunk评估相关变量
        self.chunk_evaluations = []
        self.useful_chunks = []
        self.docs_to_expand = set()
        self.user_question = user_question  # 保存用户问题供后续使用
        
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
            self._add_to_log_buffer("INFO", f"\n=== iteration {iteration + 1} ===")
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
            self._add_to_log_buffer("INFO", f"messages length: {len(messages[-1]['content'])}")
            success=False
            # 调用LLM，直到找到明确的动作
            while success==False:
                if failed_times>=self.max_failed_times:
                    self._add_to_log_buffer("WARNING", "生成动作次数达到最大限制")
                    break
                ## 如果没有找到明确的动作，则让模型重新生成一次
                response = self._call_llm(messages)     
                
                # 解析动作
                self.actionlist = self._parse_action(response)
                print(f"解析到 {len(self.actionlist)} 个动作")
                if len(self.actionlist)==0:
                    self._add_to_log_buffer("WARNING", f"actionlist长度为0，未生成动作，生成失败, 正在重新生成")
                    self._add_to_log_buffer("DEBUG", f"messages: {str(messages)[:200]}...")
                    self._add_to_log_buffer("DEBUG", f"response: {response}")
                    failed_times+=1
                    success=False
                    continue
                else:
                    success=True
                for action in self.actionlist:
                    if action.type == ActionType.SEARCH:
                        search_input=self.parse_search_input(action.content)
                        if search_input["Query"]=="":
                            self._add_to_log_buffer("WARNING", f"Search: {action.content} 搜索字符串内容为空，准备重新生成")
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
                    final_answer = action.content
                    self._add_to_log_buffer("INFO", f"Predicted Answer: {final_answer}")
                    self.conversation_history.history.append({"type": "Answer", "content": action.content})
                    # 保存完整日志并返回
                    self._save_complete_log(user_question, final_answer)
                    return final_answer, self.docidlist_iteration
                elif action.type == ActionType.SEARCH:
                    # 执行检索
                    observation = self._execute_action(action)
                    
                    self.conversation_history.history.append({"type": "Search", "content": action.content})
                    self.conversation_history.history.append({"type": "Observation", "content": observation.content,"list_format_result":observation.list_format_result})                  #这个content已经被组合成字符串了
                    self.docidlist_iteration[iteration]={"docidlist":observation.docidlist,"sql_query":observation.sql}


            self.actionlist=[]
            iteration+=1
            if iteration>=self.max_iterations:
                break


        self.conversation_history.history.append({"type": "Thought", "content": "The max iteration limit is reached, I need to give the final answer based on all the information I've got. \n\n <Answer> "})
        previous_generation_content=self.format_conversation_history()
        messages = [
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": previous_generation_content,"prefix": True}
                        ]
        response = self._call_llm(messages)
        self.actionlist = self._parse_action(response)
        for action in self.actionlist:
            if action.type == ActionType.ANSWER:
                final_answer = action.content
                self._add_to_log_buffer("INFO", f"Predicted Answer: {final_answer}")
                self.conversation_history.history.append({"type": "Answer", "content": final_answer})
                # 保存完整日志并返回
                self._save_complete_log(user_question, final_answer)
                return final_answer, self.docidlist_iteration
        
        # 保存完整日志并返回
        self._save_complete_log(user_question, response)
        
        # The knowledge graph building logic is removed here.

        return response, self.docidlist_iteration


    def _process_search_results_with_evaluation(self, results: List[Dict[str, Any]], user_question: str) -> Tuple[List[Dict[str, Any]], str]:
        """处理搜索结果，评估每个chunk并根据需要扩展上下文"""
        processed_chunks = []
        summary = []
        
        if not results:
            return [], "No results to process."
        
        self._add_to_log_buffer("INFO", f"Processing {len(results)} search results with evaluation")
        
        # 第一轮：评估所有chunks
        useful_chunks = []
        for chunk in results:
            evaluation = self._evaluate_chunk_relevance(chunk, user_question)
            self.chunk_evaluations.append(evaluation)
            
            if evaluation.relevance == ChunkRelevance.DIRECTLY_USEFUL:
                useful_chunks.append(chunk)
                summary.append(f"✓ Doc {evaluation.doc_id}: Found directly useful chunk.")
                
            elif evaluation.relevance == ChunkRelevance.PARTIALLY_USEFUL:
                self.docs_to_expand.add(evaluation.doc_id)
                useful_chunks.append(chunk)
                summary.append(f"⚠ Doc {evaluation.doc_id}: Marked for context expansion.")
                
            else:  # IRRELEVANT
                summary.append(f"✗ Doc {evaluation.doc_id}: Irrelevant chunk discarded.")
        
        # 第二轮：扩展需要上下文的文档
        expanded_chunks = []
        docs_to_expand_copy = self.docs_to_expand.copy()
        for doc_id in docs_to_expand_copy:
            relevant_evals = [e for e in self.chunk_evaluations 
                             if e.doc_id == doc_id and e.relevance == ChunkRelevance.PARTIALLY_USEFUL]
            
            for eval_item in relevant_evals:
                expanded_results = self._expand_chunk_context(eval_item, user_question)
                for exp_chunk in expanded_results:
                    exp_eval = self._evaluate_chunk_relevance(exp_chunk, user_question)
                    if exp_eval.relevance in [ChunkRelevance.DIRECTLY_USEFUL, ChunkRelevance.PARTIALLY_USEFUL]:
                        expanded_chunks.append(exp_chunk)
                        summary.append(f"↳ Expanded from Doc {doc_id}: Found useful context.")
        
        # 合并所有有用的chunks
        final_processed_chunks = useful_chunks + expanded_chunks

        # 生成处理摘要
        summary_text = "Search Results Processing Summary:\n" + "\n".join(summary)
        summary_text += f"\n\nTotal: {len(results)} chunks → {len(final_processed_chunks)} useful chunks."
        
        return final_processed_chunks, summary_text

    def _evaluate_chunk_relevance(self, chunk: Dict[str, Any], user_question: str) -> ChunkEvaluation:
        """使用LLM评估单个chunk的相关性"""
        chunk_content = chunk.get('content', '')
        doc_id = chunk.get('doc_id', '')
        chunk_id = chunk.get('chunk_id', str(chunk.get('item_id', '')))
        
        # 构造评估prompt
        evaluation_prompt = f"""
        Evaluate the relevance of this text chunk to answer the user's question.
        
        User Question: {user_question}
        
        Text Chunk (from Paper {doc_id}):
        {chunk_content}  # 限制长度
        
        Please classify this chunk as:
        1. IRRELEVANT - Completely unrelated to the question
        2. PARTIALLY_USEFUL - Contains relevant information but needs more context from the same paper
        3. DIRECTLY_USEFUL - Contains complete information to answer the question
        
        Also extract:
        - Key information related to the question (if any)
        - Whether more context is needed (yes/no)
        - If context needed, which direction (before/after/both/full)
        
        Response format:
        <Relevance>IRRELEVANT/PARTIALLY_USEFUL/DIRECTLY_USEFUL</Relevance>
        <KeyInfo>extracted key information</KeyInfo>
        <NeedsContext>yes/no</NeedsContext>
        <ContextDirection>before/after/both/full/none</ContextDirection>
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at evaluating text relevance."},
            {"role": "user", "content": evaluation_prompt}
        ]
        
        try:
            response = self._call_llm(messages, stop_token_addtional=["</ContextDirection>"])
            
            # 解析响应
            relevance_match = re.search(r'<Relevance>(.*?)</Relevance>', response, re.IGNORECASE)
            key_info_match = re.search(r'<KeyInfo>(.*?)</KeyInfo>', response, re.DOTALL)
            needs_context_match = re.search(r'<NeedsContext>(.*?)</NeedsContext>', response, re.IGNORECASE)
            context_dir_match = re.search(r'<ContextDirection>(.*?)</ContextDirection>', response, re.IGNORECASE)
            
            relevance_str = relevance_match.group(1).strip().upper() if relevance_match else "IRRELEVANT"
            
            # 映射到枚举
            relevance_map = {
                "IRRELEVANT": ChunkRelevance.IRRELEVANT,
                "PARTIALLY_USEFUL": ChunkRelevance.PARTIALLY_USEFUL,
                "DIRECTLY_USEFUL": ChunkRelevance.DIRECTLY_USEFUL
            }
            relevance = relevance_map.get(relevance_str, ChunkRelevance.IRRELEVANT)
            
            key_info = key_info_match.group(1).strip() if key_info_match else ""
            needs_context = needs_context_match.group(1).strip().lower() == "yes" if needs_context_match else False
            context_direction = context_dir_match.group(1).strip().lower() if context_dir_match else None
            
            return ChunkEvaluation(
                chunk_id=chunk_id,
                doc_id=doc_id,
                relevance=relevance,
                content=chunk_content,
                key_info=key_info,
                needs_context=needs_context,
                context_direction=context_direction,
                confidence=0.8
            )
            
        except Exception as e:
            self._add_to_log_buffer("ERROR", f"Chunk evaluation failed: {str(e)}")
            # 默认返回部分有用，以避免错过重要信息
            return ChunkEvaluation(
                chunk_id=chunk_id,
                doc_id=doc_id,
                relevance=ChunkRelevance.PARTIALLY_USEFUL,
                content=chunk_content,
                key_info="",
                needs_context=True,
                context_direction="both",
                confidence=0.3
            )





    def _expand_chunk_context(self, evaluation: ChunkEvaluation, user_question: str) -> List[Dict[str, Any]]:
        """使用游标系统扩展部分有用chunk的上下文"""
        expanded_chunks = []
        
        self._add_to_log_buffer("INFO", f"Using cursor system to expand context for doc {evaluation.doc_id}")
        
        # 初始化或获取文档游标
        if evaluation.doc_id not in self.document_cursors:
            # 假设chunk_id包含位置信息，或从检索器获取位置
            chunk_position = self._get_chunk_position_in_document(evaluation.chunk_id, evaluation.doc_id)
            cursor = self._initialize_document_cursor(evaluation.doc_id, evaluation.chunk_id, chunk_position)
        else:
            cursor = self.document_cursors[evaluation.doc_id]
            # 更新游标到当前chunk位置
            cursor.current_position = self._get_chunk_position_in_document(evaluation.chunk_id, evaluation.doc_id)
            cursor.current_chunk_id = evaluation.chunk_id
        
        # 使用游标系统提取上下文信息
        try:
            # 构造原始chunk信息
            original_chunk = {
                "doc_id": evaluation.doc_id,
                "content": evaluation.content,
                "chunk_id": evaluation.chunk_id
            }
            
            # 使用游标移动和信息提取
            extracted_info = self._move_cursor_and_extract_info(cursor, user_question, original_chunk)
            
            # 将提取的信息转换为chunk格式
            for info in extracted_info:
                expanded_chunks.append({
                    "doc_id": info["doc_id"],
                    "content": info["content"],
                    "chunk_type": "cursor_extracted",
                    "source_window": info["source_window"],
                    "match_score": 0.9  # 高置信度，因为是针对性提取
                })
            
            self._add_to_log_buffer("INFO", f"Cursor system extracted {len(expanded_chunks)} pieces of contextual information from doc {evaluation.doc_id}")
            
        except Exception as e:
            self._add_to_log_buffer("ERROR", f"Cursor-based context expansion failed: {str(e)}")
            # 回退到原有方法
            return self._expand_chunk_context_fallback(evaluation, user_question)
        
        return expanded_chunks

    def _get_chunk_position_in_document(self, chunk_id: str, doc_id: str) -> int:
        """获取chunk在文档中的位置（需要检索器支持）"""
        # 这里应该调用检索器的API获取chunk在文档中的实际位置
        # 目前返回模拟位置
        try:
            # 从chunk_id中提取位置信息，或查询数据库
            if "_" in chunk_id:
                position_str = chunk_id.split("_")[-1]
                return int(position_str)
        except:
            pass
        
        # 默认返回中间位置
        return 20

    def _expand_chunk_context_fallback(self, evaluation: ChunkEvaluation, user_question: str) -> List[Dict[str, Any]]:
        """原有的上下文扩展方法（作为回退）"""
        expanded_chunks = []
        
        # 根据需要的上下文方向构造查询
        if evaluation.context_direction == "full":
            query = f"full content from paper {evaluation.doc_id} related to: {evaluation.key_info[:100]}"
        elif evaluation.context_direction == "before":
            query = f"content before {evaluation.key_info[:50]} in paper {evaluation.doc_id}"
        elif evaluation.context_direction == "after":
            query = f"content after {evaluation.key_info[:50]} in paper {evaluation.doc_id}"
        else:
            query = f"context around {evaluation.key_info[:50]} in paper {evaluation.doc_id}"
        
        sql_constraint = [evaluation.doc_id]
        
        try:
            if self.config.get("use_blacklist", False):
                results, return_itemid = self.retriever.retrieve(
                    query, sql_constraint, blacklist=self.visited_itemid
                )
            else:
                results, return_itemid = self.retriever.retrieve(query, sql_constraint)
            
            if results:
                for i, result in enumerate(results):
                    if str(result.get('item_id', '')) not in [e.chunk_id for e in self.chunk_evaluations]:
                        expanded_chunks.append(result)
                        self.visited_itemid.add(return_itemid[i])
                        
        except Exception as e:
            self._add_to_log_buffer("ERROR", f"Fallback context expansion failed: {str(e)}")
        
        return expanded_chunks

    def _get_document_structure(self, doc_id: str) -> Dict[str, Any]:
        """获取文档结构信息（章节、目录等）"""
        # 这里可以调用检索器获取文档结构
        # 目前返回模拟结构，实际应该从数据库或检索器获取
        return {
            "doc_id": doc_id,
            "sections": [
                {"name": "Abstract", "chunk_range": [0, 2]},
                {"name": "Introduction", "chunk_range": [3, 8]},
                {"name": "Method", "chunk_range": [9, 20]},
                {"name": "Experiments", "chunk_range": [21, 35]},
                {"name": "Conclusion", "chunk_range": [36, 40]}
            ],
            "total_chunks": 40
        }

    def _initialize_document_cursor(self, doc_id: str, chunk_id: str, chunk_position: int) -> DocumentCursor:
        """初始化文档游标"""
        doc_structure = self._get_document_structure(doc_id)
        cursor = DocumentCursor(
            doc_id=doc_id,
            current_chunk_id=chunk_id,
            current_position=chunk_position,
            total_chunks=doc_structure["total_chunks"],
            context_window_size=3,
            document_structure=doc_structure
        )
        self.document_cursors[doc_id] = cursor
        return cursor

    def _get_context_window(self, cursor: DocumentCursor) -> ContextWindow:
        """获取当前游标位置的上下文窗口"""
        half_window = cursor.context_window_size // 2
        start_pos = max(0, cursor.current_position - half_window)
        end_pos = min(cursor.total_chunks - 1, cursor.current_position + half_window)
        
        # 构造SQL查询获取指定范围的chunks
        # 这里需要检索器支持按位置范围查询
        try:
            # 模拟获取窗口内的chunks
            query = f"chunks from position {start_pos} to {end_pos} in document {cursor.doc_id}"
            results, _ = self.retriever.retrieve(query, [cursor.doc_id])
            
            # 生成导航信息
            nav_info = self._generate_navigation_info(cursor, start_pos, end_pos)
            
            return ContextWindow(
                chunks=results,
                start_position=start_pos,
                end_position=end_pos,
                cursor_position=cursor.current_position,
                document_structure=cursor.document_structure,
                navigation_info=nav_info
            )
        except Exception as e:
            self._add_to_log_buffer("ERROR", f"Failed to get context window: {e}")
            return ContextWindow(chunks=[], start_position=start_pos, end_position=end_pos, cursor_position=cursor.current_position)

    def _generate_navigation_info(self, cursor: DocumentCursor, start_pos: int, end_pos: int) -> str:
        """生成导航信息"""
        doc_structure = cursor.document_structure
        if not doc_structure:
            return f"Current window: chunks {start_pos}-{end_pos} of {cursor.total_chunks}"
        
        # 找到当前位置所在的章节
        current_section = "Unknown"
        for section in doc_structure.get("sections", []):
            section_start, section_end = section["chunk_range"]
            if section_start <= cursor.current_position <= section_end:
                current_section = section["name"]
                break
        
        nav_info = f"""
                        Document Structure Navigation:
                        - Document: {cursor.doc_id}
                        - Current Section: {current_section}
                        - Current Window: chunks {start_pos}-{end_pos} (cursor at {cursor.current_position})
                        - Total Document Length: {cursor.total_chunks} chunks

                        Available Sections:
                        """


        for section in doc_structure.get("sections", []):
            section_start, section_end = section["chunk_range"]
            nav_info += f"  - {section['name']}: chunks {section_start}-{section_end}\n"
        
        return nav_info

    def _move_cursor_and_extract_info(self, cursor: DocumentCursor, user_question: str, original_chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """使用游标系统提取上下文信息"""
        extracted_info = []
        max_moves = 5  # 最大移动次数，防止无限循环
        moves_made = 0
        
        while moves_made < max_moves:
            # 获取当前窗口
            context_window = self._get_context_window(cursor)
            
            if not context_window.chunks:
                break
            
            # 构造上下文提取prompt
            window_content = ""
            for i, chunk in enumerate(context_window.chunks):
                marker = " ← [CURSOR]" if i == (cursor.current_position - context_window.start_position) else ""
                window_content += f"Chunk {context_window.start_position + i}{marker}: {chunk.get('content', '')[:300]}...\n\n"
            
            extraction_prompt = f"""
                            You are analyzing a document to find information related to this question: {user_question}

                            Original chunk that triggered this search (from Paper {cursor.doc_id}):
                            {original_chunk.get('content', '')[:500]}

                            Current context window:
                            {context_window.navigation_info}

                            Content:
                            {window_content}

                            Tasks:
                            1. Extract any information from the current window that helps answer the question
                            2. Decide if you need to move the cursor to a different location
                            3. If moving, specify which section or direction to move to

                            Response format:
                            <ExtractedInfo>information relevant to the question, or "none" if nothing useful</ExtractedInfo>
                            <NeedToMove>yes/no</NeedToMove>
                            <MoveDirection>section_name or "before" or "after" or "none"</MoveDirection>
                            <Reason>brief explanation of why you need to move or why current info is sufficient</Reason>
                            """
            
            messages = [
                {"role": "system", "content": "You are an expert at navigating and extracting information from academic documents."},
                {"role": "user", "content": extraction_prompt}
            ]
            
            try:
                response = self._call_llm(messages)
                
                # 解析响应
                info_match = re.search(r'<ExtractedInfo>(.*?)</ExtractedInfo>', response, re.DOTALL)
                move_match = re.search(r'<NeedToMove>(.*?)</NeedToMove>', response, re.IGNORECASE)
                direction_match = re.search(r'<MoveDirection>(.*?)</MoveDirection>', response, re.IGNORECASE)
                reason_match = re.search(r'<Reason>(.*?)</Reason>', response, re.DOTALL)
                
                extracted_text = info_match.group(1).strip() if info_match else ""
                need_to_move = move_match.group(1).strip().lower() == "yes" if move_match else False
                move_direction = direction_match.group(1).strip() if direction_match else ""
                reason = reason_match.group(1).strip() if reason_match else ""
                
                # 记录提取的信息
                if extracted_text and extracted_text.lower() != "none":
                    extracted_info.append({
                        "doc_id": cursor.doc_id,
                        "content": extracted_text,
                        "source_window": f"chunks {context_window.start_position}-{context_window.end_position}",
                        "extraction_reason": reason
                    })
                
                self._add_to_log_buffer("INFO", f"Cursor at position {cursor.current_position}: {reason}")
                
                # 决定是否移动游标
                if not need_to_move:
                    break
                
                # 移动游标
                new_position = self._calculate_new_cursor_position(cursor, move_direction)
                if new_position == cursor.current_position:
                    break  # 无法移动，结束
                
                cursor.current_position = new_position
                moves_made += 1
                
            except Exception as e:
                self._add_to_log_buffer("ERROR", f"Context extraction failed: {e}")
                break
        
        return extracted_info

    def _calculate_new_cursor_position(self, cursor: DocumentCursor, move_direction: str) -> int:
        """计算新的游标位置"""
        if not move_direction or move_direction.lower() == "none":
            return cursor.current_position
        
        doc_structure = cursor.document_structure
        
        if move_direction.lower() == "before":
            return max(0, cursor.current_position - cursor.context_window_size)
        elif move_direction.lower() == "after":
            return min(cursor.total_chunks - 1, cursor.current_position + cursor.context_window_size)
        elif doc_structure:
            # 移动到特定章节
            for section in doc_structure.get("sections", []):
                if section["name"].lower() == move_direction.lower():
                    section_start, section_end = section["chunk_range"]
                    return (section_start + section_end) // 2  # 移动到章节中间
        
        return cursor.current_position





if __name__ == "__main__":
    #定义logger的输出文件
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("react_agent_v2.log"))
    logger.addHandler(logging.StreamHandler())

    with open("/data4/students/zhangguangyin/chatNum/config/config.yaml","r")as f:
        config=yaml.safe_load(f)
    
    # 添加chunk评估配置
    config["use_chunk_evaluation"] = True  # 启用智能chunk评估
    
    agent=ReActAgent(config)
    question=f"""<Question> Please help me find the performance of methods proposed in different papers on the Accuracy metric for the Graph Classification task on the MNIST (MNIST) dataset, list the top three metric result. 
        Requirements:
        1.for each paper, you should only give one metric result (the highest one) of its own proposed method, since most papers will compare the performance of methods proposed in other papers and some variants of the same method in abalation study. 
        2. The top three metric result means that you need to find at least three papers that have reported the metric result .
        """
    question=question+"""
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
    
    # 使用新的性能比较方法

    result, docidlist = agent.run(question)
    print("Result:")
    print(result)
    print("Document IDs:", docidlist)
    

    
    # # 也可以使用标准方法进行比较
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