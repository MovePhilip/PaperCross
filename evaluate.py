from generation.model_generation import generationmodel
from embedding.split_doc import embeddoc
import yaml
from retrieval.retriever import Retriever
import os
import torch
import json
import json_repair
import logging
from agent.direct_qa import DirectQA
# from agent.ReAct import ReActAgent
from agent.ReAct_v1_5 import ReActAgent
from agent.resp import ReSPFramework
from agent.PaperAgent import PaperAgent
from agent.visual_agent import visual_agent
import argparse
import concurrent.futures
import threading
import traceback
import hashlib

logger = logging.getLogger("myapp")




def direct_qa_list(config):
    logger.setLevel(logging.INFO)
    #将日志同时输出都文件和控制台
    handler = logging.FileHandler(f"evaluatedirectqa_list_logger.txt")
    handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.info("Start print log")
    logger.info(config)
    num_gpus = torch.cuda.device_count()
    logger.info("当前可用GPU数量:", num_gpus)
    with open("/data4/students/zhangguangyin/chatNum/qa_selected4_200_updated-ranked-filtered.json")as f:
           doc=json.load(f)
    print(type(config))
    agent=DirectQA(config=config)
    evaluate_list=[]
    result_f=open(f"/data4/students/zhangguangyin/chatNum/evaluate_directqa_list.txt","w")
    for key, value in doc.items():
            logger.info(value["meta_info"])
            question=f"""Please give me the performance of methods proposed in different papers on the {value["meta_info"]["metric"]} metric for the {value["meta_info"]["task"]} task on the {value["meta_info"]["datasets_short"]} ({value["meta_info"]["datasets"]}) dataset, list the top three metric result. 
            Requirements:
            1.for each paper, you should only give one metric result (the highest one) of its own proposed method
            2. The top three metric result means that you need to give at least three papers that have reported the metric result .
            """
            question=question+"""
                                You need to give the result in JSON format:
                                [
                                    {"rank_id":1
                                    "method":"model name",
                                    "value": "metric value"
                                    "paper_id": "arxiv paper id"  //just give the id number, such as 1111.1111
                                    },
                                    {"rank_id":2
                                    "method":"model name",
                                    "value": "metric value"
                                    "paper_id":"arxiv paper id"
                                    },
                                    {"rank_id":3
                                    "method":"model name",
                                    "value": "metric value"
                                    "paper_id": "arxiv paper id"
                                    }
                                ]
                                
                                
                                """

            logger.info(question)
            logger.info(value["updated_answer2"][:3])
            retrieval_list=[itemd[0] for itemd in value["updated_answer3"]]
            outputtext=agent.run(question,docidlist=retrieval_list)

            json_outputtext=json_repair.loads(outputtext)
            evaluate_list.append({"question":question,"real_answer":value["updated_answer2"],"predicted_answer":outputtext,"predicted_rank_id":key,"json_outputtext":json_outputtext})
            result_f.write(json.dumps({"question":question,"real_answer":value["updated_answer2"],"predicted_answer":outputtext,"predicted_rank_id":key,"json_outputtext":json_outputtext},ensure_ascii=False)+"\n")
            result_f.flush()

    result_f.close()
    with open(f"/data4/students/zhangguangyin/chatNum/evaluate_directqa_list.json","w")as f:
        json.dump(evaluate_list,f,indent=4,ensure_ascii=False)




def evaluate_paperagent(config):
    logger.setLevel(logging.INFO)
    #将日志同时输出都文件和控制台
    handler = logging.FileHandler(f"evaluatepaperagent.txt")
    handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.info("Start print log")
    logger.info(config)
    num_gpus = torch.cuda.device_count()
    logger.info("当前可用GPU数量:", num_gpus)

    if not os.path.exists(config["doc_embedding_name"]):
        doc_embeddings,new_embedding_docid=embeddoc(config)
 
    with open("/data4/students/zhangguangyin/chatNum/qa_selected4_200_updated-ranked-filtered.json")as f:
           doc=json.load(f)

    agent=PaperAgent(config=config)
    evaluate_list=[]
    result_f=open(f"/data4/students/zhangguangyin/chatNum/evaluate_paperagent.txt","w")
    for key, value in doc.items():
        logger.info(value["meta_info"])
        question=f"""Please help me find the performance of methods proposed in different papers on the {value["meta_info"]["metric"]} metric for the {value["meta_info"]["task"]} task on the {value["meta_info"]["datasets_short"]} ({value["meta_info"]["datasets"]}) dataset, list the top three metric result. 
        Requirements:
        1.For each paper, you should only give one metric result (the highest one) of its own proposed method, not the method cited from other papers, since most papers will cite the performance of methods proposed in other papers and some variants of the same method in abalation study. 
        2. The top three metric result means that you need to find at least three papers that have reported the metric result .
        3. The value should be accord with the experiment setting as described in the question. Namely, the right dataset and the right metric.
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
                            
                            """
        logger.info(question)
        logger.info(value["updated_answer2"][:3])
        retrieval_list=[itemd[0] for itemd in value["updated_answer3"]]
        outputtext=agent.run(question,docidlist=retrieval_list)

        json_outputtext=json_repair.loads(outputtext)
        evaluate_list.append({"question":question,"real_answer":value["updated_answer2"],"predicted_answer":outputtext,"predicted_rank_id":key,"json_outputtext":json_outputtext})
        result_f.write(json.dumps({"question":question,"real_answer":value["updated_answer2"],"predicted_answer":outputtext,"predicted_rank_id":key,"json_outputtext":json_outputtext},ensure_ascii=False)+"\n")
        result_f.flush()

    result_f.close()
    with open(f"/data4/students/zhangguangyin/chatNum/evaluate_paperagent_list.json","w")as f:
        json.dump(evaluate_list,f,indent=4,ensure_ascii=False)



def evaluate_list(config):
    logger.setLevel(logging.INFO)
    #将日志同时输出都文件和控制台
    handler = logging.FileHandler(f"evaluateloglist_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"].replace("_zijie","")}_{config["simple_retrieval"]}_simple_bm25_{config["simple_bm25"]}.txt")
    handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(handler)
    # logger.addHandler(console_handler)
    logger.info("Start print log")
    logger.info(config)
    num_gpus = torch.cuda.device_count()
    logger.info("当前可用GPU数量:", num_gpus)

    if not os.path.exists(config["doc_embedding_name"]):
        doc_embeddings,new_embedding_docid=embeddoc(config)
    with open("/data4/students/zhangguangyin/chatNum/qa_selected4_200_updated-ranked-filtered_v2.json")as f:
           doc=json.load(f)

    # direct_qa=DirectQA(config,retriever=None,generationmodel=None)
    # react_agent=ReActAgent(config=config,retriever=None,generation_model=None)
    if config["agent_name"]=="ReAct":
        agent=ReActAgent(config=config,retriever=None,generation_model=None)
    elif config["agent_name"]=="resp":
        agent=ReSPFramework(config=config,)
    elif config["agent_name"]=="visual_agent":
        agent=visual_agent(config=config)
    else:
        raise ValueError(f"Invalid agent name: {config["agent_name"]}")
    
    #对config对象进行hash
    config_hash = hashlib.sha256(str(config).encode()).hexdigest()
    output_dir = f"/data4/students/zhangguangyin/chatNum/second_version_result/{config_hash}"
    os.makedirs(output_dir, exist_ok=True)
    #保存config文件到second_version_result，以config_hash为文件名
    with open(f"{output_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    existing_files = set(os.listdir(output_dir))
    questions = []
    retrieval_lists = []
    save_paths = []
    answerlists = []
    question_id=0
    for key, value in doc.items():
        base_key = f"{key.split('/')[-1].split('.')[0]}"
        per_file_name = f"{base_key}.json"
        per_file_path = os.path.join(output_dir, per_file_name)
        if per_file_name in existing_files:
            logger.info(f"Skip existing result file for key {key}")
            continue
        question=f"""<Question> Please help me find the performance of methods proposed in different papers on the {value["meta_info"]["metric"]} metric for the {value["meta_info"]["task"]} task on the {value["meta_info"]["datasets_short"]} ({value["meta_info"]["datasets"]}) dataset, list the top three metric result. 
        Requirements:
        1.for each paper, you should only give one metric result (the highest one) of its own proposed method, since most papers will compare the performance of methods proposed in other papers and some variants of the same method in abalation study. 
        2. The top three metric result means that you need to find at least three papers that have reported the metric result.
        3. The article ID must correspond to the method name, meaning the article ID should refer to the paper in which the method was originally proposed.

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
        retrieval_list = [itemd[0] for itemd in value["updated_answer3"]] if config["small_trieval_range"] else []
        questions.append(question)
        retrieval_lists.append(retrieval_list)
        save_paths.append(per_file_path)
        answerlists.append(value["updated_answer3"])
        question_id+=1

    def process_question(args):
        key, value, question, retrieval_list, agent = args
        try:
            logger.info(value["meta_info"])
            logger.info(question)
            logger.info(value["updated_answer3"][:3])
            
            if config["small_trieval_range"]:
                logger.info(f"retrieval_list:{retrieval_list}")
                save_content = agent.run(question, retrieval_list=retrieval_list)
            else:
                save_content = agent.run(question)

            result = {
                "question": question,
                "real_answer": value["updated_answer3"],
                "predicted_answer": save_content,
                "predicted_rank_id": key
            }
            return result
        except Exception as e:
            logger.error(f"Error processing question {key}: {e}")
            return {
                "question": question,
                "real_answer": value["updated_answer3"],
                "predicted_answer": "",
                "predicted_rank_id": key,
                "error": str(e)
            }



    # Use agent's built-in parallel runner to execute and save per-query results
    max_workers = min(4, os.cpu_count() or 4)
    if hasattr(agent, "parallel_run"):
        agent.parallel_run(questions, retrieval_lists=retrieval_lists, save_paths=save_paths, answerlists=answerlists, max_workers=max_workers)
    else:
        # Fallback: sequential run if agent doesn't support parallel_run
        for q, rlist, path, answerlist in zip(questions, retrieval_lists, save_paths, answerlists):
            try:
                res = agent.run(q, retrieval_list=rlist, answerlists=answerlist)
            except Exception as e:
                res = {"question": q, "predicted_answer": "", "error": str(e)}
            with open(path, "w") as pf:
                json.dump(res, pf, ensure_ascii=False, indent=1)








#
def evaluate_2hop(config, start_from=0):
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"evaluatelog2hop_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"].replace("_zijie","")}_{config["simple_retrieval"]}_{config["simple_bm25"]}.txt")
    handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.info("Start print log")
    logger.info(config)
    if not os.path.exists(config["doc_embedding_name"]):
        doc_embeddings, new_embedding_docid = embeddoc(config)
    with open("/data4/students/zhangguangyin/chatNum/qa_selected4_200_updated-ranked-2hop_question.json","r") as f:
        doc = json.load(f)
    if config["agent_name"] == "ReAct":
        agent = ReActAgent(config=config, retriever=None, generation_model=None)
    elif config["agent_name"] == "resp":
        agent = ReSPFramework(config=config,)
    elif config["agent_name"] == "direct_qa":
        agent = DirectQA(config=config, retriever=None, generation_model=None)
    elif config["agent_name"] == "visual_agent":
        agent = visual_agent(config=config)
    else:
        raise ValueError(f"Invalid agent name: {config['agent_name']}")

    # Output directory (per-run hash), store config as well
    config_hash = hashlib.sha256(str(config).encode()).hexdigest()
    output_dir = f"/data4/students/zhangguangyin/chatNum/second_version_result_2hop/{config_hash}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    existing_files = set(os.listdir(output_dir))

    question_id = 0
    tasks = []
    for key, value in doc.items():
        for key2, value2 in value.items():
            for itemid, item in enumerate(value2):
                if question_id < start_from:
                    question_id += 1
                    continue
                if "concated_question" not in item:
                    print(key, key2)
                    raise
                question = item["concated_question"]
                formatted_question = """\n\nPlease give your final response in following json format, in \"paper_id\" field, give the paper id where you find the answer in (only one paper id), in \"answer\" field, give the answer for the question.\n    {\n     \"paper_id\":\n     \"answer\":\n     }\n    """
                full_question = question + formatted_question
                result_id = f"{key2}_{itemid}"
                per_file_name = f"{result_id}.json"
                if per_file_name in existing_files:
                    logger.info(f"Skip existing result file for {per_file_name}")
                    question_id += 1
                    continue
                tasks.append((full_question, item["answer"], key2, key, itemid, result_id))
                question_id += 1

    def process_question(args):
        full_question, real_answer, question_type, doc_id, itemid, result_id = args
        try:
            logger.info(f"Question: {full_question}")
            logger.info(f"Real Answer: {real_answer}")
            outputtext, docidlist = agent.run(full_question, retrieval_list=[])
            logger.info(f"Predicted answer: {outputtext}")
            parsed_output = json_repair.loads(outputtext)
            predicted_doc_id = parsed_output.get("paper_id", "")
            predicted_answer = parsed_output.get("answer", "")
            result = {
                "question": full_question,
                "real_answer": real_answer,
                "predicted_answer": predicted_answer,
                "predicted_doc_id": predicted_doc_id,
                "question_type": question_type,
                "doc_id": doc_id,
                "docidlist": docidlist,
                "itemid": itemid,
                "result_id": result_id
            }
            return result
        except Exception as e:
            logger.error(f"Error processing question {doc_id}-{itemid}: {e}")
            return {
                "question": full_question,
                "real_answer": real_answer,
                "predicted_answer": "",
                "predicted_doc_id": "",
                "question_type": question_type,
                "doc_id": doc_id,
                "docidlist": [],
                "itemid": itemid,
                "result_id": result_id,
                "error": str(e)
            }

    max_workers = min(10, os.cpu_count() or 10)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {executor.submit(process_question, args): args for args in tasks}
        for future in concurrent.futures.as_completed(future_to_args):
            res = future.result()
            per_file_path = os.path.join(output_dir, f"{res.get('result_id')}.json")
            with lock:
                logger.info(f"Saving 2hop result to {per_file_path}")
                with open(per_file_path, "w") as pf:
                    json.dump(res, pf, ensure_ascii=False, indent=1)







def evaluate_2hop_directqa(config, start_from=0):
    #这个是直接回答，没有拼接全文，也没有做检索，就是直接的生成结果
    # logger.setLevel(logging.INFO)
    # handler = logging.FileHandler(f"evaluatelog2hop_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"]}.txt")
    # handler.setLevel(logging.INFO)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(handler)
    # logger.addHandler(console_handler)
    # logger.info("Start print log")
    # logger.info(config)
    if not os.path.exists(config["doc_embedding_name"]):
        doc_embeddings, new_embedding_docid = embeddoc(config)
    with open("/data4/students/zhangguangyin/chatNum/qa_selected4_200_updated-ranked-2hop_question.json","r") as f:
        doc = json.load(f)
    if config["agent_name"] == "ReAct":
        agent = ReActAgent(config=config, retriever=None, generation_model=None)
    elif config["agent_name"] == "resp":
        agent = ReSPFramework(config=config,)
    elif config["agent_name"] == "direct_qa":
        agent = DirectQA(config=config, retriever=None, generation_model=None)
    else:
        raise ValueError(f"Invalid agent name: {config['agent_name']}")
    evaluate_list = []
    result_path = f"/data4/students/zhangguangyin/chatNum/evaluate_2hop_json_list_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"]}.txt"
    if os.path.exists(result_path):
        with open(result_path,"r")as f:
             lines=f.readlines()
             num_lines=len(lines)
             logger.info(f"num_lines:{num_lines}, start from {num_lines}")
             start_from=num_lines
    
    question_id = 0
    tasks = []
    for key, value in doc.items():
        for key2, value2 in value.items():
            for itemid, item in enumerate(value2):
                if question_id < start_from:#这个到最后的答案部分应该还是要去个重的，因为可能会有重复的
                    question_id += 1
                    continue
                if "concated_question" not in item:
                    print(key, key2)
                    raise
                question = item["concated_question"]
                formatted_question = """\n\nPlease give your final response in following json format, in \"paper_id\" field, give the arxiv paper id where you remember the answer in (only one paper id), in \"answer\" field, give the answer for the question.\n    {\n     \"paper_id\":\n     \"answer\":\n     }\n    """
                full_question = question + formatted_question
                tasks.append((full_question, item["answer"], key2, key, itemid))
                question_id += 1




    def process_question(args):
        full_question, real_answer, question_type, doc_id, itemid = args
        try:
            logger.info(f"Question: {full_question}")
            logger.info(f"Real Answer: {real_answer}")
            outputtext, docidlist = agent.run(full_question, retrieval_list=[])
            logger.info(f"Predicted answer: {outputtext}")
            parsed_output = json_repair.loads(outputtext)
            predicted_doc_id = parsed_output.get("paper_id", "")
            predicted_answer = parsed_output.get("answer", "")
            result = {
                "question": full_question,
                "real_answer": real_answer,
                "predicted_answer": predicted_answer,
                "predicted_doc_id": predicted_doc_id,
                "question_type": question_type,
                "doc_id": doc_id,
                "docidlist": docidlist,
                "itemid": itemid
            }
            return result
        except Exception as e:
            logger.error(f"Error processing question {doc_id}-{itemid}: {e}")
            return {
                "question": full_question,
                "real_answer": real_answer,
                "predicted_answer": "",
                "predicted_doc_id": "",
                "question_type": question_type,
                "doc_id": doc_id,
                "docidlist": [],
                "itemid": itemid,
                "error": str(e)
            }
    results = []
    max_workers = min(15, os.cpu_count() or 15)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor, open(result_path, "a") as result_f:
        future_to_args = {executor.submit(process_question, args): args for args in tasks}
        for future in concurrent.futures.as_completed(future_to_args):
            res = future.result()
            results.append(res)
            with lock:
                result_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                result_f.flush()
    with open(f"/data4/students/zhangguangyin/chatNum/evaluate_2hop_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"]}.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)




def evaluate_2hop_max_table(config, start_from=0):
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"evaluatelog2hop_maxtable_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"]}_{config["simple_retrieval"]}_{config["simple_bm25"]}.txt")
    handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.info("Start print log")
    logger.info(config)
    if not os.path.exists(config["doc_embedding_name"]):
        doc_embeddings, new_embedding_docid = embeddoc(config)
    with open("/data4/students/zhangguangyin/chatNum/qa_selected4_200_updated-ranked-2hop_question_max_tabel_loacted.json","r") as f:
        doc = json.load(f)
    if config["agent_name"] == "ReAct":
        agent = ReActAgent(config=config, retriever=None, generation_model=None)
    elif config["agent_name"] == "resp":
        agent = ReSPFramework(config=config,)
    elif config["agent_name"] == "direct_qa":
        agent = DirectQA(config=config, retriever=None, generation_model=None)
    elif config["agent_name"] == "visual_agent":
        agent = visual_agent(config=config)
    else:
        raise ValueError(f"Invalid agent name: {config['agent_name']}")
    evaluate_list = []
    result_path = f"/data4/students/zhangguangyin/chatNum/evaluate_2hop_maxtable_json_list_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"]}_{config["simple_retrieval"]}_{config["simple_bm25"]}.txt"
    if os.path.exists(result_path):
        with open(result_path,"r")as f:
             lines=f.readlines()
             num_lines=len(lines)
             logger.info(f"num_lines:{num_lines}, start from {num_lines}")
             start_from=num_lines
    
    question_id = 0
    tasks = []
    for key, value in doc.items():
        for key2, value2 in value.items():
            for itemid, item in enumerate(value2):
                if question_id < start_from:#这个到最后的答案部分应该还是要去个重的，因为可能会有重复的
                    question_id += 1
                    continue
                if "concated_question" not in item:
                    print(key, key2)
                    raise
                question = item["concated_question"]
                formatted_question = """\n\nPlease give your final response in following json format, in \"paper_id\" field, give the paper id where you find the answer in (only one paper id), in \"answer\" field, give the answer for the question.\n    {\n     \"paper_id\":\n     \"answer\":\n     }\n    """
                full_question = question + formatted_question
                tasks.append((full_question, item["answer"], key2, key, itemid))
                question_id += 1




    def process_question(args):
        full_question, real_answer, question_type, doc_id, itemid = args
        try:
            logger.info(f"Question: {full_question}")
            logger.info(f"Real Answer: {real_answer}")
            outputtext, docidlist = agent.run(full_question, retrieval_list=[])
            logger.info(f"Predicted answer: {outputtext}")
            parsed_output = json_repair.loads(outputtext)
            predicted_doc_id = parsed_output.get("paper_id", "")
            predicted_answer = parsed_output.get("answer", "")
            result = {
                "question": full_question,
                "real_answer": real_answer,
                "predicted_answer": predicted_answer,
                "predicted_doc_id": predicted_doc_id,
                "question_type": question_type,
                "doc_id": doc_id,
                "docidlist": docidlist,
                "itemid": itemid
            }
            return result
        except Exception as e:
            logger.error(f"Error processing question {doc_id}-{itemid}: {traceback.format_exc(e)}")
            return {
                "question": full_question,
                "real_answer": real_answer,
                "predicted_answer": "",
                "predicted_doc_id": "",
                "question_type": question_type,
                "doc_id": doc_id,
                "docidlist": [],
                "itemid": itemid,
                "error": str(e)
            }
    results = []
    max_workers = min(10, os.cpu_count() or 10)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor, open(result_path, "a") as result_f:
        future_to_args = {executor.submit(process_question, args): args for args in tasks}
        for future in concurrent.futures.as_completed(future_to_args):
            res = future.result()
            results.append(res)
            with lock:
                result_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                result_f.flush()
    with open(f"/data4/students/zhangguangyin/chatNum/evaluate_2hop_maxtable_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"]}_{config["simple_retrieval"]}_{config["simple_bm25"]}.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)






def evaluate_2hop_max_table_directqa(config, start_from=0):
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"evaluatelog2hop_maxtable_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"]}.txt")
    handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.info("Start print log")
    logger.info(config)
    if not os.path.exists(config["doc_embedding_name"]):
        doc_embeddings, new_embedding_docid = embeddoc(config)
    with open("/data4/students/zhangguangyin/chatNum/qa_selected4_200_updated-ranked-2hop_question_max_tabel_loacted.json","r") as f:
        doc = json.load(f)
    if config["agent_name"] == "ReAct":
        agent = ReActAgent(config=config, retriever=None, generation_model=None)
    elif config["agent_name"] == "resp":
        agent = ReSPFramework(config=config,)
    elif config["agent_name"] == "direct_qa":
        agent = DirectQA(config=config, retriever=None, generation_model=None)
    else:
        raise ValueError(f"Invalid agent name: {config['agent_name']}")
    evaluate_list = []
    result_path = f"/data4/students/zhangguangyin/chatNum/evaluate_2hop_maxtable_json_list_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"]}.txt"
    if os.path.exists(result_path):
        with open(result_path,"r")as f:
             lines=f.readlines()
             num_lines=len(lines)
             logger.info(f"num_lines:{num_lines}, start from {num_lines}")
             start_from=num_lines
    
    question_id = 0
    tasks = []
    for key, value in doc.items():
        for key2, value2 in value.items():
            for itemid, item in enumerate(value2):
                if question_id < start_from:#这个到最后的答案部分应该还是要去个重的，因为可能会有重复的
                    question_id += 1
                    continue
                if "concated_question" not in item:
                    print(key, key2)
                    raise
                question = item["concated_question"]
                formatted_question = """\n\nPlease give your final response in following json format, in \"paper_id\" field, give the arxiv paper id where you remember the answer in (only one paper id), in \"answer\" field, give the answer for the question.\n    {\n     \"paper_id\":\n     \"answer\":\n     }\n    """
                full_question = question + formatted_question
                tasks.append((full_question, item["answer"], key2, key, itemid))
                question_id += 1




    def process_question(args):
        full_question, real_answer, question_type, doc_id, itemid = args
        try:
            logger.info(f"Question: {full_question}")
            logger.info(f"Real Answer: {real_answer}")
            outputtext, docidlist = agent.run(full_question, retrieval_list=[])
            logger.info(f"Predicted answer: {outputtext}")
            parsed_output = json_repair.loads(outputtext)
            predicted_doc_id = parsed_output.get("paper_id", "")
            predicted_answer = parsed_output.get("answer", "")
            result = {
                "question": full_question,
                "real_answer": real_answer,
                "predicted_answer": predicted_answer,
                "predicted_doc_id": predicted_doc_id,
                "question_type": question_type,
                "doc_id": doc_id,
                "docidlist": docidlist,
                "itemid": itemid
            }
            return result
        except Exception as e:
            logger.error(f"Error processing question {doc_id}-{itemid}: {e}")
            return {
                "question": full_question,
                "real_answer": real_answer,
                "predicted_answer": "",
                "predicted_doc_id": "",
                "question_type": question_type,
                "doc_id": doc_id,
                "docidlist": [],
                "itemid": itemid,
                "error": str(e)
            }
    results = []
    max_workers = min(10, os.cpu_count() or 6)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor, open(result_path, "a") as result_f:
        future_to_args = {executor.submit(process_question, args): args for args in tasks}
        for future in concurrent.futures.as_completed(future_to_args):
            res = future.result()
            results.append(res)
            with lock:
                result_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                result_f.flush()
    with open(f"/data4/students/zhangguangyin/chatNum/evaluate_2hop_maxtable_{config["agent_name"]}_{config["retrieval_top_k"]}_{config["embedding_model"]}_{config["small_trieval_range"]}_{config["generation_model"]}.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)







if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Evaluate different agents with config.")
    parser.add_argument('--mode', type=str, required=True, choices=['paperagent', 'list', '2hop','2hopmaxtable','directqa'], help='Evaluation mode')
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml file')
    parser.add_argument('--start_from', type=int, default=0, help='Start from question id')
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.mode == 'paperagent':
        evaluate_paperagent(config)
    elif args.mode == 'list':
        evaluate_list(config)
    elif args.mode == '2hop':
        evaluate_2hop(config,args.start_from)
    elif args.mode == '2hopmaxtable':
        evaluate_2hop_max_table(config,args.start_from)
    elif args.mode == 'directqa':
        # direct_qa_list(config)
        evaluate_2hop_directqa(config,args.start_from)
        evaluate_2hop_max_table_directqa(config,args.start_from)





#python evaluate.py --mode 2hop --config config/config_2hop.yaml --start_from 226
#nohup python evaluate.py --mode 2hop --config config/config_2hop.yaml &
#nohup python evaluate.py --mode 2hopmaxtable --config config/config_2hop.yaml &
#nohup python evaluate.py --mode list --config config/config.yaml &
#nohup python evaluate.py --mode directqa --config config/config_directqa.yaml &

#nohup python evaluate.py --mode list --config config/config_visual_agent.yaml &
#nohup python evaluate.py --mode 2hop --config config/config_visual_agent.yaml &
#nohup python evaluate.py --mode 2hopmaxtable --config config/config_visual_agent.yaml &






