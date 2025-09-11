import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import json
import shutil

from .embedding_jina_v3 import embedder_jina
from .embedding_qwen2_7b import qwen_embedder
from .embedding_colpali import embedder_colpali
from .embedding_gme import embedder_gme
import pickle
from typing import cast
import yaml
import torch
from .embedding_bge_m3 import bge_m3_embedder
import requests
import numpy as np
from transformers import  AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B',cache_dir="/data4/students/zhangguangyin/LLMs")

with open("/data4/students/zhangguangyin/chatNum/qa_selected4_200_updated-ranked-filtered_v2.json","r")as f:
       rankedquestion=json.load(f)



docpathlist=set()
# for idx,(key,value) in enumerate(rankedquestion.items()):
#     # if idx>=501:
#     #      break
#     # # print(key)
#     for docid in value["src_docs"]:
#         if docid not in doclist:
#             doclist.add(docid)
# doclist=list(doclist)
# print("文档数量一共是",len(doclist))

for idx,(key,value) in enumerate(rankedquestion.items()):
    for item in value["updated_answer3"]:#src_docs，updated_answer3
        docid=item[0]
        docidpath1=f"/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/{docid}/{docid}_content_list.json"
        docidpath2=f"/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/{docid}/ocr/{docid}_content_list.json"
        if os.path.exists(docidpath1):
            docpathlist.add(docidpath1)
        elif os.path.exists(docidpath2):
            docpathlist.add(docidpath2)
        else:
                print(f"docidpath:{docidpath1} and {docidpath2} not exists")
                continue


        # if docid not in doclist:
        #     doclist.add(docid)
docpathlist=list(docpathlist)
print("文档数量一共是",len(docpathlist))

def split_by_tokens(text: str, max_tokens:int):
    #按照断行\n，英文句号. 英文逗号,这四种符号优先级对字符串进行分割，从最大max_tokens处向左侧寻找最优分割点，当前一种符号能成功分割时，返回分割后的两个字符串
    split_symbols = ["\n", ".", ",", "\t"]
    best_split_point=0
    # 编码时返回 offset mapping
    tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    # tokens["offset_mapping"] 是一个 (start, end) 的列表
    offsets = tokens["offset_mapping"]

    if max_tokens < len(offsets):
        char_start, char_end = offsets[max_tokens]
    else:
        raise ValueError("max_tokens is too large")
    initial_split_position=char_end
    for symbol in split_symbols:
        split_point=text.rfind(symbol, 0, initial_split_position)
        if split_point>best_split_point:
            best_split_point=split_point
    if best_split_point>0:
        return text[:best_split_point], text[best_split_point:]
    else:
        raise ValueError("No split point found")



def token_len(text: str) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return len(text)




def embeddoc(config):

    segmentlist=[]#这个是用来生成输入文本的
    embedding_docid=[]#这个只是用来记录id的
    papertitledict={}

    doc_embeddingsavename=config["doc_embedding_name"]
    if os.path.exists(doc_embeddingsavename):
        return "doc embedding complete"
    text_chunk_size=config["text_chunk_size"]
    for docpath in docpathlist:
        # docpath=f"/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/{docid}/{docid}_content_list.json"
        docid=docpath.split("/")[-1].split("_")[0]
        with open(docpath,"r")as f:
            parsedcontent=json.load(f)
        segment=""
        para_count=0
        # 这个字典用来记录各级标题所在的segment的位置
        papertitlelist=[]
        # 新增：临时存储当前segment中的标题
        current_titles = []#这个列表用来存储当前chunk中包含的title列表
        firsttitle=-1

        for itemid,item in enumerate(parsedcontent):
            if item["type"]=="text":
                # 这个情况是用来处理加上chunk之后长度超过text_chunk_size
                if token_len(segment+"\n"+item["text"]+"\n")>text_chunk_size:
                    # 在添加chunk前，检查是否有标题需要记录
                    for title in current_titles:
                        papertitlelist.append([title,len(segmentlist)])
                    segmentlist.append(segment)
                    embedding_docid.append([docid,"text"])
                    segment=""
                    current_titles = []  # 清空标题列表
                
                #判断当前块大小是否超过了text_chunk_size，如果超过了，则进行分割
                if token_len(item["text"])>text_chunk_size:

                    head_text, tail_text = split_by_tokens(item["text"], text_chunk_size)
                    if "text_level" in item and item["text_level"]==1:
                        current_titles.append(head_text)
                        for title in current_titles:
                            papertitlelist.append([title,len(segmentlist)])
                        segment="\n # "+head_text+"\n"
                        segmentlist.append(segment)
                        embedding_docid.append([docid,"text"])
                        segment=tail_text
                        current_titles = []  # 清空标题列表
                        para_count=1
                    else:
                        segment="\n "+head_text+"\n"
                        for title in current_titles:
                            papertitlelist.append([title,len(segmentlist)])
                        segmentlist.append(segment)
                        embedding_docid.append([docid,"text"])
                        segment=tail_text
                        current_titles = []  # 清空标题列表
                        para_count=1
                
                else:
                    #当前块没有超过规定大小
                    if "text_level" in item and item["text_level"]==1:
                        if item["text"]!="":
                            segment+="\n # "+item["text"]+"\n"
                            para_count=1
                            if firsttitle==-1:
                                current_titles.append(f"Paper Title: {item['text']}")
                                firsttitle=0
                            else:
                                current_titles.append(item["text"])  # 记录标题
                            para_count+=1

                    else:
                        if item["text"]!="":
                            segment+="\n"+item["text"]
                            para_count+=1

                
            if item["type"]=="equation":
                if token_len(item["text"])>text_chunk_size:
                    item["text"]=item["text"][:text_chunk_size]

                if token_len(segment+"\n"+item["text"]+"\n")>text_chunk_size:
                    # 在添加chunk前，检查是否有标题需要记录
                    for title in current_titles:
                        papertitlelist.append([title,len(segmentlist)])
                    
                    segmentlist.append(segment)
                    embedding_docid.append([docid,"text"])
                    segment=item["text"]+"\n"
                    para_count=1
                    current_titles = []  # 清空标题列表
                else:
                    segment=segment+item["text"]+"\n"
                    para_count+=1


            if item["type"]=="table" and item["img_path"].strip()!="" and "table_body" in item:
                temp=""
                if "table_caption" in item and len(item["table_caption"])!=0:
                    temp=temp+"The table caption:\n\n"+"\n".join(item["table_caption"])+"\n\n"
                if "table_footnote" in item and len(item["table_footnote"])!=0:
                    temp=temp+"\n"+"\n".join(item["table_footnote"])+"\n\n"
                temp=temp+"The html body of the table: \n"+item["table_body"]+"\n"
                

                #这个本来是用来处理是否将完整的表格表述用来生成emedding的，现在实现的方式两者之间没什么区别了
                if "table_context_description" in item:
                    temp=temp+"\n"+item["table_context_description"]
                segmentlist.append(temp)
                embedding_docid.append([docid,"table",itemid])


            if item["type"]=="image":
                if "chart_parse" in item:
                    has_chart=True
                    temp=""
                    
                    if "img_caption" in item:#这里没有就是不加的意思
                        try:
                            temp=temp+"\n"+"\n".join(item["img_caption"])
                        except:
                            print(type(temp))
                            print(type(item["img_caption"]))
                    temp=temp+"\n"+str(item["chart_parse"])
                    segmentlist.append(temp)
                    embedding_docid.append([docid,"image",itemid])
            
            if para_count>=5:
                # 在添加chunk前，检查是否有标题需要记录
                for title in current_titles:
                    papertitlelist.append([title,len(segmentlist)])
                
                segmentlist.append(segment)
                embedding_docid.append([docid,"text"])
                segment=""
                para_count=0
                current_titles = []  # 清空标题列表

        # 处理文档末尾剩余的segment
        if segment.strip() != "":
            # 在添加最后一个chunk前，检查是否有标题需要记录
            for title in current_titles:
                papertitlelist.append([title,len(segmentlist)])
            
            segmentlist.append(segment)
            embedding_docid.append([docid,"text"])
        
        papertitledict[docid]=papertitlelist

    # 选择端口
    if config["embedding_model"]=="jina":
        embedding_port = config.get("jina_v3_embedding_port", 5438)
    elif config["embedding_model"]=="qwen2_7b":
        embedding_port = config.get("qwen2_7b_embedding_port", 5437)
    elif config["embedding_model"]=="bge_m3":
        embedding_port = config.get("bge_m3_embedding_port", 5439)
    else:
        raise ValueError(f"Unknown embedding model: {config['embedding_model']}")
    embedding_url = f"http://localhost:{embedding_port}/embedding"
  

    print(f"总chunk个数是{len(segmentlist)}")
    # 通过API获取embedding
    payload = {"method": "embed_document", "documents": segmentlist}
    response = requests.post(embedding_url, json=payload)
    if response.status_code == 200:
        embeddingresult = response.json()["embeddings"]
        # 将embeddingresult转换为numpy数组
        embeddingresult = np.array(embeddingresult)
    else:
        raise RuntimeError(f"Embedding API error: {response.text}")
    new_embedding_docid=[ [segmentlist[i],embedding_docid[i]]   for i in range(len(segmentlist))]

    with open(config["doc_embedding_name"],"wb")as f:#.replace(".pickle","_v2.pickle")
         pickle.dump(embeddingresult,f)
    with open(config["doc_embedding_idlistname"],"w")as f:#.replace(".json","_v2.json")
         json.dump(new_embedding_docid,f,ensure_ascii=False,indent=1)
    with open(config["papertitledictname"],"w")as f:
         json.dump(papertitledict,f,ensure_ascii=False,indent=1)
    
    print("生成完毕")

    # return embeddingresult,new_embedding_docid,papertitledict





def get_fulltext_list(config):

    # doc_embeddingsavename=f"{config["embedding_model"]}_{config["text_chunk_size"]}.pickle"

    fulltext_list={}
    for docid in doclist:
        docpath=f"/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/{docid}/{docid}_content_list.json"
        with open(docpath,"r")as f:
            parsedcontent=json.load(f)
        fulltext=""
        fulltext_onlytext=""
        
        for item in parsedcontent:
            if item["type"]=="text":
                    if "text_level" in item and item["text_level"]==1:
                        if item["text"]!="":
                            fulltext=fulltext+"\n # "+item["text"]+"\n"
                            fulltext_onlytext=fulltext_onlytext+"\n # "+item["text"]+"\n"
                    else:
                        if item["text"]!="":
                            fulltext=fulltext+"\n"+item["text"]
                            fulltext_onlytext=fulltext_onlytext+"\n"+item["text"]


            if item["type"]=="equation":
                fulltext=fulltext+"\n"+item["text"]+"\n"

            if item["type"]=="table" and item["img_path"].strip()!="" and "table_body" in item:
                fulltext=fulltext+"\n"+"\n".join(item["table_caption"])+"\n".join(item["table_footnote"])+"\n"+item["table_body"]


            if item["type"]=="image":
                if "chart_parse" in item:                   
                    if "img_caption" in item:#这里没有就是不加的意思
                        fulltext=fulltext+"\n"+"\n".join(item["img_caption"])


                    fulltext=fulltext+"\n"+str(item["chart_parse"])

                    if "img_footnote" in item:
                        fulltext=fulltext+"\n"+"\n".join(item["img_footnote"])

        fulltext_list[docid]=fulltext

    with open(config["fulltext_list_name"],"w")as f:
         json.dump(fulltext_list,f,ensure_ascii=False,indent=1)



    return fulltext_list



















def embed_image(config):

    imagepathlist=[]
    embedding_docid=[]
    stop=False
    for docid in doclist:
        imagefolder=f"/data4/students/zhangguangyin/chatNum/rank200_pdf_images/{docid}/pdf_images"
        if not os.path.exists(imagefolder):
            print(f"imagefolder:{imagefolder} not exists")
            stop=True
            continue
        for imagename in list(os.listdir(imagefolder)):
            if "dpi800" in imagename:
                os.remove(os.path.join(imagefolder,imagename))
            # if "dpi1200" not in imagename:
            #     os.remove(os.path.join(imagefolder,imagename))
            

        image_num=len(list(os.listdir(imagefolder)))
        if image_num==0:
            print(f"image_num==0,docid:{docid}")
            continue
        for imageid in range(1,image_num+1):
            imagepath=os.path.join(imagefolder,f"page_{imageid}_dpi600.png")
            if not os.path.exists(imagepath):
                print(f"imagepath:{imagepath} not exists")
                stop=True
                continue
            imagepathlist.append(imagepath)
            embedding_docid.append([docid,imageid])
    if stop:
        assert False

    # embedding_model=embedder_colpali(config)
    embedding_model=embedder_gme(config)
    print(f"imagepathlist:{len(imagepathlist)}")
    image_embeds=embedding_model.embed_document(imagepathlist)
    with open(config["gme_image_embedding_name"],"wb")as f:
        pickle.dump(image_embeds,f)
    with open(config["gme_image_embedding_idlistname"],"w")as f:
        json.dump(embedding_docid,f,ensure_ascii=False,indent=1)
        
    return image_embeds






def modify_image_docid():
    with open("/data4/students/zhangguangyin/chatNum/colpali_v1.3_docid.json","r") as f:
         doc=json.load(f)
    new_embedding_docid=[]
    for itemid,item in enumerate(doc):
        new_embedding_docid.append([item[0],item[1]+1])
    with open("/data4/students/zhangguangyin/chatNum/colpali_v1.3_docid_v2.json","w") as f:
        json.dump(new_embedding_docid,f,ensure_ascii=False,indent=1)





if __name__ == "__main__":
    # with open("/data4/students/zhangguangyin/chatNum/config/config_paperagent.yaml","r")as f:
    #     config=yaml.safe_load(f)
    with open("/data4/students/zhangguangyin/chatNum/config/config.yaml","r")as f:
        config=yaml.safe_load(f)
    embeddoc(config)
    # embed_image(config)
    # modify_image_docid()
    # get_fulltext_list(config)

