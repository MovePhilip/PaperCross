import os
import json
os.environ["DASHSCOPE_API_KEY"]="sk-8dcaeb06a7944113a00d08b94824319a"
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import json
import requests
import json_repair
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from openai import OpenAI
import logging
import io
logger=logging.getLogger("myapp")

class closemodel:
      def __init__(self,config):
          self.config=config
          self.model_name=config["generation_model"]
          if self.model_name in ["deepseek","deepseek_reasoner"]:
              if self.config["agent_name"]=="ReAct":
                  base_url="https://api.deepseek.com/beta"
              else:
                  base_url="https://api.deepseek.com"
              self.client = OpenAI(api_key="", base_url=base_url)
          elif self.model_name=="gpt-4.1":
              self.client = OpenAI(api_key="", base_url="https://api.openai.com/v1")
          elif self.model_name=="gpt-5-chat":
              self.client = OpenAI(api_key="", base_url="https://xiaoai.plus/v1") 
          elif self.model_name in ["deepseek_zijie","qwen_vl", "qwen3_235b"]:
              self.client = OpenAI(api_key="", base_url="https://uni-api.cstcloud.cn/v1")#https://uni-api.cstcloud.cn/v1,https://ark.cn-beijing.volces.com/api/v3
          else:
              raise ValueError(f"Unsupported model: {self.model_name}")
           

      def generate(self,messages,stop_token=None):
            if self.model_name=="gpt-4.1":
                return self.generate_gpt(messages,stop_token)
            if self.model_name=="gpt-5-chat":
                return self.generate_gpt_5_chat(messages,stop_token)
            elif self.model_name=="deepseek":
                return self.generate_deepseek(messages,stop_token)
            elif self.model_name=="deepseek_reasoner":
                return self.generate_deepseek_reasoner(messages,stop_token)
            elif self.model_name=="deepseek_zijie":
                return self.generate_deepseek_zijie(messages,stop_token)
            elif self.model_name=="qwen_vl":
                return self.generate_qwen_vl(messages,stop_token)
            elif self.model_name=="qwen3_235b":
                return self.generate_qwen3_235b(messages,stop_token)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")


      def generate_gpt_5_chat(self,messages,stop_token=None):
            iteration=0
            while True:
                try:
                    url="https://xiaoai.plus/v1/chat/completions"

                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer sk-nwVDjcHR7vXMzAmx4ZuLikCgnJPruToNrY3iIW7Xx2qSMAWN",
                    }

                    data = {
                        "model": "gpt-5-chat",
                        "messages": messages,
                        "max_tokens":8000,
                        "temperature":1.0,
                        "stop":stop_token   
                    }

                    response=requests.post(url, headers=headers, json=data)
                    result = response.json()
                    generated_text = result["choices"][0]["message"]["content"]
                    break
                except Exception as e:
                    if iteration>5:
                        raise e
                    logger.info(f"GPT-5-chat error: {e}")
                    iteration+=1
                    time.sleep(10*iteration)
            return generated_text   

      def generate_gpt(self,messages,stop_token=None):
            url="http://149.88.89.156:3002/v1/chat/completions"

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-cudqGwlxe0r80EpBXgqjNobf3PhQBmiqXrmMSWzB8FQYkh8H",
                # "response_format": "json_object"
            }

            data = {
                "model": self.model_name,  # 模型版本,  gemini-2.0-pro-exp, gpt-4.1, gpt-4.1-mini，gemini-1.5-pro-latest,chatgpt-4o-latest
                "messages": messages,
                "max_tokens":4500,
                # "top_p":0.4,
                "temperature":1.0,
                "stop":stop_token

                # "response_format":{ "type": "json_object" }
            }

            response=requests.post(url, headers=headers, json=data)
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            # result_json=json_repair.loads(generated_text)
            # print(generated_text)
            return generated_text
    
      def generate_deepseek(self,messages,stop_token=None):
            iteration=0
            while True:
                try:
                    response=self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        stream=False,
                        temperature=1.0,
                        stop=stop_token
                    )
                    generated_text = response.choices[0].message.content
                    break
                except Exception as e:
                    if iteration>10:
                        raise e
                    logger.info(f"Deepseek error: {e}")
                    iteration+=1
                    time.sleep(10*iteration)

            return generated_text

      def generate_deepseek_reasoner(self,messages,stop_token=None):
            iteration=0
            while True:
                try:
                    response=self.client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=messages,
                        stream=False,
                        temperature=1.0,
                        stop=stop_token
                    )
                    generated_text = response.choices[0].message.content
                    break
                except Exception as e:
                    if iteration>10:
                        raise e
                    logger.info(f"Deepseek reasoner error: {e}")
                    iteration+=1
                    time.sleep(10*iteration)

            return generated_text
            
    
      def generate_deepseek_zijie(self,messages,stop_token=None):
            iteration=0
            while True:
                try:
                    response=self.client.chat.completions.create(
                        model="deepseek-v3:671b",
                        messages=messages,
                        stream=False,
                        temperature=1.0,
                        stop=stop_token

                    )
                    generated_text = response.choices[0].message.content
                    break
                except Exception as e:
                    if iteration>10:
                        raise e
                    logger.info(f"Deepseek_zijie error: {e}")
                    iteration+=1
                    time.sleep(10*iteration)

            return generated_text
      

      def generate_qwen_vl(self,messages,stop_token=None):
            iteration=0
            while True:
                try:
                    print("开始生成")
                    response=self.client.chat.completions.create(
                        model="qwen2.5-vl:72b",
                        messages=messages,
                        stream=False,
                        temperature=1.0
                    )
                    generated_text = response.choices[0].message.content
                    print("生成完成")
                    break
                except Exception as e: 
                    if iteration>10:
                        raise e
                    print(e)
                    logger.exception(f"Qwen_vl error: {e}")
                    iteration+=1
                    time.sleep(10*iteration)
            return generated_text

      def generate_qwen3_235b(self,messages,stop_token=None):
            iteration=0
            url="https://uni-api.cstcloud.cn/v1/chat/completions"

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer e36b5397eee2bb61205e0939d143a00292cec18a0d1b14bce1077c502285d660",
                # "response_format": "json_object"
            }
            while True:
                data = {
                    "model": "qwen3:235b",  # 模型版本,  gemini-2.0-pro-exp, gpt-4.1, gpt-4.1-mini，gemini-1.5-pro-latest,chatgpt-4o-latest
                    "messages": messages,
                    "max_tokens":4500,
                    # "top_p":0.4,
                    "temperature":1.0,
                    "stop":stop_token,
                    "include_stop_str_in_output":True

                    # "response_format":{ "type": "json_object" }
                }
                
                try:
                    response=requests.post(url, headers=headers, json=data)
                    result = response.json()
                    generated_text = result["choices"][0]["message"]["content"]
                    break
                except Exception as e:
                    print(e)
                    print(response)
                    print(result)
                    logger.exception(f"Qwen3_235b error: {e}")
                    time.sleep(10)
                    continue

                

            return generated_text



def encode_image(image_path, max_side: int = 1000):
    """
    1400输入10张时没问题的，但是幻觉严重到不行，没法用

    读取图片，若最长边超过 max_side（默认 1600px）则按比例缩放，
    再返回 base64 编码后的 PNG 字节串。最终图片 DPI 设为 300。
    """
    with Image.open(image_path) as img:
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG", dpi=(300, 300))
        return base64.b64encode(buf.getvalue()).decode("utf-8")












    


if __name__=="__main__":
      config={
            "generation_model":"qwen_vl",
            "agent":"ReAct",
            "deepseek_api_key":"sk-1ce662dbcdea4e15946746f5585c975b"
      }
      model=closemodel(config)


      #预处理图片，更改图像分辨率的大小，更改为dpi300
      


      # 将xxxx/eagle.png替换为你本地图像的绝对路径
      #开始图像预处理
      print("开始图像预处理")
      pathlist=[
      "/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1704.04920/pdf_images/page_1_dpi600.png",
      "/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1704.04920/pdf_images/page_2_dpi600.png",
      "/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1704.04920/pdf_images/page_3_dpi600.png",
    #   "/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1704.04920/pdf_images/page_4_dpi600.png",
    #   "/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1704.04920/pdf_images/page_5_dpi600.png",
    #   "/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1704.04920/pdf_images/page_6_dpi600.png",
    #   "/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1704.04920/pdf_images/page_7_dpi600.png",
    #   "/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1704.04920/pdf_images/page_8_dpi600.png",
    #   "/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1704.04920/pdf_images/page_9_dpi600.png",
    #   "/data4/students/zhangguangyin/chatNum/rank1000_fulltext_paper/1704.04920/pdf_images/page_10_dpi600.png"
      ]


      
      image_list=[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"}} for path in pathlist]
      print("图像预处理完成")
      image_list.append({"type": "text", "text": "这3张图片，每张图片的第一句话分别是什么"})

      messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": image_list
                
                
                #[
                    # {
                    #     "type": "image_url",
                    #     # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                    #     # PNG图像：  f"data:image/png;base64,{base64_image}"
                    #     # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                    #     # WEBP图像： f"data:image/webp;base64,{base64_image}"
                    #     "image_url": {"url": f"data:image/png;base64,{base64_image1}"}, 
                    # },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {"url": f"data:image/png;base64,{base64_image2}"}, 
                    # },

                        # {"type": "text", "text": "这10张图片，每张图片的第一句话分别是什么"},
                    # ],
                }
            ]

      print(model.generate(messages))





