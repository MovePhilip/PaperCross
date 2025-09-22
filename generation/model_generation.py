import os
import json
os.environ["DASHSCOPE_API_KEY"]=""
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import json
import requests
import json_repair
from .close_model import closemodel
from .open_model import openmodel

class generationmodel:
      def __init__(self,config):
           if config["generation_model"]=="Qwen/Qwen3-14B":
               self.model=openmodel(config)
           elif config["generation_model"] in ["gpt-4.1","deepseek","deepseek_reasoner","deepseek_zijie","qwen_vl","qwen3_235b","gpt-5-chat"]:
               self.model=closemodel(config)


      def generate(self,messages,stop_token=None):
           result=self.model.generate(messages,stop_token=stop_token)
           return result




















