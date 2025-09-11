import os
import json
os.environ["DASHSCOPE_API_KEY"]="sk-8dcaeb06a7944113a00d08b94824319a"
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

import torch
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import json
import requests
import json_repair
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info



class openmodel:
      def __init__(self,config):
           if config["generation_model"]=="Qwen/Qwen3-14B":
                model_name = "Qwen/Qwen3-14B"
                # load the tokenizer and the model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="/data4/students/zhangguangyin/LLMs")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    # device_map="auto",
                    cache_dir="/data4/students/zhangguangyin/LLMs"
                )
           self.gpu_id = config["gpu_id"]  # 指定你想使用的GPU编号
           self.device = torch.device(f"cuda:{self.gpu_id}")
           self.model=self.model.to(self.device)

      def generate(self,messages):
            return self.generate_vllm(messages)

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # conduct text completion
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=8900,

            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            # print("thinking content:", thinking_content)
            # print("content:", content)
            
            return content

      def generate_vllm(self,messages):
            model_name="Qwen3-14B"
            url = "http://10.3.11.77:8000/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                # "Authorization": "Bearer your-api-key"
            }
            data = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 8900,
                "temperature": 0.3,
                "top_p": 0.5,
            }

            response = requests.post(url, headers=headers, json=data)

            # 提取生成的文本内容
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            if "</think>" in generated_text:
                generated_text=generated_text.split("</think>")[1]
            return generated_text





class Qwen2VL():
    def __init__(self, config):
        super().__init__(config)
        max_pixels = 2048*28*28
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_id, torch_dtype="auto", device_map="balanced_low_0"
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_id) # , max_pixels=max_pixels
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }
        
    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
        
    def create_image_message(self, images, question):
        content = []
        for image_path in images:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
    
    @torch.no_grad()
    def predict(self, question, texts = None, images = None, history = None):
        self.clean_up()
        messages = self.process_message(question, texts, images, history)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        messages.append(self.create_ans_message(output_text))
        self.clean_up()
        return output_text, messages
        
    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True

    def process_message(self, question, texts, images, history):
        if history is not None:
            assert(self.is_valid_history(history))
            messages = history
        else:
            messages = []
        
        if texts is not None:
            messages.append(self.create_text_message(texts, question))
        if images is not None:
            messages.append(self.create_image_message(images, question))
        
        if (texts is None or len(texts) == 0) and (images is None or len(images) == 0):
            messages.append(self.create_ask_message(question))
        
        return messages


class Qwen2_5VL(Qwen2VL):
    def __init__(self, config):
        self.config = config
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config["vl_model_name"], torch_dtype="auto",cache_dir="/data4/students/zhangguangyin/LLMs" 
        )
        self.processor = AutoProcessor.from_pretrained(self.config["vl_model_name"],cache_dir="/data4/students/zhangguangyin/LLMs")
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }
        self.device = torch.device(f"cuda:{config['gpu_id']}")
        self.model=self.model.to(self.device)



