import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder



class Reranker:
    def __init__(self,config):
        self.config=config
        self.model = CrossEncoder(config["reranker_model"],automodel_args={"torch_dtype": "auto"},trust_remote_code=True,cache_folder="/data4/students/zhangguangyin/LLMs")
        self.model.eval()
        self.gpu_id = config["gpu_id"]  # 指定你想使用的GPU编号
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.model=self.model.to(self.device)
        self.model.to(self.device)

    def rerank(self,pairs):
       scores=self.model.predict(pairs)
       return scores





