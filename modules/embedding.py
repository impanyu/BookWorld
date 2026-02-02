import sys
sys.path.append("../")
from chromadb.api.types import Embeddings, Documents, EmbeddingFunction, Space
from modelscope import AutoModel, AutoTokenizer
from functools import partial
from bw_utils import get_child_folders
import torch
import os


class EmbeddingModel(EmbeddingFunction[Documents]):
    def __init__(self, model_name, language='en'):
        self.model_name = model_name
        self.language = language
        cache_dir = "~/.cache/modelscope/hub"
        if "/" in model_name:
            model_provider = model_name.split("/")[0]
            model_smallname = model_name.split("/")[1]
            model_path = os.path.join(cache_dir, f"models--{model_provider}--{model_smallname}/snapshots/")
        else:
            model_path = os.path.join(cache_dir, f"models--{model_name}/snapshots/")
        
        if os.path.exists(model_path) and get_child_folders(model_path):
            try:
                model_path = os.path.join(model_path,get_child_folders(model_path)[0])
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModel.from_pretrained(model_path)
            except Exception as e:
                print(e)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, input):
        inputs = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].tolist()
        return embeddings

class OpenAIEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_name="text-embedding-ada-002", base_url = "https://api.openai.com/v1", api_key_field = "OPENAI_API_KEY"):
        from openai import OpenAI
        
        # Fix: Remove trailing /embeddings if present (client appends it automatically)
        if base_url.endswith("/embeddings"):
            base_url = base_url[:-len("/embeddings")]
            
        self.client = OpenAI(
            base_url = base_url,
            api_key = os.environ.get(api_key_field, "")
        )
        self.model_name = model_name

    def __call__(self, input):
        if isinstance(input, str):
            input = input.replace("\n", " ")
            return self.client.embeddings.create(input=[input], model=self.model_name).data[0].embedding
        elif isinstance(input,list):
            return [self.client.embeddings.create(input=[sentence.replace("\n", " ")], model=self.model_name).data[0].embedding for sentence in input]

def get_embedding_model(embed_name, language='en'):
    print(f"[Embedding] get_embedding_model called with: embed_name={embed_name}, language={language}")
    local_model_dict = {
        "bge-m3":"BAAI/bge-m3",
        "bge-large": f"BAAI/bge-large-{language}",
        "luotuo": "silk-road/luotuo-bert-medium",
        "bert": "google-bert/bert-base-multilingual-cased",
        "bge-small": f"BAAI/bge-small-{language}",
    }
    online_model_dict = {
        "openai":
            {"model_name":"text-embedding-ada-002",
             "url":"https://api.openai.com/v1",
             "api_key_field":"OPENAI_API_KEY"},
        "qwen":
            {"model_name":"text-embedding-v2",
             "url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
             "api_key_field":"DASHSCOPE_API_KEY"},
    }
    if embed_name in local_model_dict:
        model_name = local_model_dict[embed_name]
        print(f"[Embedding] Using local model: {model_name}")
        return EmbeddingModel(model_name, language=language)
    
    if embed_name in online_model_dict:
        model_name = online_model_dict[embed_name]["model_name"]
        api_key_field = online_model_dict[embed_name]["api_key_field"]
        base_url = online_model_dict[embed_name]["url"]
        return OpenAIEmbedding(model_name=model_name, base_url=base_url,api_key_field=api_key_field)
    
    if embed_name.startswith("text-embedding-v"):
        return OpenAIEmbedding(model_name=embed_name, 
                               base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                               api_key_field="DASHSCOPE_API_KEY")
    
    print(f"[Embedding] Using local model: {embed_name}")
    return EmbeddingModel(embed_name, language=language)
