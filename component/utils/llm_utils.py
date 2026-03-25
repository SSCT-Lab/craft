# ./component/llm_utils.py
import json
from openai import OpenAI

def load_api_key(path="aliyun.key"):
    return open(path).read().strip()

def get_qwen_client(key_path="../aliyun.key"):
    api_key = load_api_key(key_path)
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
