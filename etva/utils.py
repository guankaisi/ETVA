import json
from vllm import LLM
import os
os.environ["NCCL_DEBUG"] = "ERROR"
def load_llm(model_path: str='Qwen/Qwen2.5-72B-Instruct'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=4,
        dtype='bfloat16',
    )
    return llm

def load_mllm(model_path: str='Qwen/Qwen2-VL-72B-Instruct'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    mllm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        tensor_parallel_size=4,
        dtype='bfloat16',
    )
    return mllm

def load_prompts_from_json(prompt_path: str):
    with open(prompt_path, 'r') as f:
        prompts = [json.loads(line) for line in f.readlines()]
    return prompts

def load_prompts_from_txt(prompt_path: str):
    with open(prompt_path, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts






    