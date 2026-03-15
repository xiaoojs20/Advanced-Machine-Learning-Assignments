from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image
import numpy as np
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import os
import re


from diffusers import AutoPipelineForText2Image
import torch

import torch
from peft import PeftModel
import pandas as pd
from datasets import Dataset
import json
import os
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# from embedding import CustomEmbedding


# load config
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    
model_sd21 = "stabilityai/stable-diffusion-2-1"
lora_path_base = "/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora"
save_path_base = "./image/mmllm"
torch.manual_seed(1337)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.makedirs(save_path_base, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embeddings = HuggingFaceEmbeddings(model_name=config['embedding']['model'])
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=config['vector_db']['path']
)






# start
prompts = [
    "text2text: Scene-Clipping Long Video For Better Understanding作者是谁？", \
    "text2text(RAG): Scene-Clipping Long Video For Better Understanding作者是谁？", \
    "text2image(simpsons): A student sitting at a desk in a classroom taking a difficult exam. The student looks extremely stressed and overwhelmed, with tears streaming down their face, showing signs of frustration and despair. The Simpsons."
]


def parse_modal(prompt):
    """"
    根据prompt的前缀判断意图类型。
    # text2text, text2text(RAG), text-to-image(pokemon), text-to-image(simpsons), text-to-image(cartoon), text-to-image(rubberduck)
    """
    # 查看冒号前的文字，进行匹配, return 判断
    match = re.match(r'^(.*?):\s*(.*)$', prompt)
    if match:
        modal_type = match.group(1).strip().lower()
        real_prompt = match.group(2).strip()
        return modal_type, real_prompt
    else:
        return 'unknown', prompt
    

def text2image(prompt):
    modal_type, real_prompt = parse_modal(prompt)
    print(f"Q: {prompt}",end="\n\n")
    
    model_name = model_sd21
    task = "simpsons_better"
    if modal_type == "text2image(pokemon)":
        task = "pokemon_better"
    elif modal_type == "text2image(simpsons)":
        task = "simpsons_better"
    elif modal_type == "text2image(cartoon)":
        task = "cartoon_better"
    elif modal_type == "text2image(rubberduck)":
        task = "rubberduck_better"
    else:
        print(f"Unknown task: {modal_type}")
        return
    
    lora_path = os.path.join(lora_path_base, task)
    save_path = os.path.join(save_path_base, task)
    os.makedirs(save_path, exist_ok=True)
    
    pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    pipe.load_lora_weights(lora_path.format(epoch=5000), weight_name="pytorch_lora_weights.safetensors")
    images = pipe(prompt).images
    images[0].save(os.path.join(save_path, f"image_{task}.png"))

    print(f"Saved image to {save_path}")


def llm_with_rag():
    # tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['glm4-9b'], 
        trust_remote_code=True
        )
    tokenizer.pad_token = tokenizer.eos_token

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['glm4-9b'],
        # device_map="auto",
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
        ).eval().to(device)
    model = PeftModel.from_pretrained(model, model_id=os.path.join(config['lora']['path'], 'checkpoint-100'))
    
    system_prompt = """使用以下内容来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。尽量使答案简明扼要。”。内容: {context}。问题: {question}"""
    
    
    # 用API调用llm, 用hf pipeline封装finetune后的llm
    llm = ChatZhipuAI(
        model_name="glm-4-flash", 
        temperature = 0.1,
        openai_api_key=config['api_key'],
        do_sample=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=config['generation']['max_length'],            # 按需求设置推理参数
        temperature=config['generation']['temperature'],         # 如需采样
        do_sample=True                                          # 若想启用随机采样模式
    )
    llm_ft = HuggingFacePipeline(pipeline=pipe)
    
    # qa chain prompt
    qa_chain_prompt = PromptTemplate(
        input_variables=["context","question"],
        template=system_prompt
        )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt":qa_chain_prompt}
        )
    
    
    return llm, qa_chain


def text2text(prompt):
    modal_type, real_prompt = parse_modal(prompt)
    question = real_prompt
    
    # LLM only
    if modal_type == "text2text":
        llm, llm_rag = llm_with_rag()
        response = llm.invoke(question)
        print(f"Q: {prompt}",end="\n\n")
        print(f"A (LLM): {response.content}",end="\n\n")
    # LLM + Fine-tuned + RAG
    elif modal_type == "text2text(rag)":
        llm, llm_rag = llm_with_rag()
        result = llm_rag.invoke({"query": question})
        print(f"Q: {prompt}",end="\n\n")
        print(f"A (LLM ft+RAG): {result['result']}")
        print("Reference: ")
        # dict_keys(['query', 'result', 'source_documents'])
        # 查看检索到的上下文（源文档）
        source_docs = result["source_documents"]
        for i, doc in enumerate(source_docs, start=1):
            print(f"[{i}] {doc.page_content}")
    else: 
        print(f"Invalid modal type: {modal_type}")
    
        
def main():
    for prompt in prompts:
        print("--------------------------------------------")
        modal_type, real_prompt = parse_modal(prompt)
        print(f"Modal Type: {modal_type}\nReal Prompt: {real_prompt}\n")
        # print(type(modal_type))
        
        if "text2image" in modal_type:
            text2image(prompt)
        elif "text2text" in modal_type:
            text2text(prompt)
        else:
            print("Invalid modal type")

if __name__ == "__main__":
    main()
    
    