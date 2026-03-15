
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
from embedding import CustomEmbedding


# load config
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["ZHIPUAI_API_KEY"] = config['api_key']
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embeddings = HuggingFaceEmbeddings(model_name=config['embedding']['model'])
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=config['vector_db']['path']
)


def inference(example, tokenizer, model, generation_config, max_length=512):
    """
    输入 example 数据，调用模型生成响应。
    
    Args:
        example (dict): 包含 instruction 和 input 的数据。
        tokenizer (Tokenizer): 分词器实例。
        model (Model): 预训练模型实例。
        max_length (int): 最大生成序列长度。
    
    Returns:
        str: 模型生成的响应。
    """
    messages = [
        {"role": "system", "content": config['lora']['instruction']},
        {"role": "user", "content": example['question']}
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    # generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            max_length=max_length,
            # input_ids=model_inputs.input_ids,
            # attention_mask=model_inputs.attention_mask,
            generation_config=generation_config    
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


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

def main():
    # generation_config = GenerationConfig(
    #                         temperature=config['generation']['temperature'],
    #                         top_p=config['generation']['top_p'],
    #                         top_k=config['generation']['top_k'],
    #                         num_beams=config['generation']['num_beams'],
    #                         max_tokens=config['generation']['max_tokens'],
    #                         do_sample=True
    #                     )

    # example = {
    #     "question": "你是谁?"
    # }
    
    llm, llm_rag = llm_with_rag()

    for question in config['generation']['questions']:
        print('---'*50)
        print(f"Q: {question}",end="\n\n")
        # LLM only
        response = llm.invoke(question)
        print(f"A (LLM): {response.content}",end="\n\n")
        # LLM + Fine-tuned + RAG
        result = llm_rag.invoke({"query": question})
        print(f"A (LLM ft+RAG): {result['result']}")
        print("Reference: ")
        # dict_keys(['query', 'result', 'source_documents'])
        # 查看检索到的上下文（源文档）
        source_docs = result["source_documents"]
        for i, doc in enumerate(source_docs, start=1):
            print(f"[{i}] {doc.page_content}")
        
        print('---'*50)
    
    
if __name__ == '__main__':
    main()