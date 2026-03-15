from typing import List
from openai import OpenAI
from zhipuai import ZhipuAI
import re
import json
from langchain_core.documents import Document
from tqdm import tqdm

from embedding import read_txt_files, split_documents

# load config
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)



PROMPT = """
下面是上下文信息。  
```{context_str} ```。
给定上下文信息，没有先验知识。 仅根据下面的查询生成问题。 

你的任务是设置{num_questions_per_chunk}个问题，并且以问题涉及到的原文内容作为答案，问题的性质应该是多样化的，将问题限制在提供的上下文信息之内。\
严格按照下面的格式输出每个问题和答案。\
Q: 问题
A: 答案 
"""


class QaPairs():
    '''存储List[dict]类型数据'''
    def __init__(self, qa_pairs: List[dict]):
        self.qa_pairs = qa_pairs

    def save_json(self, path: str):
        '''将数据存储为json格式'''
        with open(path, "w", encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=4)

    @classmethod
    def from_json(cls, path:str) -> 'QaPairs':
        '''读取json格式数据'''
        with open(path) as f:
            data = json.load(f)
        return cls(data)


def list_generate_qa_pairs(
        texts: List[str],
        num_questions_per_chunk: int = 2,
        model: str = 'glm-4-flash',
    ) -> QaPairs:
    '''借助大模型从给定的texts里提取出问题与对应的答案'''
    llm = ZhipuAI()
    qa_pairs = []
    for text in tqdm(texts):
        if len(text) > 200:
            prompt = PROMPT.format(
                context_str=text,
                num_questions_per_chunk=num_questions_per_chunk
            )
            response = llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            matches = re.findall(
                r'Q:(.*?)A:(.*?)((?=Q)|$)',
                response.choices[0].message.content,
                re.DOTALL
            )
            for _, match in enumerate(matches):
                qa = {
                    'query': match[0].strip(),
                    'answer': match[1].strip()
                }
                qa_pairs.append(qa)

    return QaPairs(qa_pairs=qa_pairs)


def docs_generate_qa_pairs(
        docs: List[Document], 
        num_questions_per_chunk: int = 2,
        model: str = 'glm-4-flash'
    ) -> QaPairs:
    """借助大模型从给定的docs里提取出问题与对应的答案
    docs：格式为List[Document]的文档。
    num_questions_per_chunk：每页生成的QA对数量，默认为2。
    moedl：生成QA对所使用的模型，默认为智谱的glm-4-flash。
    """
    list_doc = [doc.page_content for doc in docs]
    return list_generate_qa_pairs(
      list_doc, 
      num_questions_per_chunk, 
      model=model
      )

def main():
    # load and split document
    # lines = []
    with open(config['data']['total'], 'r', encoding='utf-8') as f:
        text = f.read()
    #     lines = f.readlines()
    #     line_count = len(lines)
    # print(f"总行数：{line_count}")
    # # 将列表合并成一个字符串
    # selected_lines = lines[:100]
    # text = ''.join(selected_lines)
    # text = ''.join(lines)
    
    doc = Document(page_content=text, metadata={"filename": "data.txt"})
    documents = [doc]
    split_docs = split_documents(documents=documents,
                                chunk_size=800, 
                                chunk_overlap=300)
    # print(type(split_docs))
    # print(type(split_docs[0]))
    # print(len(split_docs))
    
    qa_pairs = docs_generate_qa_pairs(
        docs=split_docs, 
        num_questions_per_chunk=2,
        model='glm-4-flash'
        )
    # print(qa_pairs.qa_pairs[0])

    for i in range(len(qa_pairs.qa_pairs)):
        print('第{}个问题：{}'.format(i + 1, qa_pairs.qa_pairs[i]['query']))
        print('第{}个答案：{}'.format(i + 1, qa_pairs.qa_pairs[i]['answer']), end='\n\n')
  
    qa_pairs.save_json("AMLQA.json")
    
if __name__ == '__main__':
    main()
    

        