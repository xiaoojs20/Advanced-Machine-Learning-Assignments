import os
import json
from zhipuai import ZhipuAI
# from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings import ZhipuAIEmbeddings
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import faiss
import logging
from typing import Any, List
from pydantic import Extra
import torch



MODEL_PATH = "/flash2/aml/public/models/bge-m3"
os.environ["ZHIPUAI_API_KEY"] = "YOUR_API_KEY_HERE"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    
    
# read txt
# Test: pumpkin_book
def read_txt_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # 将文本包装成 Document 对象
                doc = Document(page_content=text, metadata={"filename": filename})
                documents.append(doc)
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
    return documents



def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    使用文本分割器将文档拆分成更小的块。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        is_separator_regex=False,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

## 文本向量化的两种方法：embedding API 和 预训练模型
def zhipu_embedding(text: str):
    api_key = os.environ['ZHIPUAI_API_KEY']
    client = ZhipuAI(api_key=api_key)
    response = client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return response

# text = '要生成 embedding 的输入文本，字符串形式。'
# response = zhipu_embedding(text=text)
## 使用sentence-embedding 预训练模型将文本转化为向量
def sentence_transformers_embeddings(texts: str, model_name='all-MiniLM-L6-v2'):
    """
    使用指定的模型将文本转换为向量。
    """
    model = SentenceTransformer(model_name)
    # texts = [doc['text'] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


class CustomEmbedding(Embeddings):
    client: Any#: :meta private:
    tokenizer: Any
    context_sequence_length: int = 512
    query_sequence_length: int = 512
    model_name: str = ''
    """Model name to use."""

    def __init__(self, model, **kwargs: Any):
        """Initialize the sentence_transformer."""
        # super().__init__(**kwargs)
        self.client = SentenceTransformer(
            model, 
            device=device
            )
        self.context_sequence_length = 512
        self.query_sequence_length = 512

    class Config:
        extra = Extra.forbid

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / \
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        with torch.no_grad():
            embeddings = self.client.encode(texts)
            embeddings = embeddings.astype('float32')
            return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]




def main():
    # load and split document
    with open(config['data']['total'], 'r', encoding='utf-8') as f:
        text = f.read()
    doc = Document(page_content=text, metadata={"filename": "data.txt"})
    documents = [doc]
    split_docs = split_documents(documents=documents,
                                chunk_size=800, 
                                chunk_overlap=300)
    
    print(type(split_docs))
    print(len(split_docs))
    for idx, doc in enumerate(split_docs):
        print(idx)
        print(doc.page_content)
        print("-"*50)

    
    # sentence_transformers_embeddings or zhipu_embedding
    # embeddings = ZhipuAIEmbeddings(
    #     model="embedding-3",
    #     # With the `embedding-3` class
    #     # of models, you can specify the size
    #     # of the embeddings you want returned.
    #     # dimensions=1024
    # )
    
    # test embedding
    embeddings = CustomEmbedding(model=config['embedding']['model'])
    emb = embeddings.embed_query("张三")
    print(len(emb))
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=config['vector_db']['path']  # 允许我们将persist_directory目录保存到磁盘上
    )
    vectordb.persist()
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
    
if __name__ == '__main__':
    main()

    
    


