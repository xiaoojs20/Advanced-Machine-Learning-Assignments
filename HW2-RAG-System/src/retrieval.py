import os
import json
import torch

from dotenv import find_dotenv, load_dotenv
from embedding import CustomEmbedding
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    
# embeddings = CustomEmbedding()
embeddings = HuggingFaceEmbeddings(model_name=config['embedding']['model'])
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=config['vector_db']['path']
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")


def test_retrieval():
    # query -> RAG! -> top-k docs
    query="Scene-Clipping Long Video For Better Understanding作者是谁？"
    print(f"检索问题：{query}")

    sim_docs = vectordb.similarity_search(query, k=3)
    print(f"相似度检索到的内容数：{len(sim_docs)}")
    for i, sim_doc in enumerate(sim_docs):
        print(f"相似度检索到的第{i+1}个内容: \n{sim_doc.page_content}", end="\n--------------\n")
        
    mmr_docs = vectordb.max_marginal_relevance_search(query,k=3)
    print(f"MMR 检索到的内容数：{len(mmr_docs)}")
    for i, mmr_doc in enumerate(mmr_docs):
        print(f"MMR 检索到的第{i+1}个内容: \n{mmr_doc.page_content}", end="\n--------------\n")

  
# """
# 创建检索 QA 链的方法 RetrievalQA.from_chain_type() 有如下参数：
# llm：指定使用的 LLM
# 指定 chain type : RetrievalQA.from_chain_type(chain_type="map_reduce")，也可以利用load_qa_chain()方法指定chain type。
# 自定义 prompt ：通过在RetrievalQA.from_chain_type()方法中，指定chain_type_kwargs参数，而该参数：chain_type_kwargs = {"prompt": PROMPT}
# 返回源文档：通过RetrievalQA.from_chain_type()方法中指定：return_source_documents=True参数；也可以使用RetrievalQAWithSourceChain()方法，返回源文档的引用（坐标或者叫主键、索引）
# """

## 构建检索问答链
# system_prompt = (
#     "Use the given context to answer the question. "
#     "If you don't know the answer, say you don't know. "
#     "Use three sentence maximum and keep the answer concise. "
#     "Context: {context}"
# )
system_prompt = """使用以下内容来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。尽量使答案简明扼要。”。内容: {context}。问题: {question}"""

def main():
    ## call llm
    llm = ChatZhipuAI(
        model_name="glm-4-flash", 
        temperature = 0.1,
        openai_api_key=config['api_key']
        )
    response = llm.invoke("请你自我介绍一下自己！")
    # print(response)
    print(response.content)
    
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

    print("*"*30)
    question = "Scene-Clipping Long Video For Better Understanding 作者是谁？"
    print(f"\nQ: {question}")
    # 大模型自己回答的效果      
    prompt_template = f"请回答下列问题: {question}"
    print("\nLLM Only: ")
    print(llm.invoke(prompt_template).content)
    
    # 基于召回结果和 query 结合起来构建的 prompt 效果
    print("\nLLM with RAG")
    result = qa_chain.invoke({"query": question})
    # print(type(result))
    # print(result)
    print(result['result'])
    # 查看检索到的上下文（源文档）
    source_docs = result["source_documents"]
    for i, doc in enumerate(source_docs, start=1):
        print(f"[{i}] {doc.page_content}")

qa_list = []

def main_rag_qa():
    with open('AMLQA_raw.json', 'r', encoding='utf-8') as f:
        QA_raw = json.load(f)
    
    llm = ChatZhipuAI(
        model_name="glm-4-flash", 
        temperature = 0.1,
        openai_api_key=config['api_key']
        )
    
    # 对每一个QA对中的Q，使用RAG重新生成答案
    system_prompt = """使用以下内容来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。尽量使答案简明扼要。”。内容: {context}。问题: {question}"""
    
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
    
    
    question = "Scene-Clipping Long Video For Better Understanding 作者是谁？"
    print(f"\nQ: {question}")
    # 大模型自己回答的效果      
    prompt_template = f"请回答下列问题: {question}"
    print("\nLLM Only: ")
    print(llm.invoke(prompt_template).content)
    
    # 基于召回结果和 query 结合起来构建的 prompt 效果
    print("\nLLM with RAG")
    result = qa_chain.invoke({"query": question})
    print(result['result'])
    answer = result['result']
    # 查看检索到的上下文（源文档）
    source_docs = result["source_documents"]
    for i, doc in enumerate(source_docs, start=1):
        print(f"[{i}] {doc.page_content}")
        answer += f" [{i}] {doc.page_content}"

    # qa_list.append({
    #     "question": question,
    #     "answer": answer
    # })
    # with open("/flash2/aml/xiaojs24/AML_2/src/AMLQA.json", "w", encoding="utf-8") as f:
    #     json.dump(qa_list, f, ensure_ascii=False, indent=2)

    # LLM + RAG构造新的QA pair
    for idx, qa in tqdm(enumerate(QA_raw), total=len(QA_raw)):
        print(f"Processing QA pair {idx+1}/{len(QA_raw)}")
        query = qa.get("query", "")
        if idx <= 144:
            continue
        if query:
            question = query
            result = qa_chain.invoke({"query": question})
            answer = result['result']
            source_docs = result["source_documents"]
            for i, doc in enumerate(source_docs, start=1):
                # print(f"[{i}] {doc.page_content}")
                answer += f" [{i}] {doc.page_content}"
            new_qa = {
                "question": question,
                "answer": answer
            }
            # qa_list.append(new_qa)
            # # 以 JSON Lines 形式追加写入
            # with open("/flash2/aml/xiaojs24/AML_2/src/AMLQA.json", 'a', encoding='utf-8') as f:
            #     # 将一条新数据写入一行
            #     json_str = json.dumps(new_qa, ensure_ascii=False, indent=2)
            #     f.write(json_str + "\n")
            # 读 -> 改 -> 写 回JSON文件
            with open("/flash2/aml/xiaojs24/AML_2/src/AMLQA.json", 'r', encoding='utf-8') as f:
                qa_list = json.load(f)
            qa_list.append(new_qa)
            
            with open("/flash2/aml/xiaojs24/AML_2/src/AMLQA.json", 'w', encoding='utf-8') as f:
                json.dump(qa_list, f, ensure_ascii=False, indent=2)

            print(f"写入第 {idx} 条QA: {new_qa}")
        
        
if __name__ == "__main__":
    # test_retrieval()
    # main()
    main_rag_qa()





    # ## 检索问答链效果测试
    # question_1 = "什么是CodeGeeX？"
    # question_2 = "Prompt Engineering for Developer是谁写的？"
    # # 基于召回结果和 query 结合起来构建的 prompt 效果
    # result = qa_chain({"query": question_1})
    # print("大模型+知识库后回答 question_1 的结果：")
    # print(result["result"])
    # result = qa_chain({"query": question_2})
    # print("大模型+知识库后回答 question_2 的结果：")
    # print(result["result"])
    
    # # 大模型自己回答的效果
    # prompt_template = """请回答下列问题:{}""".format(question_1)
    # print("大模型自己回答 question_1 的结果：")
    # print(llm.predict(prompt_template))
    # prompt_template = """请回答下列问题:{}""".format(question_2)
    # print("大模型自己回答 question_2 的结果：")
    # print(llm.predict(prompt_template))
    
    # """
    # 对话检索链（ConversationalRetrievalChain）在检索 QA 链的基础上，增加了处理对话历史的能力。
    # 它的工作流程是:

    # 将之前的对话与新问题合并生成一个完整的查询语句。
    # 在向量数据库中搜索该查询的相关文档。
    # 获取结果后,存储所有答案到对话记忆区。
    # 用户可在 UI 中查看完整的对话流程。
    # """
    # ## 增加历史对话的记忆功能
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    #     return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    # )
    
    # qa = ConversationalRetrievalChain.from_llm(
    #     llm,
    #     retriever=vectordb.as_retriever(),
    #     memory=memory
    # )
    # question = "我可以学习到关于提示工程的知识吗？"
    # result = qa({"question": question})
    # print(result['answer'])
    
    # question = "为什么这门课需要教这方面的知识？"
    # result = qa({"question": question})
    # print(result['answer'])
    
    
    
    