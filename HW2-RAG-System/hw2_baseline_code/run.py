import os
import logging

from lightrag import LightRAG, QueryParam
from lightrag.llm import zhipu_complete, zhipu_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./working_dir"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

os.environ["ZHIPUAI_API_KEY"] = "5b1761c66363e169fb12005b9a194baa.pSyjWBg5CeafdEKF"
api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("Please set ZHIPU_API_KEY in your environment")


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=zhipu_complete,
    llm_model_name="glm-4-flash",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    embedding_func=EmbeddingFunc(
        embedding_dim=2048, # Zhipu embedding-3 dimension
        max_token_size=8192,
        func=lambda texts: zhipu_embedding(texts),
    ),
)

db_path = "./book.txt"
# db_path = "../database/ee_data.txt"

with open(db_path, "r", encoding="utf-8") as f:
    rag.insert(f.read())

question = "What is FRP?"
# question = "MATPOWER中，如何调用求解LODF？"

# Perform naive search
print(
    rag.query(question, param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query(question, param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query(question, param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query(question, param=QueryParam(mode="hybrid"))
)
