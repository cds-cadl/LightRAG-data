import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./smollm-mxbai"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="smollm2:1.7b",
    llm_model_max_async=4,
    llm_model_max_token_size=8192,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=512,
        func=lambda texts: ollama_embedding(
            texts, embed_model="mxbai-embed-large:335m"
        ),
    ),
)

with open("./para-4.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# CHUNK_SIZE = 1024  # Adjust this value as needed

# with open("./paragraphs.txt", "r", encoding="utf-8") as f:
#     while True:
#         chunk = f.read(CHUNK_SIZE)
#         if not chunk:
#             break  # End of file
#         rag.insert(chunk)

# Perform naive search
print("NAIVE")
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform local search
print("LOCAL")
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
)

# Perform global search
print("GLOBAL")
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print("HYBRID")
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
)

# # stream response
# resp = rag.query(
#     "What are the top themes in this story?",
#     param=QueryParam(mode="hybrid", stream=True),
# )


# async def print_stream(stream):
#     async for chunk in stream:
#         print(chunk, end="", flush=True)


# if inspect.isasyncgen(resp):
#     asyncio.run(print_stream(resp))
# else:
#     print(resp)
