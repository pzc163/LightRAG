import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
import pymupdf4llm
import pathlib
from dotenv import load_dotenv 

load_dotenv()


WORKING_DIR = "./AI_Studio_Demo/宁德时代2024半年度报告"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "ernie-4.0-8k",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("AI_STUDIO_API_KEY"),
        base_url="https://aistudio.baidu.com/llm/lmapi/v3",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="bge-large-zh",
        api_key=os.getenv("AI_STUDIO_API_KEY"),
        base_url="https://aistudio.baidu.com/llm/lmapi/v3",
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())


async def main():
    try:
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=512,
                func=embedding_func,
            ),
        )
        file_path = "./data/宁德时代2024半年度报告.pdf"
        md_file_path = pathlib.Path(file_path.replace(".pdf", ".md"))
        
        # 判断是否存在对应的 markdown 文件
        if md_file_path.exists():
            # 如果存在，直接读取 markdown 文件
            md_text = md_file_path.read_text(encoding='utf-8')
        else:
            # 如果不存在，将 PDF 转换为 markdown
            md_text = pymupdf4llm.to_markdown(file_path)
            # 保存 markdown 文件
            md_file_path.write_bytes(md_text.encode())

        # Perform naive search
        print(
            await rag.aquery(
                "宁德时代 2024年上半年的财务状况如何?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print(
            await rag.aquery(
                "宁德时代 2024年上半年的财务状况如何?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print(
            await rag.aquery(
                "宁德时代 2024年上半年的财务状况如何?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
        print(
            await rag.aquery(
                "宁德时代 2024年上半年的财务状况如何?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
