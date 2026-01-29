# -*- coding: utf-8 -*-
# RAGLlama3.py — end-to-end script (completion, chat, simple RAG, tools + RAG over PDFs)

import sys
import os
import nest_asyncio

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import VectorStoreIndex

import LLMTasks
from LLMInit import initialize_LLM
from RAGMathFunctions import multiply, add, subtract, divide, compute_circle_area


def main():
    # 1) Initialize LLM (Phi-3)
    llm = initialize_LLM()
    print("----------------")

    # 2) Basic completion
    statement = "The fundamental idea in low rank adaptation is: "
    response = LLMTasks.LLM_generation_complete(llm, statement)
    print(response)

    print("----------------")
    # 3) Chat with roles
    messages = [
        ChatMessage(
            role="system",
            content="You are the new president of the Robotics club at University of Bridgeport.",
        ),
        ChatMessage(
            role="user",
            content="Introduce your club to other engineering students.",
        ),
    ]
    response = LLMTasks.LLM_generation_chat(llm, messages)
    print(response)

    print("----------------")
    # 4) Simple text RAG using your MSAI.txt (use your absolute path to avoid 'missing file')
    input_files = [r"C:\Users\saimu\OneDrive\Desktop\ragphi\RAGLlama3\data\MSAI.txt"]
    topk = 2
    prompt = "What are the specializations offered in the MS in AI at University of Bridgeport?"
    try:
        response = LLMTasks.LLM_RAG_simple(llm, prompt, input_files, topk)
        print("\n----- RAG Response based on MSAI.txt document -----\n", response)
    except FileNotFoundError as e:
        print(str(e))
        print("Tip: update 'input_files' to the correct path or place MSAI.txt under ./data")

    # 5) ReAct agent with math tools
    nest_asyncio.apply()
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)
    subtract_tool = FunctionTool.from_defaults(fn=subtract)
    divide_tool = FunctionTool.from_defaults(fn=divide)
    circle_area_tool = FunctionTool.from_defaults(fn=compute_circle_area)

    agent = ReActAgent.from_tools(
        [multiply_tool, add_tool, subtract_tool, divide_tool, circle_area_tool],
        llm=llm,
        verbose=True,
    )
    print(agent.chat("What is 100*6-5+10 ?"))             # expected 605
    print(agent.chat("What is the area of a circle with radius 10"))  # ~314.159...

    # 6) ReAct agent + RAG query-engine tools over PDFs (using safe manual loader)
    pdf_specs = [
        ("data/lyft_2021.pdf", "lyft_10k", "Provides information about Lyft financials for year 2021."),
        ("data/uber_2021.pdf", "uber_10k", "Provides information about Uber financials for year 2021."),
        ("data/catalog-2022-2023.pdf", "ub_catalog", "University of Bridgeport catalog 2022-2023."),
    ]

    tools = []
    for path, name, desc in pdf_specs:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                docs = LLMTasks.load_documents_paths([path])  # uses manual .txt/.pdf loaders (no extras)
                idx = VectorStoreIndex.from_documents(docs)
                tools.append(
                    QueryEngineTool(
                        query_engine=idx.as_query_engine(similarity_top_k=3),
                        metadata=ToolMetadata(name=name, description=desc),
                    )
                )
            except Exception as e:
                print(f"Skipping {path}: {e}")
        else:
            print(f"Missing PDF (skipping): {path}")

    if tools:
        agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
        print(agent.chat("What was Lyft's revenue in 2021?"))
        print(agent.chat("What was Uber's revenue in 2021?"))
        print(agent.chat("Give a brief description of the Computer Vision course at University of Bridgeport."))
    else:
        print("No PDF tools created (place the PDFs under ./data to enable this section).")

    return 0


if __name__ == "__main__":
    sys.exit(int(main() or 0))
