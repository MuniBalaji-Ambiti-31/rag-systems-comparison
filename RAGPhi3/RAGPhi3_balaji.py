# RAGPhi3_assignment_clean.py
# Minimal-output RAG (FAISS + LangChain + Phi-3), tailored to the assignment format.
# Setup (once):
# pip install -U "langchain>=0.2.12" "langchain-community>=0.2.12" langchain-huggingface \
#                faiss-cpu pypdf sentence-transformers accelerate transformers

import os
import sys
import argparse
import urllib.request
import warnings
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitters
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---------------- Config (edit these) ----------------
PDF_REMOTE = "https://www.bridgeport.edu/files/docs/academics/catalogs/catalog-2022-2023.pdf"
PDF_LOCAL  = "ub_catalog.pdf"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"       # 768-dim (good recall)
MODEL_ID    = "microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN    = "Add hugg tken"                           # <-- replace, then revoke after use
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 1000, 200, 10

# ---------------- Quiet all the noise ----------------
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def ensure_pdf(path: str, url: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    urllib.request.urlretrieve(url, path)

def build_index(pdf_path: str):
    # Load and chunk
    pages = PyPDFLoader(pdf_path, extract_images=False).load_and_split()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks = splitter.split_documents(pages)
    # Build FAISS in-memory
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    vectordb = FAISS.from_documents(chunks, embedding=emb)
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    return vectordb, retriever

def load_llm():
    device_idx = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
    gen = pipeline("text-generation", model=mdl, tokenizer=tok, device=device_idx, max_new_tokens=300)
    return HuggingFacePipeline(pipeline=gen)

def make_chain(llm):
    # Assignment prompt with “I dont know” fallback
    tmpl = """<|system|>
You have been provided with the context and a question, try to find out the answer to the question only using the context information. If the answer to the question is not found within the context, return "I dont know" as the response.<|end|>
<|user|>
Context:
{context}

Question: {question}<|end|>
<|assistant|>"""
    return create_stuff_documents_chain(llm=llm, prompt=ChatPromptTemplate.from_template(tmpl))

def answer_question(q: str, vectordb, retriever, chain, cite=False):
    # Retrieval (with a quiet fallback)
    try:
        docs = retriever.invoke(q)
    except Exception:
        docs = vectordb.similarity_search(q, k=TOP_K)

    if not docs:
        return "I dont know", []

    out = chain.invoke({"context": docs, "question": q})
    ans = (out.split("<|assistant|>")[-1]).strip()

    if cite:
        # Return the first few page numbers as lightweight citations
        pages = []
        for d in docs[:3]:
            m = d.metadata or {}
            pages.append(str(m.get("page_number", m.get("page", "?"))))
        return ans, pages
    return ans, []

def main():
    parser = argparse.ArgumentParser(description="Assignment-style RAG (quiet output)")
    parser.add_argument("--q", "--question", dest="question", default=None, help="Ask a single question and exit")
    parser.add_argument("--cite", action="store_true", help="Print page numbers for context used")
    args = parser.parse_args()

    # Silence perf-only attention warnings (best effort)
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            pass

    # Prepare corpus/index + model
    ensure_pdf(PDF_LOCAL, PDF_REMOTE)
    vectordb, retriever = build_index(PDF_LOCAL)
    llm = load_llm()
    chain = make_chain(llm)

    # One-shot or REPL
    if args.question:
        ans, pages = answer_question(args.question, vectordb, retriever, chain, cite=args.cite)
        print(f"Question: {args.question}")
        print("Answer:", ans)
        if args.cite and pages:
            print("Pages:", ", ".join(pages))
        return 0

    # Interactive: clean, assignment-friendly I/O
    try:
        while True:
            q = input().strip()
            if not q or q.lower() == "end":
                break
            ans, pages = answer_question(q, vectordb, retriever, chain, cite=args.cite)
            print("Answer:", ans)
            if args.cite and pages:
                print("Pages:", ", ".join(pages))
    except (EOFError, KeyboardInterrupt):
        pass
    return 0

if __name__ == "__main__":
    sys.exit(int(main() or 0))
