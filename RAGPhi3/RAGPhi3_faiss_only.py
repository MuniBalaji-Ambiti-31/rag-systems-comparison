# RAGPhi3_faiss_only.py
# Setup (once):
# pip install -U "langchain>=0.2.12" "langchain-community>=0.2.12" langchain-huggingface \
#                faiss-cpu pypdf sentence-transformers accelerate transformers

import os, sys, urllib.request, textwrap, traceback, time
from typing import List
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---------------- CONFIG ----------------
PDF_URL_REMOTE = "https://www.bridgeport.edu/files/docs/academics/catalogs/catalog-2022-2023.pdf"
PDF_PATH_LOCAL = "ub_catalog.pdf"
EMBED_MODEL    = "sentence-transformers/all-mpnet-base-v2"  # 768-dim
MODEL_ID       = "microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN       = "add token here"  # <- replace then revoke

CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
TOP_K          = 10
LOG_FILE       = "rag_error.log"

def log(msg: str):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass

def log_exc(note: str):
    log(f"⚠️ {note}")
    traceback.print_exc()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f); f.write("\n")
    except Exception:
        pass

def ensure_pdf_local():
    if os.path.exists(PDF_PATH_LOCAL) and os.path.getsize(PDF_PATH_LOCAL) > 0:
        return
    log(f"→ Downloading catalog to {PDF_PATH_LOCAL} …")
    urllib.request.urlretrieve(PDF_URL_REMOTE, PDF_PATH_LOCAL)
    log("✅ PDF downloaded")

def load_pages() -> List[Document]:
    pages = PyPDFLoader(PDF_PATH_LOCAL, extract_images=False).load_and_split()
    log(f"Pages loaded: {len(pages)}")
    if not pages:
        raise RuntimeError("No pages parsed from PDF.")
    return pages

def make_chunks(pages: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
    chunks = splitter.split_documents(pages)
    log(f"Chunks created: {len(chunks)} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    if not chunks:
        raise RuntimeError("No chunks produced from pages.")
    return chunks

def pretty_docs(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs[:3], start=1):
        meta = d.metadata or {}
        src = meta.get("source"); page = meta.get("page_number", meta.get("page"))
        preview = textwrap.shorten(d.page_content.replace("\n", " "), width=160, placeholder=" …")
        lines.append(f"[{i}] source={src} page={page} | {preview}")
    return "\n".join(lines) if lines else "(no docs)"

def make_chain(llm):
    tmpl = """<|system|>
Use only the provided context. If the answer is not found in the context, reply "I dont know".<|end|>
<|user|>
Context:
{context}

Question: {question}<|end|>
<|assistant|>"""
    return create_stuff_documents_chain(llm=llm, prompt=ChatPromptTemplate.from_template(tmpl))

def main():
    # quiet perf-only warnings
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            pass
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        ensure_pdf_local()
        pages  = load_pages()
        chunks = make_chunks(pages)

        # LLM
        device_idx = 0 if torch.cuda.is_available() else -1
        print(f"Loading Phi-3 on {'CUDA' if device_idx==0 else 'CPU'} …")
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
        mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
        gen = pipeline("text-generation", model=mdl, tokenizer=tok, device=device_idx, max_new_tokens=300)
        llm = HuggingFacePipeline(pipeline=gen)
        print("✅ Model loaded")

        # Embeddings
        print(f"→ Loading embeddings: {EMBED_MODEL}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cuda" if device_idx == 0 else "cpu"},
        )
        print("✅ Embeddings ready")

        # Build FAISS (in-memory; fast & stable on Windows)
        print("→ Building FAISS index …")
        vectordb = FAISS.from_documents(chunks, embedding=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
        print("✅ FAISS index ready")

        chain = make_chain(llm)
        print("✅ QA chain ready\n")

        print("RAG is ready. Type questions (or 'end' to quit).")
        while True:
            try:
                q = input("\nYour question: ").strip()
            except EOFError:
                break
            if not q or q.lower() == "end":
                break

            print("→ Retrieving …", flush=True)
            docs = []
            try:
                docs = retriever.invoke(q)
            except Exception:
                log_exc("retriever.invoke failed; using similarity_search")
                try:
                    docs = vectordb.similarity_search(q, k=TOP_K)
                except Exception:
                    log_exc("similarity_search failed")
                    print("⚠️ Retrieval failed. Try a different question.")
                    continue

            if not docs:
                print("⚠️ No documents retrieved. Try a different question.")
                continue

            print("Retrieved context (top 3):")
            print(pretty_docs(docs))

            print("\n→ Generating answer …", flush=True)
            try:
                out = chain.invoke({"context": docs, "question": q})
            except Exception:
                log_exc("Generation crashed; retrying on CPU …")
                # CPU fallback generation
                cpu_tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
                cpu_mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
                cpu_gen = pipeline("text-generation", model=cpu_mdl, tokenizer=cpu_tok, device=-1, max_new_tokens=300)
                cpu_llm = HuggingFacePipeline(pipeline=cpu_gen)
                cpu_chain = make_chain(cpu_llm)
                out = cpu_chain.invoke({"context": docs, "question": q})

            ans = (out.split("<|assistant|>")[-1]).strip()
            print("\nAnswer:\n", ans, flush=True)

    except Exception:
        log_exc("💥 Fatal error during setup")
    finally:
        try:
            input("\nPress Enter to exit…")
        except Exception:
            pass

if __name__ == "__main__":
    sys.exit(int(main() or 0))
