# -*- coding: utf-8 -*-
# LLMTasks.py — helpers for LlamaIndex 0.10.x
# No external reader packages required: manual loaders for .txt and .pdf.

from __future__ import annotations
import os
from typing import List, Tuple

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------- File loaders (no llama-index-readers-file needed) ----------------
def _ensure_exists(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"Missing or empty data file: {path}")

def _load_txt(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(text=text, metadata={"file_name": os.path.basename(path)})]

def _load_pdf(path: str) -> List[Document]:
    # Try PyMuPDF first (best quality), else fallback to PyPDF2.
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            pages.append(page.get_text())
        text = "\n".join(pages)
        return [Document(text=text, metadata={"file_name": os.path.basename(path)})]
    except Exception:
        try:
            import PyPDF2
            text = ""
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return [Document(text=text, metadata={"file_name": os.path.basename(path)})]
        except Exception as e:
            raise ImportError(
                "Unable to read PDF. Install one PDF backend:\n"
                "  pip install pymupdf\n"
                "or\n"
                "  pip install PyPDF2"
            ) from e

def load_documents_paths(paths: List[str]) -> List[Document]:
    """Load a list of .txt/.pdf into Documents without optional readers."""
    docs: List[Document] = []
    for p in paths:
        _ensure_exists(p)
        ext = os.path.splitext(p)[1].lower()
        if ext == ".txt":
            docs.extend(_load_txt(p))
        elif ext == ".pdf":
            docs.extend(_load_pdf(p))
        else:
            # Default: try reading as text
            docs.extend(_load_txt(p))
    return docs

# ---------------- LLM helpers (same API your main expects) ----------------
def LLM_generation_complete(llm, prompt: str) -> str:
    resp = llm.complete(prompt)
    return str(resp).strip()

def LLM_generation_chat(llm, messages: List[ChatMessage]) -> str:
    resp = llm.chat(messages)
    return str(resp).strip()

def LLM_RAG_simple(llm, question: str, input_files: List[str], topk: int = 3) -> str:
    """
    Tiny RAG: load files -> chunk -> index -> query.
    Uses manual loaders; no extra reader packages required.
    """
    # Configure global LlamaIndex
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Load files into Documents
    docs = load_documents_paths(input_files)

    # Chunk and index
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes)

    # Query
    qe = index.as_query_engine(similarity_top_k=topk)
    resp = qe.query(question)
    return str(resp).strip()
