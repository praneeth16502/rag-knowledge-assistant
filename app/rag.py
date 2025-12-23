import time
import logging
import subprocess
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import TOP_K

# Basic logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def build_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents, embeddings)


def answer_question(vectorstore, question: str):
    logger.info(f"Query received: {question}")

    # Retrieval timing
    t0 = time.time()
    docs = vectorstore.similarity_search(question, k=TOP_K)
    retrieval_time = time.time() - t0

    logger.info(
        f"Retrieved {len(docs)} chunks in {retrieval_time:.3f}s"
    )

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present, say: "I don't know."

Context:
{context}

Question:
{question}
"""

    # LLM timing
    t1 = time.time()
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        text=True,
    )
    llm_time = time.time() - t1

    logger.info(f"LLM response generated in {llm_time:.3f}s")

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "snippet": doc.page_content[:200] + "..."
        }
        for doc in docs
    ]

    return {
        "answer": result.stdout.strip(),
        "sources": sources,
        "metrics": {
            "retrieval_time_sec": round(retrieval_time, 3),
            "llm_time_sec": round(llm_time, 3),
            "chunks_used": len(docs),
        }
    }
