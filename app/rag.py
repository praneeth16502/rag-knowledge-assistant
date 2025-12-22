from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import TOP_K
import subprocess


def build_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents, embeddings)


def answer_question(vectorstore, question: str):
    docs = vectorstore.similarity_search(question, k=TOP_K)

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

    # Call local LLM via Ollama
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        text=True,
    )

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "snippet": doc.page_content[:200] + "..."
        }
        for doc in docs
    ]

    return {
        "answer": result.stdout.strip(),
        "sources": sources
    }
