from dotenv import load_dotenv
load_dotenv()
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment")

# RAG settings
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
TOP_K = 4

