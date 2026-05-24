import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
PDF_PATH = DATA_DIR / "fia_2026_formula_1_technical_regulations_issue_8_-_2024-06-24.pdf"
PARSED_TEXT_PATH = DATA_DIR / "parsed_chunks.json"

# ---- API Keys ----
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "your-dashscope-key-here")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key-here")

# ---- Embedding ----
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 10  # DashScope 单次请求最大20条

# ---- LLM ----
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com"

# ---- Retrieval ----
VECTOR_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 5

# ---- Chunking ----
CHUNK_MAX_CHARS = 1500
CHUNK_OVERLAP_CHARS = 100
