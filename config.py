import os
from dotenv import load_dotenv

load_dotenv()

# ─── OpenAI ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
FAST_MODEL       = "gpt-4o-mini"
SMART_MODEL      = "gpt-4o"
EMBEDDING_MODEL  = "text-embedding-3-small"

# ─── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ─── Celery ───────────────────────────────────────────────────────────────────
CELERY_MAX_WORKERS   = int(os.getenv("CELERY_MAX_WORKERS", "4"))   # max parallel analyses
CELERY_TASK_TIMEOUT  = int(os.getenv("CELERY_TASK_TIMEOUT", "600")) # 10 min per job

# ─── Rate Limiting ────────────────────────────────────────────────────────────
RATE_LIMIT_MAX       = int(os.getenv("RATE_LIMIT_MAX", "5"))        # max analyses
RATE_LIMIT_WINDOW    = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # per hour (seconds)

# ─── Cost Tracking ────────────────────────────────────────────────────────────
# Prices per 1M tokens (USD) — update when OpenAI changes pricing
COST_PER_1M = {
    "gpt-4o":              {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":         {"input": 0.15,  "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_BASE_DIR = os.getenv("CHROMA_BASE_DIR", "./chroma_db")

# ─── LLM Retry ────────────────────────────────────────────────────────────────
LLM_MAX_RETRIES    = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_MIN_WAIT = int(os.getenv("LLM_RETRY_MIN_WAIT", "2"))   # seconds
LLM_RETRY_MAX_WAIT = int(os.getenv("LLM_RETRY_MAX_WAIT", "30"))  # seconds
