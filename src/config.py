"""Environment-driven settings. Every model ID is pinned through .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", str(ROOT / ".models"))

CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(ROOT / "chroma_index")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "techdocs")
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", str(ROOT / "corpus")))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"

STATE_BACKEND = os.getenv("STATE_BACKEND", "sqlite")
SQLITE_STATE_PATH = os.getenv("SQLITE_STATE_PATH", str(ROOT / "state.sqlite"))
FIRESTORE_PROJECT_ID = os.getenv("FIRESTORE_PROJECT_ID", "")

MAX_SUPERVISOR_HOPS = int(os.getenv("MAX_SUPERVISOR_HOPS", "3"))
