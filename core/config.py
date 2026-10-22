import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = DATA / "docs"
ANALYTICS_DB = ROOT / "analytics.db"

# Load .env (KEY=value lines) without extra dependencies; real env vars win.
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))


def _secret(name: str, default: str = "") -> str:
    """Read from env first, then Streamlit secrets (for Streamlit Cloud)."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


STORE_BRAND = _secret("STORE_BRAND", "StrideAI Footwear")
GROQ_API_KEY = _secret("GROQ_API_KEY")
# Groq retires models regularly; keep these configurable. Check console.groq.com/docs/models.
GROQ_MODEL = _secret("GROQ_MODEL", "openai/gpt-oss-120b")          # answer generation
GROQ_FAST_MODEL = _secret("GROQ_FAST_MODEL", "openai/gpt-oss-20b")  # query understanding
EMBED_MODEL = _secret("EMBED_MODEL", "BAAI/bge-small-en-v1.5")      # fastembed (ONNX, no torch)
USE_DENSE = _secret("USE_DENSE", "1") == "1"
TOP_K = 5
