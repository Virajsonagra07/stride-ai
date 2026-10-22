# StrideAI — Grounded AI Shopping Assistant for Multi-Store Retail

An AI assistant for a family-run, 3-store footwear retailer. Customers ask in **English or Hinglish** ("shaadi ke liye mojari chahiye, size 8, 2k ke andar") and get recommendations that are **actually in stock, in their size, at a specific branch**. Every question is also logged as a **demand signal**, so the owner sees what customers wanted but couldn't buy and what to restock.

> Built on a real problem: my family runs a 25+ year multi-store footwear business. Customers walk out when their size isn't on the shelf, staff can't see stock at other branches, and nobody records what people asked for and didn't find.

**Live demo:** `<your-app>.streamlit.app` · **Stack:** Python · Streamlit · Groq (gpt-oss-120b / 20b) · SQLite · BM25 + fastembed (bge-small) · RRF · pypdf

---

## What makes it more than "chat with a PDF"

| Plain PDF RAG | StrideAI |
|---|---|
| Vector search only | **Hybrid retrieval**: SQL hard filters + BM25 + dense embeddings, fused with Reciprocal Rank Fusion |
| Can recommend things that are out of stock | **Never.** Size and stock > 0 are SQL constraints, not suggestions |
| Returns nothing when nothing matches | **Progressive relaxation**: tries other branches, then +25% budget, then colour, and so on, and tells the customer what it changed |
| Trusts the LLM output | **Post-generation verification**: any SKU the model cites that wasn't retrieved is stripped |
| English only | **English + Hinglish** (kala, chappal, "2k ke andar", "number 9", shaadi) |
| Answers and forgets | **Demand analytics**: unmet-demand rate, size-gap heatmap, revenue-at-risk, restock list |
| Dies if the API is down | **Graceful degradation** to a deterministic rule parser and templates |

## Architecture

```
Customer message (English / Hinglish)
   │
   ▼  ① Query understanding — Groq LLM in JSON mode → intent + filters
   │     (every value validated against catalog vocabulary; rule parser = fallback + guardrail)
   ▼  ② Router — product_search │ policy │ store_info │ chitchat
   │
   ├─ product_search → SQL over live inventory (size, qty>0, price, gender, store)
   │                   → 0 rows? cost-ordered relaxation plans
   │                   → rank: BM25 + dense (bge-small, ONNX) → RRF
   ├─ policy         → PDF → recursive chunks (500/80 overlap) → hybrid top-3
   ├─ store_info     → stores table
   ▼
   ③ Grounded generation — answers ONLY from retrieved context, cites [SKU-XXXX], matches the customer's language
   ▼
   ④ Verification — strip hallucinated SKUs, report grounding %
   ▼
   ⑤ Logging — analytics DB → Demand Insights tab (owner view)
```

```
app.py                Streamlit UI: chat · demand insights · how it works
core/parser.py        LLM + rule-based query understanding (Hinglish lexicon)
core/retriever.py     SQL inventory search, relaxation, hybrid BM25+dense index, PDF chunking
core/agent.py         orchestration, grounded prompt, SKU verification, fallbacks
core/analytics.py     query logging, unmet demand, revenue-at-risk
core/llm.py           Groq wrapper (timeouts and failures → None → fallback)
data/                 sample catalog / inventory / stores + policy PDF (all synthetic)
eval/                 eval set + metrics script
```

## Run in VS Code (Mac)
1. Open this folder in VS Code → install the recommended Python extension when prompted.
2. Terminal → `bash setup_mac.sh` (creates `.venv`, installs packages, creates `.env`).
3. Paste your Groq key into `.env`.
4. Press **F5** → "Run StrideAI (Streamlit)" → browser opens at localhost:8501.

## Run locally (any terminal)

```bash
git clone https://github.com/<you>/stride-ai && cd stride-ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...                              # free key: console.groq.com/keys
streamlit run app.py
```
No key? It still runs in deterministic mode. The first launch downloads the ~70 MB embedding model.

## Deploy free (Streamlit Community Cloud)

1. Push this folder to a **public GitHub repo** (`.gitignore` already excludes secrets and the DB).
2. Go to share.streamlit.io → **Create app** → pick the repo → main file `app.py` → Advanced settings → Python **3.11**.
3. In **Secrets**, paste the contents of `.streamlit/secrets.toml.example` with your real Groq key.
4. Deploy, then put the URL at the top of this README and on your resume.

> Groq retires models regularly. If answers fall back to "TEMPLATE" in the *Under the hood* panel, check console.groq.com/docs/models and update `GROQ_MODEL` in Secrets. No code change is needed.

## Evaluation

```bash
python eval/run_eval.py
```
It reports intent accuracy, slot accuracy (category, gender, colour, size, budget), **constraint precision** (the share of recommended items that truly satisfy size, stock and budget), policy hit@3 and grounding.
The 22-case set is hand-written and small, so treat it as a regression test, not a benchmark. Add real customer phrasings to `eval/eval_set.json` as you collect them.

## Use your own data
Replace `data/catalog.csv`, `data/inventory.csv` and `data/stores.csv` with a POS export that uses the same columns, and drop the real policy PDFs into `data/docs/`. Nothing else changes.

## Roadmap
- WhatsApp front-end (Gupshup / Meta Cloud API): same `agent.answer()` behind a webhook
- "Reserve for 24h" → writes a hold into inventory
- Postgres + nightly POS sync; per-branch restock alerts via n8n
- Image search ("find shoes like this photo")

*All products, brands, stores and stock in this repo are synthetic sample data.*
