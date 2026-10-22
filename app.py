"""StrideAI - grounded shopping assistant for a multi-store footwear retailer.
Run: streamlit run app.py
"""
import json

import altair as alt
import pandas as pd
import streamlit as st

from core import agent, analytics, config, llm, retriever

st.set_page_config(page_title="StrideAI - Footwear Shopping Assistant", page_icon="👟", layout="wide")

SEQ = ["#dbe9f6", "#9ecae1", "#4292c6", "#08519c"]  # single-hue sequential (magnitude)
BAR = "#2b6cb0"

EXAMPLES = [
    "Black formal shoes for office, size 9, under 3000",
    "Shaadi ke liye mojari chahiye, size 8, 2k ke andar",
    "Comfortable running shoes for women size 5",
    "Red heels size 7 under 1000",
    "Can I return shoes I bought on sale?",
    "Mall wala store kab tak khula hai?",
    "What is UK size for US 10?",
]


@st.cache_resource(show_spinner="Building search indexes (one-time)...")
def warmup():
    retriever.db()
    retriever.product_index()
    retriever.doc_index()
    analytics.seed_if_empty()
    return True


warmup()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"### 👟 {config.STORE_BRAND}")
    st.caption("AI shopping assistant · 3 stores · live stock")
    st.markdown("**System status**")
    st.write(("🟢 LLM: `" + config._secret("GROQ_MODEL", config.GROQ_MODEL) + "` on Groq") if llm.available()
             else "🟡 LLM offline - deterministic mode (add GROQ_API_KEY)")
    st.write("🟢 Hybrid search: BM25 + dense embeddings" if retriever.Embedder.active()
             else "🟡 Search: BM25 + SQL (dense embeddings off)")
    st.divider()
    st.markdown("**Try asking**")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state.pending = ex
    st.divider()
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
    st.caption("Sample data: products, brands, stores and stock are synthetic.")

tab_chat, tab_insights, tab_how = st.tabs(["Shop assistant", "Demand insights (owner view)", "How it works"])

# ------------------------------------------------------------------ chat
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_products(products):
    if not products:
        return
    cols = st.columns(min(len(products), 3))
    for c, p in zip(cols * 2, products[:6]):
        with c.container(border=True):
            st.markdown(f"**{p['name']}**  \n`{p['sku']}` · {p['category']} · {p['gender']}")
            st.markdown(f"### ₹{int(p['price_inr']):,}" + ("  🏷️ sale" if p["on_sale"] else ""))
            st.caption(f"⭐ {p['rating']} · {p['material']} · {p['features'].replace('|', ', ')}")
            for store, v in p["stock"].items():
                if isinstance(v, int):
                    badge = "🔴" if v <= 1 else "🟠" if v <= 2 else "🟢"
                    st.caption(f"{badge} {store}: {v} left")
                else:
                    st.caption(f"🟢 {store}: sizes {', '.join(map(str, v))}")


def render_trace(tr, docs):
    with st.expander("Under the hood"):
        a, b, c, d = st.columns(4)
        a.metric("Total latency", f"{tr['total_ms']} ms")
        b.metric("Parser", tr["parser"].upper(), f"{tr['parse_ms']} ms" if tr["parse_ms"] else None)
        c.metric("Answer", tr["generator"].upper(), f"{tr['gen_ms']} ms" if tr["gen_ms"] else None)
        d.metric("Grounding check", f"{int(tr['verification']['grounded'] * 100)}%",
                 f"{len(tr['verification']['rejected'])} rejected" if tr['verification']['rejected'] else None)
        st.markdown("**1 · Understood as**")
        st.json({k: v for k, v in tr["parsed"].items() if v not in (None, "")}, expanded=False)
        if "sql" in tr:
            st.markdown(f"**2 · Live inventory query** — {tr['exact']} exact in-stock matches")
            st.code(tr["sql"] + "\n-- params: " + json.dumps(tr["sql_args"]), language="sql")
            if tr["relaxed"]:
                st.warning("No exact match → relaxed: " + " → ".join(tr["relaxed"]))
        if docs:
            st.markdown("**2 · Retrieved policy chunks** (hybrid BM25 + dense, RRF)")
            for dd in docs:
                st.caption(f"{dd['source']} p.{dd['page']} · score {dd['score']}")
                st.text(dd["text"][:400])


with tab_chat:
    st.markdown("#### Ask anything — size, budget, occasion, store, returns. English or Hinglish.")
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_products(m.get("products", []))
                render_trace(m["trace"], m.get("docs", []))

    q = st.chat_input("e.g. white sneakers size 8 under 2000") or st.session_state.pop("pending", None)
    if q:
        history = [(m["role"], m["content"]) for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Checking all 3 stores..."):
                r = agent.answer(q, history)
            st.markdown(r["text"])
            render_products(r["products"])
            render_trace(r["trace"], r["docs"])
        st.session_state.messages.append({"role": "assistant", "content": r["text"], "products": r["products"],
                                          "docs": r["docs"], "trace": r["trace"]})

# ------------------------------------------------------------------ insights
with tab_insights:
    df = analytics.load()
    st.markdown("#### Every customer question is a demand signal")
    st.caption(f"{len(df):,} queries logged · {int(df.simulated.sum()):,} simulated (30-day seed) · "
               f"{int((df.simulated == 0).sum()):,} live from this demo")
    ps = df[df.intent == "product_search"]
    unmet = ps[ps.exact_matches == 0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Product searches", f"{len(ps):,}")
    k2.metric("Unmet demand rate", f"{len(unmet) / max(len(ps), 1):.0%}", help="Searches with zero exact in-stock matches")
    k3.metric("Est. revenue at risk", f"₹{analytics.lost_revenue_estimate(df):,}",
              help="Unmet searches × category avg price × 30% assumed conversion")
    k4.metric("Median response", f"{int(df.latency_ms.median())} ms")

    left, right = st.columns(2)
    with left:
        st.markdown("**Unmet demand by category** — searches with no exact in-stock match")
        uc = unmet.groupby("category").size().reset_index(name="searches").sort_values("searches", ascending=False)
        st.altair_chart(alt.Chart(uc).mark_bar(color=BAR, cornerRadiusEnd=4, size=18).encode(
            x=alt.X("searches:Q", title="Unmet searches"), y=alt.Y("category:N", sort="-x", title=None),
            tooltip=["category", "searches"]).properties(height=280), width="stretch")
    with right:
        st.markdown("**Size gaps** — where stock-outs cluster")
        hm = unmet.dropna(subset=["size"]).groupby(["category", "size"]).size().reset_index(name="searches")
        hm["size"] = hm["size"].astype(int)
        st.altair_chart(alt.Chart(hm).mark_rect(cornerRadius=3, stroke="white", strokeWidth=2).encode(
            x=alt.X("size:O", title="Size (UK/IND)"), y=alt.Y("category:N", title=None),
            color=alt.Color("searches:Q", scale=alt.Scale(range=SEQ), title="Unmet"),
            tooltip=["category", "size", "searches"]).properties(height=280), width="stretch")

    st.markdown("**Restock recommendations** (top unmet category × gender × size)")
    rec = analytics.unmet_demand(df).head(8).copy()
    if not rec.empty:
        rec["size"] = rec["size"].astype("Int64")
        rec["gender"] = rec["gender"].fillna("any")
        rec["action"] = rec.apply(lambda r: f"Review stock: {r['gender']} {r['category']} size {r['size']} — "
                                            f"{r['searches']} searches found no in-stock match", axis=1)
        st.dataframe(rec, hide_index=True, width="stretch")
    with st.expander("Raw query log"):
        st.dataframe(df.sort_values("ts", ascending=False), hide_index=True, width="stretch")

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown("""
#### Architecture

```
Customer message (English / Hinglish)
        │
        ▼
① Query understanding ── Groq LLM (JSON mode) → intent + filters
        │                 └─ validated against catalog vocabulary; rule-based parser as fallback
        ▼
② Router ── product_search │ policy │ store_info │ chitchat
        │
        ├─ product_search → SQL on live inventory (size, stock>0, price, gender, store)
        │                   → no match? progressive relaxation (colour → budget → branch → size)
        │                   → hybrid rank: BM25 + dense embeddings, fused with Reciprocal Rank Fusion
        │
        ├─ policy         → PDF → recursive chunks (500 chars, 80 overlap) → hybrid retrieval top-3
        │
        ▼
③ Grounded generation ── Groq LLM, answers ONLY from retrieved context, cites [SKU-XXXX]
        ▼
④ Verification ── any cited SKU not in the retrieved set is stripped (hallucination guard)
        ▼
⑤ Demand logging ── every query → analytics DB → unmet-demand + restock recommendations
```

#### Why this beats a plain "chat with your PDF" RAG
| Plain PDF RAG | StrideAI |
|---|---|
| Vector search only | SQL hard constraints + BM25 + dense, RRF fusion |
| Can recommend out-of-stock items | Never — stock and size are SQL filters |
| Returns nothing on no match | Relaxes constraints step by step and says so |
| Trusts the LLM | Post-generation SKU verification |
| English only | English + Hinglish |
| Answers and forgets | Logs demand → restock recommendations for the owner |
| Breaks without API | Degrades to deterministic mode |
""")
