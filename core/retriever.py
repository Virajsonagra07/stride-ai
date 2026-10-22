"""Hybrid retrieval.

Products:  SQL hard filters (live stock, size, price, gender, category)  ->  progressive relaxation
           when nothing matches  ->  hybrid ranking (BM25 + dense embeddings, fused with RRF).
Documents: PDF -> recursive chunks with overlap -> BM25 + dense -> RRF top-k.

Why hybrid: SQL guarantees we never recommend a pair that isn't in stock in the asked size;
semantic ranking handles fuzzy intent ("comfortable for long standing", "something classy").
"""
import re
import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from . import config

STORE_NAMES = {}


# ---------------------------------------------------------------- data
@lru_cache(maxsize=1)
def db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:", check_same_thread=False)
    for name in ("catalog", "inventory", "stores"):
        pd.read_csv(config.DATA / f"{name}.csv").to_sql(name, con, index=False)
    con.execute("CREATE INDEX ix_inv ON inventory(sku, size, store_id)")
    STORE_NAMES.update(dict(con.execute("SELECT store_id, store_name FROM stores").fetchall()))
    return con


def catalog() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM catalog", db())


def stores() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM stores", db())


def _tok(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())


# ---------------------------------------------------------------- embeddings
class Embedder:
    """fastembed (ONNX, CPU, no torch -> deploys on free tiers). Falls back to None silently."""
    _model, _failed = None, False

    @classmethod
    def encode(cls, texts):
        if cls._failed or not config.USE_DENSE:
            return None
        try:
            if cls._model is None:
                from fastembed import TextEmbedding
                cls._model = TextEmbedding(config.EMBED_MODEL)
            v = np.array(list(cls._model.embed(list(texts))), dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        except Exception as e:
            print(f"[embed] dense retrieval disabled ({type(e).__name__}: {e}); using BM25 only")
            cls._failed = True
            return None

    @classmethod
    def active(cls):
        return cls._model is not None and not cls._failed


def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion over lists of ids (best first)."""
    score = {}
    for ranks in rank_lists:
        for r, i in enumerate(ranks):
            score[i] = score.get(i, 0) + 1.0 / (k + r + 1)
    return score


class HybridIndex:
    def __init__(self, ids, texts):
        self.ids, self.texts = list(ids), list(texts)
        self.bm25 = BM25Okapi([_tok(t) for t in self.texts])
        self.vecs = Embedder.encode(self.texts)

    def rank(self, query, subset=None):
        idx = list(range(len(self.ids))) if subset is None else \
            [i for i, x in enumerate(self.ids) if x in subset]
        if not idx:
            return {}
        bm = self.bm25.get_scores(_tok(query))
        lists = [sorted(idx, key=lambda i: -bm[i])]
        if self.vecs is not None:
            qv = Embedder.encode([query])
            if qv is not None:
                cos = self.vecs @ qv[0]
                lists.append(sorted(idx, key=lambda i: -cos[i]))
        fused = rrf(lists)
        return {self.ids[i]: s for i, s in fused.items()}


@lru_cache(maxsize=1)
def product_index() -> HybridIndex:
    c = catalog()
    texts = (c["name"] + ". " + c["description"] + " Occasions: " + c["occasions"].str.replace("|", ", ")
             + ". Material: " + c["material"] + ".").tolist()
    return HybridIndex(c["sku"], texts)


# ---------------------------------------------------------------- product search
def _sql(f: dict):
    where, args = ["1=1"], []
    if f.get("sku"):
        where.append("c.sku = ?"); args.append(f["sku"])
    if f.get("category"):
        where.append("c.category = ?"); args.append(f["category"])
    if f.get("gender"):
        if f["gender"] in ("men", "women"):
            where.append("c.gender IN (?, 'unisex')")
        else:
            where.append("c.gender = ?")
        args.append(f["gender"])
    if f.get("color"):
        where.append("(c.color LIKE ? OR c.name LIKE ?)"); args += [f"%{f['color']}%"] * 2
    if f.get("brand"):
        where.append("c.brand LIKE ?"); args.append(f"%{f['brand']}%")
    if f.get("max_price"):
        where.append("c.price_inr <= ?"); args.append(f["max_price"])
    if f.get("min_price"):
        where.append("c.price_inr >= ?"); args.append(f["min_price"])
    inv = ["i.qty > 0"]
    if f.get("size"):
        inv.append("i.size = ?"); args.append(f["size"])
    if f.get("store"):
        inv.append("i.store_id = ?"); args.append(f["store"])
    sql = (f"SELECT DISTINCT c.sku FROM catalog c JOIN inventory i ON i.sku = c.sku "
           f"WHERE {' AND '.join(where)} AND {' AND '.join(inv)}")
    return sql, args


def _plan(f, store=False, budget=None, color=False, brand=False, size=False):
    g = dict(f)
    if store: g["store"] = None
    if budget == "+25%" and g.get("max_price"): g["max_price"] = int(g["max_price"] * 1.25)
    if budget == "drop": g["max_price"] = g["min_price"] = None
    if color: g["color"] = None
    if brand: g["brand"] = None
    if size: g["size"] = None
    return g


# Relaxation plans, ordered by how much they cost the customer. Each starts from the ORIGINAL request,
# so we never drop the size if relaxing only the budget would have worked.
RELAX_PLANS = [
    (["other branches"], dict(store=True)),
    (["budget +25%"], dict(store=True, budget="+25%")),
    (["colour"], dict(store=True, color=True)),
    (["budget +25%", "colour"], dict(store=True, budget="+25%", color=True)),
    (["brand"], dict(store=True, budget="+25%", color=True, brand=True)),
    (["budget"], dict(store=True, budget="drop")),
    (["budget", "colour"], dict(store=True, budget="drop", color=True, brand=True)),
    (["size (showing other sizes)"], dict(store=True, size=True)),
    (["size", "colour"], dict(store=True, size=True, color=True, brand=True, budget="+25%")),
    (["size", "colour", "budget"], dict(store=True, size=True, color=True, brand=True, budget="drop")),
]


def search_products(f: dict, k: int = config.TOP_K):
    original = dict(f)
    relaxed = []
    sql, args = _sql(f)
    skus = [r[0] for r in db().execute(sql, args)]
    exact_count = len(skus)
    if not skus:
        seen = [f]
        for labels, kw in RELAX_PLANS:
            g = _plan(original, **kw)
            if g in seen:
                continue
            seen.append(g)
            sql, args = _sql(g)
            skus = [r[0] for r in db().execute(sql, args)]
            if skus:
                f, relaxed = g, labels
                break

    query = " ".join(str(x) for x in [original.get("keywords"), original.get("occasion"),
                                      original.get("color"), original.get("category")] if x)
    scores = product_index().rank(query or "shoes", set(skus)) if skus else {}
    cat = catalog().set_index("sku")
    ranked = sorted(skus, key=lambda s: (-scores.get(s, 0), -cat.loc[s, "rating"]))[:k]

    results = []
    for s in ranked:
        row = cat.loc[s].to_dict()
        row["sku"] = s
        row["retrieval_score"] = round(scores.get(s, 0), 4)
        row["stock"] = stock_for(s, f.get("size"))
        results.append(row)
    return dict(results=results, exact_count=exact_count, relaxed=relaxed, sql=sql, sql_args=args,
                filters_used=f)


def stock_for(sku, size=None):
    """{store_name: qty} if size given, else {store_name: [available sizes]}"""
    con = db()
    if size:
        rows = con.execute("SELECT store_id, qty FROM inventory WHERE sku=? AND size=? AND qty>0",
                           (sku, size)).fetchall()
        return {STORE_NAMES[s]: q for s, q in rows}
    rows = con.execute("SELECT store_id, size FROM inventory WHERE sku=? AND qty>0 ORDER BY size",
                       (sku,)).fetchall()
    out = {}
    for s, z in rows:
        out.setdefault(STORE_NAMES[s], []).append(z)
    return out


# ---------------------------------------------------------------- document RAG
def split_text(text, size=500, overlap=80, seps=("\n\n", "\n", ". ", " ")):
    """Recursive character splitter (same idea as LangChain's, no dependency)."""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    for sep in seps:
        if sep in text:
            parts, chunks, cur = text.split(sep), [], ""
            for p in parts:
                cand = (cur + sep + p) if cur else p
                if len(cand) <= size:
                    cur = cand
                else:
                    if cur:
                        chunks.append(cur.strip())
                    cur = (cur[-overlap:] + sep + p) if cur and overlap else p
                    if len(cur) > size:
                        chunks.extend(split_text(cur, size, overlap, seps[seps.index(sep) + 1:] or (" ",)))
                        cur = ""
            if cur.strip():
                chunks.append(cur.strip())
            return chunks
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


@lru_cache(maxsize=1)
def doc_index():
    from pypdf import PdfReader
    chunks, meta = [], []
    for pdf in sorted(config.DOCS.glob("*.pdf")):
        for pno, page in enumerate(PdfReader(str(pdf)).pages, 1):
            text = page.extract_text() or ""
            # keep section headings attached to their paragraph
            for ch in split_text(text):
                chunks.append(ch)
                meta.append({"source": pdf.name, "page": pno})
    idx = HybridIndex(range(len(chunks)), chunks)
    return idx, chunks, meta


def search_docs(query, k=3):
    idx, chunks, meta = doc_index()
    scores = idx.rank(query)
    top = sorted(scores, key=lambda i: -scores[i])[:k]
    return [dict(text=chunks[i], score=round(scores[i], 4), **meta[i]) for i in top]
