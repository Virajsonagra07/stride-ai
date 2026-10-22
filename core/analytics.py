"""Demand analytics: every customer question is a demand signal.

Searches that return zero exact in-stock matches = 'unmet demand' -> this is the
business-analyst layer: it tells the owner what to restock, in which size, at which branch.
"""
import json
import random
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from . import config

SCHEMA = """CREATE TABLE IF NOT EXISTS queries(
  ts TEXT, query TEXT, intent TEXT, category TEXT, gender TEXT, color TEXT, size INTEGER,
  max_price INTEGER, store TEXT, exact_matches INTEGER, relaxed TEXT, top_sku TEXT,
  parser TEXT, latency_ms INTEGER, simulated INTEGER)"""


def _con():
    con = sqlite3.connect(config.ANALYTICS_DB, check_same_thread=False)
    con.execute(SCHEMA)
    return con


def log(query, f, exact, relaxed, top_sku, parser, latency_ms, simulated=0, ts=None):
    con = _con()
    con.execute("INSERT INTO queries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        ts or datetime.now().isoformat(timespec="seconds"), query, f.get("intent"), f.get("category"),
        f.get("gender"), f.get("color"), f.get("size"), f.get("max_price"), f.get("store"),
        exact, json.dumps(relaxed), top_sku, parser, latency_ms, simulated))
    con.commit()
    con.close()


def load() -> pd.DataFrame:
    con = _con()
    df = pd.read_sql("SELECT * FROM queries", con, parse_dates=["ts"])
    con.close()
    return df


def seed_if_empty(n=600):
    """Simulated 30 days of customer queries so the insights tab has something to show on day one.
    Clearly flagged simulated=1; real traffic is logged with simulated=0."""
    df = load()
    if len(df) and df.simulated.sum() > 0:
        return
    from . import parser, retriever
    rnd = random.Random(7)
    templates = [
        ("{color} formal shoes for men size {m} under {p}", 0.16), ("sports shoes women size {w}", 0.12),
        ("running shoes size {m} under {p}", 0.12), ("{color} sneakers size {m}", 0.10),
        ("mojari for wedding size {m}", 0.08), ("heels for party size {w} under {p}", 0.08),
        ("kids school shoes black size {k}", 0.07), ("chappal size {m}", 0.07),
        ("sandals for women size {w} under {p}", 0.07), ("boots size {m}", 0.05),
        ("loafers size {m} under {p}", 0.05), ("return policy on sale items", 0.03),
    ]
    weights = [w for _, w in templates]
    start = datetime.now() - timedelta(days=30)
    for _ in range(n):
        t = rnd.choices(templates, weights)[0][0]
        q = t.format(color=rnd.choice(["black", "brown", "white", "tan", "blue"]),
                     m=rnd.choices([6, 7, 8, 9, 10, 11], [5, 15, 30, 28, 15, 7])[0],
                     w=rnd.choices([3, 4, 5, 6, 7, 8], [5, 15, 30, 28, 15, 7])[0],
                     k=rnd.choice([10, 11, 12, 13, 1, 2]), p=rnd.choice([1000, 1500, 2000, 2500, 3000]))
        f = retriever_f = parser.rule_parse(q)
        if f["intent"] == "product_search":
            r = retriever.search_products(retriever_f)
            exact, relaxed = r["exact_count"], r["relaxed"]
            top = r["results"][0]["sku"] if r["results"] else None
        else:
            exact, relaxed, top = None, [], None
        ts = (start + timedelta(minutes=rnd.randint(0, 30 * 24 * 60))).isoformat(timespec="seconds")
        log(q, f, exact, relaxed, top, "rules", rnd.randint(250, 900), simulated=1, ts=ts)


def unmet_demand(df: pd.DataFrame) -> pd.DataFrame:
    s = df[(df.intent == "product_search") & (df.exact_matches == 0)]
    if s.empty:
        return pd.DataFrame(columns=["category", "gender", "size", "searches"])
    return (s.groupby(["category", "gender", "size"], dropna=False).size().reset_index(name="searches")
            .sort_values("searches", ascending=False))


def lost_revenue_estimate(df: pd.DataFrame, conversion=0.30) -> int:
    """Unmet searches x avg price of that category x assumed conversion (stated assumption)."""
    from . import retriever
    avg = retriever.catalog().groupby("category")["price_inr"].mean()
    s = df[(df.intent == "product_search") & (df.exact_matches == 0) & df.category.notna()]
    return int(sum(avg.get(c, 0) for c in s.category) * conversion)
