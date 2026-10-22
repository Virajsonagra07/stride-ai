"""Offline evaluation - run before every demo / commit.

Measures:
  * intent accuracy             - router sends the query to the right pipeline
  * filter accuracy (slot F1)   - parser extracts category/gender/colour/size/price correctly
  * constraint precision        - % of recommended products that truly satisfy size + stock + budget
  * policy retrieval hit@3      - correct policy section appears in top-3 chunks
  * grounding                   - % of cited SKUs that came from retrieval (hallucination guard)
Run:  python eval/run_eval.py            (uses LLM if GROQ_API_KEY set, else rule parser)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import agent, parser, retriever  # noqa: E402

CASES = json.loads((Path(__file__).parent / "eval_set.json").read_text())
SLOTS = ["category", "gender", "color", "size", "max_price"]


def main():
    n_int = n_slot_ok = n_slot = 0
    cons_ok = cons_total = 0
    hit = n_pol = 0
    grounded = []
    rows = []
    t0 = time.time()
    for c in CASES:
        f, src, _ = parser.parse(c["q"])
        n_int += f["intent"] == c["intent"]
        for s in SLOTS:
            if s in c:
                n_slot += 1
                n_slot_ok += f.get(s) == c[s]
        if c["intent"] == "product_search":
            res = retriever.search_products(f)
            if not res["relaxed"]:
                for p in res["results"]:
                    cons_total += 1
                    ok = (not f.get("max_price") or p["price_inr"] <= f["max_price"]) and \
                         (not f.get("size") or sum(v for v in p["stock"].values()) > 0)
                    cons_ok += ok
        if c["intent"] == "policy":
            n_pol += 1
            docs = retriever.search_docs(f.get("keywords") or c["q"])
            hit += any(c["expect_text"].lower() in d["text"].lower() for d in docs)
        r = agent.answer(c["q"])
        grounded.append(r["trace"]["verification"]["grounded"])
        rows.append((c["q"][:48], f["intent"], src, r["trace"]["generator"], r["trace"]["total_ms"]))

    print(f"{'query':50}{'intent':16}{'parser':8}{'answer':10}ms")
    for q, i, s, g, ms in rows:
        print(f"{q:50}{i:16}{s:8}{g:10}{ms}")
    print("\n=== RESULTS ===")
    print(f"cases                 {len(CASES)}")
    print(f"intent accuracy       {n_int / len(CASES):.0%}")
    print(f"slot accuracy         {n_slot_ok / max(n_slot, 1):.0%}  ({n_slot_ok}/{n_slot})")
    print(f"constraint precision  {cons_ok / max(cons_total, 1):.0%}  ({cons_ok}/{cons_total} recommended items)")
    print(f"policy hit@3          {hit / max(n_pol, 1):.0%}  ({hit}/{n_pol})")
    print(f"grounding             {sum(grounded) / len(grounded):.0%}")
    print(f"wall time             {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
