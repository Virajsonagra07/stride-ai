"""Orchestrator: understand -> route -> retrieve -> grounded generation -> verify -> log."""
import json
import re
import time

from . import analytics, config, llm, parser, retriever

ANSWER_SYSTEM = """You are the in-store shopping assistant for {brand}, a family-run multi-store footwear retailer.
Rules (strict):
- Recommend ONLY products listed in CONTEXT. Cite each one as [SKU-XXXX]. Never invent products, prices, sizes or stock.
- Say which store has the customer's size and how many pairs are left when stock is low (1-2 pairs = "only N left").
- If CONTEXT says filters were relaxed, say honestly that the exact request isn't in stock and why these are the closest options.
- For policy / store questions, answer ONLY from the provided policy excerpts or store data. If the answer isn't there, say a staff member will confirm.
- Reply in the customer's style: {language}. Hinglish = natural Roman-script Hindi-English mix.
- Be warm and brief: max ~120 words, short bullets for products (name, price in Rs, key feature, where in stock).
- End with one helpful next step (e.g. reserve on WhatsApp for 24 hours, or try-on at the nearest branch)."""


def _product_context(res):
    items = []
    for p in res["results"]:
        items.append(dict(sku=p["sku"], name=p["name"], price_inr=int(p["price_inr"]), color=p["color"],
                          material=p["material"], category=p["category"], gender=p["gender"],
                          features=p["features"].replace("|", ", "), rating=p["rating"],
                          on_sale=bool(p["on_sale"]), stock=p["stock"]))
    return json.dumps(dict(relaxed_filters=res["relaxed"], exact_matches=res["exact_count"], products=items),
                      ensure_ascii=False)


def _verify(text, allowed_skus):
    """Hallucination guard: drop any line citing a SKU we didn't retrieve."""
    cited = set(re.findall(r"SKU-\d{4}", text))
    bad = cited - set(allowed_skus)
    if bad:
        text = "\n".join(l for l in text.splitlines() if not any(b in l for b in bad))
    return text, dict(cited=sorted(cited), rejected=sorted(bad),
                      grounded=round(1 - len(bad) / len(cited), 2) if cited else 1.0)


def _fallback_products(f, res):
    hi = f.get("language") == "hinglish"
    if not res["results"]:
        return ("Maaf kijiye, abhi iske jaisa kuch stock mein nahi hai." if hi
                else "Sorry, nothing close to that is in stock right now.") + \
            " A staff member can check the next delivery for you."
    lines = []
    if res["relaxed"]:
        lines.append(("Exact match stock mein nahi hai - relaxed: " if hi else
                      "Exact match isn't in stock - closest options (relaxed: ") + ", ".join(res["relaxed"]) +
                     ("" if hi else ")") + ":")
    else:
        lines.append("Yeh options available hain:" if hi else "Here's what's in stock for you:")
    for p in res["results"][:4]:
        stock = p["stock"]
        where = ", ".join(f"{s} ({v} left)" if isinstance(v, int) else f"{s} (sizes {', '.join(map(str, v[:6]))})"
                          for s, v in stock.items()) or "check with staff"
        lines.append(f"- **{p['name']}** [{p['sku']}] - Rs {int(p['price_inr']):,} - "
                     f"{p['features'].split('|')[0]} - {where}")
    lines.append("Reserve any pair on WhatsApp for 24 hours." if not hi else
                 "WhatsApp pe 24 ghante ke liye reserve kar sakte hain.")
    return "\n".join(lines)


def _store_text(f):
    s = retriever.stores()
    if f.get("store"):
        s = s[s.store_id == f["store"]]
    return "\n".join(f"- **{r.store_name}** - {r.area} ({r.landmark}) - {r.hours} - {r.phone}"
                     for r in s.itertuples())


def answer(q: str, history=None) -> dict:
    t0 = time.perf_counter()
    f, parser_src, parse_ms = parser.parse(q, history)
    trace = dict(parsed=f, parser=parser_src, parse_ms=parse_ms, llm=llm.available(),
                 dense=retriever.Embedder.active())
    products, docs, gen_ms, res = [], [], 0, None
    lang = "Hinglish" if f.get("language") == "hinglish" else "English"

    if f["intent"] == "product_search":
        res = retriever.search_products(f)
        products = res["results"]
        trace.update(sql=res["sql"], sql_args=res["sql_args"], exact=res["exact_count"], relaxed=res["relaxed"])
        context = "CONTEXT (live inventory search result):\n" + _product_context(res)
        fallback = _fallback_products(f, res)
    elif f["intent"] == "policy":
        docs = retriever.search_docs(f.get("keywords") or q)
        context = "POLICY EXCERPTS:\n" + "\n---\n".join(f"(p{d['page']}) {d['text']}" for d in docs)
        fallback = "From our policy:\n\n> " + docs[0]["text"][:600] if docs else "A staff member will confirm."
    elif f["intent"] == "store_info":
        context = "STORES:\n" + _store_text(f)
        fallback = _store_text(f)
    else:
        context = "Greet the customer and ask what they're shopping for (occasion, size, budget)."
        fallback = ("Namaste! Kis occasion ke liye joote dekh rahe hain? Size aur budget bata dijiye."
                    if lang == "Hinglish" else
                    "Hi! What are you shopping for today? Tell me the occasion, your size and budget.")

    msgs = [{"role": "system", "content": ANSWER_SYSTEM.format(brand=config.STORE_BRAND, language=lang)}]
    for role, text in (history or [])[-4:]:
        msgs.append({"role": role, "content": text[:600]})
    msgs.append({"role": "user", "content": f"{context}\n\nCustomer: {q}"})
    text, gen_ms = llm.chat(msgs)
    trace["generator"] = "llm" if text else "template"
    text = text or fallback

    text, check = _verify(text, [p["sku"] for p in products])
    trace.update(gen_ms=gen_ms, verification=check, total_ms=int((time.perf_counter() - t0) * 1000))

    try:
        analytics.log(q, f, res["exact_count"] if res else None, res["relaxed"] if res else [],
                      products[0]["sku"] if products else None, parser_src, trace["total_ms"])
    except Exception as e:
        print("[analytics]", e)
    return dict(text=text, products=products, docs=docs, trace=trace)
