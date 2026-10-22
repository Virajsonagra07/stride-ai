"""Query understanding: free text (English / Hinglish) -> structured intent + filters.

Two layers:
  1. LLM (Groq, JSON mode) - handles messy phrasing, synonyms, multi-constraint queries.
  2. Rule-based parser - deterministic fallback + sanity layer. Every LLM value is
     validated against the catalog vocabulary so the model can't invent a filter.
"""
import json
import re

from . import llm

CATEGORIES = ["formal", "loafers", "casual", "sports", "sandals", "slippers",
              "ethnic", "boots", "heels", "school"]
GENDERS = ["men", "women", "kids", "unisex"]
INTENTS = ["product_search", "policy", "store_info", "chitchat"]

CAT_SYNONYMS = {
    "formal": ["formal", "oxford", "derby", "brogue", "office shoe", "office shoes", "monk", "pump", "dress shoe"],
    "loafers": ["loafer", "loafers", "moccasin", "slip on formal"],
    "casual": ["casual", "sneaker", "sneakers", "canvas", "slip-on", "slip on", "college"],
    "sports": ["sports", "sport", "running", "runner", "gym", "trainer", "training", "walking shoe",
               "jogging", "badminton", "tennis"],
    "sandals": ["sandal", "sandals", "floater", "floaters", "gladiator"],
    "slippers": ["slipper", "slippers", "chappal", "chappals", "flip flop", "flip-flop", "slide", "slides",
                 "hawai"],
    "ethnic": ["ethnic", "mojari", "jutti", "juttis", "kolhapuri", "traditional", "sherwani"],
    "boots": ["boot", "boots", "chelsea", "chukka"],
    "heels": ["heel", "heels", "stiletto", "wedge", "block heel", "sandal heels"],
    "school": ["school", "school shoe", "pt shoe"],
}
COLOR_SYNONYMS = {
    "black": ["black", "kala", "kaala", "kali", "kaale"], "white": ["white", "safed", "sufed"],
    "brown": ["brown", "bhura"], "tan": ["tan"], "red": ["red", "laal", "lal"],
    "blue": ["blue", "neela", "nila"], "navy": ["navy"], "grey": ["grey", "gray"],
    "gold": ["gold", "golden", "sunehri"], "silver": ["silver"], "pink": ["pink", "gulabi"],
    "maroon": ["maroon"], "beige": ["beige", "cream"], "olive": ["olive"], "nude": ["nude"],
    "burgundy": ["burgundy", "wine"], "neon green": ["neon", "green", "hara"],
}
GENDER_SYNONYMS = {
    "men": ["men", "mens", "men's", "man", "male", "gents", "boys' formal", "husband", "papa", "father",
            "bhai", "for him", "ladka"],
    "women": ["women", "womens", "women's", "woman", "ladies", "lady", "female", "girl", "girls", "wife",
              "mummy", "mother", "for her", "ladki", "didi", "behen"],
    "kids": ["kid", "kids", "child", "children", "son", "daughter", "baby", "bachcha", "bacche", "beta",
             "beti", "school"],
}
OCCASIONS = ["office", "wedding", "interview", "party", "casual", "college", "travel", "running", "gym",
             "walking", "sports", "monsoon", "home", "festival", "winter", "school"]
OCC_SYNONYMS = {"wedding": ["shaadi", "shadi", "marriage", "sangeet", "reception"],
                "festival": ["diwali", "navratri", "garba", "eid", "puja", "festive"],
                "monsoon": ["rain", "rainy", "baarish", "barish", "waterproof"],
                "office": ["work", "corporate", "naukri"], "running": ["marathon", "jog"]}
POLICY_WORDS = ["return", "exchange", "refund", "warranty", "guarantee", "defect", "emi", "upi", "payment",
                "loyalty", "points", "care", "clean", "polish", "wash", "size guide", "size chart",
                "convert", "us size", "eu size", "uk size", "delivery", "reserve", "gst", "invoice", "bill",
                "wapas", "badal"]
STORE_WORDS = ["store", "shop", "dukaan", "dukan", "branch", "timing", "timings", "open", "close", "address",
               "where", "kahan", "location", "phone", "contact", "hours"]

SYSTEM = f"""You convert a footwear-store customer's message (English or Hinglish) into JSON.
Return ONLY JSON with these keys (use null when not mentioned):
intent: one of {INTENTS}
category: one of {CATEGORIES} or null
gender: one of {GENDERS} or null
color: lowercase colour word or null
size: integer Indian/UK size or null (convert US/EU sizes to UK if stated)
max_price: integer INR or null
min_price: integer INR or null
occasion: one of {OCCASIONS} or null
brand: string or null
store: one of ["S1","S2","S3"] or null  (S1=Main Bazaar, S2=Station Road, S3=City Mall)
sku: like "SKU-0012" or null
keywords: short English search phrase capturing style/features (e.g. "comfortable lightweight breathable")
language: "hinglish" or "english"
Hints: "chappal" = slippers; "joote"/"jute" = shoes; "number 9"/"no. 9" = size 9; "ke andar"/"se kam" = under;
"2k" = 2000; "shaadi" = wedding; "mojari"/"jutti"/"kolhapuri" = ethnic (NOT slippers); "floaters" = sandals. Policy questions (returns, exchange, warranty, size chart, payment, care) -> intent "policy".
Store timing/address questions -> "store_info"."""


def _find(text, synonyms: dict):
    for key, words in synonyms.items():
        for w in words:
            if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", text):
                return key
    return None


def _num(s: str) -> int:
    s = s.lower().replace(",", "").strip()
    mult = 1000 if s.endswith("k") else 1
    return int(float(s.rstrip("k")) * mult)


def rule_parse(q: str) -> dict:
    t = " " + q.lower() + " "
    out = dict(intent="product_search", category=None, gender=None, color=None, size=None,
               max_price=None, min_price=None, occasion=None, brand=None, store=None, sku=None,
               keywords=q.strip(), language="english")

    if any(w in t for w in [" hai ", " chahiye", " kya ", " mein ", " ke ", " ka ", " ki ", " joote", "dikhao",
                            " koi ", " wala", " wale", " kitne", " kitna", " hain"]):
        out["language"] = "hinglish"

    sku = re.search(r"sku[-\s]?(\d{1,4})", t)
    if sku:
        out["sku"] = f"SKU-{int(sku.group(1)):04d}"

    size = re.search(r"(?:size|sz|number|no\.?|uk)\s*[:#]?\s*(\d{1,2})(?!\d)", t) or \
        re.search(r"(?<!\d)(\d{1,2})\s*(?:size|number|no\b|uk)", t)
    if size:
        out["size"] = int(size.group(1))
    us = re.search(r"us\s*(?:size)?\s*(\d{1,2})", t)
    if us:
        out["size"] = int(us.group(1)) - 1  # US men ~ UK + 1
    eu = re.search(r"eu\s*(?:size)?\s*(\d{2})", t)
    if eu:
        out["size"] = int(eu.group(1)) - 34

    rng = re.search(r"(\d[\d,]*k?)\s*(?:-|to|se)\s*(\d[\d,]*k?)", t)
    under = re.search(r"(?:under|below|less than|upto|up to|within|max|budget|<|₹|rs\.?|inr)\s*(\d[\d,]*k?)", t) \
        or re.search(r"(\d[\d,]*k?)\s*(?:ke andar|ke under|se kam|tak|budget|rs|rupees|inr)", t)
    above = re.search(r"(?:above|over|more than|min)\s*(\d[\d,]*k?)", t)
    if rng and _num(rng.group(2)) > 100:
        out["min_price"], out["max_price"] = _num(rng.group(1)), _num(rng.group(2))
    elif under and _num(under.group(1)) >= 100:
        out["max_price"] = _num(under.group(1))
    if above and _num(above.group(1)) >= 100:
        out["min_price"] = _num(above.group(1))

    out["category"] = _find(t, CAT_SYNONYMS)
    out["color"] = _find(t, COLOR_SYNONYMS)
    out["gender"] = _find(t, GENDER_SYNONYMS)
    out["occasion"] = _find(t, {o: [o] + OCC_SYNONYMS.get(o, []) for o in OCCASIONS})
    if out["occasion"] == out["category"]:
        out["occasion"] = None
    if "kids" == out["gender"] and out["category"] is None and "school" in t:
        out["category"] = "school"

    for sid, words in {"S1": ["main bazaar", "bazaar"], "S2": ["station"], "S3": ["mall"]}.items():
        if any(w in t for w in words):
            out["store"] = sid

    if re.search(r"(us|eu|uk)\s*(size)?\s*\d+.*(in|to|for)?\s*(uk|us|eu|indian)|size (chart|guide|conversion)|convert", t) \
            and re.search(r"what|which|kya|convert|equal|=|mean", t):
        out["intent"] = "policy"
        return out
    has_product_signal = any(out[k] for k in ("category", "color", "size", "max_price", "sku"))
    if any(w in t for w in POLICY_WORDS) and not (out["category"] and not out["sku"] and "return" not in t
                                                     and "exchange" not in t and "warranty" not in t):
        out["intent"] = "policy"
    elif any(f" {w} " in t or f" {w}?" in t for w in STORE_WORDS) and not has_product_signal:
        out["intent"] = "store_info"
    elif not has_product_signal and not out["gender"] and not out["occasion"] and \
            re.fullmatch(r"\s*(hi|hello|hey|namaste|thanks|thank you|ok|okay|bye)[!. ]*\s*", q.lower()):
        out["intent"] = "chitchat"
    return out


def _clean(llm_out: dict, base: dict) -> dict:
    """Validate LLM output against vocabularies; fall back to rule values on anything invalid."""
    out = dict(base)
    if not isinstance(llm_out, dict):
        return out
    checks = {"intent": INTENTS, "category": CATEGORIES, "gender": GENDERS,
              "occasion": OCCASIONS, "store": ["S1", "S2", "S3"]}
    for k, allowed in checks.items():
        v = llm_out.get(k)
        if isinstance(v, str) and v.lower() in [a.lower() for a in allowed]:
            out[k] = next(a for a in allowed if a.lower() == v.lower())
    for k in ("size", "max_price", "min_price"):
        v = llm_out.get(k)
        try:
            if v is not None and str(v).strip() not in ("", "null"):
                out[k] = int(float(v))
        except (TypeError, ValueError):
            pass
    for k in ("color", "brand", "keywords", "language"):
        v = llm_out.get(k)
        if isinstance(v, str) and v.strip() and v.lower() != "null":
            out[k] = v.strip().lower() if k in ("color", "language") else v.strip()
    sku = llm_out.get("sku")
    if isinstance(sku, str) and re.fullmatch(r"SKU-\d{4}", sku.upper()):
        out["sku"] = sku.upper()
    if out["size"] is not None and not (1 <= out["size"] <= 13):
        out["size"] = base["size"]
    # Exact lexicon hits (e.g. "mojari" -> ethnic) are high-precision: they beat the LLM's guess.
    # The LLM fills the gaps the lexicon can't (paraphrases, implied gender, follow-ups).
    for k in ("category", "color", "gender", "size", "max_price"):
        if base.get(k) is not None:
            out[k] = base[k]
    return out


def parse(q: str, history=None) -> tuple[dict, str, int]:
    """Returns (filters, source, latency_ms). history = last few (role, text) turns for follow-ups."""
    base = rule_parse(q)
    ctx = ""
    if history:
        ctx = "Previous conversation (for follow-ups like 'size 8 mein?' or 'cheaper ones'):\n" + \
              "\n".join(f"{r}: {t[:200]}" for r, t in history[-4:]) + "\n\n"
    data, ms = llm.chat_json([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": ctx + "Customer message: " + q}])
    if data is None:
        return base, "rules", 0
    return _clean(data, base), "llm", ms


if __name__ == "__main__":
    for q in ["black formal shoes size 9 under 3000", "shaadi ke liye mojari chahiye 2k ke andar",
              "can I return sale items?", "Station road store timing?", "running shoes for women no. 5",
              "kids school shoes white", "hi"]:
        print(q, "->", json.dumps({k: v for k, v in rule_parse(q).items() if v and k != "keywords"}))
