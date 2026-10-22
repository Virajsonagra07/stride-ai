"""Generate the SAMPLE dataset for the demo.

All products, brands, stores and stock figures are synthetic. Swap these CSVs
for a real POS export (same columns) to run the assistant on live data.

Run:  python data/generate_data.py
"""
import random
from pathlib import Path

import pandas as pd

random.seed(42)
OUT = Path(__file__).parent

BRANDS = ["Stride", "UrbanWalk", "Kraft & Co", "Pavo", "Trekker", "Bliss", "Nimbus"]
STORES = [
    {"store_id": "S1", "store_name": "Main Bazaar Store", "area": "Main Bazaar",
     "hours": "10:00 AM - 9:30 PM (all days)", "phone": "+91-00000-00001",
     "landmark": "Opposite Clock Tower"},
    {"store_id": "S2", "store_name": "Station Road Store", "area": "Station Road",
     "hours": "10:30 AM - 9:00 PM (closed Tuesday)", "phone": "+91-00000-00002",
     "landmark": "Next to Railway Station gate 2"},
    {"store_id": "S3", "store_name": "City Mall Store", "area": "City Mall, 1st floor",
     "hours": "11:00 AM - 10:00 PM (all days)", "phone": "+91-00000-00003",
     "landmark": "1st floor, near food court"},
]

# category -> (genders, price range, materials, colors, occasions, name stems)
CATS = {
    "formal": (["men", "women"], (1299, 4499), ["genuine leather", "synthetic leather", "patent leather"],
               ["black", "brown", "tan", "burgundy"], ["office", "wedding", "interview", "party"],
               ["Oxford", "Derby", "Brogue", "Monk Strap", "Pump"]),
    "loafers": (["men", "women"], (999, 3499), ["suede", "genuine leather", "synthetic leather"],
                ["brown", "tan", "navy", "black", "beige"], ["casual", "office", "party"],
                ["Penny Loafer", "Tassel Loafer", "Driving Loafer"]),
    "casual": (["men", "women", "unisex"], (799, 2999), ["canvas", "mesh", "synthetic leather", "suede"],
               ["white", "black", "grey", "navy", "olive", "beige"], ["casual", "college", "travel"],
               ["Sneaker", "Slip-On", "Canvas Low", "Street Runner"]),
    "sports": (["men", "women", "unisex"], (1199, 4999), ["mesh", "knit", "synthetic"],
               ["black", "grey", "blue", "white", "red", "neon green"], ["running", "gym", "walking", "sports"],
               ["Running Shoe", "Trainer", "Walking Shoe", "Court Shoe"]),
    "sandals": (["men", "women", "kids"], (399, 1999), ["synthetic", "genuine leather", "rubber"],
                ["brown", "black", "tan", "pink", "gold"], ["casual", "travel", "monsoon"],
                ["Floater", "Strap Sandal", "Gladiator", "Comfort Sandal"]),
    "slippers": (["men", "women", "kids", "unisex"], (199, 899), ["rubber", "EVA", "synthetic"],
                 ["black", "blue", "grey", "pink", "white"], ["home", "monsoon", "casual"],
                 ["Flip Flop", "Slide", "Home Slipper"]),
    "ethnic": (["men", "women"], (699, 2999), ["embroidered fabric", "genuine leather", "velvet"],
               ["gold", "maroon", "beige", "red", "silver"], ["wedding", "festival", "party"],
               ["Mojari", "Jutti", "Kolhapuri", "Kolhapuri Chappal"]),
    "boots": (["men", "women"], (1999, 5499), ["genuine leather", "suede", "synthetic leather"],
              ["brown", "black", "tan"], ["winter", "travel", "party"],
              ["Chelsea Boot", "Chukka Boot", "Ankle Boot"]),
    "heels": (["women"], (899, 3499), ["synthetic leather", "patent leather", "satin"],
              ["black", "nude", "red", "gold", "silver"], ["party", "wedding", "office"],
              ["Block Heel", "Stiletto", "Wedge", "Kitten Heel"]),
    "school": (["kids"], (499, 1299), ["synthetic leather", "canvas"],
               ["black", "white"], ["school"], ["School Shoe", "PT Shoe"]),
}
FEATURES = {
    "formal": ["cushioned insole", "anti-slip sole", "hand-stitched welt", "breathable lining"],
    "loafers": ["memory-foam footbed", "flexible sole", "padded collar"],
    "casual": ["lightweight", "memory-foam insole", "washable", "padded collar"],
    "sports": ["shock-absorbing midsole", "breathable upper", "high-grip outsole", "lightweight"],
    "sandals": ["adjustable velcro strap", "water-friendly", "cushioned footbed"],
    "slippers": ["waterproof", "soft EVA", "anti-skid"],
    "ethnic": ["handcrafted", "cushioned footbed", "zari embroidery"],
    "boots": ["side zip", "water-resistant", "lug sole"],
    "heels": ["padded insole", "ankle strap", "stable block base"],
    "school": ["scuff-resistant toe", "velcro closure", "non-marking sole"],
}
SIZES = {"men": list(range(6, 12)), "women": list(range(3, 9)),
         "kids": list(range(8, 14)) + [1, 2, 3], "unisex": list(range(4, 11))}

rows, inv = [], []
i = 0
for cat, (genders, (lo, hi), mats, colors, occs, stems) in CATS.items():
    n = 14 if cat in ("formal", "casual", "sports", "sandals") else 10
    for _ in range(n):
        i += 1
        g = random.choice(genders)
        stem = random.choice(stems)
        if g == "men" and stem in ("Pump",):
            stem = "Derby"
        color = random.choice(colors)
        mat = random.choice(mats)
        brand = random.choice(BRANDS)
        price = int(round(random.randint(lo, hi) / 100.0) * 100 - 1)
        occ = random.sample(occs, k=min(len(occs), random.randint(1, 3)))
        feats = random.sample(FEATURES[cat], k=2)
        sku = f"SKU-{i:04d}"
        name = f"{brand} {color.title()} {stem}"
        label = {"men": "Men's", "women": "Women's", "kids": "Kids'", "unisex": "Unisex"}[g]
        desc = (f"{label} {color} {mat} {stem.lower()} by {brand}. "
                f"Great for {', '.join(occ)}. Features {feats[0]} and {feats[1]}.")
        rows.append(dict(sku=sku, name=name, brand=brand, category=cat, gender=g, color=color,
                         material=mat, price_inr=price, occasions="|".join(occ),
                         features="|".join(feats), rating=round(random.uniform(3.6, 4.9), 1),
                         on_sale=random.random() < 0.15, description=desc))
        for s in STORES:
            if random.random() < 0.2:  # some stores don't carry the article
                continue
            for size in SIZES[g]:
                # mid sizes sell fastest -> deliberately thin stock on 8/9 (men) & 5/6 (women)
                hot = (g == "men" and size in (8, 9)) or (g == "women" and size in (5, 6))
                qty = max(0, int(random.gauss(1.2 if hot else 3, 1.5)))
                inv.append(dict(sku=sku, store_id=s["store_id"], size=size, qty=qty))

pd.DataFrame(rows).to_csv(OUT / "catalog.csv", index=False)
pd.DataFrame(inv).to_csv(OUT / "inventory.csv", index=False)
pd.DataFrame(STORES).to_csv(OUT / "stores.csv", index=False)
print(f"catalog: {len(rows)} products | inventory rows: {len(inv)} | stores: {len(STORES)}")
