#!/usr/bin/env python3
"""Generates transactions_1m.csv with 1 million rows.

Most rows have a blank category (unlabeled) — the active-learning workflow.
A small number of seed rows carry a label to bootstrap the model.
Run:  python3 generate_transactions.py
"""

import random
import sys

ROWS       = 1_000_000
SEEDS_PER_CATEGORY = 5   # labeled rows per category to seed the model
OUT        = "transactions_1m.csv"

# ── vocabulary ────────────────────────────────────────────────────────────────

INSURANCE_COMPANIES = [
    "AIG-Life", "Allianz", "Allstate", "AXA", "Barmenia", "Ergo",
    "Geico-Auto", "Generali", "Gothaer", "HanseMerkur", "HUK-Coburg",
    "LVM", "NÜRNBERGER", "R+V", "Signal-Iduna", "TK", "Zurich",
]
INSURANCE_TYPES = [
    "insurance premium", "insurance annual", "policy payment",
    "liability insurance", "accident insurance", "health insurance",
]
LANDLORD_NAMES = [
    "Immobilia", "WohnBau", "CityFlats", "PrimeRent",
    "HausverwaltungMüller", "NordRent", "SüdWohn",
]
RENT_DESCRIPTIONS = [
    "monthly rent", "rent apartment", "rent payment",
    "flat rental", "apartment rental", "housing rent",
]
EMPLOYERS = [
    "FinTech GmbH", "Digital AG", "InnoSoft KG", "DataCorp", "TechWorks",
    "AlphaBank", "BetaSystems", "GammaLogistics", "DeltaMedia", "EpsilonRetail",
    "Bundesanstalt", "Stadtwerke",
]
SALARY_WORDS = [
    "salary", "wage", "payroll", "monthly salary", "net salary", "Gehalt", "Lohn",
]
SUPERMARKETS = [
    "Aldi", "Edeka", "Lidl", "Netto", "Penny", "Rewe", "Kaufland", "Norma",
]
GROCERY_WORDS = [
    "supermarket", "food", "groceries", "Lebensmittel", "weekly shop", "provisions",
]
UTILITIES = [
    "Stadtwerke München", "Vattenfall", "E.ON", "EnBW",
    "RheinEnergie", "Telekom", "Vodafone", "O2", "1&1",
]
UTILITY_TYPES = [
    "electricity bill", "gas bill", "water bill",
    "internet bill", "broadband", "energy payment", "utility",
]
TRANSPORT_CO = [
    "Deutsche Bahn", "BVG", "MVG", "HVV", "VGN",
    "Flixbus", "Shell", "Aral", "BP", "Esso", "Sixt",
]
TRANSPORT_WORDS = [
    "train ticket", "monthly pass", "fuel", "parking",
    "car rental", "bus ticket", "transport", "Tankstelle",
]
STREAMING_SVC = [
    "Netflix", "Spotify", "Amazon Prime", "Disney+",
    "Apple TV+", "YouTube Premium", "Deezer", "Sky",
]
SUB_WORDS = [
    "monthly subscription", "subscription", "streaming", "premium plan", "membership fee",
]
PHARMACIES = [
    "dm Drogerie", "Rossmann", "Apotheke am Markt",
    "DocMorris", "Stadtapotheke", "Rathaus-Apotheke",
]
MEDICAL_WORDS = [
    "pharmacy", "Apotheke", "prescription", "doctor visit",
    "hospital payment", "medical", "health payment",
]

# ── row generators ────────────────────────────────────────────────────────────

def insurance_row(r):
    return f"{r.choice(INSURANCE_COMPANIES)} {r.randint(10, 300)}.00 {r.choice(INSURANCE_TYPES)}", "insurance"

def rent_row(r):
    return (f"{r.choice(RENT_DESCRIPTIONS)} {r.choice(LANDLORD_NAMES)} "
            f"{r.randint(400, 2500)}.00 Ref{r.randint(1000, 9999)}"), "rent"

def salary_row(r):
    return f"{r.choice(SALARY_WORDS)} {r.choice(EMPLOYERS)} {r.randint(1, 12):02d}/2024", "salary"

def groceries_row(r):
    return (f"{r.choice(SUPERMARKETS)} {r.choice(GROCERY_WORDS)} "
            f"{r.randint(5, 200)}.{r.randint(0, 99):02d}"), "groceries"

def utilities_row(r):
    return (f"{r.choice(UTILITIES)} {r.choice(UTILITY_TYPES)} "
            f"ACC{r.randint(10000, 99999)}"), "utilities"

def transport_row(r):
    return (f"{r.choice(TRANSPORT_CO)} {r.choice(TRANSPORT_WORDS)} "
            f"{r.randint(2, 150)}.{r.randint(0, 99):02d}"), "transport"

def subscription_row(r):
    return f"{r.choice(STREAMING_SVC)} {r.choice(SUB_WORDS)}", "subscription"

def medical_row(r):
    return (f"{r.choice(PHARMACIES)} {r.choice(MEDICAL_WORDS)} "
            f"{r.randint(3, 500)}.{r.randint(0, 99):02d}"), "medical"

GENERATORS = [
    (insurance_row,    0.12),
    (rent_row,         0.10),
    (salary_row,       0.10),
    (groceries_row,    0.20),
    (utilities_row,    0.12),
    (transport_row,    0.15),
    (subscription_row, 0.11),
    (medical_row,      0.10),
]

_cumulative = []
_total = 0.0
for _fn, _w in GENERATORS:
    _total += _w
    _cumulative.append((_total, _fn))

def pick_generator(r):
    x = r.random()
    for threshold, fn in _cumulative:
        if x < threshold:
            return fn
    return _cumulative[-1][1]

def iban(r):
    return f"DE{r.randint(10, 99)}{r.randint(10_000_000_000_000_000, 99_999_999_999_999_999_999)}"

def csv_field(s):
    return f'"{s}"' if "," in s else s

# ── seed indices: first SEEDS_PER_CATEGORY rows of each category are labeled ─

def main():
    r = random.Random(42)

    # Pre-generate all rows so we can assign seeds deterministically.
    # For 1M rows this is ~300 MB in memory briefly; use chunked streaming if that matters.
    seeds_remaining = {fn.__name__: SEEDS_PER_CATEGORY for fn, _ in GENERATORS}
    chunk = 50_000

    with open(OUT, "w", buffering=1 << 20) as f:
        f.write("purpose,iban,category\n")
        buf = []
        written = 0
        for _ in range(ROWS):
            gen = pick_generator(r)
            purpose, category = gen(r)
            key = gen.__name__
            # label only the first SEEDS_PER_CATEGORY rows of each category
            label = ""
            if seeds_remaining.get(key, 0) > 0:
                label = category
                seeds_remaining[key] -= 1
            buf.append(f"{csv_field(purpose)},{iban(r)},{label}\n")
            if len(buf) >= chunk:
                f.writelines(buf)
                written += len(buf)
                buf.clear()
                print(f"\r  {written:>9,} / {ROWS:,}", end="", file=sys.stderr)
        if buf:
            f.writelines(buf)
            written += len(buf)

    total_seeds = SEEDS_PER_CATEGORY * len(GENERATORS)
    print(f"\r  {written:,} rows → {OUT}  ({total_seeds} labeled seeds, rest unlabeled)",
          file=sys.stderr)

if __name__ == "__main__":
    main()
