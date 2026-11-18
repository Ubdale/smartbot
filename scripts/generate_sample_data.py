"""Generate the synthetic demo datasets in sample_data/.

Deterministic (fixed seed, stdlib only): the golden-question eval depends on
these exact files, so re-running must produce identical output.

    uv run python scripts/generate_sample_data.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "sample_data"
SEED = 42

FIRST = ["Aisha", "Ben", "Carlos", "Dana", "Elif", "Farah", "Gwen", "Hiro", "Imran", "Jade",
         "Kofi", "Lena", "Mateo", "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sami", "Tara",
         "Usman", "Vera", "Wei", "Xena", "Yusuf", "Zara"]  # fmt: skip
LAST = ["Ahmed", "Brown", "Chen", "Diaz", "Evans", "Fischer", "Garcia", "Haddad", "Ito",
        "Jensen", "Khan", "Lopez", "Malik", "Novak", "Okafor", "Patel", "Rossi", "Silva",
        "Tanaka", "Weber"]  # fmt: skip

DEPARTMENTS = {  # name: (weight, salary range)
    "Engineering": (30, (65_000, 140_000)),
    "Sales": (22, (45_000, 110_000)),
    "Operations": (18, (38_000, 80_000)),
    "Customer Support": (14, (32_000, 60_000)),
    "Marketing": (9, (45_000, 95_000)),
    "Finance": (7, (50_000, 115_000)),
}
CITIES = ["London", "Manchester", "Berlin", "Lisbon", "Dubai", "Karachi"]
LEVELS = ["Junior", "Mid", "Senior", "Lead"]

REGIONS = {"North": ["London", "Manchester"], "Europe": ["Berlin", "Lisbon"],
           "Middle East": ["Dubai"], "Asia": ["Karachi"]}  # fmt: skip
PRODUCTS = {  # product: (category, unit price)
    "Starter Plan": ("Subscriptions", 29.0),
    "Team Plan": ("Subscriptions", 99.0),
    "Enterprise Plan": ("Subscriptions", 499.0),
    "Onboarding Package": ("Services", 750.0),
    "Training Day": ("Services", 1200.0),
    "Data Connector": ("Add-ons", 149.0),
    "Extra Storage": ("Add-ons", 19.0),
}
CUSTOMERS = [f"{a} {b}" for a in ("Apex", "Blue", "Cedar", "Delta", "Echo", "Falcon", "Granite",
             "Harbor", "Iris", "Juniper", "Kestrel", "Lumen", "Maple", "Nova", "Orbit")
             for b in ("Ltd", "Group", "Labs")]  # fmt: skip

START, END = date(2024, 1, 1), date(2026, 9, 30)


def _rand_date(rng: random.Random, lo: date, hi: date) -> date:
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


def hr_rows(rng: random.Random, n: int = 400) -> list[dict[str, object]]:
    names, weights = list(DEPARTMENTS), [w for w, _ in DEPARTMENTS.values()]
    rows = []
    for i in range(1, n + 1):
        dept = rng.choices(names, weights)[0]
        lo, hi = DEPARTMENTS[dept][1]
        hired = _rand_date(rng, date(2018, 1, 1), END)
        left = None
        if rng.random() < 0.18:
            left = _rand_date(rng, min(hired + timedelta(days=60), END), END)
        rows.append(
            {
                "employee_id": f"E{i:04d}",
                "full_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                "department": dept,
                "city": rng.choice(CITIES),
                "job_level": rng.choices(LEVELS, [35, 35, 22, 8])[0],
                "hire_date": hired.isoformat(),
                "termination_date": left.isoformat() if left else "",
                "annual_salary": round(rng.uniform(lo, hi), -2),
                "is_remote": rng.random() < 0.3,
            }
        )
    return rows


def sales_rows(rng: random.Random, n: int = 3000) -> list[dict[str, object]]:
    rows = []
    for i in range(1, n + 1):
        region = rng.choice(list(REGIONS))
        product = rng.choice(list(PRODUCTS))
        category, price = PRODUCTS[product]
        qty = rng.randint(1, 3) if category == "Services" else rng.randint(1, 25)
        discount = rng.choice([0, 0, 0, 0.05, 0.1, 0.15])
        rows.append(
            {
                "order_id": f"SO-{i:05d}",
                "order_date": _rand_date(rng, START, END).isoformat(),
                "customer": rng.choice(CUSTOMERS),
                "region": region,
                "city": rng.choice(REGIONS[region]),
                "product": product,
                "category": category,
                "quantity": qty,
                "unit_price": price,
                "discount": discount,
                "revenue": round(qty * price * (1 - discount), 2),
            }
        )
    rows.sort(key=lambda r: (str(r["order_date"]), str(r["order_id"])))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    write_csv(OUT / "hr_employees.csv", hr_rows(rng))
    write_csv(OUT / "sales_orders.csv", sales_rows(rng))
    for p in sorted(OUT.glob("*.csv")):
        print(f"wrote {p.relative_to(OUT.parent)} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
