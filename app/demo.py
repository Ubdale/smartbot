"""A built-in sample dataset so new users can ask questions immediately."""

import csv
import io
import random
from datetime import date, timedelta

DEMO_NAME = "Sample: Online Store Orders"
DEMO_FILENAME = "sample-online-store.csv"

_REGIONS = ["North America", "Europe", "Asia", "Middle East"]
_PRODUCTS = {
    "Team Plan": "Subscriptions",
    "Pro Plan": "Subscriptions",
    "Starter Plan": "Subscriptions",
    "Extra Storage": "Add-ons",
    "Priority Support": "Add-ons",
    "Onboarding": "Services",
}
_CUSTOMERS = [
    "Apex Labs",
    "Granite Group",
    "Blue Harbor",
    "Nimbus Co",
    "Vertex Retail",
    "Oak & Pine",
    "Summit Foods",
    "Delta Works",
    "Crescent Bank",
    "Harbor Media",
]
_STATUSES = ["Paid", "Paid", "Paid", "Pending", "Refunded"]


def demo_csv() -> bytes:
    """Deterministic, realistic sample orders across a year (so trends and groups work)."""
    rng = random.Random(42)  # noqa: S311 (demo data, not security)
    start = date(2025, 1, 1)
    rows = []
    for i in range(120):
        product = rng.choice(list(_PRODUCTS))
        units = rng.randint(1, 40)
        unit_price = {"Subscriptions": 99.0, "Add-ons": 19.0, "Services": 250.0}[_PRODUCTS[product]]
        discount = rng.choice([0.0, 0.0, 0.1, 0.15])
        revenue = round(units * unit_price * (1 - discount), 2)
        rows.append(
            {
                "order_id": f"SO-{1000 + i}",
                "order_date": (start + timedelta(days=rng.randint(0, 364))).isoformat(),
                "customer": rng.choice(_CUSTOMERS),
                "region": rng.choice(_REGIONS),
                "product": product,
                "category": _PRODUCTS[product],
                "units": units,
                "revenue": revenue,
                "status": rng.choice(_STATUSES),
            }
        )
    rows.sort(key=lambda r: str(r["order_date"]))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")
