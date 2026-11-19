"""Charts for exports (PDF, Word, PNG), drawn by code from the answer's chart spec.

Uses matplotlib's object API (no pyplot), so it is safe to call from worker threads.
"""

import io
from typing import Any

from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

PALETTE = ["#4F46E5", "#0D9488", "#EA580C", "#DB2777", "#CA8A04", "#2563EB", "#16A34A", "#9333EA"]
MAX_CATEGORIES = 30
MAX_SERIES = 8
MAX_SLICES = 10


def chart_series(
    chart: dict[str, Any] | None, columns: list[str], rows: list[list[Any]]
) -> tuple[list[str], dict[str, list[float | None]]] | None:
    """Categories (x) and one list of values per series, or None if there's nothing to draw."""
    if not chart or not rows or chart.get("x") not in columns or chart.get("y") not in columns:
        return None
    xi, yi = columns.index(chart["x"]), columns.index(chart["y"])
    si = columns.index(chart["series"]) if chart.get("series") in columns else None
    cats: list[str] = []
    series: dict[str, dict[str, float]] = {}
    for r in rows:
        v = r[yi]
        if not isinstance(v, int | float) or isinstance(v, bool):
            continue
        x = "" if r[xi] is None else str(r[xi])
        name = str(r[si]) if si is not None else str(chart["y"])
        if x not in cats:
            cats.append(x)
        series.setdefault(name, {})[x] = float(v)
    if not cats:
        return None
    cats = cats[:MAX_CATEGORIES]
    return cats, {n: [vals.get(c) for c in cats] for n, vals in list(series.items())[:MAX_SERIES]}


def _number(v: float, _pos: object = None) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.1f}B"
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e4:
        return f"{v / 1e3:.0f}K"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def render_chart_png(
    chart: dict[str, Any] | None,
    columns: list[str],
    rows: list[list[Any]],
    *,
    title: str = "",
    y_label: str = "",
    width: float = 9.0,
    height: float = 4.6,
    dpi: int = 160,
) -> bytes | None:
    """A PNG of the answer's chart (bar, line or pie), or None when there is no chart."""
    data = chart_series(chart, columns, rows)
    if data is None or chart is None:
        return None
    cats, series = data
    kind = chart.get("type", "bar")

    fig = Figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot()
    if title:
        ax.set_title(title[:90], loc="left", fontsize=12, fontweight="bold", color="#18181B")

    if kind == "pie":
        values = [v or 0.0 for v in next(iter(series.values()))]
        pairs = sorted(zip(cats, values, strict=True), key=lambda p: -p[1])
        if len(pairs) > MAX_SLICES:
            rest = sum(v for _, v in pairs[MAX_SLICES - 1 :])
            pairs = [*pairs[: MAX_SLICES - 1], ("Other", rest)]
        pairs = [p for p in pairs if p[1] > 0]
        if not pairs:
            return None
        ax.pie(
            [v for _, v in pairs],
            labels=[c[:25] for c, _ in pairs],
            colors=(PALETTE * 2)[: len(pairs)],
            autopct="%1.0f%%",
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
            textprops={"fontsize": 8, "color": "#3F3F46"},
        )
        ax.axis("equal")
    else:
        positions = list(range(len(cats)))
        names = list(series)
        if kind == "line":
            for k, name in enumerate(names):
                ys = [float("nan") if v is None else v for v in series[name]]
                ax.plot(positions, ys, color=PALETTE[k % len(PALETTE)], linewidth=2,
                        marker="o" if len(cats) <= 24 else None, markersize=3.5,
                        label=name)  # fmt: skip
        else:
            group = 0.8
            bar_w = group / len(names)
            for k, name in enumerate(names):
                xs = [p - group / 2 + bar_w * (k + 0.5) for p in positions]
                ys = [v or 0.0 for v in series[name]]
                ax.bar(xs, ys, width=bar_w * 0.92, color=PALETTE[k % len(PALETTE)], label=name)
        ax.set_xticks(positions)
        long = max(len(c) for c in cats) > 8 or len(cats) > 10
        ax.set_xticklabels([c[:22] for c in cats], rotation=35 if long else 0,
                           ha="right" if long else "center", fontsize=8)  # fmt: skip
        ax.yaxis.set_major_formatter(FuncFormatter(_number))
        ax.tick_params(axis="y", labelsize=8, colors="#52525B")
        ax.tick_params(axis="x", colors="#52525B", length=0)
        ax.grid(axis="y", color="#E4E4E7", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#D4D4D8")
        if y_label:
            ax.set_ylabel(y_label, fontsize=9, color="#52525B")
        if len(names) > 1:
            ax.legend(frameon=False, fontsize=8, ncols=min(4, len(names)))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()
