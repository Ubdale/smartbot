"""Delivery channels for digests and alerts. Email lives in digests.py; this adds Slack
(an incoming webhook — free, no app approval needed)."""

import logging

import httpx

log = logging.getLogger(__name__)


def post_slack(
    webhook_url: str, text: str, *, client: httpx.Client | None = None
) -> tuple[bool, str | None]:
    """Post a message to a Slack incoming webhook. Returns (sent, note)."""
    if not webhook_url.startswith("https://hooks.slack.com/"):
        return False, "That doesn't look like a Slack incoming webhook URL."
    own = client is None
    http = client or httpx.Client(timeout=15)
    try:
        resp = http.post(webhook_url, json={"text": text[:39000]})
        if resp.status_code >= 400:
            return False, f"Slack refused it ({resp.status_code}): {resp.text[:140]}"
        return True, None
    except httpx.HTTPError as exc:
        log.warning("Slack post failed: %s", exc)
        return False, f"Couldn't reach Slack: {exc}"
    finally:
        if own:
            http.close()


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}" if not v.is_integer() else f"{int(v):,}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return "" if v is None else str(v)


def digest_slack_text(title: str, dataset_name: str, content: list[dict[str, object]]) -> str:
    """A compact Slack message (Markdown) for a digest run."""
    lines = [f"*{title}*  _(from “{dataset_name}”)_"]
    for item in content:
        q = str(item.get("question", ""))
        summary = str(item.get("summary", ""))
        lines.append(f"\n*{q}*\n{summary}")
        now, before = item.get("headline"), item.get("previous")
        if isinstance(now, int | float) and isinstance(before, int | float) and before != now:
            diff = now - before
            pct = f" ({diff / before:+.0%})" if before else ""
            arrow = "▲" if diff > 0 else "▼"
            lines.append(f"{arrow} {_fmt(abs(diff))}{pct} since last time")
    return "\n".join(lines)
