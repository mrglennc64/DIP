"""Adapter: Polymarket — crypto up/down windows as a prediction source.

Upstream: https://gamma-api.polymarket.com  (public, no auth, no headers)

A market price IS a prediction: a crowd forecast, with a probability, about a
question that resolves. So Polymarket enters DIP as a source like any other and
is judged the same way — by its own graded history through
calibration/evidence.py — rather than being treated as a yardstick that other
sources are measured against. That distinction matters: a "market baseline"
subtracted from a model is not evidence about either one, whereas two sources
side by side in the ledger, each with its own calibration curve, is.

Polymarket resolves its own markets, so grading arrives WITH the source
(`to_results`), the same way Fantasy ships graded rows. DIP never reaches for
Chainlink or a spot feed to second-guess it; distrust of a source's grading is
expressed as a reliability score, not as a second oracle.

Scope: the machine-generated short-interval family only —
    {asset}-updown-{interval}-{unix_ts}      e.g. btc-updown-5m-1784562900
whose timestamp is the window START, UTC epoch, floor-aligned to the interval.
The human-readable daily family (`bitcoin-up-or-down-july-22-2026-12pm-et`) uses
Eastern Time boundaries and is DST-sensitive; it is rejected by name rather than
parsed on a guess.
"""
from __future__ import annotations

import datetime
import json
import re
import urllib.parse
import urllib.request

from ..contract import CONF_PROBABILITY, Prediction, norm_entity

SOURCE = "polymarket"
GAMMA = "https://gamma-api.polymarket.com"
DOMAIN = "crypto"

# Enumerating live windows. `closed=false` ALONE IS NOT ENOUGH — the flag goes
# stale and the endpoint will hand back months-old markets that were never
# flipped. `end_date_min` set to now is what actually bounds the result to the
# current wave, so it is not optional here.
TAG = "up-or-down"

# {asset}-updown-{interval}-{window_start_unix}
_SLUG = re.compile(r"^([a-z0-9]+)-updown-(\d+[mh])-(\d+)$")

_INTERVAL_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}

# outcomePrices always sums to exactly 1.0000 because it is a NORMALIZED
# synthetic mid, not two independent quotes. There is nothing to de-vig — and
# that is precisely the trap, because the normalization hides the book beneath
# it. Observed live: a market quoting [0.251, 0.749] whose real book was
# bid 0.002 / ask 0.500. A "25.1% probability" you could only buy at 0.50 is not
# a 25.1% probability. So the spread is carried alongside and gates ingestion.
MAX_SPREAD = 0.10

# What a complete row carries. Measured against a FIXED expectation rather than
# however many keys happened to arrive, so a degraded upstream that stops
# promising half its fields cannot score as complete simply by promising less.
_EXPECTED = ("outcomePrices", "bestBid", "bestAsk", "spread", "liquidityNum",
             "volumeNum", "eventStartTime", "conditionId")


def _get(url: str, timeout: float = 30.0) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "DIP/1 (+ingestion)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _jloads(v, default):
    """outcomes / outcomePrices / clobTokenIds arrive as JSON-ENCODED STRINGS,
    not arrays. Indexing one without decoding it silently yields characters."""
    if isinstance(v, list):
        return v
    if not v:
        return default
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


def fetch_open(now: str | None = None, limit: int = 200,
               base: str = GAMMA, fetch=_get) -> list[dict]:
    """Every currently-open up/down event, soonest-ending first."""
    now = now or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    q = urllib.parse.urlencode({
        "tag_slug": TAG, "closed": "false", "end_date_min": now,
        "limit": limit, "order": "endDate", "ascending": "true"})
    out = fetch(f"{base}/events?{q}")
    return out if isinstance(out, list) else []


def fetch_by_slug(slug: str, base: str = GAMMA, fetch=_get) -> dict | None:
    """One market by slug, in whatever state it is in.

    Deliberately /events, not /markets. NEITHER /markets form is state-agnostic:
    the bare query returns only OPEN markets, and `closed=true` returns only
    SETTLED ones — so both answer "empty list" for half the lifecycle, which is
    indistinguishable from "no such market". A grader built on either one cannot
    tell a window still in flight from one it should have graded and lost, and
    would report clean runs while the ledger silently filled with predictions
    that never resolve. /events?slug= returns the market in both states, so the
    lifecycle flags on the row are what decide, not the shape of the response.
    """
    out = fetch(f"{base}/events?slug={urllib.parse.quote(slug)}")
    if not (isinstance(out, list) and out):
        return None
    markets = out[0].get("markets") or []
    return markets[0] if markets else None


def parse_slug(slug: str) -> tuple[str, str, int] | None:
    """slug -> (asset, interval, window_start_unix), or None if not our family."""
    m = _SLUG.match(slug or "")
    if not m:
        return None
    asset, interval, ts = m.group(1), m.group(2), int(m.group(3))
    step = _INTERVAL_SECONDS.get(interval)
    # A window start that is not floor-aligned means the slug convention moved
    # under us. Refuse rather than derive a window boundary that is off by
    # seconds — every outcome near the open would grade against the wrong bar.
    if step is None or ts % step != 0:
        return None
    return asset, interval, ts


def _iso(ts: int) -> str:
    return datetime.datetime.fromtimestamp(
        ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _up_index(outcomes: list) -> int | None:
    for i, o in enumerate(outcomes):
        if str(o).strip().lower() == "up":
            return i
    return None


def _markets(events: list[dict]):
    """Flatten events -> markets, since one event wraps one window."""
    for ev in events or []:
        for mk in ev.get("markets") or []:
            yield mk


def to_predictions(events: list[dict]) -> tuple[list, list]:
    """Map open up/down markets to contract rows. Returns (predictions, rejected).

    Rejections are RETURNED, not swallowed. A row vanishing quietly between an
    upstream and the ledger is the ingestion bug that never announces itself.
    """
    preds, rejected = [], []

    for mk in _markets(events):
        slug = mk.get("slug") or ""
        parsed = parse_slug(slug)
        if parsed is None:
            rejected.append({"slug": slug, "why": "not the machine-generated "
                             "{asset}-updown-{interval}-{unix} family"})
            continue
        asset, interval, ts = parsed

        outcomes = _jloads(mk.get("outcomes"), [])
        up = _up_index(outcomes)
        if up is None:
            rejected.append({"slug": slug,
                             "why": f"no 'Up' outcome in {outcomes!r}"})
            continue

        # A brand-new market has NO outcomePrices key at all, with bestBid=0 /
        # bestAsk=1 / spread=1. That is "no signal" — emphatically not 50%.
        # Defaulting it to even money would manufacture a confident-looking
        # coin flip out of an empty book, on every window, forever.
        prices = _jloads(mk.get("outcomePrices"), [])
        if len(prices) != len(outcomes):
            rejected.append({"slug": slug, "why": "no outcomePrices yet — "
                             "empty book, not an even-money market"})
            continue

        spread = mk.get("spread")
        if spread is not None and float(spread) > MAX_SPREAD:
            rejected.append({"slug": slug, "why": f"spread {float(spread):.3f} "
                             f"> {MAX_SPREAD} — normalized mid hides the book"})
            continue

        try:
            p_up = float(prices[up])
        except (TypeError, ValueError):
            rejected.append({"slug": slug, "why": f"unparseable price {prices!r}"})
            continue

        present = sum(1 for k in _EXPECTED if mk.get(k) not in (None, ""))

        p = Prediction(
            source=SOURCE,
            # Polymarket exposes no model build. Recording the empty string is
            # honest; inventing one would let a scoring change be attributed to
            # a version that never existed.
            source_version="",
            domain=DOMAIN,
            entity=norm_entity(asset),
            entity_display=asset.upper(),
            # Every asset trading the SAME window shares an event_key, so the
            # correlation engine sees simultaneous BTC/ETH/SOL exposure as
            # same-event rather than independent. That is the BTC/ETH guard —
            # structural, not a bolted-on rule.
            event_key=f"poly:{interval}:{ts}",
            market=f"updown_{interval}",
            # The window START, not a calendar day. `line` and `event_date` are
            # both inside the identity hash and a 5-minute market has no line
            # to vary, so the date is the only slot that can distinguish 288
            # windows a day — a calendar date would collapse all of them onto
            # ONE prediction id. Cost: date filters must match by prefix
            # (event_date LIKE '2026-07-20%'), which is why grading below
            # joins on the exact string it wrote.
            event_date=_iso(ts),
            # "Will close exceed open" — the threshold is the window's own open,
            # which the API does not publish. 0.0 is the truthful encoding of
            # "change vs open", and it varies nothing, hence the note above.
            line=0.0,
            predicted_value=p_up,
            prob_over=p_up,            # over == Up
            prob_under=1.0 - p_up,
            # For a binary outcome the variance IS p(1-p) — not an estimate, an
            # identity. This is a real input to Module 2's 15% variance weight
            # and the "Variance Low" clause, rather than a missing field.
            variance=p_up * (1.0 - p_up),
            side="over" if p_up >= 0.5 else "under",
            confidence=max(p_up, 1.0 - p_up),
            confidence_kind=CONF_PROBABILITY,
            feature_completeness=present / len(_EXPECTED),
            raw=mk,
        )
        problems = p.validate()
        if problems:
            rejected.append({"slug": slug, "why": "; ".join(problems)})
            continue
        preds.append(p)

    return preds, rejected


def to_results(markets: list[dict]) -> list[dict]:
    """Outcomes for markets Polymarket has settled.

    The gate is `closed AND umaResolutionStatus == "resolved"`, because there is
    a lag between endDate passing and the flags flipping, and both fields are
    ABSENT (not false) while a market is live. Notably `active` stays true after
    resolution and is useless as a liveness signal.

    Graded off outcomePrices only. lastTradePrice settles at 0.99/0.999 and
    would grade a won market as a near-miss.
    """
    out = []
    for mk in markets or []:
        parsed = parse_slug(mk.get("slug") or "")
        if parsed is None:
            continue
        if not (mk.get("closed") and mk.get("umaResolutionStatus") == "resolved"):
            continue

        outcomes = _jloads(mk.get("outcomes"), [])
        prices = _jloads(mk.get("outcomePrices"), [])
        up = _up_index(outcomes)
        if up is None or len(prices) != len(outcomes):
            continue
        try:
            vals = [float(x) for x in prices]
        except (TypeError, ValueError):
            continue

        # An exact tie resolves 50-50 by the market's own rules. That is a PUSH,
        # not a loss: there was no side of the line to have been on. Folding it
        # into a loss would bias every accuracy rate DIP derives.
        if max(vals) < 0.9:
            result, actual = "X", None
        else:
            up_won = vals[up] == max(vals)
            result, actual = ("1" if up_won else "0"), (1.0 if up_won else 0.0)

        asset, interval, ts = parsed
        out.append({
            "domain": DOMAIN,
            "entity": norm_entity(asset),
            "market": f"updown_{interval}",
            "event_date": _iso(ts),
            "line": 0.0,
            "result": result,
            "actual": actual,
        })
    return out


def upstream_stamp(events: list[dict]) -> str:
    """Freshness marker — how the stale-source alert notices an upstream that
    keeps answering 200 while its content stops moving."""
    mks = list(_markets(events))
    ends = sorted({m.get("endDate") for m in mks if m.get("endDate")})
    return (f"events={len(events or [])} markets={len(mks)} "
            f"next_end={ends[0] if ends else 'none'}")
