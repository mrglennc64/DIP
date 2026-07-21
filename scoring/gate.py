"""The green-light gate — the three questions a user actually has.

    1. Can I trust this system yet?      -> a light
    2. Is anything worth buying today?   -> a count, usually 0
    3. For each: what, where, how much?  -> a slip

Everything below the light already existed (quality, calibration, correlation,
recommendation). What was missing was a surface that answers those three
questions in one sentence each, instead of a table the reader has to infer
"do nothing" from by scanning 25 rows of dashes.

Two deliberate departures from the obvious implementation:

ONE LIGHT PER MARKET, NOT ONE GLOBAL LIGHT. A single number ("990 graded")
pooled across every source and market would let one market's history unlock
another's recommendations — tennis evidence greenlighting a crypto trade. That
is the same pooling the contract already refuses for confidence_kind and that
quality.index_history refuses per market, so it is refused here too. The page
still shows one headline light, but it is a ROLL-UP of the per-market lights
(the weakest one wins), never the gate itself. Unlocking is always per market.

BREAKEVEN IS COMPUTED, NOT ASSUMED. 52.4% is the -110 sportsbook number and is
wrong everywhere else. On a binary contract bought at price p, paying p to win
1 with fee f on winnings, the hit rate that breaks even is

    p / (p + (1 - p)(1 - f))

which at p=0.50, f=0 is 50%, and at p=0.505 with Polymarket's 7% taker fee is
52.3%. Using a flat 52.4% would flatter cheap markets and punish expensive ones,
and the whole point of the light is that it does not flatter anything.
"""
from __future__ import annotations

from calibration.evidence import wilson

# Below this many graded outcomes nothing is trusted, however good the rate
# looks. A 70% hit rate on 20 picks is not evidence; it is a small sample.
MIN_GRADED = 300

# Fee on winnings, per venue. Polymarket's crypto markets ship
# feeType "crypto_fees_v2" with a 7% taker rate; a venue absent here is
# assumed fee-free, which is optimistic and therefore recorded in the output
# rather than hidden.
VENUE_FEE = {"polymarket": 0.07}
DEFAULT_FEE = 0.0

# Layer 2 thresholds for a row to be worth showing at all.
MIN_EDGE = 0.04          # 4c cheaper than fair
MIN_QUALITY = 60.0
MIN_COVERAGE = 0.70

RED, AMBER, GREEN = "red", "amber", "green"


def breakeven_rate(price: float, fee: float = DEFAULT_FEE) -> float:
    """Hit rate needed to break even buying at `price` with `fee` on winnings."""
    price = min(max(price, 1e-6), 1 - 1e-6)
    return price / (price + (1.0 - price) * (1.0 - fee))


def market_light(rows: list[dict], venue: str = "", min_graded: int = MIN_GRADED
                 ) -> dict:
    """One market's light, from its own graded history.

    `rows` are graded records: {"p": probability of the side taken, "hit": 0|1}.
    The test is deliberately the pessimistic end of the interval — the LOWER
    Wilson bound against breakeven — because the question is not "did we do
    well" but "can we rule out having been lucky".
    """
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    fee = VENUE_FEE.get(venue, DEFAULT_FEE)
    mean_price = (sum(r["p"] for r in rows) / n) if n else 0.5
    be = breakeven_rate(mean_price, fee)

    lo, hi = wilson(hits, n) if n else (0.0, 1.0)
    realized = (hits / n) if n else None

    if n < min_graded:
        light, headline = RED, (
            f"{n} graded. Need {min_graded} before this market can be trusted. "
            f"Paper only.")
    elif lo <= 0.50:
        light, headline = RED, (
            f"{n} graded, {realized:.1%} hit rate. Still can't rule out luck "
            f"(could be as low as {lo:.1%}). No real money — this is the "
            f"system protecting you, not failing you.")
    elif lo < be:
        light, headline = AMBER, (
            f"{n} graded, {realized:.1%} hit rate. Beating a coin flip but not "
            f"yet the {be:.1%} you need to cover the vig. Edge forming — "
            f"paper only.")
    else:
        light, headline = GREEN, (
            f"{n} graded, {realized:.1%} hit rate. Clears the {be:.1%} "
            f"breakeven even at the pessimistic end ({lo:.1%}). "
            f"Real-money recommendations unlocked.")

    return {"light": light, "headline": headline, "n": n, "realized": realized,
            "ci95": [lo, hi], "breakeven": be, "fee": fee,
            "fee_assumed": venue not in VENUE_FEE, "mean_price": mean_price}


def rollup(lights: dict) -> dict:
    """The one headline light: the weakest market wins.

    A roll-up cannot be an average. One trusted market beside three unproven
    ones is not "mostly trusted" — it is three markets that must not be traded,
    and a headline that averaged them would say the opposite.
    """
    if not lights:
        return {"light": RED, "headline": "Nothing graded yet. Paper only.",
                "trusted": 0, "total": 0}
    order = {RED: 0, AMBER: 1, GREEN: 2}
    worst = min(lights.values(), key=lambda l: order[l["light"]])
    trusted = sum(1 for l in lights.values() if l["light"] == GREEN)
    return {"light": worst["light"], "headline": worst["headline"],
            "trusted": trusted, "total": len(lights)}


def opportunity(pred, qual: dict, light: dict, fair: float | None,
                venue_price: float | None) -> dict | None:
    """One row of Layer 2, or None if it does not qualify.

    Returns the row with a filled-in WHY in human words. The empty WHY column
    was the actual missing piece: a light with no sentence behind it is just a
    coloured dot, and a user cannot audit a coloured dot.
    """
    if fair is None or venue_price is None:
        return None
    edge = fair - venue_price
    score = qual.get("score")
    cov = qual.get("coverage") or 0.0

    fails = []
    if edge < MIN_EDGE:
        fails.append(f"edge {edge*100:.1f}c < {MIN_EDGE*100:.0f}c")
    if score is None or score < MIN_QUALITY:
        fails.append(f"quality {score if score is not None else 'unmeasured'} "
                     f"< {MIN_QUALITY:.0f}")
    if cov < MIN_COVERAGE:
        fails.append(f"coverage {cov:.0%} < {MIN_COVERAGE:.0%}")
    if fails:
        return None

    # Amber when the market itself is not yet trusted: the row is real, the
    # track record behind it is not, so it is shown as paper rather than hidden.
    tone = GREEN if light["light"] == GREEN else AMBER
    stake = "1 unit" if tone == GREEN else "paper only"

    why = (f"fair {fair*100:.0f}c, {pred.source} asks {venue_price*100:.0f}c "
           f"-> {edge*100:.1f}c cheap. Quality {score:.0f}, "
           f"{light['n']} graded {pred.market} behind this.")
    if tone == AMBER:
        why += " Track record not proven yet, so paper."

    return {"tone": tone, "subject": pred.entity_display, "market": pred.market,
            "source": pred.source, "fair": fair, "price": venue_price,
            "edge": edge, "quality": score, "coverage": cov,
            "why": why, "stake": stake}


def nothing_today(n_checked: int, n_markets: int) -> str:
    """The sentence that replaces a wall of dashes."""
    return (f"Nothing worth buying today. Checked {n_checked} markets across "
            f"{n_markets} venues — all fairly priced or too noisy. Doing "
            f"nothing just saved you the house's margin.")
