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


# Correlated same-(city, day) locks are one weather event observed through
# several buckets, not several independent tests — counting them at face value
# is exactly the "four Dallas locks in an afternoon" trap this gate exists to
# catch. Within a cluster the first lock counts full; each additional one counts
# only this fraction. Documented, not hidden, and surfaced on the dashboard.
CLUSTER_EXTRA = 0.25


def effective_independent_n(cluster_sizes) -> float:
    """De-clustered sample count. `cluster_sizes` is the list of k per
    correlation group (e.g. per (city, event_date)). Each group yields
    1 + CLUSTER_EXTRA*(k-1): the first observation full, the rest discounted."""
    return float(sum(1 + CLUSTER_EXTRA * (k - 1) for k in cluster_sizes if k > 0))


def market_light(rows: list[dict], venue: str = "", min_graded: int = MIN_GRADED,
                 effective_n: float | None = None) -> dict:
    """One market's light, from its own graded history.

    `rows` are graded records: {"p": probability of the side taken, "hit": 0|1}.
    The test is deliberately the pessimistic end of the interval — the LOWER
    Wilson bound against breakeven — because the question is not "did we do
    well" but "can we rule out having been lucky".

    `effective_n`, when supplied, is the de-clustered independent count (see
    effective_independent_n). It gates trust AND widens the interval: correlated
    bursts must not manufacture a falsely narrow "proven" band. The realized
    rate stays the raw observed rate; only the EVIDENCE it constitutes shrinks.
    """
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    fee = VENUE_FEE.get(venue, DEFAULT_FEE)
    mean_price = (sum(r["p"] for r in rows) / n) if n else 0.5
    be = breakeven_rate(mean_price, fee)

    eff = float(n) if effective_n is None else min(float(effective_n), float(n))
    # Scale counts to the effective sample size: same observed rate, wider band.
    lo, hi = wilson(hits * (eff / n), eff) if n else (0.0, 1.0)
    realized = (hits / n) if n else None
    declustered = effective_n is not None and eff < n
    eff_note = f" ({eff:.0f} effective after de-clustering)" if declustered else ""

    if eff < min_graded:
        light, headline = RED, (
            f"{n} graded{eff_note}. Need {min_graded} before this market can be "
            f"trusted. Paper only.")
    elif lo <= 0.50:
        light, headline = RED, (
            f"{n} graded{eff_note}, {realized:.1%} hit rate. Still can't rule "
            f"out luck (could be as low as {lo:.1%}). No real money — this is "
            f"the system protecting you, not failing you.")
    elif lo < be:
        light, headline = AMBER, (
            f"{n} graded{eff_note}, {realized:.1%} hit rate. Beating a coin "
            f"flip but not yet the {be:.1%} you need to cover the vig. Edge "
            f"forming — paper only.")
    else:
        light, headline = GREEN, (
            f"{n} graded{eff_note}, {realized:.1%} hit rate. Clears the "
            f"{be:.1%} breakeven even at the pessimistic end ({lo:.1%}). "
            f"Real-money recommendations unlocked.")

    return {"light": light, "headline": headline, "n": n, "realized": realized,
            "effective_n": round(eff, 1), "ci95": [lo, hi], "breakeven": be,
            "fee": fee, "fee_assumed": venue not in VENUE_FEE,
            "mean_price": mean_price}


# ---- Tradeability: a SECOND, independent light -----------------------------
# "Trustable" (market_light) asks: is the track record real? This asks: is there
# harvestable, slippage-survivable edge? A market can be one without the other —
# a calibrated model whose edge evaporates into the spread is trustable but not
# tradeable; a fat mechanical edge with no graded history is tradeable-looking
# but not yet trustable. Real money needs BOTH lights green.

MIN_MEDIAN_FILL = 25.0   # $ that a realistic $200 order actually clears at lock
MIN_MEDIAN_LAG = 300     # seconds of window before the market reprices to match
MAX_AT_RISK_FRAC = 0.34  # more than a third thin -> a revision could flip them


def _median(xs):
    s = sorted(x for x in xs if x is not None)
    return s[len(s) // 2] if s else None


def tradeable_light(locks: list[dict]) -> dict:
    """One market's tradeability light from SUPPLIED per-lock attributes.

    Each lock: {"hit":0|1, "p":price_paid, "fill_200":$profit_if_correct,
    "lag_s":int|None, "at_risk":0|1, "recon_delta_max":float|None}. DIP consumes
    these; it never recomputes them (it has no METAR feed, no order book).

    EV/downside, not hit-rate (Prompt improvement #5): a wrong lock forfeits the
    cost paid, which on a binary bought at price p to win 1 is fill*p/(1-p) — far
    more than the pennies a win earns on an expensive bucket. A high hit rate
    with negative EV must NOT read green, so the gate is mean P&L, floored by the
    worst single outcome, not the win count.
    """
    usable = [l for l in locks
              if l.get("fill_200") is not None and l.get("p") not in (None, 0, 1)]
    n = len(usable)
    if n == 0:
        return {"light": RED, "n": 0, "ev": None,
                "headline": "No tradeability data yet — these locks predate the "
                            "attribute export (lag/fill/edge). Paper only."}

    pnls = []
    for l in usable:
        p = l["p"]
        upside = l["fill_200"]                 # profit if the lock's side wins
        downside = upside * p / (1.0 - p)      # cost forfeited if it loses
        pnls.append(upside if l["hit"] else -downside)

    ev = sum(pnls) / n
    worst = min(pnls)
    med_fill = _median([l["fill_200"] for l in usable])
    med_lag = _median([l.get("lag_s") for l in usable])
    at_risk_frac = sum(1 for l in usable if l.get("at_risk")) / n
    max_bias = max((abs(l["recon_delta_max"]) for l in usable
                    if l.get("recon_delta_max") is not None), default=None)

    if ev <= 0:
        light, headline = RED, (
            f"EV ${ev:+.0f}/order after slippage & downside over {n} locks — a "
            f"wrong lock (worst ${worst:+.0f}) costs more than a right one "
            f"earns. Hit rate hides this. No real money.")
    elif (med_fill or 0) < MIN_MEDIAN_FILL or (med_lag or 0) < MIN_MEDIAN_LAG:
        light, headline = AMBER, (
            f"EV ${ev:+.0f}/order positive but thin: median ${med_fill or 0:.0f} "
            f"fillable, median lag {(med_lag or 0)//60:.0f}m. Edge real, depth "
            f"or window marginal — paper only.")
    elif at_risk_frac > MAX_AT_RISK_FRAC:
        light, headline = AMBER, (
            f"EV ${ev:+.0f}/order positive but {at_risk_frac:.0%} of locks are "
            f"AT_RISK (thin clearance a revision could flip) — paper only.")
    else:
        light, headline = GREEN, (
            f"EV ${ev:+.0f}/order over {n} locks, median ${med_fill or 0:.0f} "
            f"fillable with a {(med_lag or 0)//60:.0f}m window. Slippage-"
            f"survivable edge.")

    return {"light": light, "headline": headline, "n": n, "ev": round(ev, 2),
            "worst": round(worst, 2), "median_fill": med_fill,
            "median_lag_s": med_lag, "at_risk_frac": round(at_risk_frac, 3),
            "max_bias": max_bias}


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
