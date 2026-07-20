"""Adapter: a prediction FILE — CSV or JSON, any schema, any domain.

This is the front door. DIP consumes predictions it did not make, and the
general case is not an API you can poll: it is a file somebody exports. A
forecasting spreadsheet, a model's nightly dump, a vendor's CSV.

Column names are matched by alias rather than dictated, because the whole point
is that the producer does not have to know DIP exists. Nothing is guessed
silently: `describe_mapping()` reports exactly which column was read as which
field, and anything unmatched is listed so a wrong guess is visible rather than
buried in a score three layers later.
"""
from __future__ import annotations

import csv
import json
import os

from ..contract import CONF_PROBABILITY, Prediction, norm_entity

SOURCE = "file"

# Alias -> canonical field. Lowercased, non-alphanumerics stripped before match,
# so "Predicted Value", "predicted_value" and "predictedvalue" all land together.
ALIASES = {
    "entity": ("entity", "player", "name", "subject", "ticker", "symbol",
               "asset", "pitcher", "team", "id", "item"),
    "event_key": ("eventkey", "game", "event", "matchup", "contest", "period",
                  "eventid", "gameid"),
    "market": ("market", "metric", "stat", "target", "measure", "question",
               "category", "type"),
    "event_date": ("eventdate", "date", "day", "asof", "timestamp", "when"),
    "line": ("line", "threshold", "strike", "benchmark", "cutoff", "target_value"),
    "predicted_value": ("predictedvalue", "predicted", "projection", "forecast",
                        "mu", "lam", "lambda", "expected", "estimate", "mean",
                        "pointestimate"),
    "prob_over": ("probover", "pover", "probability", "prob", "p", "pmore",
                  "modelp", "probabilityover", "confidence_prob", "phat"),
    "side": ("side", "direction", "lean", "pick", "position", "bet"),
    "confidence": ("confidence", "conf", "certainty", "score"),
    "variance": ("variance", "var", "uncertainty"),
    "dispersion": ("dispersion", "r", "size", "overdispersion", "sd", "stdev",
                   "std", "sigma"),
    "source_version": ("sourceversion", "version", "modelversion", "muversion",
                       "build"),
    "domain": ("domain", "sport", "asset_class", "vertical", "field"),
    # Outcome columns — present in a HISTORY file, absent in a live one.
    "actual": ("actual", "result_value", "outcome_value", "observed", "truth",
               "realized", "actualvalue"),
    "result": ("result", "hit", "won", "correct", "outcome", "graded"),
}

_CANON = {a: field for field, al in ALIASES.items() for a in al}

# Real column names are compounds — `expected_ks`, `model_prob`, `p_more_raw`.
# Exact aliasing misses all of them, so a fuzzy pass follows the exact one.
# Only aliases at least this long are eligible: without the floor, the "p"
# alias would claim `pitcher`, `park` and `platform`.
_MIN_FUZZY = 4

# Headers that must NEVER be read as the model's probability, however much they
# look like one. `fair_prob`, `book_prob`, `implied_prob` and `market_prob` are
# the BOOK's de-vigged price — the thing a model is measured AGAINST. Reading
# one as the model's belief would make every prediction look perfectly
# calibrated to the market and drive the measured edge to zero, silently. This
# is the single most damaging mis-map available, so it is blocked by name.
_NOT_MODEL_PROB = ("fair", "book", "implied", "market", "clos", "consensus",
                   "vig", "devig", "line", "settle")

# Prefixes that NEGATE or qualify the field they precede. `low_confidence` is a
# boolean "small sample, do not trust" flag; fuzzy matching would read it as a
# confidence VALUE, which is the inverse of what it means. Likewise `is_stale`,
# `has_error`, `not_graded`. A negated flag must never become the thing it
# negates, so these are refused outright rather than mapped and misread.
_NEGATING = ("low", "is", "has", "no", "not", "un", "flag", "min", "max")


def _key(name: str) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum())


def _fuzzy_field(k: str) -> str | None:
    """Longest alias that prefixes or is contained in the header key."""
    best, best_len = None, 0
    for alias, field in _CANON.items():
        if len(alias) < _MIN_FUZZY or len(alias) <= best_len:
            continue
        if k.startswith(alias):
            best, best_len = field, len(alias)
        elif alias in k:
            # Contained rather than leading: check what sits in front of it.
            # "lowconfidence" ends in a real alias but means its opposite.
            prefix = k[:k.index(alias)]
            if prefix and any(prefix.startswith(n) for n in _NEGATING):
                continue
            best, best_len = field, len(alias)
    return best


def map_columns(headers: list[str]) -> tuple[dict, list[str]]:
    """headers -> ({field: column}, unmatched columns).

    Two passes so that an exact name always beats a fuzzy one regardless of
    column order: with a single pass, a file carrying both `model_prob` and
    `prob` would resolve by whichever appeared first in the header row.
    """
    mapping, claimed = {}, set()

    for h in headers:                                   # pass 1: exact
        field = _CANON.get(_key(h))
        if field and field not in mapping:
            mapping[field], _ = h, claimed.add(h)

    for h in headers:                                   # pass 2: fuzzy
        if h in claimed:
            continue
        k = _key(h)
        field = _fuzzy_field(k)
        if field == "prob_over" and any(b in k for b in _NOT_MODEL_PROB):
            continue                                    # market price, not belief
        if field and field not in mapping:
            mapping[field], _ = h, claimed.add(h)

    return mapping, [h for h in headers if h not in claimed]


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(path: str) -> list[dict]:
    """Read CSV or JSON into a list of dicts. JSON may be a bare list, or an
    object with the rows under any of the usual container keys."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        for k in ("predictions", "rows", "data", "records", "results", "items"):
            if isinstance(data.get(k), list):
                return data[k]
        raise ValueError(
            f"{path}: JSON object has no recognizable row list "
            f"(looked for predictions/rows/data/records/results/items)")
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _side_and_probs(row: dict, m: dict, line, predicted):
    """Resolve (side, prob_over, prob_under) from whatever the file gave us.

    A file may carry a probability, a side, both, or neither. The one thing
    never done here is inventing a probability from a point estimate: without a
    distribution there is no defensible mapping from "predicted 6.1 vs line 5.5"
    to a number, and a fabricated one would flow straight into the quality score
    wearing the same clothes as a real one.
    """
    raw_side = (row.get(m.get("side", "")) or "").strip().lower()
    side = None
    if raw_side in ("over", "more", "above", "up", "long", "yes", "1", "true"):
        side = "over"
    elif raw_side in ("under", "less", "below", "down", "short", "no", "0", "false"):
        side = "under"

    p = _num(row.get(m.get("prob_over", "")))
    if p is not None and p > 1.0:
        p = p / 100.0                      # a file giving 61.2 means 61.2%

    if p is None:
        return side, None, None

    # A probability column paired with a side column is ambiguous: it is
    # usually P(the leaned side), not P(over). Treat it as the leaned side's
    # probability when a side is stated and the value is >= 0.5 — the same
    # convention mlb-edge uses, and the source of a sign error if assumed away.
    if side == "under" and p >= 0.5:
        return side, 1.0 - p, p
    if side is None:
        side = "over" if p >= 0.5 else "under"
    return side, p, 1.0 - p


def to_predictions(path: str, domain: str = "", source: str = SOURCE,
                   overrides: dict | None = None) -> tuple[list, list, dict]:
    """Read a prediction file. Returns (predictions, rejected, mapping_info)."""
    rows = load_rows(path)
    if not rows:
        return [], [], {"mapping": {}, "unmatched": [], "n_rows": 0}

    headers = list(rows[0].keys())
    mapping, unmatched = map_columns(headers)

    # Every producer names its subject column something different — sku, ticker,
    # account, case_id. Rather than chase that list forever, fall back to the
    # first unmatched non-numeric column: in practice the identifier is the one
    # text column nothing else claimed. Recorded in the mapping so the guess is
    # visible in the report, not silent — without an entity the correlation
    # engine cannot tell two predictions about the same subject apart.
    inferred_entity = None
    if "entity" not in mapping and unmatched:
        for h in unmatched:
            if _num(rows[0].get(h)) is None and str(rows[0].get(h, "")).strip():
                mapping["entity"] = h
                inferred_entity = h
                unmatched = [u for u in unmatched if u != h]
                break

    mapping.update(overrides or {})

    preds, rejected = [], []
    stem = os.path.splitext(os.path.basename(path))[0]

    for i, row in enumerate(rows):
        display = str(row.get(mapping.get("entity", ""), "") or "").strip()
        line = _num(row.get(mapping.get("line", "")))
        predicted = _num(row.get(mapping.get("predicted_value", "")))
        side, p_over, p_under = _side_and_probs(row, mapping, line, predicted)

        # A prediction with neither a probability nor a point estimate is not a
        # prediction. Say so on the row rather than scoring an empty shell.
        if p_over is None and predicted is None:
            rejected.append({"row_index": i, "why": "no probability and no "
                             "point estimate — nothing to evaluate"})
            continue
        if line is None and p_over is None:
            rejected.append({"row_index": i, "why": "point estimate with no "
                             "line and no probability — no question to answer"})
            continue

        # event_key is required by the contract (correlation joins on it). When
        # a file has no game/event column, each row is its own event: that is
        # the truthful default, and it makes independent rows behave as
        # independent rather than silently sharing a bucket.
        ev = str(row.get(mapping.get("event_key", ""), "") or "").strip()
        if not ev:
            ev = f"{stem}:row{i}"

        dom = (str(row.get(mapping.get("domain", ""), "") or "").strip()
               or domain or "unspecified")
        completeness = sum(
            1 for f in ("predicted_value", "prob_over", "line", "event_key",
                        "market", "side", "source_version")
            if mapping.get(f) and str(row.get(mapping[f], "") or "").strip()
        ) / 7.0

        conf = _num(row.get(mapping.get("confidence", "")))
        if conf is not None and conf > 1.0:
            conf = conf / 100.0
        if conf is None and p_over is not None:
            conf = max(p_over, 1.0 - p_over)

        p = Prediction(
            source=source,
            source_version=str(row.get(mapping.get("source_version", ""), "") or ""),
            domain=dom,
            entity=norm_entity(display) or f"row{i}",
            entity_display=display or f"row {i}",
            event_key=ev,
            market=str(row.get(mapping.get("market", ""), "") or "value").strip(),
            event_date=str(row.get(mapping.get("event_date", ""), "") or "").strip(),
            line=line if line is not None else 0.0,
            predicted_value=predicted,
            prob_over=p_over,
            prob_under=p_under,
            variance=_num(row.get(mapping.get("variance", ""))),
            dispersion=_num(row.get(mapping.get("dispersion", ""))),
            side=side,
            confidence=conf,
            confidence_kind=CONF_PROBABILITY if conf is not None else None,
            feature_completeness=completeness,
            raw=dict(row),
        )
        problems = [x for x in p.validate()
                    if "event_key" not in x]     # defaulted above, deliberately
        if problems:
            rejected.append({"row_index": i, "why": "; ".join(problems)})
            continue
        preds.append(p)

    return preds, rejected, {"mapping": mapping, "unmatched": unmatched,
                             "n_rows": len(rows),
                             "inferred_entity": inferred_entity}


def to_history(path: str, overrides: dict | None = None) -> list[dict]:
    """Read a GRADED file: predictions that already have outcomes.

    Rows without an outcome are skipped rather than defaulted — an ungraded row
    counted as a loss is the single easiest way to manufacture a pessimistic
    calibration curve out of nothing.
    """
    rows = load_rows(path)
    if not rows:
        return []
    mapping, _ = map_columns(list(rows[0].keys()))
    mapping.update(overrides or {})

    out = []
    for row in rows:
        res = str(row.get(mapping.get("result", ""), "") or "").strip().lower()
        actual = _num(row.get(mapping.get("actual", "")))
        line = _num(row.get(mapping.get("line", "")))
        side, p_over, _ = _side_and_probs(row, mapping, line, None)

        # Normalize however the file spells an outcome. 'X'/'push'/'void' is a
        # PUSH and stays distinct: folding it into a loss biases every rate.
        if res in ("1", "win", "won", "hit", "true", "yes", "correct"):
            hit = 1
        elif res in ("0", "loss", "lost", "miss", "false", "no", "incorrect"):
            hit = 0
        elif res in ("x", "push", "void", "tie", "draw"):
            hit = None
        elif res == "" and actual is not None and line is not None and side:
            if actual == line and float(line).is_integer():
                hit = None
            else:
                high = side in ("over", "more")
                hit = int(actual > line if high else actual < line)
        else:
            continue

        if hit is None:
            continue                       # push: excluded from rates, per grade.py

        # p is the probability of the side that was taken, which is what a
        # calibration curve is about.
        p_side = None
        if p_over is not None:
            p_side = p_over if side in ("over", "more", None) else 1.0 - p_over
            p_side = max(p_side, 1.0 - p_side) if side is None else p_side

        out.append({
            "entity": norm_entity(str(row.get(mapping.get("entity", ""), "") or "")),
            "market": str(row.get(mapping.get("market", ""), "") or "value").strip(),
            "date": str(row.get(mapping.get("event_date", ""), "") or "").strip(),
            "source_version": str(row.get(mapping.get("source_version", ""), "") or ""),
            "p": p_side,
            "predicted": _num(row.get(mapping.get("predicted_value", ""))),
            "actual": actual,
            "line": line,
            "side": side,
            "hit": hit,
        })
    return out
