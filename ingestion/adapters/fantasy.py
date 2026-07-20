"""Adapter: Fantasy pick6 — owned MLB prop projection system.

Upstream: https://fantasy.perfecthold.online/dip.json  (written by
pick6/export_dip.py, published hourly by deploy/cron_daily.sh)

This is the higher-information source of the two. It ships a negative-binomial
distribution rather than Poisson, a fitted probability calibration, the
UNcalibrated probability alongside it, and its own graded outcomes. All four are
carried through: the pre-calibration probability is what lets DIP measure
whether the calibrator is actually earning its keep, and the dispersion
parameter is what stops DIP's simulator defaulting to Poisson on a model that is
deliberately overdispersed.
"""
from __future__ import annotations

import json
import urllib.request

from ..contract import CONF_PROBABILITY, Prediction, norm_entity

SOURCE = "fantasy"
URL = "https://fantasy.perfecthold.online/dip.json"
DOMAIN = "MLB"

# Payload major version this adapter was written against. A bump upstream means
# a shape change a consumer must react to, so fail loudly rather than parse a
# payload whose meaning has moved.
SUPPORTED_MAJOR = 1

# The export always sends these; a None among them is a real gap in the row, not
# an optional extra. Kept explicit so completeness is measured against a fixed
# promise rather than against whatever happened to arrive.
_EXPECTED = ("predicted", "p", "p_more", "p_uncal", "p_more_raw",
             "mu_source", "bench_proj", "rw_proj")


def fetch(url: str = URL, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _check_schema(payload: dict) -> None:
    schema = payload.get("schema", "")
    name, _, ver = schema.partition("/")
    if name != "dip-export" or not ver:
        raise ValueError(f"unrecognized payload schema {schema!r}")
    if int(ver.split(".")[0]) != SUPPORTED_MAJOR:
        raise ValueError(
            f"dip-export major {ver} != supported {SUPPORTED_MAJOR}; "
            "adapter must be updated before ingesting")


def to_predictions(payload: dict) -> tuple[list, list]:
    """Map dip.json predictions to contract rows. Returns (predictions, rejected)."""
    _check_schema(payload)
    date = payload.get("date") or ""
    dispersion = payload.get("dispersion_r")
    preds, rejected = [], []

    for row in payload.get("predictions", []):
        if row.get("line") is None or row.get("p_more") is None:
            rejected.append({"row": row, "why": "no line/p_more"})
            continue

        present = sum(1 for k in _EXPECTED if row.get(k) is not None)
        # entity arrives pre-normalized by the exporter (same normalizer), but
        # it is re-normalized here rather than trusted. The adapter owns the
        # join key; a future exporter change must not be able to split an
        # entity in DIP without DIP noticing.
        display = row.get("player") or ""

        p = Prediction(
            source=SOURCE,
            source_version=row.get("mu_version") or "",
            domain=DOMAIN,
            entity=norm_entity(display),
            entity_display=display,
            event_key=row.get("event_key") or "",
            market=row.get("market") or "strikeouts",
            event_date=date,
            line=float(row["line"]),
            predicted_value=row.get("predicted"),
            prob_over=row.get("p_more"),
            prob_under=row.get("p_less"),
            # p_uncal is the probability calibrate() was about to be applied to.
            # Keeping it is what makes the calibrator's lift measurable rather
            # than assumed — upstream learned this the hard way when the fit was
            # being made on a different quantity than the one it was applied to.
            prob_uncalibrated=row.get("p_uncal"),
            dispersion=dispersion,
            side=row.get("side"),
            # This one IS a probability: the leaned side's calibrated p. Tagged
            # as such so it is never pooled with mlb-edge's categorical label.
            confidence=row.get("p"),
            confidence_kind=CONF_PROBABILITY,
            feature_completeness=present / len(_EXPECTED),
            raw=row,
        )
        problems = p.validate()
        if problems:
            rejected.append({"row": row, "why": "; ".join(problems)})
            continue
        preds.append(p)

    return preds, rejected


def to_results(payload: dict) -> list[dict]:
    """Graded rows from the payload, keyed for matching against the ledger.

    `result` is passed through verbatim — including 'X', the push where a stat
    landed exactly on a whole line. Upstream's grade.py excludes those from hit
    rates; collapsing one to a loss here would bias every accuracy number DIP
    computes, and it would do so invisibly.
    """
    _check_schema(payload)
    out = []
    for r in payload.get("results", []):
        if not r.get("result"):
            continue
        out.append({
            "domain": DOMAIN,
            "entity": norm_entity(r.get("player") or ""),
            "market": r.get("market") or "strikeouts",
            "event_date": r.get("date") or "",
            "line": r.get("line"),
            "actual": r.get("actual"),
            "result": r["result"],
            "side": r.get("side"),
        })
    return out


def upstream_stamp(payload: dict) -> str:
    """Freshness from the data itself. The export carries no wall-clock stamp on
    purpose, so an unchanged hour is byte-identical: `frozen_at` that stops
    advancing is a stalled board, which is precisely what needs alerting."""
    return (f"board={payload.get('board_status')} "
            f"frozen_at={payload.get('frozen_at')} "
            f"graded_through={payload.get('graded_through')}")
