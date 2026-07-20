"""File in -> decision out, from the command line.

    python -m api.cli predictions.csv
    python -m api.cli predictions.csv --history graded.csv --json out.json

Runs the same chain as the HTTP API (api/pipeline.py), so a file on disk and a
POSTed batch get identical treatment. Exit 0 when the portfolio is non-empty
and the evidence supports acting; 2 otherwise — a cron can branch on the answer
without parsing text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api import pipeline                       # noqa: E402
from explainability import attribution         # noqa: E402

BAR = "=" * 74


def render(a: dict, path: str) -> str:
    o, ev = [], a["assessment"]
    m = a.get("mapping", {})

    o.append(BAR)
    o.append(f"DECISION REPORT  -  {os.path.basename(path)}")
    o.append(BAR)

    if m:
        o.append("\nHOW THE FILE WAS READ")
        for f, c in sorted(m.get("mapping", {}).items()):
            note = ("  (inferred - no alias matched)"
                    if f == "entity" and m.get("inferred_entity") else "")
            o.append(f"  {f:<20} <- column {c!r}{note}")
        if m.get("unmatched"):
            o.append(f"  ignored columns: {', '.join(m['unmatched'])}")
        o.append(f"  {m.get('n_rows', 0)} rows read, "
                 f"{len(a['scored'])} usable, {len(a.get('rejected', []))} rejected")
        for r in a.get("rejected", [])[:5]:
            o.append(f"    row {r['row_index']}: {r['why']}")

    o.append("\nEVIDENCE  (Module 7 — what the graded history supports)")
    o.append(f"  {ev['headline']}")
    if ev.get("brier") is not None:
        o.append(f"  Brier {ev['brier']:.4f} vs 0.2500   "
                 f"log-loss {ev['log_loss']:.4f} vs 0.6931")
    for b in ev.get("curve", []):
        flag = "  <-- significantly worse than stated" if b["significant"] else ""
        o.append(f"    {b['bucket']}  n={b['n']:<4} stated {b['stated']:.1%}"
                 f"  realized {b['realized']:.1%}  {b['gap_pts']:+.1f} pts{flag}")
    if ev.get("ranking", {}).get("status") not in (None, "unknown"):
        o.append(f"  RANKING: {ev['ranking']['status'].upper()} — "
                 f"{ev['ranking']['detail']}")

    o.append("\nQUALITY + RECOMMENDATION  (Modules 2, 6, 9)")
    o.append(f"  {'rec':<7} {'score':>5} {'cov':>5}  subject")
    for s in a["scored"][:25]:
        p, q, r = s["pred"], s["quality"], s["recommendation"]
        o.append(f"  {str(r['recommendation']):<7} "
                 f"{(f'{q['score']:.1f}' if q['score'] is not None else '-'):>5} "
                 f"{q['coverage']:>4.0%}  {p.entity_display}"
                 + (f"  {p.market} {p.side or ''} {p.line:g}" if p.line else ""))
        o.append(f"          {r['why']}")
    if len(a["scored"]) > 25:
        o.append(f"  ... {len(a['scored']) - 25} more")

    folio = a["portfolio"]
    o.append("\nPORTFOLIO  (Module 3 — top selections under correlation discount)")
    if folio["portfolio"]:
        for h in folio["portfolio"]:
            o.append(f"  {h['pred'].entity_display:<24} score "
                     f"{h['quality']['score']}  marginal {h['marginal']}"
                     f"  max-rho {h['max_rho_at_selection']}")
        c = folio["concentration"]
        o.append(f"  {c['n']} picks / {c['unique_events']} events — "
                 f"effective independent positions "
                 f"{c['effective_independent_bets']}")
    else:
        o.append("  empty — nothing scored high enough to select")

    sim = a["simulation"]
    if sim.get("num_sims"):
        o.append(f"\nSIMULATION  (Module 5 — {sim['num_sims']:,} runs over the "
                 f"portfolio)")
        o.append(f"  expected {sim['expected']} of {sim['n_predictions']} hit   "
                 f"best {sim['best_case']}  worst {sim['worst_case']}   "
                 f"P(all) {sim['p_all_hit']:.1%}  P(none) {sim['p_none_hit']:.1%}"
                 + ("" if sim["correlation_used"] is False else
                    "   (correlation respected)"))
        weakest = sim["sensitivity"][0] if sim["sensitivity"] else None
        if weakest:
            o.append(f"  most sensitive to: {weakest['entity']} "
                     f"(P(all) +{weakest['p_all_uplift_if_certain']:.1%} if certain)")

    if a["alerts"]:
        o.append("\nALERTS  (Module 10)")
        for al in a["alerts"]:
            o.append(f"  [{al['severity']}] {al['subject']}")

    recs = [s for s in a["scored"]
            if s["recommendation"]["recommendation"] in ("High", "Medium")]
    o.append("\n" + BAR)
    if recs:
        o.append(f"VERDICT: {len(recs)} recommendation(s) at Medium or above, "
                 f"{len(folio['portfolio'])} in portfolio")
    else:
        o.append("VERDICT: nothing recommended at Medium or above")
        why = next((s["recommendation"]["why"] for s in a["scored"]), None)
        if why:
            o.append(f"  {why}")
    o.append(BAR)
    return "\n".join(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dip")
    ap.add_argument("predictions", help="CSV or JSON prediction file")
    ap.add_argument("--history", help="CSV or JSON of graded predictions")
    ap.add_argument("--domain", default="")
    ap.add_argument("--target-n", type=int, default=200)
    ap.add_argument("--quality-version", choices=["m2_v1", "pqi_v1"],
                    default=None, help="Module 2 weighting (default) or the "
                                       "page-11 PQI variant")
    ap.add_argument("--json", help="write the full analysis here")
    ap.add_argument("--explain", metavar="ENTITY",
                    help="print the Module 6 panel for one subject and exit")
    a = ap.parse_args(argv)

    res = pipeline.analyze_file(a.predictions, a.history, a.domain,
                                target_n=a.target_n,
                                quality_version=a.quality_version)

    if a.explain:
        for s in res["scored"]:
            if a.explain.lower() in s["pred"].entity_display.lower():
                print(attribution.render(s["explanation"]))
                return 0
        print(f"no prediction matching {a.explain!r}")
        return 2

    print(render(res, a.predictions))
    if a.json:
        def default(o):
            if hasattr(o, "__dict__") or hasattr(o, "_asdict"):
                return str(o)
            return str(o)
        slim = {k: v for k, v in res.items() if k not in ("scored", "graph",
                                                          "portfolio")}
        slim["predictions"] = [{
            "entity": s["pred"].entity_display, "market": s["pred"].market,
            "line": s["pred"].line, "quality": s["quality"],
            "recommendation": s["recommendation"],
            "explanation": s["explanation"],
        } for s in res["scored"]]
        slim["portfolio"] = [{"entity": h["pred"].entity_display,
                              "marginal": h["marginal"]}
                             for h in res["portfolio"]["portfolio"]]
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2, default=default)
        print(f"\nfull analysis -> {a.json}")

    acted = (res["portfolio"]["portfolio"]
             and res["assessment"]["signal"] == "positive")
    return 0 if acted else 2


if __name__ == "__main__":
    sys.exit(main())
