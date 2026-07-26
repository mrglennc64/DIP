"""Module-by-module verification against the document.

Run:  python -m tests.test_modules     (from decision-platform/)

Each section is named for the module it checks. No network.
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api import pipeline                                    # noqa: E402
from alerts import triggers                                 # noqa: E402
from calibration import evidence                            # noqa: E402
from explainability import attribution                      # noqa: E402
from ingestion.adapters.file import map_columns, to_history, to_predictions  # noqa: E402
from ingestion.contract import Prediction, norm_entity      # noqa: E402
from optimization import correlation, portfolio             # noqa: E402
from scoring import quality, recommendation                 # noqa: E402
from simulation import montecarlo                           # noqa: E402

FAILS = []
TMP = tempfile.mkdtemp()


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if extra else ""))
    if not cond:
        FAILS.append(name)
    return cond


def write(name, header, rows):
    p = os.path.join(TMP, name)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return p


def P(entity, event="E1", market="strikeouts", line=5.5, p_over=0.62,
      predicted=6.1, variance=None, conf=None, **kw):
    return Prediction(
        source="test", source_version="", domain="MLB",
        entity=norm_entity(entity), entity_display=entity, event_key=event,
        market=market, event_date="2026-07-20", line=line,
        predicted_value=predicted, prob_over=p_over,
        prob_under=None if p_over is None else 1 - p_over,
        variance=variance, confidence=conf if conf is not None
        else (max(p_over, 1 - p_over) if p_over is not None else None),
        confidence_kind="probability", feature_completeness=0.8, **kw)


# =========================================================== Module 1 =======
print("--- Module 1: ingestion (any source, variance carried) ---")
m, _ = map_columns(["player", "market", "predictedValue", "line",
                    "probabilityOver", "confidence", "variance", "timestamp",
                    "source"])
check("the document's own interface maps 1:1",
      m.get("variance") == "variance" and m.get("prob_over") == "probabilityOver"
      and m.get("predicted_value") == "predictedValue")
vf = write("v.csv", ["player", "line", "probabilityOver", "variance"],
           [["Judge", 1.5, 0.61, 2.4]])
vp, _, _ = to_predictions(vf)
check("variance flows from file to contract", vp[0].variance == 2.4)

# =========================================================== Module 2 =======
print("\n--- Module 2: quality score ---")
check("Module 2 weighting is the documented one",
      quality.WEIGHTS["m2_v1"] == {"calibration": .30, "historical_accuracy": .30,
                                   "variance": .15, "feature_completeness": .15,
                                   "recent_performance": .10})
check("page-11 PQI variant also present",
      quality.WEIGHTS["pqi_v1"]["model_confidence"] == 0.25)

good_hist = ([{"p": 0.52, "hit": i % 2, "market": "strikeouts"} for i in range(60)]
             + [{"p": 0.62, "hit": 1 if i % 3 else 0, "market": "strikeouts"}
                for i in range(90)])
ga = evidence.assess(good_hist)
gidx = quality.index_history(good_hist)

pv = P("Judge", variance=1.0)          # sd 1.0 on 6.1 -> cv .16 -> high comp
q = quality.score(pv, history_index=gidx, assessment=ga)
check("all five components computed with history + variance",
      not q["missing"], q["missing"])
check("variance component present and high",
      q["components"]["variance"] is not None and q["components"]["variance"] > 0.8,
      q["components"]["variance"])
qn = quality.score(P("NoVar"))
check("missing variance is None, never defaulted",
      qn["components"]["variance"] is None)
check("coverage reported", 0 < qn["coverage"] < 1, qn["coverage"])
check("NB dispersion fallback fills variance",
      quality.score(P("Disp", dispersion=16.6))["components"]["variance"]
      is not None)

# =========================================================== Module 4 =======
print("\n--- Module 4: correlation graph ---")
judge_h = P("Judge", event="NYY@BOS", market="hits", line=1.5)
judge_tb = P("Judge", event="NYY@BOS", market="total_bases", line=2.5)
yank = P("Yankees", event="NYY@BOS", market="runs", line=4.5)
other = P("Skubal", event="DET@CLE")
g = correlation.build_graph([judge_h, judge_tb, yank, other])
check("the document's example: Judge hits <-> Judge TB strongest",
      correlation.rho_between(g, judge_h.id, judge_tb.id) == 0.80)
check("Judge TB <-> Yankees runs weaker but linked",
      correlation.rho_between(g, judge_tb.id, yank.id) == 0.40)
check("unrelated game uncorrelated",
      correlation.rho_between(g, judge_h.id, other.id) == 0.0)
check("graph has nodes and weighted edges",
      len(g["nodes"]) == 4 and len(g["edges"]) == 3)

# =========================================================== Module 3 =======
print("\n--- Module 3: portfolio optimizer ---")
# 30 in -> 5 out, exactly the document's numbers. Ten games, three picks each.
many = [P(f"Pitcher{i}", event=f"G{i % 10}", p_over=0.52 + (i % 20) * 0.01)
        for i in range(30)]
gm = correlation.build_graph(many)
scored = [{"pred": p, "quality": quality.score(p, history_index=gidx,
                                               assessment=ga)} for p in many]
folio = portfolio.select(scored, gm)
check("30 predictions in, top 5 out", len(folio["portfolio"]) == 5)
events = [h["pred"].event_key for h in folio["portfolio"]]
check("no event exceeds the concentration cap",
      max(events.count(e) for e in set(events)) <= portfolio.MAX_PER_EVENT)
check("every selection carries its marginal value and rho",
      all("marginal" in h and "max_rho_at_selection" in h
          for h in folio["portfolio"]))
check("every exclusion carries a why",
      all(x["why"] for x in folio["passed_over"]))
# Correlation discount actually binds: same-event twin of the top pick must
# rank below an uncorrelated alternative with slightly lower base score.
a = P("Star", event="GX", p_over=0.70)
twin = P("Star", event="GX", market="total_bases", p_over=0.69)
alt = P("Elsewhere", event="GY", p_over=0.66)
g2 = correlation.build_graph([a, twin, alt])
sc2 = [{"pred": x, "quality": quality.score(x, history_index=gidx,
                                            assessment=ga)}
       for x in (a, twin, alt)]
f2 = portfolio.select(sc2, g2, top_n=2)
names = [h["pred"].entity_display for h in f2["portfolio"]]
check("correlated twin loses to uncorrelated alternative",
      names[0] == "Star" and "Elsewhere" in names, names)

# =========================================================== Module 5 =======
print("\n--- Module 5: scenario simulator ---")
sim_preds = [judge_h, judge_tb, yank, other]
rho = correlation.rho_matrix(correlation.build_graph(sim_preds), sim_preds)
s = montecarlo.run(sim_preds, rho, seed=42)
check("10,000 simulations by default", s["num_sims"] == 10_000)
check("best/worst/expected/distribution all present",
      all(k in s for k in ("best_case", "worst_case", "expected",
                           "distribution", "sensitivity")))
check("distribution sums to 1",
      abs(sum(d["p"] for d in s["distribution"]) - 1) < 1e-9)
check("expected consistent with marginals (within MC noise)",
      abs(s["expected"] - sum(max(p.prob_over, 1 - p.prob_over)
                              for p in sim_preds)) < 0.1, s["expected"])
s2 = montecarlo.run(sim_preds, rho, seed=42)
check("seeded run reproduces exactly", s == s2)
# The stike bug: correlation must CHANGE the joint distribution.
indep = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
si = montecarlo.run(sim_preds, indep, seed=42)
check("correlated P(all hit) > independent P(all hit) for same-side legs",
      s["p_all_hit"] > si["p_all_hit"], f"{s['p_all_hit']} vs {si['p_all_hit']}")
check("paths are not identical (the portfolio_service.py:92 bug)",
      len([d for d in s["distribution"] if d["p"] > 0]) > 1)

# =========================================================== Module 6 =======
print("\n--- Module 6: explainability ---")
ex = attribution.explain(q, pv)
check("signed contributions present and sorted by magnitude",
      ex["contributions"]
      and all(abs(ex["contributions"][i]["points"])
              >= abs(ex["contributions"][i + 1]["points"])
              for i in range(len(ex["contributions"]) - 1)))
check("unmeasured components declared, not hidden",
      attribution.explain(qn)["unmeasured"])
panel = attribution.render(ex)
check("renders the document's panel shape",
      panel.startswith("Prediction Score") and "Why?" in panel)

# =========================================================== Module 7 =======
print("\n--- Module 7: confidence calibration ---")
d = evidence.display(0.63, good_hist)
check("the document's four fields",
      all(k in d for k in ("historical_accuracy", "model_confidence",
                           "calibration_error", "reliability")))
d_none = evidence.display(0.95, good_hist)
check("no data near a confidence level -> Unknown, not a label",
      d_none["reliability"] == "Unknown")
inv = ([{"p": 0.52, "hit": 1} for _ in range(40)]
       + [{"p": 0.52, "hit": 0} for _ in range(30)]
       + [{"p": 0.72, "hit": 1} for _ in range(12)]
       + [{"p": 0.72, "hit": 0} for _ in range(28)])
ia = evidence.assess(inv)
check("inverted ranking detected", ia["signal"] == "inverted")
check("empty history is a first-class answer",
      evidence.assess([])["status"] == "no_history")

# =========================================================== Module 9 =======
print("\n--- Module 9: recommendation engine ---")
rule = recommendation.RULES["doc_v1"][0]
check("the document's High rule at the document's thresholds",
      ("quality_score", ">", 90.0) in rule["clauses"]
      and ("calibration", ">", 0.95) in rule["clauses"]
      and ("variance_low", "is", True) in rule["clauses"]
      and ("correlation_low", "is", True) in rule["clauses"])
r = recommendation.recommend(pv, q, {"max_rho": 0.0}, ga)
check("evaluation returns full audit trail",
      r["evaluated"] and all("checks" in e for e in r["evaluated"]))
r_inv = recommendation.recommend(pv, q, {"max_rho": 0.0}, ia)
check("inverted evidence -> no recommendation, reason stated",
      r_inv["recommendation"] is None and "inverted" in r_inv["why"])
r_no = recommendation.recommend(pv, q, {"max_rho": 0.0}, evidence.assess([]))
check("no history -> no recommendation", r_no["recommendation"] is None)
# Rules are data: a versioned rule set change is additive.
check("rules are versioned data", "doc_v1" in recommendation.RULES)

# ========================================================== Module 10 =======
print("\n--- Module 10: alerts ---")
al = triggers.highest_quality_today(scored, "2026-07-20")
check("1: highest-quality prediction fires", len(al) == 1
      and al[0]["dedupe_key"] == "highest_quality:2026-07-20")
drift = triggers.calibration_drift(ia, "2026-07-20")
check("2: calibration drift + inverted ranking fire",
      {a["kind"] for a in drift} >= {"ranking_inverted"})
imp = triggers.accuracy_improved({"n": 60, "realized": 0.58},
                                 {"n": 55, "realized": 0.52}, "2026-07-20")
check("3: accuracy-improved fires on a real gain", len(imp) == 1)
noise = triggers.accuracy_improved({"n": 10, "realized": 0.70},
                                   {"n": 8, "realized": 0.40}, "2026-07-20")
check("3b: small samples never fire", noise == [])

# ================================================== end to end (CLI path) ===
print("\n--- end to end: file -> full chain ---")
live = write("live.csv",
             ["player", "game", "market", "line", "probability", "side",
              "variance"],
             [[f"P{i}", f"G{i % 4}", "strikeouts", 5.5,
               0.55 + (i % 5) * 0.02, "over", 1.5] for i in range(12)])
hist = write("hist.csv", ["player", "market", "probability", "side", "result"],
             [["x", "strikeouts", 0.55, "over", "1"] for _ in range(50)]
             + [["x", "strikeouts", 0.55, "over", "0"] for _ in range(30)]
             + [["x", "strikeouts", 0.62, "over", "1"] for _ in range(40)]
             + [["x", "strikeouts", 0.62, "over", "0"] for _ in range(20)])
res = pipeline.analyze_file(live, hist)
check("chain produces all module outputs",
      all(k in res for k in ("assessment", "scored", "graph", "portfolio",
                             "simulation", "alerts")))
check("every scored row has quality+explanation+recommendation",
      all(("quality" in s and "explanation" in s and "recommendation" in s)
          for s in res["scored"]))
check("portfolio respects top-5", len(res["portfolio"]["portfolio"]) <= 5)
check("simulation ran over the portfolio",
      res["simulation"]["n_predictions"]
      == max(1, len(res["portfolio"]["portfolio"])))

print("\n--- Trigger gate: de-clustering + tradeable light ---")
from scoring import gate                                    # noqa: E402

check("effective_independent_n discounts a cluster",
      gate.effective_independent_n([4]) == 1.0 + 0.25 * 3)
check("effective_independent_n leaves singletons whole",
      gate.effective_independent_n([1, 1, 1]) == 3.0)

# A correlated burst must NOT unlock trust: 400 rows at 75% clears the raw gate,
# but 100 effective (de-clustered) keeps it red. This is the whole point.
burst = ([{"p": 0.5, "hit": 1}] * 300) + ([{"p": 0.5, "hit": 0}] * 100)
raw_L = gate.market_light(burst)
eff_L = gate.market_light(burst, effective_n=100.0)
check("raw count would have gone green", raw_L["light"] == gate.GREEN)
check("de-clustered count stays red (below min_graded)",
      eff_L["light"] == gate.RED and eff_L["effective_n"] == 100.0)
check("de-clustering widens the interval",
      (eff_L["ci95"][1] - eff_L["ci95"][0]) > (raw_L["ci95"][1] - raw_L["ci95"][0]))

# Tradeable light: EV/downside, not hit rate. A 75%-hit market whose wins earn
# pennies and losses forfeit the expensive stake must read RED.
neg = ([{"hit": 1, "p": 0.9, "fill_200": 10.0, "lag_s": 600, "at_risk": 0}] * 3
       + [{"hit": 0, "p": 0.9, "fill_200": 10.0, "lag_s": 600, "at_risk": 0}])
negT = gate.tradeable_light(neg)
check("high hit rate but negative EV reads red",
      negT["light"] == gate.RED and negT["ev"] < 0)

good = [{"hit": 1, "p": 0.6, "fill_200": 50.0, "lag_s": 1200, "at_risk": 0,
         "recon_delta_max": 0.5, "fill_verified": True}] * 8
goodT = gate.tradeable_light(good)
check("positive EV with a VERIFIED real fill reads green",
      goodT["light"] == gate.GREEN and goodT["ev"] > 0)
# Same locks but fill unverified (displayed depth only) must NOT green — the
# phantom-liquidity guard: displayed depth can be spoofed, so it caps at amber.
unver = [{**l, "fill_verified": False} for l in good]
unverT = gate.tradeable_light(unver)
check("positive EV but UNVERIFIED (displayed depth) caps at amber",
      unverT["light"] == gate.AMBER and unverT["fill_basis"] == "displayed")

check("no supplied attrs -> not tradeable (honest), not a crash",
      gate.tradeable_light([{"hit": 1, "p": 0.6}])["light"] == gate.RED)

# Identity: two cities, same bucket label, same day must NOT collide for an
# event-keyed market — but must still ignore event_key for ordinary markets.
from ingestion.contract import prediction_id as _pid                # noqa: E402
check("event-keyed market: different event_key -> different id",
      _pid("ce", "w", "92 93f", "temp_lock", 0.5, "2026-07-25", "dallas-evt")
      != _pid("ce", "w", "92 93f", "temp_lock", 0.5, "2026-07-25", "houston-evt"))
check("ordinary market: event_key ignored in identity",
      _pid("ce", "mlb", "deGrom", "strikeouts", 5.5, "2026-07-25", "g1")
      == _pid("ce", "mlb", "deGrom", "strikeouts", 5.5, "2026-07-25", "g2"))

print("\n" + (f"{len(FAILS)} FAILURES: " + ", ".join(FAILS) if FAILS else "ALL PASS"))
sys.exit(1 if FAILS else 0)
