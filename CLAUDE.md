# DIP — standing boundary rule (read every session)

DIP consumes predictions. It does not audit producers. The upstream systems
(e.g. Contest Edge, github.com/mrglennc64/my-predictions) own producing
predictions AND proving them — their own ledgers, freeze timestamps, and
grading. DIP's job begins at the handoff: a file or API payload arriving
through ingestion/.

Rules:
1. Ingest what arrives. Judge a source ONLY by its graded history, through
   calibration/evidence.py — never by reviewing its code, its dashboard
   screenshots, or its methodology. The evidence gate IS the skepticism;
   one line ("evidence insufficient -> recommendations None") replaces any
   audit essay.
2. If rows fail contract.validate(), or a history file is missing or
   internally inconsistent, log it in the ingestion report with the reason
   and score the source accordingly. Do NOT investigate upstream internals,
   re-derive its models, or write critiques of it. Flag, score, move on.
3. Never re-grade upstream predictions against your own data feeds or
   oracles. Outcome truth arrives WITH the graded history the source
   supplies. If you distrust a source's grading, that distrust is expressed
   as a low reliability score — not as an investigation.
4. Time spent analyzing where information comes from is time stolen from
   DIP's actual job: quality scoring, correlation, portfolio, simulation,
   recommendation. When in doubt, do the module's job on the data in hand.

Why this exists: on 2026-07-20 a DIP session audited the upstream from a
dashboard screenshot instead of its ledger and got the facts wrong (called
23 existing Down-window grades nonexistent; mistook a UI slug truncation
for inverted token reads). Judging sources by graded data is not only DIP's
job — it is more accurate than inspection.

Owner's architecture: ONE prediction system (Contest Edge) and ONE decision
system (DIP), connected only by the data contract. The codebases must never
mix; neither system does the other's job.
