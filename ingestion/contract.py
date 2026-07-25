"""The Prediction contract — the one shape every source is normalized into.

DIP never makes predictions; it consumes them. That only works if two engines
that share nothing can be laid side by side, so this module owns the shape and
the identity rule, and every adapter is judged against it.

Domain-agnostic on purpose. The spec's `sport: "MLB"` is split into `domain` +
`entity` so tennis, hockey, crypto, earnings and business forecasting plug in
later with no migration — the v1 build is MLB-only, but nothing here knows that.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import unicodedata

# Bumped when the contract gains or changes a field in a way an adapter must
# react to. Adapters declare which they were written against.
CONTRACT_VERSION = 1

# What a source's `confidence` number actually MEANS. Recorded per prediction
# because the two v1 sources disagree: mlb-edge ships a categorical
# High/Medium/Low, while stike's numeric field is abs(p - 0.5) * 100 — distance
# from even money, not an uncertainty estimate. Both normalize to 0-1 and would
# look identical in the column; pooling them anyway would fabricate
# comparability, so downstream scoring reads this tag and refuses to.
CONF_CATEGORICAL = "categorical"     # bucketed label mapped onto 0-1
CONF_DISTANCE = "distance_from_even" # NOT uncertainty — a restatement of p
CONF_PROBABILITY = "probability"     # the leaned side's calibrated probability


def norm_entity(name: str) -> str:
    """Accent-folded lowercase match key: "Martín Pérez" -> "martin perez".

    Matches Fantasy's pick6/feed.py:norm and grade.py:norm for any name input.
    That is a requirement, not a coincidence — it is the key the two sources
    join on, and a normalizer that disagreed by one character would silently
    split a pitcher into two entities and destroy every cross-source comparison
    DIP exists to make.

    ONE deliberate difference: digits are kept. Fantasy's version drops them,
    which is harmless for people's names but catastrophic for every other
    domain — "SKU-1181", "SKU-2043" and "SKU-4412" all normalize to "sku",
    merging unrelated subjects into one entity and making the correlation
    engine report them as the same thing. Since no MLB player name contains a
    digit, keeping them is byte-identical on the join path and correct
    everywhere else.
    """
    nk = unicodedata.normalize("NFKD", name)
    nk = "".join(c for c in nk if not unicodedata.combining(c))
    return "".join(c for c in nk.lower()
                   if c.isalpha() or c.isdigit() or c == " ").strip()


# Markets whose identity MUST also include event_key. For player-prop markets
# one entity plays one event per day, so (entity, market, line, date) is unique
# and event_key stays out (see below). But bucket markets reuse the SAME label
# across events — two cities can both have a "92-93°F" bucket the same day — so
# without event_key their locks collide to one id and INSERT OR IGNORE silently
# drops the second. These markets fold the event key in to stay distinct.
EVENT_KEYED_MARKETS = {"temp_lock"}


def prediction_id(source: str, domain: str, entity: str, market: str,
                  line: float, event_date: str, event_key: str = "") -> str:
    """Stable identity for a prediction.

    `line` is part of the identity deliberately. When a book moves 5.5 -> 6.5,
    that is not an update to the earlier prediction — it is a prediction about a
    different question, and the original still has to be graded on the line it
    was made against. Including line keeps both on the record; excluding it
    would let the later one overwrite the earlier and make the ledger flatter
    for exactly the days when line movement mattered most.

    `event_key` is folded in ONLY for EVENT_KEYED_MARKETS. Elsewhere it is
    excluded on purpose: player-prop identity is (entity, market, line, date),
    and mixing the venue's event id in would fragment a player's day across
    however many books quote it. Grading is unaffected either way — it resolves
    by the stored id, not by re-deriving this hash.
    """
    key = f"{source}|{domain}|{entity}|{market}|{line:g}|{event_date}"
    if market in EVENT_KEYED_MARKETS and event_key:
        key += f"|{event_key}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class Prediction:
    source: str
    source_version: str
    domain: str
    entity: str                    # normalized key
    entity_display: str            # as the source spelled it
    event_key: str                 # game id — structural correlation joins on this
    market: str
    event_date: str

    line: float
    predicted_value: float | None = None
    prob_over: float | None = None
    prob_under: float | None = None
    prob_uncalibrated: float | None = None
    # Module 1 carries `variance` directly. It is 15% of the Module 2 weighting
    # and the "Variance Low" clause of the Module 9 rule, so it is a first-class
    # field rather than something inferred at scoring time.
    variance: float | None = None
    dispersion: float | None = None          # NB size; None = Poisson assumed
    side: str | None = None
    confidence: float | None = None          # 0-1
    confidence_kind: str | None = None
    feature_completeness: float | None = None

    raw: dict = dataclasses.field(default_factory=dict)

    @property
    def id(self) -> str:
        return prediction_id(self.source, self.domain, self.entity,
                             self.market, self.line, self.event_date,
                             self.event_key)

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the row is ingestible.

        Adapters call this and DIP drops offenders WITH THE REASON LOGGED
        rather than silently. A row quietly vanishing between an upstream and
        the ledger is the one ingestion bug that never announces itself — the
        totals just come out slightly low and nobody notices for a month.
        """
        p = []
        if not self.entity:
            p.append("entity empty after normalization")
        if not self.event_key:
            # Required, not optional: without it two legs in the same game look
            # independent, and the portfolio optimizer will happily concentrate
            # the whole card on one game while reporting low correlation.
            p.append("event_key missing — structural correlation impossible")
        if not self.market:
            p.append("market missing")
        if self.line is None:
            p.append("line missing")
        for f in ("prob_over", "prob_under", "confidence",
                  "prob_uncalibrated", "feature_completeness"):
            v = getattr(self, f)
            if v is not None and not (0.0 <= v <= 1.0):
                p.append(f"{f}={v} outside [0,1]")
        if (self.prob_over is not None and self.prob_under is not None
                and abs(self.prob_over + self.prob_under - 1.0) > 1e-6):
            # Not pedantry: an un-de-vigged pair sums above 1, and treating the
            # book's margin as model probability overstates edge on every row.
            # That is bug #1 in mlb-edge's own README; DIP refuses to re-import
            # it through a side door.
            p.append(f"prob_over + prob_under = "
                     f"{self.prob_over + self.prob_under:.4f}, expected 1.0")
        if self.confidence is not None and self.confidence_kind is None:
            p.append("confidence set without confidence_kind")
        return p

    def to_row(self, first_seen_at: str) -> dict:
        d = dataclasses.asdict(self)
        d.pop("raw")
        d["id"] = self.id
        d["first_seen_at"] = first_seen_at
        d["raw"] = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return d
