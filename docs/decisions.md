# Decisions

`llm.decide` asks a decision model one typed question and gets back a label
with a probability, not text. The call is recorded on the same ledger as
every LLM call: workload, provider, model, latency, tokens, cost, data tier.

(Not to be confused with the `decisions` table, which holds sommelier model
choices. Decision calls are `calls` rows with `kind = 'decide'`.)

## When to use it instead of an LLM call

Use `decide` when the answer is one of:

- whether a condition holds (yes/no),
- one label from a fixed set (route, category, tone),
- a level on an ordered scale (urgency, severity, quality).

An LLM prompted to "answer with one word" and parsed afterwards does the same
job slower and dearer, and its "confidence" is whatever text it chose to
write. A decision model returns a probability for every label. Keep using
`generate` / `generate_structured` when you need text, extraction of values
that are not in a fixed set, or reasoning you will read.

## The three question shapes

```python
import somm

llm = somm.llm(project="support")
ticket = "I was charged twice for my subscription this month."

# Yes/no: `ask` alone. Labels are "yes" and "no".
r = llm.decide(ticket, ask="Is this message about billing?", workload="ticket_billing")
r.label, r.p          # ("yes", 0.98)

# Choice: `labels`, a list of names or a dict of name -> when it applies.
r = llm.decide(
    ticket,
    ask="What is the tone of this message?",
    labels={"angry": "An upset or hostile message",
            "calm": "A neutral or polite message",
            "excited": "An enthusiastic or eager message"},
    workload="ticket_tone",
)
r.label, r.p, r.probabilities   # ("calm", 0.8, {"angry": 0.15, "calm": 0.8, "excited": 0.05})

# Score: `scale`, ordered level descriptions, lowest first (2 to 10 levels).
r = llm.decide(
    ticket,
    ask="How urgent is this message?",
    scale=["Can wait", "Needs attention this week", "Needs attention today"],
    workload="ticket_urgency",
)
r.label, r.p, r.score   # ("Needs attention today", 0.8, 1.7)
```

`context=` adds structured facts beside the text (the request's state
becomes `{"text": ..., "context": ...}`). `model=` pins a model id such as
`jev-1.14.0`; the default is the vendor alias `jev-latest`, which moves when
the vendor ships a release. Pin a versioned id once you have tuned `min_p`
against it.

The result (`somm.decide.DecisionResult`) is a plain value:

| Field | Meaning |
|---|---|
| `label` | the chosen label (a choice name, a scale level, or `yes` / `no`) |
| `p` | the probability the model gave `label` |
| `probabilities` | every label's probability |
| `abstained` | `p` fell below your `min_p` |
| `confidence` | the vendor's concentration statistic (choice and score only) |
| `score` | the expected level index, probability-weighted (score only) |
| `provider`, `model`, `call_id`, `latency_ms`, `tokens_in`, `cost_usd` | as for any call |

Errors raise somm's typed errors after the row is written: `SommAuthError`,
`SommBadRequest`, `SommRateLimited`, `SommTransientError` (overloaded or a
malformed answer), `SommTimeout`; `SommProviderNotConfigured` when no key
is set; `SommPrivacyViolation` when the workload's data boundary does not
admit the route; `SommBudgetExceeded` from the daily budget gate.

## Abstain and escalate

`min_p=` marks a result abstained when `p` is below it. The label is still
reported and the row records `abstained = 1`. Nothing else happens on its
own: somm does not escalate for you. Escalating is two lines, on the same
workload so both calls are counted together:

```python
r = llm.decide(ticket, ask="Which team owns this?", labels=["billing", "auth", "other"],
               workload="ticket_route", min_p=0.8)
if r.abstained:
    value, result = llm.generate_structured(ticket, schema=Route, workload="ticket_route",
                                            parent_call_id=r.call_id)
```

`parent_call_id` links the LLM row to the decision it rescued.

## What is recorded

One `calls` row per decision, with the usual columns and five more
(migration 0030):

| Column | Value |
|---|---|
| `kind` | `decide` (NULL on every ordinary call) |
| `label` | the chosen label |
| `p_top` | its probability |
| `abstained` | 1 below `min_p`, else 0 |
| `result_json` | the typed answer: shape, `min_p`, label, p, all probabilities, confidence, score |

The question text and the input are not stored, only their hash
(`prompt_hash`). Cost: the vendor bills input tokens only, so `tokens_in`
carries the meter and `cost_usd = tokens_in x price_in`. The price is in the
bundled price table (`$0.042` per million input tokens, output free, per
https://docs.typesafe.ai/models.md), so cost is known, not
`unknown:no-price`. A response naming a model the table does not list is
recorded with cost unknown.

The data tier is classified like any route. TypeSafe is not in the shipped
tier catalog, so it is `unvetted`: a workload with a stricter boundary is
refused (and the refusal recorded) until you declare the route in
`~/.somm/boundary.toml`. See [data-boundary.md](data-boundary.md).

## Calibration

A decision model's probabilities are only useful if they are calibrated on
your data: of the decisions it made at `p = 0.8`, about 80% should be right.
The vendor says its models are trained for calibrated decisions; somm lets
you check.

Record a verdict whenever you learn the right label:

```python
from somm.decide import record_verdict

r.verdict("auth")                      # on the result
record_verdict(llm, call_id, "auth")   # later, by call id
```

A verdict is a `call_updates` row (field `verdict`, value the right label);
the latest one per call wins. Then:

```python
from somm_core.read import decision_calibration

decision_calibration(llm.repo.db_path, "support", "ticket_route", since_days=30)
# The shape (numbers illustrative):
# {"n": 412, "brier": 0.081, "ece": 0.034, "accuracy": 0.91,
#  "bins": [{"lo": 0.9, "hi": 1.0, "n": 300, "mean_p": 0.95, "accuracy": 0.96}, ...],
#  "decisions": 5000, "unverified": 4588, "workload": "ticket_route", "model": None}
```

- **Brier score**: the mean of `(p - correct)^2` over decisions with a
  verdict. 0 is perfect; answering 0.5 every time scores 0.25.
- **ECE** (expected calibration error): `p` split into 10 equal bins; the
  gap between mean `p` and the share right in each bin, weighted by the
  bin's size. Read the `bins` rows to see where the model is over- or
  under-confident, and set `min_p` where accuracy in the bins above it is
  what you need.

The pure function is `somm_core.calibration.calibration(pairs, bins=10)`
over `(p, correct)` pairs. Pass `model=` to compare versions. Abstained rows
count: calibration belongs to the model, not to your threshold.

## Limits

- **One provider, pinned.** TypeSafe System One is the only decision
  provider; there is no fallback or routing across decision models.
- **One question per call.** The API accepts many questions over one state
  in one request; `decide` sends one. A caller fanning out many questions
  pays one round trip each.
- **Text and JSON only.** No images and no `box` questions; no long-context
  header.
- **No pre-call hooks or short-circuit cache.** Post-call hooks fire as for
  any call.
- **Not verified against the live API.** The adapter is built from the
  vendor's documented request and response shapes
  (https://docs.typesafe.ai/api/system-one.md) and status codes
  (https://docs.typesafe.ai/api/errors.md) and tested against fakes; no
  live call was made while building it.
- **No shadow evaluation.** Decision rows are not graded by the shadow-eval
  worker; verdicts come from you.
