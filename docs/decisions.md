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

## Several questions in one request

The API answers any number of named questions about one `state` in a single
request, under the names you gave them, as long as the request fits the
model's context limits (https://docs.typesafe.ai/api/system-one.md,
https://docs.typesafe.ai/models.md). `decide_many` sends them together:
one round trip and one bill instead of one per question.

```python
out = llm.decide_many(
    ticket,
    questions={
        "billing": {"ask": "Is this message about billing?"},
        "tone": {"ask": "What is the tone?", "labels": ["angry", "calm", "excited"],
                 "min_p": 0.8},
        "urgency": {"ask": "How urgent is it?",
                    "scale": ["Can wait", "This week", "Today"]},
    },
    workload="ticket_triage",
    min_p=0.6,        # the default for questions without their own
)
out["tone"].label, out["tone"].p, out["tone"].abstained
```

Each question takes the same arguments as `decide` (`ask`, then `labels` or
`scale`, and optionally its own `min_p`). The result is a dict of name ->
`DecisionResult`, in the order given; each result's `question` is its name.
All results of one request share its `call_id`, `latency_ms`, `tokens_in`
and `cost_usd`: those belong to the request, so do not add them up.

Two request options exist only on `decide_many`, so `decide` keeps exactly
its 0.25.0 signature (to use them for one question, pass one question):

- `timeout=` seconds for this request (default 20).
- `long_context=True` sends the vendor's
  `TypeSafe-Danger-Zone-Long-Context: true` header, which raises the
  per-request and per-question context to 262k tokens; the vendor says it is
  slower and less accurate, so use it only when the input needs it
  (https://docs.typesafe.ai/models.md, "Long context").

## What is recorded

One `calls` row per request, with the usual columns and five more
(migration 0030):

| Column | `decide` | `decide_many` |
|---|---|---|
| `kind` | `decide` (NULL on every ordinary call) | `decide` |
| `label` | the chosen label | NULL |
| `p_top` | its probability | NULL |
| `abstained` | 1 below `min_p`, else 0 | NULL |
| `result_json` | the typed answer: shape, `min_p`, label, p, all probabilities, confidence, score | `{"questions": {name: <the same, plus abstained>}}` |

A `decide_many` row leaves `label`, `p_top` and `abstained` NULL even with one
question, so a row is one or the other by its shape: NULL `label` and a
`questions` key mean several answers, read from `result_json`. Tokens and
cost are per request, so they stay on the one row.

The question text and the input are not stored, only their hash
(`prompt_hash`), unless the workload opted in to sampling for grading (see
below). Cost: the vendor bills input tokens only, so `tokens_in`
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
the latest one per call wins. A `decide_many` result's verdict names its
question: `out["tone"].verdict("angry")`, or later
`record_verdict(llm, call_id, "angry", question="tone")`, writes field
`verdict:tone`. Then:

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

Without `question=`, `decision_calibration` reads `decide` rows. Pass
`question="tone"` to read that question across a workload's `decide_many`
rows.

## Grading against an LLM

Verdicts can also come from an LLM asked the same question. Then
calibration answers "how often does the decision model agree with the LLM,
and is its confidence calibrated against it?". It is a function you call,
not a worker:

```python
from somm.decide import grade_decisions

# Once: the opt-in that stores sampled inputs, and the grader route.
llm.enable_shadow("ticket_route", "openrouter", "<a model id>", sample_rate=0.05)

# Any time later (a cron line, a notebook, the end of a batch job):
grade_decisions(llm, "ticket_route", limit=20)
# {"graded": 20, "skipped": 0, "errors": [], "grader": "openrouter/<a model id>",
#  "cost_usd": 0.0123, "grader_call_ids": [...], "stopped": None}
```

As a one-liner:
`python -c 'import somm; from somm.decide import grade_decisions; print(grade_decisions(somm.llm(project="support"), "ticket_route"))'`.
It is not wired into `somm loop run`: the loop's workers run without a
client, and the grader needs one.

How it behaves:

- **Sampling.** A decide row's input (the state and the questions) is stored
  only for a workload with a shadow config, at its `sample_rate` (default
  2%), chosen by call id, and never for a private workload: the same gate,
  and the same `samples` table, as every other call's capture. Each run
  grades at most `limit` stored rows, newest first (default: the config's
  `max_grades_per_run`, 20).
- **Never twice.** A row with any verdict (yours or the grader's), or with a
  successful grader call, is skipped.
- **One LLM call per row.** All of a row's questions go in one
  `generate_structured` call whose schema allows only that question's labels,
  on the same workload, so its data boundary and budget gate apply as to
  any call. The route is `provider=`/`model=`, defaulting to the shadow
  config's gold route. Grading stops for the day at the config's
  `budget_usd_daily` of grading spend.
- **What it writes.** The grader's call is recorded with
  `observation_role = 'shadow_judge'` and `source_call_id` = the decide row,
  so it stays out of serving rollups and its cost is visible. Its answers
  become verdicts (`verdict`, or `verdict:<question>`), and its call id is
  written beside them as a `verdict_by` row.
- The shadow-eval worker grades text calls only: it skips decide rows and
  grading calls.

## Reading the comparison

`decision_economics` puts a decision workload's agreement and cost beside
the LLM calls tied to it (the escalations, `parent_call_id` = a decide row,
and the grader's calls):

```python
from somm_core.read import decision_economics

decision_economics(llm.repo.db_path, "support", "ticket_route", since_days=30)
# The shape (numbers invented):
# {"workload": "ticket_route", "since_days": 30,
#  "decide_calls": 5000, "questions": 5000, "abstained": 400, "abstained_share": 0.08,
#  "verified": 250, "agreement": 0.93,
#  "decide_cost": {"mean_usd": 0.0000168, "known": 5000, "unknown": 0},
#  "escalations": {"calls": 400, "mean_usd": 0.0021, "known": 400, "unknown": 0},
#  "grader": {"calls": 250, "mean_usd": 0.0019, "known": 250, "unknown": 0},
#  "per_1000_inputs": {"with_decide_usd": 0.1848, "llm_only_usd": 1.9,
#                      "llm_only_basis": "grader"}}
```

`with_decide_usd` is 1,000 x (mean decide cost + escalation rate x mean
escalation cost); `llm_only_usd` is 1,000 x the mean cost of the grader
answering the same input (or of the escalations when there is no grader,
which are the hard inputs only: `llm_only_basis` says which). A mean over no
known costs is `"unknown"`, and so is any per-1,000 figure that needs it.
`agreement` counts every verdict, yours and the grader's.

## Moving a direct caller onto decide

A caller that posts to the API itself keeps its own key lookup, HTTP code,
error mapping and price constant, and writes nothing to the ledger. Before:

```python
body = {"model": "jev-latest", "state": {"task": task, "project": facts},
        "questions": {f"c{i}": {"type": "noul", "instructions": q(c)}
                      for i, c in enumerate(candidates)}}
answers = post_json(API_URL, body, key, timeout=10)["answers"]
scores = {c["ref"]: answers[f"c{i}"]["noul"] for i, c in enumerate(candidates)}
cost = usage["input_tokens"] / 1e6 * 0.042
```

After:

```python
out = llm.decide_many({"task": task, "project": facts},
                      questions={f"c{i}": {"ask": q(c)} for i, c in enumerate(candidates)},
                      workload="skill_judge", timeout=10)
scores = {c["ref"]: out[f"c{i}"].probabilities["yes"] for i, c in enumerate(candidates)}
```

The key comes from `TYPESAFE_API_KEY`, the cost is on the ledger, errors are
somm's typed errors, and the workload can be calibrated and graded. The wire
`noul` value is `probabilities["yes"]`; a wire `choice` question is
`labels=` (its `criteria` dict), a wire `score` question is `scale=` (its
`criteria` list).

## Limits

- **One provider, pinned.** TypeSafe System One is the only decision
  provider; there is no fallback or routing across decision models.
- **Text and JSON only.** No images, no `box` questions, no Noul `criteria`
  (the optional yes/no descriptions), and `ask` is text (the API also takes
  an object or array as instructions).
- **`timeout` and `long_context` are on `decide_many` only.**
- **No retries.** A 429 or 529 raises `SommRateLimited` /
  `SommTransientError` after the row is written; the vendor's SDKs retry
  with backoff (https://docs.typesafe.ai/api/errors.md), somm does not.
- **No pre-call hooks or short-circuit cache.** Post-call hooks fire as for
  any call.
- **Not verified against the live API.** The adapter is built from the
  vendor's documented request and response shapes
  (https://docs.typesafe.ai/api/system-one.md), status codes
  (https://docs.typesafe.ai/api/errors.md) and the long-context header
  (https://docs.typesafe.ai/models.md) and tested against fakes; no live
  call was made while building it.
