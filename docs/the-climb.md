# The climb

A workload is not always a model call. In a chipsocket "socket", rules, lexical
cuts and a cache answer first and only the remainder reaches a model. somm's ledger
holds the model calls only, so it cannot say how much of a workload never needed
one. `somm climb` adds that picture by reading the receipts chipsocket already
writes. It is read-only: it never writes to the receipts or to the ledger, and it
imports nothing from chipsocket.

## What it reads, and from where

```
somm climb [PATH] [--workload NAME] [--since DAYS] [--project P] [--json]
```

`PATH` is a receipts directory or one file; if omitted, `SOMM_CLIMB_RECEIPTS`.
Plain `.jsonl` and gzipped `.jsonl.gz` are both read, streamed line by line.
The window defaults to 30 days, by each decision's own start time.

Schema `chip.socket-receipt/v0alpha1`, one JSON line per decision. Fields used:

| Field | Used for |
|---|---|
| `schema` | other versions and non-receipt lines are skipped and counted; a file whose first line is not a receipt (a harvest file) is skipped whole |
| `socket` | `--workload` filter (exact match) |
| `answeredBy` | the rung: `code:<name>`, `model:<name>`, `baseline:<name>`, or none (`exhausted`) |
| `timing.startedAt` | the window |
| `attempts[].meta.call_id` | on model-answered decisions, the somm call id, when the gateway recorded one |

Unknown fields are ignored. The receipt schema is marked as changing, so a new
version is skipped and counted rather than guessed at.

**No decision is counted twice.** chipsocket gzips a finished day in place by
writing the whole `.gz` and only then deleting the plain file. If both are present
(a crash in between) the plain file is the complete one, so the `.gz` twin is
skipped and counted. Each receipt is one decision.

## The model rung and the join

For decisions answered by a model, the report adds calls, tokens, known cost and
mean latency from the somm ledger (opened read-only), and says which join it used:

1. **call id**, when the model-answered receipts carry `meta.call_id`: exactly those
   ledger rows, and how many of the receipts had an id and matched.
2. **exact workload name**, when they do not and `--workload NAME` is given: the
   production calls of the somm workload with that name in the window. This is a
   whole-workload total, not receipt-to-call matching; shadow calls are excluded.
3. **none**, otherwise. The report says so and shows no model cost.

Calls whose price is unknown are counted, not priced at zero.

## Reading the three derived lines

- **answered without a model**: decisions answered by `code:` members divided by all
  decisions. A cache that replays an earlier model answer is a code member, so it
  counts here; that is the point of the climb. `baseline:` and `exhausted` decisions
  are in the denominator only.
- **model cost per 1,000 decisions**: known model cost / all decisions x 1,000. What
  the workload actually costs per thousand, rungs and all.
- **per 1,000 if all went to model**: known model cost / model calls x 1,000. What
  it would cost if every decision paid the observed per-call price.

The last two need a join and are blank without one. With a partial join (some
model receipts without an id) the cost is a floor.

## What it does not know

- The cost of running the rules themselves (compute, a local model, the engineer).
  Rungs other than the model are shown as free; they are not.
- Receipts written by other tools, or by sockets in other directories.
- Which ledger calls belong to which decision under a workload-name join.
- A socket that declines to somm's provider chain (the workload-handler setup)
  writes an `exhausted` receipt and the model call that follows has no receipt of
  its own; read that rung as "went on to the model", not as failure.
