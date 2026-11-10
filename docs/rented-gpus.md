# Rented GPUs

somm can show what you spent renting GPUs, next to what it already records for
API and seat calls. It does this by reading the receipt files that
gipper (a separate batch GPU rental tool)
writes. somm never rents, starts or stops anything; this page is about reading.

## What somm reads, and from where

gipper appends one JSON line per event to monthly files, schema
`gipper.receipt/v1`: `{schema, id, ts, kind, job, refs, payload}`. somm looks in,
first match wins:

1. `--path DIR` on the command,
2. `SOMM_GIPPER_RECEIPTS`,
3. `$GIPPER_DATA/receipts`,
4. `~/.local/share/gipper/receipts`.

It reads these fields and nothing else:

| Receipt | Field | Used for |
|---|---|---|
| any | `job`, `ts`, `schema` | grouping and dating; other schema versions are skipped and counted |
| `estimate`, `spend`, `run.end` | `refs.name`, `refs.workload` | the job's name and workload |
| `pod.create` | `payload.gpu` | GPU type |
| `spend` | `payload.usd`, `payload.seconds`, `payload.provider`, `payload.pod` | gipper's computed cost and lifetime, per machine |
| `spend.billed` | `payload.usd`, `payload.provider`, `payload.pod` | the provider's own charge; wins over `spend` for that machine |
| `run.end` | `payload.outcome` | status (`succeeded`, `failed`, `error`, `capped`, ...) |
| `reap` | `job` | status `reaped`, when the receipt names a job |

A job that rented several machines is one row whose cost is the sum. A job with
only an `estimate` (nothing rented or ended) is not counted; the count of those
is shown. A `reap` receipt that names no job cannot be attributed and is only
counted. Unknown fields and unknown receipt kinds are ignored.

## The commands

```
somm gpu spend [--since DAYS] [--path DIR] [--project P] [--json]
```

Totals for the window (default 30 days), by UTC day (the day a job ended,
which is how gipper dates its own spend), by workload, and the jobs that did
not succeed with what they cost. `somm doctor` prints one `gipper:` line when a
receipts directory exists, and nothing when it does not.

## Joining the two ledgers by workload name

gipper receipts and somm calls share no id. The only join is the name: give
your gipper job the `workload` of the somm workload it serves. Where a gipper
workload equals a registered somm workload name in the current project's
ledger, exactly, `somm gpu spend` prints the ledger's spend for that workload in
the same window beside the GPU spend. There is no fuzzy matching; a typo shows
as no match. The two figures are not summed: one is API or seat spend, the
other is rented hardware.

## Calling a model served on a rented box

somm can call an OpenAI-compatible server (vLLM and similar) that you started on
a rented machine. This is documentation of what exists, not a feature of this
page; it needs no change to somm.

Simplest: repoint the built-in `openai` provider. It already takes a base URL.

```bash
export SOMM_OPENAI_BASE_URL=http://BOX_HOST:8000/v1
export OPENAI_API_KEY=the-key-you-gave-the-server
export SOMM_OPENAI_MODEL=the-served-model-name
```

This replaces api.openai.com for that process; you cannot have both. To keep
both, or to give the box its own provider name, register a second
OpenAI-compatible endpoint with a small plugin package, using the
`somm.providers` entry point ([plugins](plugins.md)). The base class is
`OpenAICompatProvider` in `packages/somm/src/somm/providers/_openai_compat.py`
and the contract is `ProviderSpec` in `providers/registry.py`. There is no
configuration-only way to add a second endpoint. A sketch (not exercised against
a live server):

```python
import os
from somm.providers._openai_compat import OpenAICompatProvider
from somm.providers.registry import ProviderSpec

class RentedBox(OpenAICompatProvider):
    name = "rented-box"

def _make(config, tracker):
    url = os.environ.get("RENTED_BOX_URL")
    if not url:
        return None
    return RentedBox(api_key=os.environ.get("RENTED_BOX_KEY", "none"), base_url=url,
                     default_model=os.environ.get("RENTED_BOX_MODEL", ""))

provider_spec = ProviderSpec(name="rented-box", factory=_make, default_order_rank=None)
```

with `[project.entry-points."somm.providers"] rented-box = "yourpkg:provider_spec"`.
`default_order_rank=None` keeps it out of default routing; ask for it by name.

Two things to know:

- **Cost.** somm has no price for a model you host yourself, so those calls are
  recorded with `cost_source = "unknown:no-price"` and a cost of 0 (unless the
  served model name happens to match a priced model in somm's intel, in which
  case it is priced as that model, which is wrong for your hardware). The GPU
  rental cost lives only in gipper's receipts; `somm gpu spend` is how you see
  it.
- **Data boundary.** Such an endpoint's tier is `unvetted` until you declare it.
  Declare it by host in `~/.somm/boundary.toml`, with a note stating why you trust
  it ([data boundary](data-boundary.md#writing-boundarytoml)):

  ```toml
  [[route]]
  host = "BOX_HOST"
  tier = "local"
  note = "my rented box, torn down after each job"
  ```

  Choose the tier your arrangement with the host actually supports; a box on a
  public IP at a marketplace provider is not the same as a machine you own.

Limits that come from gipper, not somm: it runs one command per rental, blocks
until it ends, and tears the machine down at `max_hours`, so a served endpoint
is short-lived and has no readiness signal. somm will see connection errors and
cool the provider down as it does for any provider.
