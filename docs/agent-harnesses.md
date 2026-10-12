# Coding-agent harnesses

Somm exposes a provider-neutral API for executing one autonomous coding-agent
attempt with Claude Code, Codex, or OpenCode. This is distinct from
`SommLLM.generate()`: an LLM call returns model output, while a harness can use
tools, edit a workspace, emit events, and resume a native agent session.

```python
from pathlib import Path

from somm import harnesses
from somm.harnesses import HarnessRequest

request = HarnessRequest(
    prompt="Fix the failing parser tests and explain the change",
    cwd=Path("~/src/my-project").expanduser(),
    capture_dir=Path("./run-001"),
    model=None,                    # let the harness choose its default
    reasoning_effort=None,         # e.g. "medium" or "high" when supported
    prompt_via_stdin=False,        # use for large Codex evidence packets
    allow_unsafe=False,            # permission bypass is always explicit
    correlation_id="task-001",
)

result = harnesses.run("codex", request, timeout=1800)
print(result.outcome, result.final_text, result.session_id)
```

Use `harnesses.start()` when an outer runner needs to supervise the process
itself. It returns a `HarnessHandle` containing the `Popen` object and durable
stdout/stderr capture paths. After the process finishes, call
`harnesses.inspect()` to obtain a normalized `HarnessResult`.

```python
handle = harnesses.start("claude-cli", request)
# Poll, stream, cancel, or enforce an activity timeout in the outer runner.
exit_code = handle.proc.wait()
handle.close_captures()
result = harnesses.inspect(
    "claude-cli",
    handle.stdout_path,
    handle.stderr_path,
    exit_code=exit_code,
    correlation_id=request.correlation_id,
)
```

## Stable names and capabilities

| Name | Executable | Resume | Turn limit | Agent selection | Reasoning effort | Stdin prompt | Reports cost | Reports model |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `claude-cli` | `claude` | yes | yes | no | yes | no | yes | yes |
| `codex` (`codex-cli` alias) | `codex` | yes | no | no | yes | yes | no | no |
| `opencode` | `opencode` | yes | no | yes | no | no | yes | yes |
| `pi` | `pi` | yes | no | no | yes | yes | no | yes |

`harnesses.capability_profile(name)` returns this row as a versioned record
(plus `available` and the installed CLI version) so an outer runner can make an
admission decision without branching on the harness name.
`harnesses.validate_request(name, request)` fails closed: a request field the
harness cannot honor — `max_turns` on Codex, `agent` on Claude, `session_id` on
a harness without resume — is rejected before launch rather than silently
dropped. `start()` and `run()` call it for you.

## Result contract (envelope 2)

`HarnessResult` is versioned (`harnesses.ENVELOPE_VERSION`, currently 2) and
changes are additive: a field keeps its name and meaning once published, and an
older consumer can read a newer result. `result.as_dict()` is the JSON view.
`None` always means "the harness did not say", never a guess.

- `outcome`: one of the portable terminations below;
- `final_text`: the final agent-facing response;
- `session_id`: the native session/thread identifier for resumption;
- `model`: the model that actually served the attempt, as the harness
  reported it (Claude's `modelUsage`, OpenCode's `metadata.model`, Pi's
  message `model`; Codex does not report one);
- `harness_version`: the CLI's version (from the capture when it writes one,
  else `<cli> --version`, cached per process);
- `usage`: the harness's own token block, plus portable aliases where the
  native names differ (`reasoning_tokens`, `cache_read_input_tokens`,
  `prompt_tokens_total` for Claude's three-way prompt split);
- `billing_usage`: a per-step rollup where steps are billed separately
  (OpenCode);
- `cost_usd`: what the harness said the attempt cost (Claude
  `total_cost_usd`, OpenCode step costs, Pi `usage.cost.total`);
- `duration_ms`, `turns`: wall time and agent turns, from the harness or the
  caller's timing;
- `retry_after_s`: seconds until a limit lifts, when a rate-limit or seat
  message said so (`retry-after: N`, or Claude's "resets 3pm
  (America/New_York)");
- `terminal_event`: the untouched native terminal event for diagnostics;
- `correlation_id`: the caller's task/trace/job identifier.

Outcomes: `completed`, `turn_limit`, `context_limit`, `refused`,
`rate_limit`, `network_error`, `provider_error`, `auth_error`,
`budget_exhausted`, `input_error` (the request itself was rejected; the same
input will fail again), `agent_blocked` (the agent ended with
`<stuck>reason</stuck>`; retrying the same prompt is futile), `interrupted`
(no terminal event and an unclean exit — a supervisor usually knows why),
`failed`, `unknown`.

Unknown remains distinct from failure: a clean exit with no terminal event is
`unknown`, because the outer supervisor may have better evidence.

## The attempt in the ledger

A coding-agent attempt is the most expensive call a project makes, and until
0.17 it was the one call the ledger never saw. `record_attempt()` writes one
`calls` row per finished attempt with `origin="harness"`, `provider` set to
the harness name (which is also the seat it ran on), the served model, the
reported cost (`cost_basis="reported"`) or a computed one, the session id, and
the caller's correlation id — so a runner's receipt and the ledger row name the
same attempt and `somm status` shows seat spend beside API spend.

```python
result = harnesses.run("claude-cli", request, timeout=1800, record=True)
# or, when the runner owns the process:
result = harnesses.inspect("claude-cli", out, err, exit_code=rc,
                           started_at=handle.started_at, record=True,
                           request=request, project="my-project")
# or explicitly:
harnesses.record_attempt(result, request=request, project="my-project")
```

Recording is opt-in because a library must not write to a project's ledger
behind a runner's back; a runner that wants the row asks for it once.

## Process environment

`start()` launches the CLI with the process environment minus the seat's API
key (`ANTHROPIC_API_KEY` for Claude Code, `OPENAI_API_KEY` for Codex): a
subscription attempt must not quietly bill the API account and attribute the
spend to the wrong plan. `HarnessRequest.env` layers extra variables on top
(a workspace path, a feature flag); `start_new_session=True` puts the child in
its own session so a supervisor can cancel or reap the whole tree.

## Deliberate boundary

The harness API executes and interprets exactly one attempt. It does **not**:

- maintain a persistent job queue;
- retry, back off, or choose another harness;
- decide whether an inactive process is stuck;
- create worktrees, verify changes, merge, or release work;
- escalate failures to a human.

Those are task-runner responsibilities. This keeps Somm reusable by Fab and
other project runners without making Somm depend on any of them.

## Permission safety

`allow_unsafe=False` is the default. When true, adapters translate the request
to each CLI's explicit permission-bypass flag. Only an outer runner that has
already established workspace isolation and authorization should enable it.

`executable=` can pin a specific CLI path when PATH is not authoritative (for
example, when a maintained per-user installation should win over an older
system binary).

`prompt_via_stdin=True` keeps large prompts out of the process argument list.
The Codex adapter translates it to `codex exec -` and writes the prompt to stdin.

`extra=` passes harness-specific arguments through unchanged. Treat it as an
escape hatch; portable integrations should prefer typed request fields and
capability checks.
