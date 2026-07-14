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

| Name | Executable | Resume | Turn limit | Agent selection |
|---|---|---:|---:|---:|
| `claude-cli` | `claude` | yes | yes | no |
| `codex` (`codex-cli` alias) | `codex` | yes | no | no |
| `opencode` | `opencode` | yes | no | yes |

Use `harnesses.available()` for installed canonical names and
`harnesses.get(name).capabilities` before relying on an optional feature.

## Result contract

Somm normalizes native JSON/NDJSON streams into:

- `outcome`: completed, turn/context limit, refusal, rate limit, network,
  provider, authentication, generic failure, or unknown;
- `final_text`: the final agent-facing response;
- `session_id`: the native session/thread identifier for resumption;
- `usage`: tokens reported by the harness when available;
- `terminal_event`: the untouched native terminal event for diagnostics;
- `correlation_id`: the caller's task/trace/job identifier.

Unknown remains distinct from failure. An interrupted process may have no
terminal event, and an outer supervisor may have better evidence (for example,
that its activity watchdog killed the attempt).

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

`extra=` passes harness-specific arguments through unchanged. Treat it as an
escape hatch; portable integrations should prefer typed request fields and
capability checks.
