# The `somm` command line

`somm --help` lists the commands; `somm <command> --help` lists a command's
flags. This page records the conventions that hold across all of them.

## `--project`

`--project NAME` is accepted before the command, between a group and its
subcommand, or after the subcommand; the last one given wins:

```bash
somm --project keel inbox list
somm inbox --project keel list
somm inbox list --project keel
```

Without the flag, `SOMM_PROJECT` is used, then the project of the current
directory. `SOMM_PROJECT` is treated exactly like `--project`, including
looking up the ledger registered for that project when you run from another
checkout. Commands that act on every ledger (`ledgers`, `fleet`, `plugin`,
`mcp`, `skill`) have no project of their own.

## `--json`

Commands that print a report or a list take `--json` and then print JSON
only: `status`, `cache-advice`, `generate`, `bench`, `compare`, `frontier`,
`workload list`, `prompt list`, `boundary routes|audit`, `eval import|run`,
`campaign run`, `inbox`, `loop run|status`, `ledgers`, `fleet status`,
`spend`, `gpu spend`, `plans`, `alias list`. Lists print an array (an empty
list prints `[]`); `prompt list --json` leaves out prompt bodies, which
`prompt show` prints.

Commands that print a single line, a diff, a stream or a mixed set of
sections (`tail`, `doctor`, `plugin list`, `prompt diff`, `workload show`,
`prune`, `backfill-costs`, `drain-spool`, `mirror sync`) do not have it.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | The command ran and the outcome was a failure: a failed call (`generate`), a `bench` or `compare` in which no call succeeded, a failing `eval run` or `campaign run` gate, any workload with a data-boundary violation (`boundary audit`), an unhealthy `doctor` (or `--max-exhausted-rate` exceeded), a loop worker error, an invalid `plans.toml`, or a missing prerequisite: no telemetry database (`backfill-costs`, `drain-spool`, and `alias seed|set`), no receipts directory (`gpu spend`), no harness on PATH (`mcp`, `skill`). |
| `2` | Bad usage, or something you named does not exist: an unknown workload, prompt version, recommendation or project; a value out of range; also what argparse returns for an unknown flag. |
| `66` | `--json` only: an input file was not found. |
| `77` | `--json` only: permission denied. |

With `--json`, an unhandled exception is printed to stderr as
`{"ok": false, "error": {"type", "message", "exit_code"}}` and exits with the
code above (`2` for a bad value, `66`, `77`, otherwise `1`). Without `--json`
it is a normal Python traceback and exit `1`.

`compare` exits `0` when at least one model answered, even if others failed
or were refused by the data boundary; the failures are in the output.
`bench` exits `0` when at least one measured call succeeded.
