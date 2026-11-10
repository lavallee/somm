# Ledgers on this machine

Every project writes its calls to its own ledger, a `calls.sqlite` file. When
cross-project mirroring is on (the default), each call is also copied to the
mirror at `~/.somm/global.sqlite`, which is what `--global` commands read.
`~/.somm/registry.json` records which project writes to which ledger.

## Seeing them

```bash
somm ledgers                      # every registered ledger and the mirror
somm ledgers --scan ~/Projects    # also calls.sqlite files nobody registered
somm ledgers --json
```

Use this instead of `find ~ -name calls.sqlite`. It opens every file
read-only and changes nothing. `--scan` searches only the directories you
name, never your whole home directory. It does not descend into `.git`,
`node_modules`, or virtualenvs.

```
projects                       size schema      rows oldest     last call        clients (30d)        mirrored through  flags / path
alpha                        1.4 GB     25   1500000 2026-04-18 2026-11-09 16:32 0.19.3, pre-0.17     2026-09-19 11:59  mirror-behind  ~/work/alpha/.somm/calls.sqlite
beta,beta-scratch            2.9 MB     29      2078 2026-10-21 2026-10-29 02:33 0.23.0               —                 shared not-mirrored  ~/work/beta/.somm/calls.sqlite
probe                             —      —         —                                                                     missing  /tmp/probe/.somm/calls.sqlite
(mirror)                     1.6 GB     29           2026-06-01                                                          ~/.somm/global.sqlite

3 registered ledger file(s), 1.4 GB: 1 missing, 1 shared by several names, 0 in a temp directory; installed somm 0.23.0 (schema 29)
mirror: 2 ledger(s) hold calls the mirror does not; `somm mirror sync --dry-run` counts them, without --dry-run copies them
stale registrations: `somm ledgers forget PROJECT [--path PATH] --dry-run`
retention: nothing prunes on a schedule. Largest ledger since 2026-04-18:
  SOMM_DB_DIR=~/work/alpha/.somm somm prune --older-than 90 --dry-run   (then without --dry-run; --vacuum gives the space back)
  somm prune --global --older-than 90 --dry-run   (the mirror)
```

The flags are:

| flag | meaning |
|---|---|
| `missing` | the registered path no longer exists |
| `shared` | several project names are registered to this one file |
| `temp` | the file is under a temporary directory and will vanish |
| `mirror-behind` | the ledger has calls newer than the mirror's last call for that project |
| `not-mirrored` | the ledger has calls and the mirror has none for that project (or the project runs with `SOMM_CROSS_PROJECT=0`) |
| `mirror-has-more` | the mirror holds more of the project's calls than its registered ledgers; it probably writes to a ledger that is not registered (`--scan` finds it) |

The `schema` column is the highest migration applied to that file. A file
below the installed schema has not been opened by a current somm.

## Keeping the registry right

A project that writes to several ledgers (a scratch run, a moved checkout)
keeps every path in the registry. Older releases kept only the last one.

```bash
somm ledgers forget PROJECT [--path PATH] [--dry-run]   # remove a registration; never deletes a file
somm ledgers adopt PATH [--project NAME] [--dry-run]    # register an existing ledger
```

Without `--project`, `adopt` registers the file under every project name that
its calls carry. Adopting never changes which ledger `--project NAME` reads.
That stays the project's most recent path.

## Fleet status

```bash
somm fleet status [--json]
```

This prints one line per registered project, problems first. Each line shows
the somm versions that wrote to the project's ledgers in the last 30 days and
whether they are behind the installed version. It also shows the ledger's
schema, the last successful loop worker, the share of calls with an unknown
cost, and the share with no `client_version` (rows from clients before 0.17).
`somm doctor --global` prints the same lines in its fleet section.

## Mirror gaps

The mirror can fall behind a ledger for three reasons:

- the mirror was locked, full, or unreadable when the batch was written. Through
  0.23 the writer dropped such a batch silently. It now retries a busy
  mirror, counts what it could not write, and says so once on stderr;
- rows replayed from the spool (`somm drain-spool`, automatic drains) go to
  the project ledger only;
- the project runs with `SOMM_CROSS_PROJECT=0`, or its client could not open
  the mirror when it started.

None of these loses data: the rows are in the project ledger. To copy them
across:

```bash
somm mirror sync --dry-run              # count what is missing, per ledger
somm mirror sync [--project NAME]       # copy it
```

`sync` attaches each registered ledger read-only. It copies call rows (and
their workload rows) whose id the mirror lacks, in batches of 5,000 that each
commit on their own. Stopping it part-way keeps what it has already copied,
and running it again copies only the rest.

## Retention

Nothing prunes a ledger by itself. `somm prune --older-than DAYS` deletes old
calls with their samples, updates and evals. It keeps decisions, workloads,
prompts and recommendations. `somm loop install` schedules only the
intelligence loop. To prune on a schedule, add a line to `crontab -e`:

```cron
# 03:30 every Sunday: keep 90 days in one project ledger, and in the mirror
30 3 * * 0  SOMM_DB_DIR=/path/to/project/.somm somm prune --older-than 90 --vacuum
45 3 * * 0  somm prune --global --older-than 90 --vacuum
```

Run each line once with `--dry-run` first. `--vacuum` rewrites the file to give
the space back. While it runs it needs as much free disk as the file uses.

## Workload attribution in global views

A workload id is a hash of its name and schemas. It does not include the
project. When two projects declare the same workload, the mirror holds one
`workloads` row for both, owned by whichever registered first. Every global
report attributes calls by `calls.project` and takes only the name from the
workload row. Per-project settings, such as budgets, SLOs and boundaries,
come from the project's own ledger.
