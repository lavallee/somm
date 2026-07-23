"""somm CLI entry point — grouped subcommands.

Commands:
  somm status          roll-up per (workload, provider, model)
  somm cache-advice    find low prefix-cache reuse opportunities
  somm bench           run latency/throughput probes through somm
  somm generate        one-shot LLM call through somm
    somm tail            live call stream
  somm compare         run a prompt through N models side-by-side
  somm frontier        adequacy frontier per (provider, model) for a workload
  somm doctor          health check (config, ollama, db, model_intel, workers, cooldowns)
  somm serve           run the web admin + HTTP API (requires somm-service)
  somm spend           today's LLM spend vs daily budget cap per workload
  somm backfill-costs  recompute cost_usd for calls missing pricing
  somm plans           metered-plan quota usage + pacing
  somm drain-spool     replay spooled telemetry into the DB
  somm workload        register, constrain, and inspect project workloads
  somm prompt          manage prompt versions, labels, and A/B variants
  somm eval            promote calls to datasets and run eval gates
  somm optimize        propose a prompt fork from failing graded calls
  somm campaign        run durable experiment campaigns
  somm inbox           list, apply, and dismiss recommendations
  somm plugin          list and inspect plugins, hooks, and providers
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import inspect
import json
import sqlite3
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm_core import VERSION, list_intel
from somm_core.config import load as load_config
from somm_core.models import Outcome, PrivacyClass
from somm_core.repository import Repository

from somm.providers.ollama import OllamaProvider

# ---------------------------------------------------------------------------
# somm workload


WORKLOAD_EXAMPLES: dict[str, dict] = {
    "structured-extraction": {
        "description": "Structured extraction workload",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Source text to extract from"}
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["label", "value"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "quality_criteria": [
            "Return valid JSON matching the output schema.",
            "Extract only facts supported by the input text.",
        ],
    },
    "freeform": {
        "description": "",
        "input_schema": None,
        "output_schema": None,
        "quality_criteria": [],
    },
}


def _cmd_workload_add(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    template = WORKLOAD_EXAMPLES[args.from_example]
    description = args.description
    if description is None:
        description = template["description"]
    wl = repo.register_workload(
        name=args.name,
        project=cfg.project,
        description=description,
        input_schema=template["input_schema"],
        output_schema=template["output_schema"],
        quality_criteria=template["quality_criteria"],
        privacy_class=PrivacyClass(args.privacy_class),
        max_p95_latency_ms=args.max_p95_latency_ms,
        max_p95_ttft_ms=args.max_p95_ttft_ms,
        max_tpot_ms=args.max_tpot_ms,
        max_capability_failure_rate=args.max_capability_failure_rate,
        max_cost_per_call_usd=args.max_cost_per_call_usd,
    )
    print(f"registered workload {wl.name!r} for project {cfg.project!r}")
    print(f"privacy_class: {wl.privacy_class.value}")
    print(f"input_schema: {'yes' if wl.input_schema else 'no'}")
    print(f"output_schema: {'yes' if wl.output_schema else 'no'}")
    return 0


def _workload_rows(repo: Repository, project: str) -> list[dict]:
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT
                w.id,
                w.name,
                w.description,
                w.input_schema_json,
                w.output_schema_json,
                w.quality_criteria_json,
                w.budget_cap_usd_daily,
                w.privacy_class,
                w.capabilities_required_json,
                w.max_p95_latency_ms,
                w.max_p95_ttft_ms,
                w.max_tpot_ms,
                w.max_capability_failure_rate,
                w.max_cost_per_call_usd,
                w.created_at,
                COUNT(c.id) AS call_count
            FROM workloads w
            LEFT JOIN calls c ON c.workload_id = w.id AND c.project = w.project
            WHERE w.project = ?
            GROUP BY w.id
            ORDER BY w.created_at DESC, w.name
            """,
            (project,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "description": r[2] or "",
            "input_schema": json.loads(r[3]) if r[3] else None,
            "output_schema": json.loads(r[4]) if r[4] else None,
            "quality_criteria": json.loads(r[5]) if r[5] else [],
            "budget_cap_usd_daily": r[6],
            "privacy_class": r[7],
            "capabilities_required": json.loads(r[8]) if r[8] else [],
            "max_p95_latency_ms": r[9],
            "max_p95_ttft_ms": r[10],
            "max_tpot_ms": r[11],
            "max_capability_failure_rate": r[12],
            "max_cost_per_call_usd": r[13],
            "created_at": r[14],
            "call_count": r[15],
        }
        for r in rows
    ]


def _cmd_workload_list(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    rows = _workload_rows(repo, cfg.project)
    if not rows:
        print(f"No workloads registered for project {cfg.project!r}.")
        print("Register one with `somm workload add <name>`.")
        return 0
    print(f"Project: {cfg.project}")
    print(f"{'name':<28} {'privacy':<10} {'calls':>8}")
    for row in rows:
        print(f"{row['name'][:27]:<28} {row['privacy_class']:<10} {row['call_count']:>8}")
    return 0


def _cmd_workload_show(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    rows = [row for row in _workload_rows(repo, cfg.project) if row["name"] == args.name]
    if not rows:
        print(f"No workload {args.name!r} registered for project {cfg.project!r}.", file=sys.stderr)
        return 2
    row = rows[0]
    print(f"name: {row['name']}")
    print(f"project: {cfg.project}")
    print(f"id: {row['id']}")
    print(f"description: {row['description'] or '—'}")
    print(f"privacy_class: {row['privacy_class']}")
    print(f"call_count: {row['call_count']}")
    print(f"created_at: {row['created_at']}")
    print(
        f"budget_cap_usd_daily: {row['budget_cap_usd_daily'] if row['budget_cap_usd_daily'] is not None else '—'}"
    )
    print(f"capabilities_required: {', '.join(row['capabilities_required']) or '—'}")
    print("constraints:")
    print(
        f"  max_p95_latency_ms: {row['max_p95_latency_ms'] if row['max_p95_latency_ms'] is not None else '—'}"
    )
    print(
        f"  max_p95_ttft_ms: {row['max_p95_ttft_ms'] if row['max_p95_ttft_ms'] is not None else '—'}"
    )
    print(f"  max_tpot_ms: {row['max_tpot_ms'] if row['max_tpot_ms'] is not None else '—'}")
    print(
        f"  max_capability_failure_rate: {row['max_capability_failure_rate'] if row['max_capability_failure_rate'] is not None else '—'}"
    )
    print(
        f"  max_cost_per_call_usd: {row['max_cost_per_call_usd'] if row['max_cost_per_call_usd'] is not None else '—'}"
    )
    print("input_schema:")
    print(json.dumps(row["input_schema"], indent=2) if row["input_schema"] else "  —")
    print("output_schema:")
    print(json.dumps(row["output_schema"], indent=2) if row["output_schema"] else "  —")
    if row["quality_criteria"]:
        print("quality_criteria:")
        for item in row["quality_criteria"]:
            print(f"  - {item}")
    else:
        print("quality_criteria: —")
    return 0


def _cmd_workload_set_constraints(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = repo.workload_by_name(args.name, cfg.project)
    if wl is None:
        print(f"No workload {args.name!r} registered for project {cfg.project!r}.", file=sys.stderr)
        return 2
    if not args.clear and all(
        value is None
        for value in (
            args.max_p95_latency_ms,
            args.max_p95_ttft_ms,
            args.max_tpot_ms,
            args.max_capability_failure_rate,
            args.max_cost_per_call_usd,
        )
    ):
        print(
            "No constraints supplied; pass at least one --max-* option or --clear.", file=sys.stderr
        )
        return 2
    repo.set_workload_constraints(
        wl.id,
        max_p95_latency_ms=args.max_p95_latency_ms,
        max_p95_ttft_ms=args.max_p95_ttft_ms,
        max_tpot_ms=args.max_tpot_ms,
        max_capability_failure_rate=args.max_capability_failure_rate,
        max_cost_per_call_usd=args.max_cost_per_call_usd,
        clear=args.clear,
    )
    print(f"updated constraints for workload {wl.name!r}")
    return 0


# ---------------------------------------------------------------------------
# somm prompt


def _prompt_workload(repo: Repository, project: str, name: str):
    workload = repo.workload_by_name(name, project)
    if workload is None:
        print(f"No workload {name!r} registered for project {project!r}.", file=sys.stderr)
    return workload


def _prompt_rows(repo: Repository, workload_id: str) -> list[dict]:
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT id, workload_id, version, hash, body,
                   created_at, retired_at, parent_prompt_id
            FROM prompts
            WHERE workload_id = ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (workload_id,),
        ).fetchall()
    return [
        {
            "id": row[0],
            "workload_id": row[1],
            "version": row[2],
            "hash": row[3],
            "body": row[4],
            "created_at": row[5],
            "retired_at": row[6],
            "parent_prompt_id": row[7],
        }
        for row in rows
    ]


def _prompt_ref(repo: Repository, workload_id: str, ref: str):
    from somm.prompts import PromptNotFound, get_label, get_prompt

    labeled = get_label(repo, workload_id, ref)
    if labeled is not None:
        return labeled
    try:
        return get_prompt(repo, workload_id, version=ref)
    except PromptNotFound:
        raise PromptNotFound(f"no prompt version or label {ref!r}") from None


def _body_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "body", None) is not None:
        return args.body
    return Path(args.body_file).read_text()


def _truncate_body(body: str, full: bool) -> str:
    if full or len(body) <= 600:
        return body
    return body[:600].rstrip() + "\n... (truncated; pass --full)"


def _format_label_pointer(meta: dict) -> str:
    weights = meta.get("weights") or {}
    if weights:
        versions = meta.get("versions") or {}
        parts = []
        for prompt_id, weight in sorted(
            weights.items(),
            key=lambda item: (-float(item[1]), versions.get(item[0], item[0])),
        ):
            label = versions.get(prompt_id, prompt_id[:8])
            parts.append(f"{label}={weight * 100:.1f}%")
        return ", ".join(parts)
    return meta.get("version") or meta.get("prompt_id", "")[:8]


def _parse_weights(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid weight item {item!r}; expected v1=90")
        ref, value = item.split("=", 1)
        ref = ref.strip()
        if not ref:
            raise ValueError(f"invalid weight item {item!r}; missing version")
        out[ref] = float(value.strip())
    if not out:
        raise ValueError("weights must include at least one version")
    return out


def _prompt_score(repo: Repository, prompt_id: str) -> dict:
    with repo._open() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(DISTINCT CASE
                    WHEN e.structural_score IS NOT NULL
                      OR e.embedding_score IS NOT NULL
                      OR e.judge_score IS NOT NULL
                    THEN c.id END) AS graded_count,
                AVG(e.structural_score) AS mean_structural,
                AVG(e.embedding_score) AS mean_embedding,
                AVG(e.judge_score) AS mean_judge,
                AVG(COALESCE(e.judge_score, e.embedding_score, e.structural_score))
                    AS mean_gate
            FROM calls AS c
            LEFT JOIN eval_results AS e ON e.call_id = c.id
            WHERE c.prompt_id = ?
            """,
            (prompt_id,),
        ).fetchone()
    return {
        "graded_count": int(row[0] or 0),
        "mean_structural": row[1],
        "mean_embedding": row[2],
        "mean_judge": row[3],
        "mean_gate": row[4],
    }


def _fmt_score(value) -> str:
    return f"{float(value):.3f}" if value is not None else "—"


def _cmd_prompt_list(args: argparse.Namespace) -> int:
    from somm.prompts import list_label_pointers

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = _prompt_workload(repo, cfg.project, args.workload)
    if wl is None:
        return 2
    prompts = _prompt_rows(repo, wl.id)
    labels = list_label_pointers(repo, wl.id)
    labels_by_prompt: dict[str, list[str]] = {}
    for label, meta in labels.items():
        labels_by_prompt.setdefault(meta["prompt_id"], []).append(label)

    print(f"Workload: {args.workload}")
    if labels:
        print("labels:")
        for label, meta in labels.items():
            print(f"  {label:<14} -> {_format_label_pointer(meta)}")
    else:
        print("labels: —")
    print()
    if not prompts:
        print("No prompts registered.")
        return 0
    print(f"{'version':<10} {'id':<10} {'created_at':<20} {'retired':<8} labels")
    for row in prompts:
        retired = "yes" if row["retired_at"] else "no"
        row_labels = ", ".join(sorted(labels_by_prompt.get(row["id"], []))) or "—"
        print(
            f"{row['version']:<10} {row['id'][:8]:<10} "
            f"{str(row['created_at'])[:19]:<20} {retired:<8} {row_labels}"
        )
    return 0


def _cmd_prompt_show(args: argparse.Namespace) -> int:
    from somm.prompts import PromptNotFound, get_prompt

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = _prompt_workload(repo, cfg.project, args.workload)
    if wl is None:
        return 2
    try:
        prompt = (
            _prompt_ref(repo, wl.id, args.label)
            if args.label
            else get_prompt(repo, wl.id, version=args.version or "latest")
        )
    except PromptNotFound as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"workload: {args.workload}")
    print(f"version: {prompt.version}")
    print(f"id: {prompt.id}")
    print(f"parent_prompt_id: {prompt.parent_prompt_id or '—'}")
    print(f"created_at: {prompt.created_at.isoformat() if prompt.created_at else '—'}")
    print("body:")
    print(_truncate_body(prompt.body, args.full))
    return 0


def _cmd_prompt_register(args: argparse.Namespace) -> int:
    from somm.prompts import register_prompt

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = _prompt_workload(repo, cfg.project, args.workload)
    if wl is None:
        return 2
    prompt = register_prompt(repo, wl.id, _body_from_args(args), bump=args.bump)
    print(f"registered {prompt.version} ({prompt.id[:8]})")
    return 0


def _cmd_prompt_fork(args: argparse.Namespace) -> int:
    from somm.prompts import PromptNotFound, fork_prompt

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = _prompt_workload(repo, cfg.project, args.workload)
    if wl is None:
        return 2
    try:
        prompt = fork_prompt(
            repo,
            wl.id,
            args.from_ref,
            Path(args.body_file).read_text(),
            updated_by="somm prompt fork",
        )
    except PromptNotFound as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"forked {prompt.version} ({prompt.id[:8]})")
    print(f"parent_prompt_id: {prompt.parent_prompt_id}")
    return 0


def _cmd_prompt_diff(args: argparse.Namespace) -> int:
    from somm.prompts import PromptNotFound

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = _prompt_workload(repo, cfg.project, args.workload)
    if wl is None:
        return 2
    try:
        left = _prompt_ref(repo, wl.id, args.a)
        right = _prompt_ref(repo, wl.id, args.b)
    except PromptNotFound as exc:
        print(str(exc), file=sys.stderr)
        return 2
    diff = difflib.unified_diff(
        left.body.splitlines(keepends=True),
        right.body.splitlines(keepends=True),
        fromfile=f"{args.a} ({left.version})",
        tofile=f"{args.b} ({right.version})",
    )
    sys.stdout.writelines(diff)
    return 0


def _cmd_prompt_label(args: argparse.Namespace) -> int:
    from somm.prompts import (
        PromptNotFound,
        get_prompt,
        list_label_pointers,
        set_label,
        set_label_weights,
    )

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = _prompt_workload(repo, cfg.project, args.workload)
    if wl is None:
        return 2
    try:
        if args.weights:
            set_label_weights(
                repo,
                wl.id,
                args.label,
                _parse_weights(args.weights),
                updated_by="somm prompt label",
            )
        else:
            prompt = get_prompt(repo, wl.id, version=args.version)
            set_label(repo, wl.id, args.label, prompt.id, updated_by="somm prompt label")
    except (PromptNotFound, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    meta = list_label_pointers(repo, wl.id)[args.label]
    print(f"{args.label} -> {_format_label_pointer(meta)}")
    return 0


def _cmd_prompt_promote(args: argparse.Namespace) -> int:
    from somm.prompts import PromptNotFound, get_prompt, set_label

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = _prompt_workload(repo, cfg.project, args.workload)
    if wl is None:
        return 2
    try:
        prompt = get_prompt(repo, wl.id, version=args.version)
    except PromptNotFound as exc:
        print(str(exc), file=sys.stderr)
        return 2

    score = _prompt_score(repo, prompt.id)
    min_graded_ok = args.min_graded is None or score["graded_count"] >= int(args.min_graded)
    mean_gate = score["mean_gate"]
    min_score_ok = args.min_score is None or (
        mean_gate is not None and float(mean_gate) >= float(args.min_score)
    )
    if not args.force and (not min_graded_ok or not min_score_ok):
        print(
            "promotion gate failed: "
            f"graded={score['graded_count']} mean_score={_fmt_score(mean_gate)}",
            file=sys.stderr,
        )
        return 2

    set_label(repo, wl.id, args.to, prompt.id, updated_by="somm prompt promote")
    print(f"{args.to} -> {prompt.version}")
    print(f"graded: {score['graded_count']}  mean_score: {_fmt_score(mean_gate)}")
    if args.force and (not min_graded_ok or not min_score_ok):
        print("forced: yes")
    return 0


def _cmd_prompt_score(args: argparse.Namespace) -> int:
    from somm.prompts import PromptNotFound

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = _prompt_workload(repo, cfg.project, args.workload)
    if wl is None:
        return 2
    try:
        if args.label:
            prompts = [_prompt_ref(repo, wl.id, args.label)]
        elif args.version:
            prompts = [_prompt_ref(repo, wl.id, args.version)]
        else:
            from somm.prompts import get_prompt

            prompts = [
                get_prompt(repo, wl.id, version=row["version"]) for row in _prompt_rows(repo, wl.id)
            ]
    except PromptNotFound as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"{'version':<10} {'id':<10} {'graded':>8} {'struct':>8} {'text-sim':>8} {'judge':>8}")
    for prompt in prompts:
        score = _prompt_score(repo, prompt.id)
        print(
            f"{prompt.version:<10} {prompt.id[:8]:<10} "
            f"{score['graded_count']:>8} "
            f"{_fmt_score(score['mean_structural']):>8} "
            f"{_fmt_score(score['mean_embedding']):>8} "
            f"{_fmt_score(score['mean_judge']):>8}"
        )
    return 0


# ---------------------------------------------------------------------------
# somm eval


def _cmd_eval_promote_call(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    try:
        dataset, item = repo.promote_call_to_dataset(
            args.call_id,
            args.dataset,
            project=cfg.project,
            description=args.description or "",
            created_by="somm eval promote-call",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    n_items = len(repo.dataset_items(dataset.id))
    print(f"promoted call {args.call_id} to dataset {dataset.name!r}")
    print(f"project: {dataset.project}")
    print(f"workload_id: {dataset.workload_id}")
    print(f"dataset_id: {dataset.id}")
    print(f"item_id: {item.id}")
    print(f"items: {n_items}")
    return 0


def _cmd_eval_import(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    workload = repo.workload_by_name(args.workload, cfg.project)
    if workload is None:
        print(
            f"No workload {args.workload!r} registered for project {cfg.project!r}.",
            file=sys.stderr,
        )
        return 2
    rows: list[dict] = []
    try:
        with Path(args.file).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number}: expected a JSON object")
                rows.append(row)
        dataset, items = repo.import_dataset_items(
            project=cfg.project,
            workload_id=workload.id,
            name=args.dataset,
            items=rows,
            description=args.description or "",
            created_by="somm eval import",
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = {
        "project": dataset.project,
        "workload": args.workload,
        "workload_id": dataset.workload_id,
        "dataset": dataset.name,
        "dataset_id": dataset.id,
        "imported_items": len(items),
        "total_items": len(repo.dataset_items(dataset.id)),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"imported {payload['imported_items']} reviewed item(s) into "
            f"{dataset.project}/{args.workload}/{dataset.name}"
        )
        print(f"dataset_id: {dataset.id}")
        print(f"total_items: {payload['total_items']}")
    return 0


def _load_eval_judge_config(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid judge config {path!r}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("judge config must be a JSON object")
    panel = raw.get("panel")
    if panel is None and raw.get("provider") and raw.get("model"):
        panel = [{"provider": raw["provider"], "model": raw["model"]}]
    if not isinstance(panel, list) or not panel:
        raise ValueError("judge config needs a non-empty panel")
    for index, spec in enumerate(panel):
        if not isinstance(spec, dict) or not spec.get("provider") or not spec.get("model"):
            raise ValueError(f"judge panel[{index}] needs provider and model")
    raw["panel"] = panel
    min_judges = int(raw.get("min_judges", len(panel)))
    if min_judges < 1 or min_judges > len(panel):
        raise ValueError("judge min_judges must be between 1 and the panel size")
    raw["min_judges"] = min_judges
    return raw


def _aggregate_eval_judges(criteria, receipts: list[dict]) -> dict:
    rows = []
    for criterion in criteria:
        votes = []
        reasons = []
        for receipt in receipts:
            by_name = {
                row.get("name"): row
                for row in receipt["result"].get("criteria", [])
                if isinstance(row, dict)
            }
            result = by_name.get(criterion.name)
            if not isinstance(result, dict):
                continue
            passed = bool(result.get("pass"))
            votes.append(passed)
            reasons.append({
                "provider": receipt["provider"],
                "model": receipt["model"],
                "pass": passed,
                "reason": result.get("reason") or "",
            })
        votes_for = sum(1 for vote in votes if vote)
        rows.append({
            "name": criterion.name,
            "pass": bool(votes) and votes_for > len(votes) / 2,
            "votes_for": votes_for,
            "votes_total": len(votes),
            "reasons": reasons,
        })
    return {
        "criteria": rows,
        "score": sum(1 for row in rows if row["pass"]) / len(rows),
    }


def _dataset_judge(llm, *, workload: str, config: dict):
    from somm_core.graders import (
        build_binary_judge_prompt,
        normalize_binary_criteria,
        parse_binary_judge_response,
    )

    from somm.evals import JudgeGrade

    criteria = normalize_binary_criteria(config.get("criteria"))

    def judge(item, generated):
        prompt = build_binary_judge_prompt(
            original_prompt=item.prompt_body,
            production_text=generated.text,
            gold_text=item.expected_response_body,
            criteria=criteria,
        )
        receipts = []
        failures = []
        all_call_ids = []
        for spec in config["panel"]:
            result = llm.generate(
                prompt=prompt,
                workload=workload,
                max_tokens=int(spec.get("max_tokens") or config.get("max_tokens") or 800),
                temperature=0.0,
                provider=str(spec["provider"]),
                model=str(spec["model"]),
                no_fallback=True,
            )
            writer = getattr(llm, "_writer", None)
            if writer is not None:
                writer.flush(timeout=5.0)
            all_call_ids.append(result.call_id)
            if result.outcome != Outcome.OK:
                failures.append({
                    "call_id": result.call_id,
                    "provider": spec["provider"],
                    "model": spec["model"],
                    "outcome": result.outcome.value,
                    "error": result.error_detail,
                })
                continue
            receipts.append({
                "call_id": result.call_id,
                "provider": spec["provider"],
                "model": spec["model"],
                "result": parse_binary_judge_response(result.text, criteria),
            })
        quorum = len(receipts) >= int(config["min_judges"])
        aggregate = (
            _aggregate_eval_judges(criteria, receipts)
            if quorum
            else {
                "criteria": [
                    {
                        "name": criterion.name,
                        "pass": False,
                        "votes_for": 0,
                        "votes_total": len(receipts),
                        "reasons": [],
                    }
                    for criterion in criteria
                ],
                "score": 0.0,
            }
        )
        aggregate.update({
            "mode": "panel" if len(receipts) > 1 else "single",
            "quorum": quorum,
            "min_judges": int(config["min_judges"]),
            "judges": receipts,
            "failures": failures,
        })
        return JudgeGrade(
            score=float(aggregate["score"]),
            reason=aggregate,
            call_ids=tuple(all_call_ids),
        )

    return judge


def _cmd_eval_run(args: argparse.Namespace) -> int:
    from somm import SommLLM
    from somm.evals import run_dataset_eval

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    llm = SommLLM(config=cfg)
    try:
        try:
            judge_config = _load_eval_judge_config(args.judge_config)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        def generate(item):
            result = llm.generate(
                prompt=item.prompt_body,
                workload=args.workload,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                provider=args.provider,
                model=args.model,
                no_fallback=bool(args.provider),
            )
            writer = getattr(llm, "_writer", None)
            if writer is not None:
                writer.flush(timeout=5.0)
            return result

        try:
            result = run_dataset_eval(
                repo,
                project=cfg.project,
                workload=args.workload,
                dataset=args.dataset,
                generate=generate,
                judge=(
                    _dataset_judge(llm, workload=args.workload, config=judge_config)
                    if judge_config is not None
                    else None
                ),
                implementation=args.implementation,
                threshold=args.threshold,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    finally:
        llm.close()

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        _print_eval_run(result)
    return 0 if result.passed else 1


def _print_eval_run(result) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(
        f"{status} eval {result.project}/{result.workload}/{result.dataset} "
        f"mean={result.mean_score:.3f} threshold={result.threshold:.3f} "
        f"passed={result.n_passed}/{result.n_items} errors={result.n_errors}"
    )
    print(f"{'item':<10} {'call':<10} {'score':>7} {'status':<6} error")
    for item in result.items:
        call = (item.generated_call_id or "-")[:8]
        score = f"{item.score:.3f}"
        item_status = "PASS" if item.passed else "FAIL"
        print(f"{item.item_id[:8]:<10} {call:<10} {score:>7} {item_status:<6} {item.error or ''}")


# ---------------------------------------------------------------------------
# somm optimize


def _cmd_optimize(args: argparse.Namespace) -> int:
    from somm import SommLLM
    from somm.optimize import propose_prompt_optimization
    from somm.prompts import PromptNotFound

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = repo.workload_by_name(args.workload, cfg.project)
    if wl is None:
        print(
            f"No workload {args.workload!r} registered for project {cfg.project!r}.",
            file=sys.stderr,
        )
        return 2

    llm = SommLLM(config=cfg)
    try:

        def proposer(prompt: str) -> str:
            result = llm.generate(
                prompt=prompt,
                workload="somm_optimize",
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                provider=args.provider,
                model=args.model,
                no_fallback=bool(args.provider),
            )
            writer = getattr(llm, "_writer", None)
            if writer is not None:
                writer.flush(timeout=5.0)
            if result.outcome != Outcome.OK:
                raise RuntimeError(
                    f"optimizer call failed: {result.outcome.value} {result.error_detail or ''}"
                )
            return result.text

        try:
            result = propose_prompt_optimization(
                repo,
                workload_id=wl.id,
                from_ref=args.from_ref,
                proposer=proposer,
                threshold=args.threshold,
                limit=args.limit,
                label=args.label,
            )
        except (PromptNotFound, ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    finally:
        llm.close()

    print(f"{result.label} -> {result.proposed_prompt.version} ({result.proposed_prompt.id[:8]})")
    print(f"source: {result.source_prompt.version} ({result.source_prompt.id[:8]})")
    print(f"cases: {len(result.cases)}")
    if result.rationale:
        print(f"rationale: {result.rationale}")
    return 0


# ---------------------------------------------------------------------------
# somm campaign


def _cmd_campaign_run(args: argparse.Namespace) -> int:
    from somm import SommLLM
    from somm.campaigns import MetricContract, run_eval_campaign, write_campaign_jsonl

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    direction = args.direction
    if direction is None:
        direction = "lte" if args.metric == "error_rate" else "gte"
    try:
        contract = MetricContract(
            metric=args.metric,
            threshold=args.threshold,
            direction=direction,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    llm = SommLLM(config=cfg)
    try:

        def generate(item):
            result = llm.generate(
                prompt=item.prompt_body,
                workload=args.workload,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                provider=args.provider,
                model=args.model,
                no_fallback=bool(args.provider),
            )
            writer = getattr(llm, "_writer", None)
            if writer is not None:
                writer.flush(timeout=5.0)
            return result

        try:
            result = run_eval_campaign(
                repo,
                project=cfg.project,
                workload=args.workload,
                dataset=args.dataset,
                generate=generate,
                contract=contract,
                name=args.name,
                max_rounds=args.max_rounds,
                token_budget=args.token_budget,
                plateau_window=args.plateau_window,
                min_delta=args.min_delta,
                eval_threshold=args.eval_threshold,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    finally:
        llm.close()

    if args.log:
        try:
            write_campaign_jsonl(result, args.log)
        except OSError as exc:
            print(f"failed to write campaign log {args.log!r}: {exc}", file=sys.stderr)
            return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        _print_campaign_run(result, log_path=args.log)
    return 0 if result.passed else 1


def _print_campaign_run(result, *, log_path: str | None = None) -> None:
    campaign = result.campaign
    status = "PASS" if result.passed else "FAIL"
    best = "-" if result.best_score is None else f"{result.best_score:.3f}"
    print(
        f"{status} campaign {campaign.name} ({campaign.id[:8]}) "
        f"status={campaign.status} stop={result.stop_reason} best={best}"
    )
    print(
        f"metric: {campaign.metric} {campaign.direction} {campaign.threshold:.3f} "
        f"tokens={result.total_tokens} cost=${result.total_cost_usd:.6f}"
    )
    if log_path:
        print(f"log: {log_path}")
    print(f"{'seq':>3} {'event':<18} {'action':<7} {'score':>7} {'tokens':>8} {'run':<10}")
    for event in result.events:
        score = "-" if event.metric_score is None else f"{event.metric_score:.3f}"
        run = (event.run_id or "-")[:8]
        print(
            f"{event.sequence:>3} {event.event_type:<18} {event.action:<7} "
            f"{score:>7} {event.total_tokens:>8} {run:<10}"
        )


# ---------------------------------------------------------------------------
# somm inbox


def _decision_mirror_repo(cfg):
    if not getattr(cfg, "cross_project_enabled", False):
        return None
    try:
        return Repository(cfg.global_db_path)
    except Exception:
        return None


def _cmd_inbox_list(args: argparse.Namespace) -> int:
    from somm.recommendations import list_recommendations

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    recs = list_recommendations(
        repo,
        project=cfg.project,
        workload=args.workload,
        open_only=not args.all,
    )
    if args.json:
        print(json.dumps([rec.as_dict() for rec in recs], indent=2))
        return 0
    if not recs:
        print("no recommendations")
        return 0
    print(f"{'id':>5} {'workload':<24} {'action':<18} {'conf':>6} {'state':<9} impact")
    for rec in recs:
        if rec.applied_at:
            state = "applied"
        elif rec.dismissed_at:
            state = "dismissed"
        else:
            state = "open"
        confidence = "-" if rec.confidence is None else f"{rec.confidence:.2f}"
        print(
            f"{rec.id:>5} {rec.workload[:23]:<24} {rec.action:<18} "
            f"{confidence:>6} {state:<9} {rec.expected_impact}"
        )
    return 0


def _cmd_inbox_apply(args: argparse.Namespace) -> int:
    from somm.recommendations import apply_recommendation

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    try:
        result = apply_recommendation(
            repo,
            args.recommendation_id,
            actor="somm inbox",
            mirror_repo=_decision_mirror_repo(cfg),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(
            f"applied recommendation {result.recommendation.id} "
            f"for {result.recommendation.workload}"
        )
        print(f"decision_id: {result.decision.id}")
        if result.revision is not None:
            print(f"workload_revision: {result.revision}")
    return 0


def _cmd_inbox_dismiss(args: argparse.Namespace) -> int:
    from somm.recommendations import dismiss_recommendation

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    try:
        rec = dismiss_recommendation(repo, args.recommendation_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rec.as_dict(), indent=2))
    else:
        print(f"dismissed recommendation {rec.id} for {rec.workload}")
    return 0


# ---------------------------------------------------------------------------
# somm plugin


def _provider_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "pinned-only"


def _reference_plugin_register_name(name: str) -> str:
    return f"somm.plugins.{name}.register()"


def _reference_plugin_register_fn(spec: dict):
    module = importlib.import_module(spec["module"])
    return module.register


def _cmd_plugin_list(args: argparse.Namespace) -> int:
    from somm import hooks, plugins
    from somm.providers.registry import (
        BUILTIN_PROVIDER_SPECS,
        load_entrypoint_provider_specs,
    )

    print("REFERENCE PLUGINS")
    if not plugins.REFERENCE_PLUGINS:
        print("  no reference plugins available")
    else:
        print(f"  {'name':<16} {'phase':<14} {'enable':<42} needs")
        for name, spec in sorted(plugins.REFERENCE_PLUGINS.items()):
            extra = spec.get("extra")
            needs = f"pip install somm[{extra}]" if extra else "-"
            print(
                f"  {name:<16} {spec['phase']:<14} "
                f"{_reference_plugin_register_name(name):<42} {needs}"
            )
            print(f"    {spec['summary']}")

    print()
    print("ACTIVE HOOKS")
    hooks.load_entry_points()
    active = hooks.registered_hooks()
    if not any(active.get(phase) for phase in hooks.HOOK_PHASES):
        print("  no active hooks")
    else:
        for phase in hooks.HOOK_PHASES:
            rows = active.get(phase, [])
            print(f"  {phase}")
            if not rows:
                print("    none")
                continue
            print(f"    {'priority':>8}  callable")
            for qualname, priority in rows:
                print(f"    {priority:>8}  {qualname}")

    print()
    print("PROVIDERS")
    print("  built-in")
    print(f"    {'name':<18} rank")
    for spec in BUILTIN_PROVIDER_SPECS:
        print(f"    {spec.name:<18} {_provider_rank(spec.default_order_rank)}")
    entrypoint_specs = load_entrypoint_provider_specs()
    print("  entry-point")
    if not entrypoint_specs:
        print("    no entry-point providers")
    else:
        print(f"    {'name':<18} rank")
        for spec in entrypoint_specs:
            print(f"    {spec.name:<18} {_provider_rank(spec.default_order_rank)}")
    return 0


def _cmd_plugin_info(args: argparse.Namespace) -> int:
    from somm import plugins

    spec = plugins.REFERENCE_PLUGINS.get(args.name)
    if spec is None:
        valid = ", ".join(sorted(plugins.REFERENCE_PLUGINS)) or "none"
        print(
            f"Unknown reference plugin {args.name!r}. Valid plugins: {valid}.",
            file=sys.stderr,
        )
        return 2

    register_fn = _reference_plugin_register_fn(spec)
    signature = inspect.signature(register_fn)
    extra = spec.get("extra")
    print(f"name: {args.name}")
    print(f"summary: {spec['summary']}")
    print(f"phase: {spec['phase']}")
    print(f"module: {spec['module']}")
    print(f"register: register{signature}")
    print(f"enable: {_reference_plugin_register_name(args.name)}")
    print(f"extra: {f'pip install somm[{extra}]' if extra else '-'}")
    return 0


# ---------------------------------------------------------------------------
# somm status


def _fmt_stat_int(value: object) -> str:
    if value is None:
        return "-"
    return str(int(round(float(value))))


def _fmt_stat_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def _fmt_stat_pct(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.0f}%"


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    if getattr(args, "global_view", False):
        db_path = cfg.global_db_path
        if not db_path.exists():
            if getattr(args, "json", False):
                print(
                    json.dumps(
                        {
                            "scope": "global",
                            "project": cfg.project,
                            "db_path": str(db_path),
                            "window_days": args.since,
                            "count": 0,
                            "rows": [],
                        },
                        indent=2,
                    )
                )
                return 0
            print(f"No global mirror at {db_path}.")
            print("Enable via SOMM_CROSS_PROJECT=1 and run a project with somm.")
            return 0
        repo = Repository(db_path)
        # Global status sums across projects; call with project=None semantics
        # via a single-table query.
        stats = _stats_global(repo, since_days=args.since)
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "scope": "global",
                        "project": cfg.project,
                        "db_path": str(db_path),
                        "window_days": args.since,
                        "count": len(stats),
                        "rows": stats,
                    },
                    indent=2,
                )
            )
            return 0
        if not stats:
            print(f"Global mirror at {db_path} has no rows in the last {args.since} days.")
            return 0
        print(f"GLOBAL ({db_path})  ({args.since}d window)")
        print(
            f"{'project':<18} {'workload':<20} {'provider':<10} {'model':<16} "
            f"{'n':>6} {'tok_in':>8} {'tok_out':>8} {'cost':>10} {'fail':>6} "
            f"{'p95':>7} {'ttft95':>7} {'tpot':>7} {'out/s':>8} {'cache':>6} {'good':>6}"
        )
        for s in stats:
            cost_s = f"${(s['cost_usd'] or 0):.4f}"
            print(
                f"{s['project'][:17]:<18} {s['workload'][:19]:<20} {s['provider'][:9]:<10} "
                f"{s['model'][:15]:<16} {s['n_calls']:>6} {(s['tokens_in'] or 0):>8} "
                f"{(s['tokens_out'] or 0):>8} {cost_s:>10} {s['n_failed']:>6} "
                f"{_fmt_stat_int(s.get('p95_latency_ms')):>7} "
                f"{_fmt_stat_int(s.get('p95_ttft_ms')):>7} "
                f"{_fmt_stat_float(s.get('tpot_ms')):>7} "
                f"{_fmt_stat_float(s.get('output_tokens_per_second')):>8} "
                f"{_fmt_stat_pct(s.get('cache_read_ratio')):>6} "
                f"{_fmt_stat_pct(s.get('goodput_under_slo')):>6}"
            )
        return 0

    repo = Repository(cfg.db_path)
    stats = repo.stats_by_workload(cfg.project, since_days=args.since)
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "scope": "project",
                    "project": cfg.project,
                    "db_path": str(cfg.db_path),
                    "window_days": args.since,
                    "count": len(stats),
                    "rows": stats,
                },
                indent=2,
            )
        )
        return 0
    if not stats:
        print(f"No calls yet for project {cfg.project!r} in the last {args.since} days.")
        print(f"Run `somm.llm({cfg.project!r}).generate(...)` in your Python code.")
        return 0
    print(f"Project: {cfg.project}  ({args.since}d window)")
    print(
        f"{'workload':<24} {'provider':<12} {'model':<18} "
        f"{'n':>6} {'tok_in':>8} {'tok_out':>8} {'cost':>10} {'fail':>6} "
        f"{'p95':>7} {'ttft95':>7} {'tpot':>7} {'out/s':>8} {'cache':>6} {'good':>6}"
    )
    for s in stats:
        cost_s = f"${(s['cost_usd'] or 0):.4f}"
        print(
            f"{s['workload'][:23]:<24} {s['provider'][:11]:<12} {s['model'][:17]:<18} "
            f"{s['n_calls']:>6} {(s['tokens_in'] or 0):>8} {(s['tokens_out'] or 0):>8} "
            f"{cost_s:>10} {s['n_failed']:>6} "
            f"{_fmt_stat_int(s.get('p95_latency_ms')):>7} "
            f"{_fmt_stat_int(s.get('p95_ttft_ms')):>7} "
            f"{_fmt_stat_float(s.get('tpot_ms')):>7} "
            f"{_fmt_stat_float(s.get('output_tokens_per_second')):>8} "
            f"{_fmt_stat_pct(s.get('cache_read_ratio')):>6} "
            f"{_fmt_stat_pct(s.get('goodput_under_slo')):>6}"
        )
    return 0


def _prefix_cache_advice(
    stats: list[dict],
    *,
    min_tokens_in: int,
    max_cache_read_ratio: float,
) -> list[dict]:
    out: list[dict] = []
    for row in stats:
        tokens_in = int(row.get("tokens_in") or 0)
        if tokens_in < min_tokens_in:
            continue
        cache_read_ratio = row.get("cache_read_ratio")
        effective_ratio = float(cache_read_ratio or 0.0)
        if effective_ratio > max_cache_read_ratio:
            continue
        cache_tokens_in = int(row.get("cache_tokens_in") or 0)
        n_calls = int(row.get("n_calls") or 0)
        issue = "no_cache_reads" if cache_tokens_in == 0 else "low_cache_reuse"
        out.append(
            {
                "workload": row["workload"],
                "provider": row["provider"],
                "model": row["model"],
                "n_calls": n_calls,
                "tokens_in": tokens_in,
                "cache_tokens_in": cache_tokens_in,
                "cache_read_ratio": effective_ratio,
                "issue": issue,
                "advice": (
                    "Stabilize repeated system/context prefixes and batch calls with "
                    "shared prefixes together so provider prefix caches can hit."
                ),
            }
        )
    return sorted(out, key=lambda row: (row["cache_read_ratio"], -row["tokens_in"]))


def _cmd_cache_advice(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    stats = repo.stats_by_workload(cfg.project, since_days=args.since)
    rows = _prefix_cache_advice(
        stats,
        min_tokens_in=args.min_tokens_in,
        max_cache_read_ratio=args.max_cache_read_ratio,
    )
    payload = {
        "project": cfg.project,
        "window_days": args.since,
        "count": len(rows),
        "rows": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if not rows:
        print("No prefix-cache opportunities found for the current thresholds.")
        return 0
    print(f"Prefix-cache advice: {cfg.project}  ({args.since}d window)")
    print(
        f"{'workload':<24} {'provider':<12} {'model':<18} {'n':>6} {'tok_in':>9} {'cache':>7} issue"
    )
    for row in rows:
        print(
            f"{row['workload'][:23]:<24} {row['provider'][:11]:<12} "
            f"{row['model'][:17]:<18} {row['n_calls']:>6} "
            f"{row['tokens_in']:>9} {_fmt_stat_pct(row['cache_read_ratio']):>7} "
            f"{row['issue']}"
        )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    if args.prompt_file and args.prompt:
        raise ValueError("pass either PROMPT or --prompt-file, not both")
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt == "-":
        prompt = sys.stdin.read()
    else:
        prompt = args.prompt
    if not prompt:
        raise ValueError("prompt is required (pass PROMPT, --prompt-file, or '-')")

    cfg = load_config(project=args.project)
    from somm.client import SommLLM

    llm = SommLLM(config=cfg)
    try:
        result = llm.generate(
            prompt,
            system=args.system,
            workload=args.workload,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    finally:
        llm.close()

    payload = {
        "ok": result.outcome == Outcome.OK,
        "text": result.text,
        "provider": result.provider,
        "model": result.model,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
        "call_id": result.call_id,
        "outcome": result.outcome.value,
        "error_kind": result.error_kind,
        "ttft_ms": result.ttft_ms,
        "cache_tokens_in": result.cache_tokens_in,
        "cache_tokens_out": result.cache_tokens_out,
        "citations": result.citations,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(result.text)
    return 0


def _bench_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file and args.prompt:
        raise ValueError("pass either PROMPT or --prompt-file, not both")
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt == "-":
        return sys.stdin.read()
    if not args.prompt:
        raise ValueError("prompt is required (pass PROMPT, --prompt-file, or '-')")
    return args.prompt


def _bench_tpot_ms(result) -> float | None:
    if result.ttft_ms is None or result.tokens_out <= 1:
        return None
    if result.latency_ms < result.ttft_ms:
        return None
    return (result.latency_ms - result.ttft_ms) / (result.tokens_out - 1)


def _bench_percentile(values: list[float | int | None], pct: int) -> float | None:
    ordered = sorted(float(v) for v in values if v is not None)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, (pct * len(ordered) + 99) // 100 - 1))
    return ordered[index]


def _bench_summary(rows: list[dict], *, wall_seconds: float) -> dict:
    ok_rows = [row for row in rows if row["outcome"] == Outcome.OK.value]
    total_latency_ms = sum(row["latency_ms"] for row in ok_rows)
    total_input_tokens = sum(row["tokens_in"] for row in ok_rows)
    total_output_tokens = sum(row["tokens_out"] for row in ok_rows)
    total_tokens = total_input_tokens + total_output_tokens
    total_cost = sum(row["cost_usd"] for row in rows)
    return {
        "calls": len(rows),
        "ok_calls": len(ok_rows),
        "failed_calls": len(rows) - len(ok_rows),
        "wall_seconds": wall_seconds,
        "latency_ms": {
            "p50": _bench_percentile([row["latency_ms"] for row in ok_rows], 50),
            "p95": _bench_percentile([row["latency_ms"] for row in ok_rows], 95),
            "p99": _bench_percentile([row["latency_ms"] for row in ok_rows], 99),
            "mean": (total_latency_ms / len(ok_rows)) if ok_rows else None,
        },
        "ttft_ms": {
            "p50": _bench_percentile([row["ttft_ms"] for row in ok_rows], 50),
            "p95": _bench_percentile([row["ttft_ms"] for row in ok_rows], 95),
            "p99": _bench_percentile([row["ttft_ms"] for row in ok_rows], 99),
        },
        "tpot_ms": {
            "mean": (
                sum(row["tpot_ms"] for row in ok_rows if row["tpot_ms"] is not None)
                / len([row for row in ok_rows if row["tpot_ms"] is not None])
            )
            if any(row["tpot_ms"] is not None for row in ok_rows)
            else None,
            "p95": _bench_percentile([row["tpot_ms"] for row in ok_rows], 95),
        },
        "throughput": {
            "requests_per_second": (len(ok_rows) / wall_seconds) if wall_seconds > 0 else None,
            "input_tokens_per_second": (total_input_tokens / wall_seconds)
            if wall_seconds > 0
            else None,
            "output_tokens_per_second": (total_output_tokens / wall_seconds)
            if wall_seconds > 0
            else None,
            "total_tokens_per_second": (total_tokens / wall_seconds) if wall_seconds > 0 else None,
        },
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
            "total": total_tokens,
        },
        "cost_usd": total_cost,
    }


def _cmd_bench(args: argparse.Namespace) -> int:
    if args.iterations < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("--warmup must be >= 0", file=sys.stderr)
        return 2
    cfg = load_config(project=args.project)
    prompt = _bench_prompt(args)
    workload = args.workload or f"bench_{args.bench_cmd}"
    from somm.client import SommLLM

    llm = SommLLM(config=cfg)
    rows: list[dict] = []
    start: float | None = None
    try:
        for idx in range(args.warmup + args.iterations):
            if idx == args.warmup:
                start = time.monotonic()
            result = llm.generate(
                prompt,
                workload=workload,
                provider=args.provider,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                no_fallback=args.no_fallback,
            )
            if idx < args.warmup:
                continue
            rows.append(
                {
                    "call_id": result.call_id,
                    "provider": result.provider,
                    "model": result.model,
                    "outcome": result.outcome.value,
                    "latency_ms": result.latency_ms,
                    "ttft_ms": result.ttft_ms,
                    "tpot_ms": _bench_tpot_ms(result),
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "cost_usd": result.cost_usd,
                    "error_kind": result.error_kind,
                }
            )
    finally:
        llm.close()
    wall_seconds = time.monotonic() - (start or time.monotonic())
    summary = _bench_summary(rows, wall_seconds=wall_seconds)
    payload = {
        "mode": args.bench_cmd,
        "project": cfg.project,
        "workload": workload,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "summary": summary,
        "runs": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    lat = summary["latency_ms"]
    ttft = summary["ttft_ms"]
    tpot = summary["tpot_ms"]
    thr = summary["throughput"]
    print(
        f"bench {args.bench_cmd}: {cfg.project}/{workload}  n={args.iterations} warmup={args.warmup}"
    )
    print(
        "latency_ms "
        f"p50={_fmt_stat_float(lat['p50'])} "
        f"p95={_fmt_stat_float(lat['p95'])} "
        f"p99={_fmt_stat_float(lat['p99'])} "
        f"ttft95={_fmt_stat_float(ttft['p95'])} "
        f"tpot_mean={_fmt_stat_float(tpot['mean'])}"
    )
    print(
        "throughput "
        f"req/s={_fmt_stat_float(thr['requests_per_second'])} "
        f"out_tok/s={_fmt_stat_float(thr['output_tokens_per_second'])} "
        f"total_tok/s={_fmt_stat_float(thr['total_tokens_per_second'])} "
        f"cost=${summary['cost_usd']:.4f}"
    )
    return 0


def _stats_global(repo, since_days: int) -> list[dict]:
    """Cross-project roll-up from the global mirror. Same shape as
    stats_by_workload but with `project` column."""
    return repo.stats_global_by_workload(since_days=since_days)


# ---------------------------------------------------------------------------
# somm tail


def _cmd_tail(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)

    # Seek cursor: start from now unless --since-minutes specified.
    since = datetime.now(UTC) - timedelta(minutes=args.since_minutes)
    seen_ids: set[str] = set()

    print(f"Tailing calls for project {cfg.project!r} (Ctrl-C to exit)")
    print(
        f"{'time':<20} {'workload':<20} {'provider':<10} {'model':<22} "
        f"{'lat_ms':>7} {'tok_i':>6} {'tok_o':>6} {'cost':>8}  outcome"
    )
    try:
        while True:
            rows = _fetch_since(repo, cfg.project, since, workload=args.workload)
            for r in rows:
                if r["id"] in seen_ids:
                    continue
                seen_ids.add(r["id"])
                ts = r["ts"][:19].replace("T", " ")
                cost_s = f"${r['cost_usd']:.4f}"
                print(
                    f"{ts:<20} {r['workload'][:19]:<20} {r['provider'][:9]:<10} "
                    f"{r['model'][:21]:<22} {r['latency_ms']:>7} {r['tokens_in']:>6} "
                    f"{r['tokens_out']:>6} {cost_s:>8}  {r['outcome']}"
                )
                # Advance cursor past the newest seen row
                row_ts = datetime.fromisoformat(r["ts"])
                if row_ts.tzinfo is None:
                    row_ts = row_ts.replace(tzinfo=UTC)
                if row_ts > since:
                    since = row_ts
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("")
        return 0


def _fetch_since(
    repo: Repository, project: str, since: datetime, workload: str | None = None
) -> list[dict]:
    q = [
        "SELECT c.id, c.ts, COALESCE(w.name, '(unregistered)') AS workload, "
        "       c.provider, c.model, c.latency_ms, c.tokens_in, c.tokens_out, "
        "       c.cost_usd, c.outcome "
        "FROM calls c LEFT JOIN workloads w ON w.id = c.workload_id "
        "WHERE c.project = ? AND c.ts > ? "
    ]
    params: list = [project, since.isoformat()]
    if workload:
        q.append("AND w.name = ? ")
        params.append(workload)
    q.append("ORDER BY c.ts ASC LIMIT 200")

    with repo._open() as conn:
        rows = conn.execute("".join(q), params).fetchall()
    return [
        {
            "id": r[0],
            "ts": r[1],
            "workload": r[2],
            "provider": r[3],
            "model": r[4],
            "latency_ms": r[5],
            "tokens_in": r[6],
            "tokens_out": r[7],
            "cost_usd": r[8] or 0.0,
            "outcome": r[9],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# somm compare


def _cmd_compare(args: argparse.Namespace) -> int:
    """Run a prompt through N models side-by-side. Non-routed, explicit picks.

    Use: somm compare "Summarize X" --models minimax/MiniMax-M3,openai/gpt-4o-mini
    """
    from somm import SommLLM

    specs = _parse_model_specs(args.models)
    if not specs:
        print("no --models supplied. example:", file=sys.stderr)
        print("  somm compare 'hi' --models minimax/MiniMax-M3,openai/gpt-4o-mini", file=sys.stderr)
        return 2

    llm = SommLLM(project=args.project or "compare")
    try:
        results = []
        for provider_name, model in specs:
            # Ensure this provider is in the chain
            if provider_name not in {p.name for p in llm.providers}:
                results.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "error": f"provider {provider_name!r} not configured (missing env key?)",
                    }
                )
                continue
            try:
                r = llm.generate(
                    prompt=args.prompt,
                    workload=args.workload,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    model=model,
                    provider=provider_name,
                )
                results.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "text": r.text,
                        "tokens_in": r.tokens_in,
                        "tokens_out": r.tokens_out,
                        "latency_ms": r.latency_ms,
                        "cost_usd": r.cost_usd,
                        "outcome": r.outcome.value,
                        "call_id": r.call_id,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
    finally:
        llm.close()

    _print_comparison(results, truncate=args.truncate)
    return 0


def _parse_model_specs(raw: list[str] | None) -> list[tuple[str, str]]:
    """Parse --models. Accepts 'provider/model' or 'provider:model'. Comma-
    separated or repeated --models flag."""
    if not raw:
        return []
    specs: list[tuple[str, str]] = []
    for item in raw:
        for token in item.split(","):
            token = token.strip()
            if not token:
                continue
            if "/" in token:
                p, _, m = token.partition("/")
            elif ":" in token:
                p, _, m = token.partition(":")
            else:
                p, m = token, ""
            specs.append((p, m))
    return specs


def _print_comparison(results: list[dict], truncate: int = 240) -> None:
    print(
        f"{'provider':<12} {'model':<26} {'lat_ms':>7} {'tok_i':>6} "
        f"{'tok_o':>6} {'cost':>10}  response"
    )
    for r in results:
        provider = r["provider"][:11]
        model = r["model"][:25]
        if "error" in r:
            print(
                f"{provider:<12} {model:<26} {'—':>7} {'—':>6} {'—':>6} "
                f"{'—':>10}  ERROR: {r['error']}"
            )
            continue
        cost_s = f"${r['cost_usd']:.4f}"
        text = r["text"].replace("\n", "\\n")
        if len(text) > truncate:
            text = text[:truncate] + "…"
        print(
            f"{provider:<12} {model:<26} {r['latency_ms']:>7} {r['tokens_in']:>6} "
            f"{r['tokens_out']:>6} {cost_s:>10}  {text}"
        )
        print(
            f"{'':<12} {'':<26} {'':>7} {'':>6} {'':>6} {'':>10}  [{r['outcome']} · {r['call_id']}]"
        )


# ---------------------------------------------------------------------------
# somm doctor (enhanced)


def _cmd_frontier(args: argparse.Namespace) -> int:
    """Print the adequacy frontier for one workload.

    For each (provider, model) the workload has touched in the window:
    capability-failure rate, detractor rate, p50/p95 latency, mean cost
    per ok call, and whether each workload constraint is exceeded.
    """
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = repo.workload_by_name(args.workload, cfg.project)
    if wl is None:
        print(
            f"No workload {args.workload!r} registered for project {cfg.project!r}.\n"
            f"Register one with somm.workload(...) in your code, then re-run.",
            file=sys.stderr,
        )
        return 2

    rows = repo.workload_frontier(wl.id, since_days=args.since)
    if not rows:
        print(
            f"No calls for workload {wl.name!r} in the last {args.since}d. "
            f"Run the workload, then come back."
        )
        return 0

    cons = {
        "max_p95_latency_ms": wl.max_p95_latency_ms,
        "max_p95_ttft_ms": wl.max_p95_ttft_ms,
        "max_tpot_ms": wl.max_tpot_ms,
        "max_capability_failure_rate": wl.max_capability_failure_rate,
        "max_cost_per_call_usd": wl.max_cost_per_call_usd,
    }
    print(f"Workload: {wl.name}  ({args.since}d window)")
    cons_pretty = ", ".join(
        f"{k.removeprefix('max_')}≤{v}" for k, v in cons.items() if v is not None
    )
    print(f"Constraints: {cons_pretty or '(none set — inspect with `somm workload show <name>`)'}")
    print(
        f"\n{'provider':<14} {'model':<28} {'n':>5} {'cap%':>6} {'det%':>6} "
        f"{'p50ms':>7} {'p95ms':>7} {'ttft95':>7} {'tpot':>7} {'$/ok':>9} fitness"
    )
    for r in rows:
        cap_pct = 100.0 * r["capability_failure_rate"]
        det_pct = 100.0 * r["detractor_rate"]
        p50 = r["p50_latency_ms"] if r["p50_latency_ms"] is not None else "-"
        p95 = r["p95_latency_ms"] if r["p95_latency_ms"] is not None else "-"
        ttft95 = r["p95_ttft_ms"] if r["p95_ttft_ms"] is not None else "-"
        tpot = _fmt_stat_float(r["tpot_ms"])
        cost = r["mean_cost_per_ok_call"]
        cost_s = f"${cost:.5f}" if cost is not None else "-"
        flags = []
        f = r["fitness"]
        if f["exceeds_max_capability_failure_rate"]:
            flags.append("UNFIT(cap)")
        if f["exceeds_max_p95_latency_ms"]:
            flags.append("UNFIT(slow)")
        if f["exceeds_max_p95_ttft_ms"]:
            flags.append("UNFIT(ttft)")
        if f["exceeds_max_tpot_ms"]:
            flags.append("UNFIT(tpot)")
        if f["exceeds_max_cost_per_call_usd"]:
            flags.append("UNFIT($)")
        fitness_s = ",".join(flags) if flags else "ok"
        print(
            f"{r['provider'][:13]:<14} {r['model'][:27]:<28} {r['n_calls']:>5} "
            f"{cap_pct:>5.1f}% {det_pct:>5.1f}% {str(p50):>7} {str(p95):>7} "
            f"{str(ttft95):>7} {tpot:>7} {cost_s:>9} {fitness_s}"
        )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo_exists = cfg.db_path.exists()
    print(f"somm v{VERSION}")
    print(f"project: {cfg.project}  mode: {cfg.mode}")
    print(f"db_path: {cfg.db_path}   exists: {repo_exists}")
    if repo_exists:
        mode = oct(cfg.db_path.stat().st_mode)[-3:]
        ok = mode in ("600",)
        print(f"db perms: {mode} {'(ok)' if ok else '(WARN — expect 600)'}")

    ok_overall = True

    # Ollama
    p = OllamaProvider(base_url=cfg.ollama_url, default_model=cfg.ollama_model)
    h = p.health()
    ok_ollama = h.available
    print(f"ollama:   {'ok' if ok_ollama else 'UNAVAILABLE'}  ({h.detail})")
    ok_overall = ok_overall and ok_ollama

    if not repo_exists:
        print("db missing — skipping intel/workers/cooldowns checks")
        return 0 if ok_overall else 1

    repo = Repository(cfg.db_path)

    # Model intel freshness
    rows = list_intel(repo)
    if not rows:
        print("model_intel: empty (run `somm-serve admin refresh-intel` to populate)")
    else:
        # Bucket by source; show latest last_seen
        by_src: dict[str, list] = {}
        for r in rows:
            by_src.setdefault(r["source"], []).append(r)
        print(f"model_intel: {len(rows)} entries across {len(by_src)} source(s)")
        for src, entries in sorted(by_src.items()):
            latest = max((e["last_seen"] or "") for e in entries)
            age = _age_since(latest) if latest else "—"
            print(f"  {src:<16} {len(entries):>5} models   latest {age}")

    # Worker heartbeats
    with repo._open() as conn:
        heartbeat_rows = conn.execute(
            "SELECT worker_name, last_run_at, last_success_at, consecutive_failures "
            "FROM worker_heartbeat ORDER BY worker_name"
        ).fetchall()
    if not heartbeat_rows:
        print("worker_heartbeat: no heartbeats recorded")
    else:
        print("worker_heartbeat:")
        print(
            f"  {'worker_name':<16} {'last_run_at':<19} "
            f"{'last_success_at':<19} consecutive_failures"
        )
        for name, last_run_at, last_success_at, failures in heartbeat_rows:
            print(
                f"  {name[:15]:<16} {(last_run_at or 'never'):<19} "
                f"{(last_success_at or 'never'):<19} {failures}"
            )

    # Cooldowns
    with repo._open() as conn:
        now = datetime.now(UTC).isoformat()
        cool_rows = conn.execute(
            "SELECT provider, model, cooldown_until, consecutive_failures "
            "FROM provider_health WHERE cooldown_until > ? "
            "ORDER BY cooldown_until",
            (now,),
        ).fetchall()
    if not cool_rows:
        print("cooldowns: none active")
    else:
        print(f"cooldowns: {len(cool_rows)} active")
        for provider, model, until, failures in cool_rows:
            remaining = _age_until(until)
            slot = f"{provider}/{model}" if model else provider
            print(f"  {slot:<42} expires in {remaining}   (failures={failures})")

    return 0 if ok_overall else 1


def _age_since(iso: str) -> str:
    if not iso:
        return "never"
    try:
        dt = datetime.fromisoformat(iso.replace(" ", "T"))
    except (ValueError, TypeError):
        return iso[:19]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - dt
    return _fmt_delta(delta) + " ago"


def _age_until(iso: str) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace(" ", "T"))
    except (ValueError, TypeError):
        return iso[:19]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = dt - datetime.now(UTC)
    return _fmt_delta(delta) if delta.total_seconds() > 0 else "now"


def _fmt_delta(delta: timedelta) -> str:
    s = int(abs(delta.total_seconds()))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60}m"
    return f"{s // 86400}d{(s % 86400) // 3600}h"


# ---------------------------------------------------------------------------
# somm spend


def spend_today(
    db_path: Path,
    project: str,
    default_cap: float | None,
) -> list[dict]:
    """Query today's (UTC) spend per workload. Read-only; no writes.

    Returns list of dicts sorted by spent_usd desc:
      {"workload": str, "spent_usd": float, "cap_usd": float | None}

    cap_usd is the workload's budget_cap_usd_daily; falls back to
    default_cap (config-level ceiling); None when neither is set.
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            SELECT
                COALESCE(w.name, '(unregistered)') AS workload,
                SUM(c.cost_usd) AS spent_usd,
                MAX(w.budget_cap_usd_daily) AS cap_usd
            FROM calls c
            LEFT JOIN workloads w ON w.id = c.workload_id
            WHERE c.project = ?
              AND date(c.ts) = date('now')
              AND c.budget_eligible != 0
            GROUP BY COALESCE(w.name, '(unregistered)')
            ORDER BY spent_usd DESC
            """,
            (project,),
        ).fetchall()

    result = []
    for name, spent, cap in rows:
        effective_cap = cap if cap is not None else default_cap
        result.append(
            {
                "workload": name,
                "spent_usd": float(spent or 0.0),
                "cap_usd": effective_cap,
            }
        )
    return result


_CATALOG_STALE_DAYS = 90


def _cmd_plans(args: argparse.Namespace) -> int:
    from somm_core.plans import (
        learn_observed_limits,
        limit_statuses,
        load_catalog,
        load_plans,
        plans_path,
    )
    from somm_core.registry import fleet_db_paths

    catalog = load_catalog()
    if args.catalog:
        if not catalog:
            print("plan catalog is empty")
            return 0
        print("Known plans (bundled catalog — verify against source before trusting):\n")
        for key in sorted(catalog):
            e = catalog[key]
            age = e.age_days()
            age_s = f"verified {age}d ago" if age is not None else "unverified"
            lims = (
                "; ".join(f"{lim.quota:g} {lim.unit}/{lim.window}" for lim in e.limits)
                or "no limits recorded"
            )
            print(f"  {key:<28} {e.display}")
            print(f"    {lims}  [{age_s}]")
            if e.notes:
                print(f"    note: {e.notes}")
            if e.source:
                print(f"    source: {e.source}")
        return 0

    cfg = load_config(project=args.project)
    try:
        plans = load_plans(catalog=catalog)
    except Exception as exc:
        print(f"plans.toml is invalid: {exc}")
        return 1
    cfg = load_config(project=args.project)
    dbs = (
        [cfg.db_path]
        if args.project_only
        else fleet_db_paths(include=cfg.db_path if cfg.db_path.exists() else None)
    )

    if args.learn:
        updates = learn_observed_limits(
            dbs,
            plans,
            dry_run=args.dry_run,
            min_events=args.min_events,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "path": str(plans_path()),
                        "dry_run": args.dry_run,
                        "updates": [
                            {
                                "provider": u.provider,
                                "window": u.window,
                                "unit": u.unit,
                                "old_quota": u.old_quota,
                                "new_quota": u.new_quota,
                                "n_events": u.n_events,
                                "last_event": u.last_event,
                                "action": u.action,
                            }
                            for u in updates
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        if not updates:
            print("no observed quota ceilings to learn")
        else:
            verb = "would update" if args.dry_run else "updated"
            changed = [u for u in updates if u.action in {"added", "updated"}]
            print(f"{verb} {len(changed)} learned plan limit(s) in {plans_path()}")
            for u in updates:
                old = "-" if u.old_quota is None else f"{u.old_quota:g}"
                print(
                    f"  {u.action:<9} {u.provider:<12} {u.window:<5} {u.unit:<12} "
                    f"{old} -> {u.new_quota:g} ({u.n_events} event(s))"
                )
        if not args.dry_run:
            try:
                plans = load_plans(catalog=catalog)
            except Exception as exc:
                print(f"plans.toml is invalid after learning: {exc}")
                return 1

    if not plans:
        print(f"No plans declared. Create {plans_path()} — example:")
        print(
            "\n  [minimax]"
            '\n  mode = "metered"'
            '\n  plan = "coding-pro"'
            "\n  soft_target_pct = 80"
            "\n  enforce = false"
            "\n  [[minimax.limits]]"
            '\n  window = "month"      # or rolling: "5h", "7d", "1w"'
            "\n  anchor_day = 1        # calendar reset day"
            "\n  quota = 40.0"
            '\n  unit = "usd_equiv"    # requests | tokens_in | tokens_out | tokens_total | usd_equiv'
            "\n\n  [gemini]"
            '\n  mode = "payg"'
        )
        return 0
    statuses = limit_statuses(dbs, plans)
    scope = "this project only" if args.project_only else f"fleet ({len(dbs)} project DBs)"

    if args.json:
        out = []
        for st in statuses:
            out.append(
                {
                    "provider": st.provider,
                    "plan": st.plan_name,
                    "window": st.limit.window,
                    "unit": st.limit.unit,
                    "used": round(st.used, 4),
                    "quota": st.limit.quota,
                    "used_pct": round(st.used_pct, 1),
                    "elapsed_pct": round(st.elapsed_pct, 1),
                    "pace_ratio": round(st.pace_ratio, 2),
                    "projected_pct": round(st.projected_pct, 1),
                    "window_end": st.window_end.isoformat(),
                    "state": st.state,
                    "mode": st.mode,
                }
            )
        from somm_core.plans import payg_burn_rates

        burn = [
            {
                "provider": b.provider,
                "spend_1d": round(b.spend_1d, 4),
                "spend_7d": round(b.spend_7d, 4),
                "spend_30d": round(b.spend_30d, 4),
                "per_day": round(b.per_day, 4),
                "projected_month": round(b.projected_month, 2),
            }
            for b in payg_burn_rates(dbs, plans)
        ]
        print(json.dumps({"scope": scope, "limits": out, "payg_burn": burn}, indent=1))
        return 0

    print(f"Plan usage — {scope}\n")
    if statuses:
        print(
            f"{'provider':<12} {'plan/budget':<14} {'window':<7} {'used / quota':>26} "
            f"{'used%':>6} {'elapsed%':>8} {'pace':>6} {'proj%':>6}  state"
        )
        for st in statuses:
            used_s = (
                f"${st.used:,.2f} / ${st.limit.quota:,.2f}"
                if st.limit.unit == "usd_equiv"
                else f"{st.used:,.0f} / {st.limit.quota:,.0f} {st.limit.unit}"
            )
            label = st.plan_name or ("budget" if st.mode == "payg" else "—")
            print(
                f"{st.provider:<12} {label:<14} {st.limit.window:<7} "
                f"{used_s:>26} {st.used_pct:>5.0f}% {st.elapsed_pct:>7.0f}% "
                f"{st.pace_ratio:>5.1f}x {st.projected_pct:>5.0f}%  {st.state}"
            )

    # Value multiple: what a metered subscription delivered vs. its price,
    # in notional list-price dollars this calendar month.
    from somm_core.plans import PlanLimit as _PL
    from somm_core.plans import usage_in_window as _usage

    value_lines = []
    for prov, pl in sorted(plans.items()):
        if pl.mode != "metered" or pl.price_usd_month <= 0:
            continue
        notional = _usage(dbs, prov, _PL(window="month", quota=1, unit="usd_equiv"))
        if notional > 0:
            mult = notional / pl.price_usd_month
            value_lines.append(
                f"  {prov:<12} ${notional:,.2f} list-price value this month "
                f"on a ${pl.price_usd_month:,.0f}/mo plan (≈{mult:.1f}x)"
            )
    if value_lines:
        print("\nplan value (notional list-price consumed vs subscription price):")
        for line in value_lines:
            print(line)

    metered_no_limits = [p for p, pl in plans.items() if pl.mode == "metered" and not pl.limits]
    if metered_no_limits:
        print(
            f"\nmetered, no limits declared (labelled only): {', '.join(sorted(metered_no_limits))}"
        )

    # PAYG burn rates: no vendor window here — the number that matters is
    # velocity and where it lands by month-end.
    from somm_core.plans import payg_burn_rates

    burn = payg_burn_rates(dbs, plans)
    if burn:
        print("\npayg burn (real dollars):")
        print(f"  {'provider':<12} {'1d':>9} {'7d':>9} {'30d':>9} {'$/day':>8} {'→month':>9}")
        for b in burn:
            print(
                f"  {b.provider:<12} ${b.spend_1d:>8.2f} ${b.spend_7d:>8.2f} "
                f"${b.spend_30d:>8.2f} ${b.per_day:>7.2f} "
                f"${b.projected_month:>8.2f}"
            )
    free = sorted(p for p, pl in plans.items() if pl.mode == "free")
    if free:
        print(f"\nfree/local: {', '.join(free)}")

    # Empirical ceilings: vendors stopped publishing numeric limits, but
    # every quota-429 in your own telemetry is ground truth.
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from somm_core.plans import observed_ceilings, recent_ok_calls

    all_ceilings: dict[str, list] = {}
    printed_header = False
    for prov, pl in sorted(plans.items()):
        if pl.mode != "metered":
            continue
        ceilings = observed_ceilings(dbs, prov)
        if not ceilings:
            continue
        all_ceilings[prov] = ceilings
        if not printed_header:
            print(
                "\nobserved ceilings — inferred from your own quota errors "
                "(median trailing-window usage at each 429):"
            )
            printed_header = True
        for c in ceilings:
            est = f"${c.estimate:,.2f}" if c.unit == "usd_equiv" else f"{c.estimate:,.0f} {c.unit}"
            print(
                f"  {prov:<12} ~{est}/{c.window}  "
                f"(from {c.n_events} event(s), last {c.last_event[:10]})"
            )

    # Quota drift: declared limits are guesses/marketing copy; vendors
    # reset and resize quotas without notice. Two tell-tales, both from
    # your own telemetry:
    #   1. usage exceeds a declared quota while calls keep succeeding
    #      → the real limit is higher (raised, reset, or wrong guess);
    #   2. recent 429-derived ceilings diverge >25% from the declared
    #      quota → the real limit moved.
    for st in statuses:
        if st.mode != "metered":
            continue
        if st.used_pct >= 100 and recent_ok_calls(dbs, st.provider, hours=24) > 0:
            print(
                f"\n⚠ {st.provider}: usage is {st.used_pct:.0f}% of the declared "
                f"{st.limit.window} quota but calls are still succeeding — the "
                f"real limit is likely higher (raised or reset). Update the "
                f"quota in plans.toml, or trust the observed ceilings above."
            )
            continue
        for c in all_ceilings.get(st.provider, []):
            if c.window != st.limit.window or c.unit != st.limit.unit:
                continue
            try:
                last = _dt.fromisoformat(c.last_event)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=_UTC)
                if (_dt.now(_UTC) - last).days > 14:
                    continue  # stale evidence; don't second-guess with it
            except ValueError:
                continue
            drift = abs(c.estimate - st.limit.quota) / st.limit.quota if st.limit.quota else 0
            if drift > 0.25:
                direction = "lower" if c.estimate < st.limit.quota else "higher"
                print(
                    f"\n⚠ {st.provider}: recent 429s put the real "
                    f"{st.limit.window} ceiling ~{drift * 100:.0f}% {direction} "
                    f"than the declared quota ({c.estimate:,.0f} vs "
                    f"{st.limit.quota:,.0f} {st.limit.unit}) — the vendor may "
                    f"have changed it; consider updating plans.toml."
                )

    # Staleness: plan limits are vendor marketing copy, not an API.
    # Nudge re-verification when a referenced catalog entry ages out.
    for pl in plans.values():
        if not pl.catalog_ref:
            continue
        entry = catalog.get(pl.catalog_ref)
        if entry is None:
            continue
        age = entry.age_days()
        if age is None or age > _CATALOG_STALE_DAYS:
            age_s = f"{age}d ago" if age is not None else "never"
            print(
                f"\n⚠ catalog entry {pl.catalog_ref} last verified {age_s} — "
                f"limits may have changed; check {entry.source or 'the vendor page'}"
            )
    return 0


def _cmd_drain_spool(args: argparse.Namespace) -> int:
    from somm_core.repository import Repository

    from somm.telemetry import drain_spool

    cfg = load_config(project=args.project)
    if not cfg.db_path.exists():
        print("no telemetry database found")
        return 1
    n = drain_spool(Repository(cfg.db_path), cfg.spool_dir)
    print(f"drained {n} spooled call(s) into {cfg.db_path}")
    return 0


def _cmd_backfill_costs(args: argparse.Namespace) -> int:
    from somm_core.pricing import backfill_costs, sync_bundled_pricing
    from somm_core.repository import Repository

    cfg = load_config(project=args.project)
    if not cfg.db_path.exists():
        print("no telemetry database found")
        return 1
    repo = Repository(cfg.db_path)
    # Make sure current pricing intel is present before joining against it,
    # so backfill works even on a DB the library hasn't opened since upgrade.
    synced = sync_bundled_pricing(repo)
    if synced:
        print(f"synced {synced} pricing row(s) from the bundled snapshot")
    n, total = backfill_costs(repo, since_days=args.since, dry_run=args.dry_run)
    verb = "would update" if args.dry_run else "updated"
    print(f"{verb} {n} call(s), ${total:.4f} in previously untracked spend")
    if args.dry_run and n:
        print("re-run without --dry-run to apply")
    return 0


def _cmd_spend(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    use_json = getattr(args, "json", False)

    if not cfg.db_path.exists():
        print("[]" if use_json else "no spend recorded today")
        return 0

    rows = spend_today(cfg.db_path, cfg.project, cfg.budget_default_cap_usd_daily)

    if use_json:
        out = []
        for r in rows:
            cap = r["cap_usd"]
            pct = None if cap is None or cap == 0.0 else round(100.0 * r["spent_usd"] / cap, 4)
            out.append(
                {
                    "workload": r["workload"],
                    "spent_usd": r["spent_usd"],
                    "cap_usd": cap,
                    "pct_of_cap": pct,
                }
            )
        print(json.dumps(out, indent=2))
        return 0

    if not rows:
        print("no spend recorded today")
        return 0

    print(f"{'workload':<28} {'spent':>10} {'cap':>10} {'pct':>8}")
    print("-" * 60)
    for r in rows:
        name = r["workload"][:27]
        spent_s = f"${r['spent_usd']:.2f}"
        cap = r["cap_usd"]
        if cap is None:
            cap_s = "—"
            pct_s = "—"
        elif cap == 0.0:
            cap_s = f"${cap:.2f}"
            pct_s = "∞"
        else:
            cap_s = f"${cap:.2f}"
            pct_s = f"{100.0 * r['spent_usd'] / cap:.1f}%"
        print(f"{name:<28} {spent_s:>10} {cap_s:>10} {pct_s:>8}")

    return 0


# ---------------------------------------------------------------------------
# somm serve (thin shim to somm-service)


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from somm_service.cli import main as serve_main
    except ImportError:
        print(
            "somm serve requires somm-service.\n"
            "  uv add somm-service    # or: pip install somm-service",
            file=sys.stderr,
        )
        return 2
    forwarded = []
    if args.project:
        forwarded += ["--project", args.project]
    forwarded += ["--host", args.host, "--port", str(args.port)]
    return serve_main(forwarded)


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="somm", description="somm — self-hosted LLM telemetry")
    p.add_argument("--version", action="version", version=f"somm {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("status", help="show call roll-up for the current project")
    ps.add_argument("--project", default=None)
    ps.add_argument("--since", type=int, default=7, help="window in days (default 7)")
    ps.add_argument("--json", action="store_true", help="emit stable JSON")
    ps.add_argument(
        "--global",
        dest="global_view",
        action="store_true",
        help="read from ~/.somm/global.sqlite (requires SOMM_CROSS_PROJECT)",
    )
    ps.set_defaults(func=_cmd_status)

    pcache = sub.add_parser("cache-advice", help="find low prefix-cache reuse opportunities")
    pcache.add_argument("--project", default=None)
    pcache.add_argument("--since", type=int, default=7, help="window in days (default 7)")
    pcache.add_argument("--min-tokens-in", type=int, default=1_000)
    pcache.add_argument("--max-cache-read-ratio", type=float, default=0.20)
    pcache.add_argument("--json", action="store_true", help="emit stable JSON")
    pcache.set_defaults(func=_cmd_cache_advice)

    pg = sub.add_parser("generate", help="one-shot LLM call through somm")
    pg.add_argument("prompt", nargs="?", help="prompt text, or '-' to read stdin")
    pg.add_argument("--prompt-file", default=None, help="read prompt text from a file")
    pg.add_argument("--project", default=None)
    pg.add_argument("--workload", default="generate")
    pg.add_argument("--system", default="")
    pg.add_argument("--provider", default=None)
    pg.add_argument("--model", default=None)
    pg.add_argument("--max-tokens", type=int, default=1024)
    pg.add_argument("--temperature", type=float, default=0.0)
    pg.add_argument("--json", action="store_true", help="emit a machine-readable result")
    pg.set_defaults(func=_cmd_generate)

    pb = sub.add_parser("bench", help="run latency and throughput probes")
    pb_sub = pb.add_subparsers(dest="bench_cmd", required=True)
    for bench_cmd, default_iterations in (("latency", 5), ("throughput", 10)):
        pbench = pb_sub.add_parser(bench_cmd, help=f"run a {bench_cmd} benchmark")
        pbench.add_argument("prompt", nargs="?", help="prompt text, or '-' to read stdin")
        pbench.add_argument("--prompt-file", default=None, help="read prompt text from a file")
        pbench.add_argument("--project", default=None)
        pbench.add_argument("--workload", default=None)
        pbench.add_argument("--provider", default=None)
        pbench.add_argument("--model", default=None)
        pbench.add_argument("--iterations", type=int, default=default_iterations)
        pbench.add_argument("--warmup", type=int, default=0)
        pbench.add_argument("--max-tokens", type=int, default=256)
        pbench.add_argument("--temperature", type=float, default=0.0)
        pbench.add_argument("--no-fallback", action="store_true")
        pbench.add_argument("--json", action="store_true", help="emit a machine-readable result")
        pbench.set_defaults(func=_cmd_bench)

    pt = sub.add_parser("tail", help="stream new calls as they land")
    pt.add_argument("--project", default=None)
    pt.add_argument("--workload", default=None, help="filter to a single workload")
    pt.add_argument(
        "--since-minutes", type=int, default=0, help="start from N minutes ago (default: now)"
    )
    pt.add_argument("--poll-interval", type=float, default=0.5)
    pt.set_defaults(func=_cmd_tail)

    pc = sub.add_parser("compare", help="run a prompt through N models side-by-side")
    pc.add_argument("prompt", help="the prompt text")
    pc.add_argument(
        "--models",
        action="append",
        required=True,
        help="provider/model (repeatable or comma-separated)",
    )
    pc.add_argument("--project", default=None)
    pc.add_argument("--workload", default="compare")
    pc.add_argument("--max-tokens", type=int, default=256)
    pc.add_argument("--temperature", type=float, default=0.0)
    pc.add_argument(
        "--truncate",
        type=int,
        default=240,
        help="truncate response text at N chars (0 = no truncate)",
    )
    pc.set_defaults(func=_cmd_compare)

    pf = sub.add_parser(
        "frontier",
        help="adequacy frontier per (provider, model) for one workload",
    )
    pf.add_argument("--workload", required=True, help="workload name to inspect")
    pf.add_argument("--project", default=None)
    pf.add_argument("--since", type=int, default=30, help="window in days (default 30)")
    pf.set_defaults(func=_cmd_frontier)

    pw = sub.add_parser("workload", help="register and inspect project workloads")
    pw_sub = pw.add_subparsers(dest="workload_cmd", required=True)

    pwa = pw_sub.add_parser("add", help="register a workload in the project DB")
    pwa.add_argument("name", help="workload name")
    pwa.add_argument("--project", default=None)
    pwa.add_argument("--description", default=None)
    pwa.add_argument(
        "--privacy-class",
        choices=[pc.value for pc in PrivacyClass],
        default=PrivacyClass.INTERNAL.value,
    )
    pwa.add_argument(
        "--from-example",
        choices=sorted(WORKLOAD_EXAMPLES),
        default="freeform",
        help="built-in workload template (default: freeform)",
    )
    pwa.add_argument("--max-p95-latency-ms", type=int, default=None)
    pwa.add_argument("--max-p95-ttft-ms", type=int, default=None)
    pwa.add_argument("--max-tpot-ms", type=float, default=None)
    pwa.add_argument("--max-capability-failure-rate", type=float, default=None)
    pwa.add_argument("--max-cost-per-call-usd", type=float, default=None)
    pwa.set_defaults(func=_cmd_workload_add)

    pwl = pw_sub.add_parser("list", help="list registered workloads")
    pwl.add_argument("--project", default=None)
    pwl.set_defaults(func=_cmd_workload_list)

    pws = pw_sub.add_parser("show", help="show one registered workload")
    pws.add_argument("name", help="workload name")
    pws.add_argument("--project", default=None)
    pws.set_defaults(func=_cmd_workload_show)

    pwc = pw_sub.add_parser("set-constraints", help="update workload adequacy constraints")
    pwc.add_argument("name", help="workload name")
    pwc.add_argument("--project", default=None)
    pwc.add_argument("--max-p95-latency-ms", type=int, default=None)
    pwc.add_argument("--max-p95-ttft-ms", type=int, default=None)
    pwc.add_argument("--max-tpot-ms", type=float, default=None)
    pwc.add_argument("--max-capability-failure-rate", type=float, default=None)
    pwc.add_argument("--max-cost-per-call-usd", type=float, default=None)
    pwc.add_argument("--clear", action="store_true", help="clear all workload constraints")
    pwc.set_defaults(func=_cmd_workload_set_constraints)

    pprompt = sub.add_parser("prompt", help="manage prompt versions, labels, and A/B variants")
    pprompt_sub = pprompt.add_subparsers(dest="prompt_cmd", required=True)

    pplst = pprompt_sub.add_parser("list", help="list prompt versions for a workload")
    pplst.add_argument("--workload", required=True)
    pplst.add_argument("--project", default=None)
    pplst.set_defaults(func=_cmd_prompt_list)

    ppsh = pprompt_sub.add_parser("show", help="show a prompt version or label")
    ppsh.add_argument("--workload", required=True)
    ppsh.add_argument("--project", default=None)
    ppsh_ref = ppsh.add_mutually_exclusive_group()
    ppsh_ref.add_argument("--version", default=None)
    ppsh_ref.add_argument("--label", default=None)
    ppsh.add_argument("--full", action="store_true")
    ppsh.set_defaults(func=_cmd_prompt_show)

    ppreg = pprompt_sub.add_parser("register", help="register a new prompt version")
    ppreg.add_argument("--workload", required=True)
    ppreg.add_argument("--project", default=None)
    ppreg_body = ppreg.add_mutually_exclusive_group(required=True)
    ppreg_body.add_argument("--body-file", default=None)
    ppreg_body.add_argument("--body", default=None)
    ppreg.add_argument("--bump", choices=["minor", "major"], default="minor")
    ppreg.set_defaults(func=_cmd_prompt_register)

    ppfork = pprompt_sub.add_parser("fork", help="fork a prompt from a version or label")
    ppfork.add_argument("--workload", required=True)
    ppfork.add_argument("--project", default=None)
    ppfork.add_argument("--from", dest="from_ref", required=True)
    ppfork.add_argument("--body-file", required=True)
    ppfork.set_defaults(func=_cmd_prompt_fork)

    ppdiff = pprompt_sub.add_parser("diff", help="diff two prompt versions or labels")
    ppdiff.add_argument("--workload", required=True)
    ppdiff.add_argument("--project", default=None)
    ppdiff.add_argument("a")
    ppdiff.add_argument("b")
    ppdiff.set_defaults(func=_cmd_prompt_diff)

    pplabel = pprompt_sub.add_parser("label", help="move a label to a version or weights")
    pplabel.add_argument("--workload", required=True)
    pplabel.add_argument("--project", default=None)
    pplabel.add_argument("--label", required=True)
    pplabel_target = pplabel.add_mutually_exclusive_group(required=True)
    pplabel_target.add_argument("--version", default=None)
    pplabel_target.add_argument("--weights", default=None)
    pplabel.set_defaults(func=_cmd_prompt_label)

    ppprom = pprompt_sub.add_parser("promote", help="promote a version to a label")
    ppprom.add_argument("--workload", required=True)
    ppprom.add_argument("--project", default=None)
    ppprom.add_argument("--version", required=True)
    ppprom.add_argument("--to", required=True)
    ppprom.add_argument("--min-graded", type=int, default=None)
    ppprom.add_argument("--min-score", type=float, default=None)
    ppprom.add_argument("--force", action="store_true")
    ppprom.set_defaults(func=_cmd_prompt_promote)

    ppscore = pprompt_sub.add_parser("score", help="show per-version eval rollups")
    ppscore.add_argument("--workload", required=True)
    ppscore.add_argument("--project", default=None)
    ppscore_ref = ppscore.add_mutually_exclusive_group()
    ppscore_ref.add_argument("--version", default=None)
    ppscore_ref.add_argument("--label", default=None)
    ppscore.set_defaults(func=_cmd_prompt_score)

    peval = sub.add_parser("eval", help="promote datasets and run eval gates")
    peval_sub = peval.add_subparsers(dest="eval_cmd", required=True)

    peval_promote = peval_sub.add_parser(
        "promote-call",
        help="copy a sampled call into a durable eval dataset",
    )
    peval_promote.add_argument("call_id")
    peval_promote.add_argument("--project", default=None)
    peval_promote.add_argument("--dataset", required=True)
    peval_promote.add_argument("--description", default="")
    peval_promote.set_defaults(func=_cmd_eval_promote_call)

    peval_import = peval_sub.add_parser(
        "import",
        help="import reviewed JSONL prompt/expected-response pairs",
    )
    peval_import.add_argument("--workload", required=True)
    peval_import.add_argument("--dataset", required=True)
    peval_import.add_argument("--file", required=True)
    peval_import.add_argument("--project", default=None)
    peval_import.add_argument("--description", default="")
    peval_import.add_argument("--json", action="store_true")
    peval_import.set_defaults(func=_cmd_eval_import)

    peval_run = peval_sub.add_parser(
        "run",
        help="run a workload against a durable eval dataset",
    )
    peval_run.add_argument("--workload", required=True)
    peval_run.add_argument("--dataset", required=True)
    peval_run.add_argument("--project", default=None)
    peval_run.add_argument("--threshold", type=float, default=0.8)
    peval_run.add_argument("--max-tokens", type=int, default=1024)
    peval_run.add_argument("--temperature", type=float, default=0.0)
    peval_run.add_argument("--provider", default=None)
    peval_run.add_argument("--model", default=None)
    peval_run.add_argument(
        "--implementation",
        default=None,
        help="explicit implementation coordinate, normally a Git commit SHA",
    )
    peval_run.add_argument(
        "--judge-config",
        default=None,
        help="JSON binary-rubric criteria and explicit judge panel",
    )
    peval_run.add_argument("--json", action="store_true")
    peval_run.set_defaults(func=_cmd_eval_run)

    popt = sub.add_parser(
        "optimize",
        help="propose a prompt fork from failing graded calls",
    )
    popt.add_argument("--workload", required=True)
    popt.add_argument("--project", default=None)
    popt.add_argument("--from", dest="from_ref", default="production")
    popt.add_argument("--threshold", type=float, default=0.8)
    popt.add_argument("--limit", type=int, default=8)
    popt.add_argument("--label", default="proposed")
    popt.add_argument("--max-tokens", type=int, default=2048)
    popt.add_argument("--temperature", type=float, default=0.2)
    popt.add_argument("--provider", default=None)
    popt.add_argument("--model", default=None)
    popt.set_defaults(func=_cmd_optimize)

    pcamp = sub.add_parser("campaign", help="run durable experiment campaigns")
    pcamp_sub = pcamp.add_subparsers(dest="campaign_cmd", required=True)
    pcamp_run = pcamp_sub.add_parser(
        "run",
        help="run repeated eval rounds with keep/revert logging",
    )
    pcamp_run.add_argument("--workload", required=True)
    pcamp_run.add_argument("--dataset", required=True)
    pcamp_run.add_argument("--project", default=None)
    pcamp_run.add_argument("--name", default=None)
    pcamp_run.add_argument(
        "--metric",
        choices=("mean_score", "pass_rate", "error_rate"),
        default="mean_score",
    )
    pcamp_run.add_argument("--direction", choices=("gte", "lte"), default=None)
    pcamp_run.add_argument("--threshold", type=float, default=0.8)
    pcamp_run.add_argument("--eval-threshold", type=float, default=None)
    pcamp_run.add_argument("--max-rounds", type=int, default=5)
    pcamp_run.add_argument("--token-budget", type=int, default=None)
    pcamp_run.add_argument("--plateau-window", type=int, default=2)
    pcamp_run.add_argument("--min-delta", type=float, default=0.0)
    pcamp_run.add_argument("--max-tokens", type=int, default=1024)
    pcamp_run.add_argument("--temperature", type=float, default=0.0)
    pcamp_run.add_argument("--provider", default=None)
    pcamp_run.add_argument("--model", default=None)
    pcamp_run.add_argument("--log", default=None)
    pcamp_run.add_argument("--json", action="store_true")
    pcamp_run.set_defaults(func=_cmd_campaign_run)

    pinbox = sub.add_parser("inbox", help="list, apply, and dismiss recommendations")
    pinbox_sub = pinbox.add_subparsers(dest="inbox_cmd", required=True)
    pinbox_list = pinbox_sub.add_parser("list", help="list recommendations")
    pinbox_list.add_argument("--project", default=None)
    pinbox_list.add_argument("--workload", default=None)
    pinbox_list.add_argument("--all", action="store_true")
    pinbox_list.add_argument("--json", action="store_true")
    pinbox_list.set_defaults(func=_cmd_inbox_list)

    pinbox_apply = pinbox_sub.add_parser("apply", help="apply a recommendation")
    pinbox_apply.add_argument("recommendation_id", type=int)
    pinbox_apply.add_argument("--project", default=None)
    pinbox_apply.add_argument("--json", action="store_true")
    pinbox_apply.set_defaults(func=_cmd_inbox_apply)

    pinbox_dismiss = pinbox_sub.add_parser("dismiss", help="dismiss a recommendation")
    pinbox_dismiss.add_argument("recommendation_id", type=int)
    pinbox_dismiss.add_argument("--project", default=None)
    pinbox_dismiss.add_argument("--json", action="store_true")
    pinbox_dismiss.set_defaults(func=_cmd_inbox_dismiss)

    pp = sub.add_parser("plugin", help="list and inspect plugins, hooks, and providers")
    pp_sub = pp.add_subparsers(dest="plugin_cmd", required=True)

    pplg = pp_sub.add_parser("list", help="list reference plugins, active hooks, and providers")
    pplg.set_defaults(func=_cmd_plugin_list)

    ppi = pp_sub.add_parser("info", help="show one reference plugin")
    ppi.add_argument("name", help="reference plugin name")
    ppi.set_defaults(func=_cmd_plugin_info)

    pd = sub.add_parser("doctor", help="check config + ollama + db + intel + workers + cooldowns")
    pd.add_argument("--project", default=None)
    pd.set_defaults(func=_cmd_doctor)

    psr = sub.add_parser("serve", help="run the web admin + HTTP API (localhost:7878)")
    psr.add_argument("--project", default=None)
    psr.add_argument("--host", default="127.0.0.1")
    psr.add_argument("--port", type=int, default=7878)
    psr.set_defaults(func=_cmd_serve)

    pspend = sub.add_parser("spend", help="today's LLM spend vs daily budget cap per workload")
    pspend.add_argument("--project", default=None)
    pspend.add_argument(
        "--json",
        dest="json",
        action="store_true",
        default=False,
        help="emit a JSON array instead of the aligned table",
    )
    pspend.set_defaults(func=_cmd_spend)

    pbf = sub.add_parser(
        "backfill-costs",
        help="recompute cost_usd for $0 calls that now have pricing intel",
    )
    pbf.add_argument("--project", default=None)
    pbf.add_argument("--since", type=int, default=None, help="only calls from the last N days")
    pbf.add_argument("--dry-run", action="store_true", help="report without writing")
    pbf.set_defaults(func=_cmd_backfill_costs)

    ppl = sub.add_parser(
        "plans",
        help="metered-plan quota usage + pacing (PAYG vs metered, fleet-wide)",
    )
    ppl.add_argument("--project", default=None)
    ppl.add_argument(
        "--project-only",
        action="store_true",
        help="count only this project's usage (default: whole fleet — quotas are shared)",
    )
    ppl.add_argument("--json", action="store_true")
    ppl.add_argument(
        "--catalog",
        action="store_true",
        help="list known plans from the bundled catalog (with sources + verified dates)",
    )
    ppl.add_argument(
        "--learn",
        action="store_true",
        help="write observed quota ceilings from recent 429s into plans.toml",
    )
    ppl.add_argument("--dry-run", action="store_true", help="show learned limits without writing")
    ppl.add_argument(
        "--min-events",
        type=int,
        default=1,
        help="minimum quota-error events required to learn a ceiling",
    )
    ppl.set_defaults(func=_cmd_plans)

    pds = sub.add_parser(
        "drain-spool",
        help="replay spooled JSONL telemetry (written during DB outages) into the DB",
    )
    pds.add_argument("--project", default=None)
    pds.set_defaults(func=_cmd_drain_spool)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        if getattr(args, "json", False):
            code = _json_error_exit_code(exc)
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                            "exit_code": code,
                        },
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return code
        raise


def _json_error_exit_code(exc: Exception) -> int:
    if isinstance(exc, (ValueError, argparse.ArgumentError)):
        return 2
    if isinstance(exc, FileNotFoundError):
        return 66
    if isinstance(exc, PermissionError):
        return 77
    return 1


if __name__ == "__main__":
    sys.exit(main())
