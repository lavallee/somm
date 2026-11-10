"""somm prompt: prompt versions, labels and A/B variants."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from somm_core.repository import Repository

from somm.cli import _common


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

    cfg = _common.load_config(project=args.project)
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

    cfg = _common.load_config(project=args.project)
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

    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = _prompt_workload(repo, cfg.project, args.workload)
    if wl is None:
        return 2
    prompt = register_prompt(repo, wl.id, _body_from_args(args), bump=args.bump)
    print(f"registered {prompt.version} ({prompt.id[:8]})")
    return 0


def _cmd_prompt_fork(args: argparse.Namespace) -> int:
    from somm.prompts import PromptNotFound, fork_prompt

    cfg = _common.load_config(project=args.project)
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

    cfg = _common.load_config(project=args.project)
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

    cfg = _common.load_config(project=args.project)
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

    cfg = _common.load_config(project=args.project)
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

    cfg = _common.load_config(project=args.project)
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


def add_parsers(sub: argparse._SubParsersAction) -> None:
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
