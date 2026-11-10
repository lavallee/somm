"""somm eval, optimize, campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from somm_core.models import Outcome
from somm_core.repository import Repository

from somm.cli import _common


def _cmd_eval_promote_call(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
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
    cfg = _common.load_config(project=args.project)
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

    cfg = _common.load_config(project=args.project)
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
                reasoning_effort=args.reasoning_effort,
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


def _cmd_optimize(args: argparse.Namespace) -> int:
    from somm import SommLLM
    from somm.optimize import propose_prompt_optimization
    from somm.prompts import PromptNotFound

    cfg = _common.load_config(project=args.project)
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


def _cmd_campaign_run(args: argparse.Namespace) -> int:
    from somm import SommLLM
    from somm.campaigns import MetricContract, run_eval_campaign, write_campaign_jsonl

    cfg = _common.load_config(project=args.project)
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


def add_parsers(sub: argparse._SubParsersAction) -> None:
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
        "--reasoning-effort",
        default=None,
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        help=(
            "reasoning budget for models that dial it per request; sweeping this"
            " across runs is how a model gets evaluated rather than assumed"
        ),
    )
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
