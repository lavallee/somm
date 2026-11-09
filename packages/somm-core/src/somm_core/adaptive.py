"""Opt-in, conservative task routing. Unknown work takes the stronger route.

The caller supplies application-specific fast patterns. Tool results are never
classified as new human requests; short continuations inherit their task.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Literal

_CONTINUATION = re.compile(r"^(yes|yeah|yep|ok|okay|sure|go for it|go ahead|do it|please do|continue|thanks|thank you)[.!\s]*$", re.I)
_FACTUAL = re.compile(r"\b(bio(?:graphy)?|résumé|resume|career|employment|employer|professional (?:profile|background)|fact.?check|verify|research|sources?|citations?|medical|legal|financial)\b", re.I)


@dataclass(frozen=True)
class WeeklyQuota:
    used_percent: float | None
    resets_at: int | None = None
    observed_at: float = 0
    source: str = "codex:account/rateLimits/read"
    error: str | None = None

    def usable(self, now: float, max_age: float) -> bool:
        return (self.used_percent is not None and 0 <= self.used_percent <= 100
                and 0 <= now - self.observed_at <= max_age
                and self.resets_at is not None and self.resets_at > now)


@dataclass(frozen=True)
class RouteTarget:
    provider: str
    model: str
    reasoning_effort: str | None = None

    @classmethod
    def parse(cls, value: dict) -> RouteTarget:
        if not isinstance(value, dict) or set(value) - {"provider", "model", "reasoning_effort"}:
            raise ValueError("adaptive targets accept provider, model and reasoning_effort")
        if any(not isinstance(value.get(k), str) or not value[k].strip() for k in ("provider", "model")):
            raise ValueError("adaptive targets require nonempty provider and model")
        effort = value.get("reasoning_effort")
        if effort is not None and not isinstance(effort, str):
            raise ValueError("adaptive reasoning_effort must be a string")
        return cls(value["provider"].strip(), value["model"].strip(), effort)


@dataclass(frozen=True)
class AdaptivePolicy:
    fast: RouteTarget
    strong: RouteTarget
    fallback: RouteTarget
    fast_patterns: tuple[str, ...] = ()
    weekly_limit_percent: float = 90
    quota_max_age_seconds: float = 300

    @classmethod
    def parse(cls, value: dict) -> AdaptivePolicy:
        keys = {"fast", "strong", "fallback", "fast_patterns", "weekly_limit_percent", "quota_max_age_seconds"}
        if not isinstance(value, dict) or set(value) - keys:
            raise ValueError("adaptive policy contains unknown fields")
        targets = [RouteTarget.parse(value.get(k)) for k in ("fast", "strong", "fallback")]
        if targets[2].provider in {"chatgpt", "codex-cli"}:
            raise ValueError("adaptive fallback must not consume the conserved ChatGPT quota")
        patterns = value.get("fast_patterns", [])
        if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
            raise ValueError("fast_patterns must be a list of regular expressions")
        for pattern in patterns:
            try:
                re.compile(pattern, re.I)
            except re.error as exc:
                raise ValueError("invalid adaptive fast pattern") from exc
        limit = value.get("weekly_limit_percent", 90)
        age = value.get("quota_max_age_seconds", 300)
        if isinstance(limit, bool) or not isinstance(limit, (float, int)) or not 0 < limit <= 100:
            raise ValueError("weekly_limit_percent must be greater than zero and at most 100")
        if isinstance(age, bool) or not isinstance(age, (float, int)) or not 0 < age <= 3600:
            raise ValueError("quota_max_age_seconds must be greater than zero and at most 3600")
        return cls(*targets, tuple(patterns), float(limit), float(age))


@dataclass(frozen=True)
class RouteDecision:
    tier: Literal["fast", "strong"]
    target: RouteTarget | None
    reason: str
    quota: WeeklyQuota
    deferred: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def human_turns(messages: list[dict]) -> list[str]:
    turns = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("messages must contain objects")
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(v.get("text", "")) for v in content
                                if isinstance(v, dict) and v.get("type") in {"text", "input_text"})
        if isinstance(content, str) and content.strip():
            turns.append(content.strip())
    return turns


def choose_route(policy: AdaptivePolicy, messages: list[dict], quota: WeeklyQuota,
                 *, now: float | None = None) -> RouteDecision:
    now = time.time() if now is None else now
    turns = human_turns(messages)
    text = ""
    for turn in reversed(turns):
        text = turn
        if not _CONTINUATION.fullmatch(turn):
            break
    if _FACTUAL.search(text):
        tier, reason = "strong", "factual_or_research"
    elif text and (re.fullmatch(r"(?:hi|hello|hey|good (?:morning|evening))[!.\s]*", text, re.I)
                   or any(re.fullmatch(pattern, text, re.I) for pattern in policy.fast_patterns)):
        tier, reason = "fast", "known_simple_task"
    else:
        tier, reason = "strong", "ambiguous_or_complex"
    target = policy.fast if tier == "fast" else policy.strong
    if target.provider != "chatgpt":
        return RouteDecision(tier, target, reason, quota)
    usable = quota.usable(now, policy.quota_max_age_seconds)
    if not usable or quota.used_percent >= policy.weekly_limit_percent:
        reason = "weekly_quota_unknown" if not usable else "weekly_quota_conservation"
        return RouteDecision(tier, policy.fallback if tier == "fast" else None, reason, quota,
                             deferred=tier == "strong")
    return RouteDecision(tier, target, reason, quota)
