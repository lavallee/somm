"""Call-site capture (schema 23).

A workload name says what kind of work a call is; it does not say which code
asked for it. These tests pin the capture behavior that makes telemetry
joinable to source, and — as much as anything — the abstentions: a site that
cannot be determined must read as unknown rather than as an answer.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from somm import hooks
from somm_core.models import Call, Outcome
from somm_core.repository import Repository


@pytest.fixture(autouse=True)
def _restore_provider():
    yield
    hooks.set_call_site_provider(None)
    hooks._site_path_cache.clear()


# -- capture ----------------------------------------------------------------


def test_default_capture_names_the_calling_file_and_line() -> None:
    site = hooks.current_call_site()
    assert site is not None
    path, _, line = site.rpartition(":")
    assert path.endswith("tests/test_call_site.py")
    assert line.isdigit()


def test_capture_walks_out_of_somm_frames() -> None:
    """The walk is by module name, not a fixed depth, so it stays correct as
    somm's internal call path changes."""

    def through_a_somm_frame() -> str | None:
        # Masquerade as a somm module: the walk must skip this frame.
        globals_ = {"__name__": "somm.client"}
        return eval("current()", {**globals_, "current": hooks.current_call_site})

    site = through_a_somm_frame()
    assert site is not None
    assert "test_call_site.py" in site


def test_provider_overrides_the_default() -> None:
    hooks.set_call_site_provider(lambda: "svc/dispatch.py:42")
    assert hooks.current_call_site() == "svc/dispatch.py:42"


def test_clearing_the_provider_restores_the_default() -> None:
    hooks.set_call_site_provider(lambda: "svc/dispatch.py:42")
    hooks.set_call_site_provider(None)
    site = hooks.current_call_site()
    assert site is not None and "test_call_site.py" in site


def test_a_provider_returning_none_disables_capture() -> None:
    hooks.set_call_site_provider(lambda: None)
    assert hooks.current_call_site() is None


def test_a_broken_provider_never_raises_into_the_call_path() -> None:
    def boom() -> str | None:
        raise RuntimeError("provider is broken")

    hooks.set_call_site_provider(boom)
    assert hooks.current_call_site() is None


# -- path resolution --------------------------------------------------------


def test_site_is_repo_relative_and_names_the_repo(tmp_path) -> None:
    """A site is only joinable to a static audit if it says which repo it is
    in, so the repository directory itself stays in the path."""
    repo = tmp_path / "acme-service"
    (repo / ".git").mkdir(parents=True)
    src = repo / "acme_service" / "ingest"
    src.mkdir(parents=True)
    target = src / "tools.py"
    target.write_text("x = 1\n")

    assert hooks._relative_site_path(str(target)) == "acme-service/acme_service/ingest/tools.py"


def test_repo_marker_beats_a_nearer_package_marker(tmp_path) -> None:
    """In a monorepo the nearest pyproject.toml is a *package* root. Stopping
    there yields `somm/src/somm/client.py` where the joinable answer is
    `somm/packages/somm/src/somm/client.py`."""
    repo = tmp_path / "somm"
    (repo / ".git").mkdir(parents=True)
    pkg = repo / "packages" / "somm"
    pkg.mkdir(parents=True)
    (pkg / "pyproject.toml").write_text("[project]\n")
    src = pkg / "src" / "somm"
    src.mkdir(parents=True)
    target = src / "client.py"
    target.write_text("x = 1\n")

    assert hooks._relative_site_path(str(target)) == "somm/packages/somm/src/somm/client.py"


def test_package_marker_is_used_when_there_is_no_repo(tmp_path) -> None:
    pkg = tmp_path / "loose"
    (pkg / "src").mkdir(parents=True)
    (pkg / "pyproject.toml").write_text("[project]\n")
    target = pkg / "src" / "mod.py"
    target.write_text("x = 1\n")

    assert hooks._relative_site_path(str(target)) == "loose/src/mod.py"


def test_a_file_outside_any_project_does_not_leak_the_filesystem(tmp_path) -> None:
    """A stdlib or site-packages path says more about this machine than about
    the call."""
    stray = tmp_path / "deep" / "nested" / "mod.py"
    stray.parent.mkdir(parents=True)
    stray.write_text("x = 1\n")

    resolved = hooks._relative_site_path(str(stray))
    assert resolved == "nested/mod.py"
    assert str(tmp_path) not in resolved


def test_resolution_is_cached_per_file(tmp_path) -> None:
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)
    target = repo / "mod.py"
    target.write_text("x = 1\n")

    first = hooks._relative_site_path(str(target))
    target.unlink()  # resolution must not re-stat
    assert hooks._relative_site_path(str(target)) == first


def test_the_path_cache_is_bounded(tmp_path) -> None:
    """Generated or exec'd code can produce unbounded distinct filenames."""
    hooks._site_path_cache.clear()
    for i in range(hooks._MAX_SITE_CACHE + 50):
        hooks._relative_site_path(str(tmp_path / f"gen_{i}.py"))
    assert len(hooks._site_path_cache) <= hooks._MAX_SITE_CACHE


# -- persistence ------------------------------------------------------------


def _call(call_id: str, site: str | None) -> Call:
    return Call(
        id=call_id,
        ts=datetime.now(UTC),
        project="p",
        workload_id=None,
        prompt_id=None,
        provider="ollama",
        model="m",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        cost_usd=0.0,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash="h",
        response_hash="h",
        call_site=site,
    )


def test_call_site_round_trips_through_a_single_insert(tmp_path) -> None:
    repo = Repository(tmp_path / "t.sqlite")
    repo.write_call(_call("c1", "acme-service/acme_service/ingest/tools.py:88"))
    assert repo.get_call("c1").call_site == "acme-service/acme_service/ingest/tools.py:88"


def test_call_site_round_trips_through_a_batch_insert(tmp_path) -> None:
    repo = Repository(tmp_path / "t.sqlite")
    repo.write_calls_batch([_call("c1", "a/b.py:1"), _call("c2", None)])
    assert repo.get_call("c1").call_site == "a/b.py:1"
    assert repo.get_call("c2").call_site is None


def test_an_undetermined_site_is_stored_as_null_not_empty(tmp_path) -> None:
    repo = Repository(tmp_path / "t.sqlite")
    repo.write_call(_call("c1", None))
    with repo._open() as conn:
        stored = conn.execute("SELECT call_site FROM calls WHERE id = 'c1'").fetchone()[0]
    assert stored is None
