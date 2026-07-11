from __future__ import annotations

import sqlite3

from somm_core.repository import Repository
from somm_core.schema import _list_migrations, current_schema_version, ensure_schema
from somm_core.version import SCHEMA_VERSION


def test_v10_database_upgrades_to_current_schema(tmp_path):
    db_path = tmp_path / "v10.sqlite"
    with sqlite3.connect(db_path) as conn:
        for version, path in _list_migrations():
            if version > 10:
                continue
            conn.executescript(path.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

        assert current_schema_version(conn) == 10

        upgraded = ensure_schema(conn)

        assert upgraded == SCHEMA_VERSION == 20
        assert current_schema_version(conn) == 20
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "prompt_labels" in tables
        assert "prompt_label_history" in tables

        prompt_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(prompts)").fetchall()
        }
        assert "parent_prompt_id" in prompt_columns

        indexes = {
            row[1]
            for row in conn.execute(
                "SELECT type, name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert "idx_prompt_label_history_lookup" in indexes
        assert "idx_prompts_parent_prompt_id" in indexes


def test_v11_database_upgrades_to_current_schema(tmp_path):
    db_path = tmp_path / "v11.sqlite"
    with sqlite3.connect(db_path) as conn:
        for version, path in _list_migrations():
            if version > 11:
                continue
            conn.executescript(path.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

        assert current_schema_version(conn) == 11

        upgraded = ensure_schema(conn)

        assert upgraded == SCHEMA_VERSION == 20
        assert current_schema_version(conn) == 20
        call_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()
        }
        assert {
            "ttft_ms",
            "session_id",
            "parent_call_id",
            "cache_tokens_in",
            "cache_tokens_out",
            "citations_json",
        }.issubset(call_columns)

        indexes = {
            row[1]
            for row in conn.execute(
                "SELECT type, name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert "idx_calls_session_ts" in indexes
        assert "idx_calls_parent_call" in indexes


def test_v12_database_upgrades_to_v13_workload_revisions(tmp_path):
    db_path = tmp_path / "v12.sqlite"
    with sqlite3.connect(db_path) as conn:
        for version, path in _list_migrations():
            if version > 12:
                continue
            conn.executescript(path.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

        conn.execute(
            "INSERT INTO workloads (id, name, project) VALUES ('w1', 'work', 'proj')"
        )
        conn.commit()

        assert current_schema_version(conn) == 12

        upgraded = ensure_schema(conn)

        assert upgraded == SCHEMA_VERSION == 20
        assert current_schema_version(conn) == 20

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "workload_revisions" in tables

        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(workload_revisions)").fetchall()
        }
        assert {
            "id",
            "workload_id",
            "revision",
            "config_json",
            "created_at",
            "created_by",
        }.issubset(columns)

        indexes = {
            row[1]
            for row in conn.execute(
                "SELECT type, name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert "idx_workload_revisions_wl_rev" in indexes
        assert "idx_workload_revisions_wl" in indexes


def test_v13_database_upgrades_to_v14_prompt_label_weights(tmp_path):
    db_path = tmp_path / "v13.sqlite"
    with sqlite3.connect(db_path) as conn:
        for version, path in _list_migrations():
            if version > 13:
                continue
            conn.executescript(path.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

        assert current_schema_version(conn) == 13

        upgraded = ensure_schema(conn)

        assert upgraded == SCHEMA_VERSION == 20
        assert current_schema_version(conn) == 20
        label_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(prompt_labels)").fetchall()
        }
        assert "weights_json" in label_columns


def test_v14_database_upgrades_to_v15_workload_policy(tmp_path):
    db_path = tmp_path / "v14.sqlite"
    with sqlite3.connect(db_path) as conn:
        for version, path in _list_migrations():
            if version > 14:
                continue
            conn.executescript(path.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

        assert current_schema_version(conn) == 14

        upgraded = ensure_schema(conn)

        assert upgraded == SCHEMA_VERSION == 20
        assert current_schema_version(conn) == 20
        workload_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(workloads)").fetchall()
        }
        assert "policy_json" in workload_columns


def test_v15_database_upgrades_to_v16_datasets(tmp_path):
    db_path = tmp_path / "v15.sqlite"
    with sqlite3.connect(db_path) as conn:
        for version, path in _list_migrations():
            if version > 15:
                continue
            conn.executescript(path.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

        assert current_schema_version(conn) == 15

        upgraded = ensure_schema(conn)

        assert upgraded == SCHEMA_VERSION == 20
        assert current_schema_version(conn) == 20
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"datasets", "dataset_items"}.issubset(tables)

        dataset_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(datasets)").fetchall()
        }
        assert {
            "id",
            "project",
            "workload_id",
            "name",
            "description",
            "created_at",
            "updated_at",
        }.issubset(dataset_columns)
        item_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(dataset_items)").fetchall()
        }
        assert {
            "id",
            "dataset_id",
            "source_call_id",
            "prompt_body",
            "expected_response_body",
            "metadata_json",
            "created_at",
        }.issubset(item_columns)

        indexes = {
            row[1]
            for row in conn.execute(
                "SELECT type, name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert "idx_datasets_project_workload_name" in indexes
        assert "idx_dataset_items_dataset" in indexes


def test_v16_database_upgrades_to_v17_eval_receipts(tmp_path):
    db_path = tmp_path / "v16.sqlite"
    with sqlite3.connect(db_path) as conn:
        for version, path in _list_migrations():
            if version > 16:
                continue
            conn.executescript(path.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

        assert current_schema_version(conn) == 16

        upgraded = ensure_schema(conn)

        assert upgraded == SCHEMA_VERSION == 20
        assert current_schema_version(conn) == 20
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "eval_receipts" in tables

        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(eval_receipts)").fetchall()
        }
        assert {
            "id",
            "eval_result_id",
            "run_id",
            "receipt_type",
            "call_id",
            "dataset_id",
            "dataset_item_id",
            "candidate_a_call_id",
            "candidate_b_call_id",
            "winner",
            "score",
            "threshold",
            "payload_json",
            "created_at",
        }.issubset(columns)

        indexes = {
            row[1]
            for row in conn.execute(
                "SELECT type, name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert "idx_eval_receipts_eval_result" in indexes
        assert "idx_eval_receipts_pair" in indexes


def test_v17_database_upgrades_to_v18_campaigns(tmp_path):
    db_path = tmp_path / "v17.sqlite"
    with sqlite3.connect(db_path) as conn:
        for version, path in _list_migrations():
            if version > 17:
                continue
            conn.executescript(path.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

        assert current_schema_version(conn) == 17

        upgraded = ensure_schema(conn)

        assert upgraded == SCHEMA_VERSION == 20
        assert current_schema_version(conn) == 20
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"campaigns", "campaign_events"}.issubset(tables)

        campaign_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(campaigns)").fetchall()
        }
        assert {
            "id",
            "project",
            "workload_id",
            "dataset_id",
            "metric",
            "direction",
            "threshold",
            "token_budget",
            "plateau_window",
            "best_score",
            "metadata_json",
            "completed_at",
        }.issubset(campaign_columns)
        event_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(campaign_events)").fetchall()
        }
        assert {
            "id",
            "campaign_id",
            "sequence",
            "run_id",
            "event_type",
            "action",
            "metric_score",
            "payload_json",
        }.issubset(event_columns)

        indexes = {
            row[1]
            for row in conn.execute(
                "SELECT type, name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert "idx_campaigns_project_workload" in indexes
        assert "idx_campaign_events_campaign" in indexes


def test_v18_database_upgrades_to_v19_model_aliases(tmp_path):
    db_path = tmp_path / "v18.sqlite"
    with sqlite3.connect(db_path) as conn:
        for version, path in _list_migrations():
            if version > 18:
                continue
            conn.executescript(path.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

        assert current_schema_version(conn) == 18

        upgraded = ensure_schema(conn)

        assert upgraded == SCHEMA_VERSION == 20
        assert current_schema_version(conn) == 20
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "model_aliases" in tables
        alias_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(model_aliases)").fetchall()
        }
        assert {
            "provider",
            "model",
            "canonical_id",
            "source",
            "created_at",
            "updated_at",
        }.issubset(alias_columns)
        indexes = {
            row[1]
            for row in conn.execute(
                "SELECT type, name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert "idx_model_aliases_canonical" in indexes


def test_v19_database_upgrades_to_v20_workload_serving_slos(tmp_path):
    db_path = tmp_path / "v19.sqlite"
    with sqlite3.connect(db_path) as conn:
        for version, path in _list_migrations():
            if version > 19:
                continue
            conn.executescript(path.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

        assert current_schema_version(conn) == 19

        upgraded = ensure_schema(conn)

        assert upgraded == SCHEMA_VERSION == 20
        assert current_schema_version(conn) == 20
        workload_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(workloads)").fetchall()
        }
        assert {"max_p95_ttft_ms", "max_tpot_ms"}.issubset(workload_columns)


def test_repository_model_alias_roundtrip(tmp_path):
    repo = Repository(tmp_path / "aliases.sqlite")

    repo.set_model_alias(
        "google/gemini-2.5-flash",
        "openrouter",
        "google/gemini-2.5-flash",
        source="test",
    )

    assert (
        repo.canonical_model_id("openrouter", "google/gemini-2.5-flash")
        == "google/gemini-2.5-flash"
    )
    assert repo.canonical_model_id("anthropic", "claude-sonnet") == (
        "anthropic/claude-sonnet"
    )
    assert repo.model_alias_map()[
        ("openrouter", "google/gemini-2.5-flash")
    ] == "google/gemini-2.5-flash"
    aliases = repo.model_aliases("google/gemini-2.5-flash")
    assert len(aliases) == 1
    assert aliases[0].provider == "openrouter"
    assert aliases[0].source == "test"


def test_migration_and_version_stamp_are_atomic(tmp_path, monkeypatch):
    """A migration that fails partway must leave neither the schema change nor
    the version bump — otherwise a re-run hits a duplicate ADD COLUMN and wedges
    the DB. The DDL and the schema_version stamp share one transaction."""
    import somm_core.schema as schema_mod

    db = tmp_path / "atomic.sqlite"
    conn = sqlite3.connect(db)
    # Bring it to a known good state first.
    ensure_schema(conn)
    good_version = current_schema_version(conn)

    # Inject a fake pending migration whose second statement fails.
    bad_version = good_version + 1
    bad_sql = (
        "CREATE TABLE _atomic_probe (x INTEGER);\n"
        "INSERT INTO _atomic_probe (nonexistent_column) VALUES (1);\n"  # fails
    )

    class _FakePath:
        def read_text(self):
            return bad_sql

    orig = schema_mod._list_migrations
    monkeypatch.setattr(
        schema_mod, "_list_migrations", lambda: [(bad_version, _FakePath())]
    )
    try:
        raised = False
        try:
            ensure_schema(conn)
        except Exception:
            raised = True
        assert raised
    finally:
        monkeypatch.setattr(schema_mod, "_list_migrations", orig)

    # The failed migration rolled back entirely: version unchanged AND the
    # half-created probe table is gone.
    assert current_schema_version(conn) == good_version
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "_atomic_probe" not in tables
    conn.close()
