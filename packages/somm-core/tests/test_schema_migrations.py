from __future__ import annotations

import sqlite3

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

        assert upgraded == SCHEMA_VERSION == 13
        assert current_schema_version(conn) == 13
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

        assert upgraded == SCHEMA_VERSION == 13
        assert current_schema_version(conn) == 13
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

        assert upgraded == SCHEMA_VERSION == 13
        assert current_schema_version(conn) == 13

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
