from __future__ import annotations

import sqlite3

from somm_core.schema import _list_migrations, current_schema_version, ensure_schema
from somm_core.version import SCHEMA_VERSION


def test_v10_database_upgrades_to_v11_prompt_labels(tmp_path):
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

        assert upgraded == SCHEMA_VERSION == 11
        assert current_schema_version(conn) == 11
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
