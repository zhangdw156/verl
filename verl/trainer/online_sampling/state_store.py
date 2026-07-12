# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .types import SampleObservation

_SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    fingerprint_json    TEXT NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS samples (
    run_id          TEXT NOT NULL,
    sample_id       TEXT NOT NULL,
    dense_index     INTEGER NOT NULL,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    eligible        INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (run_id, sample_id),
    UNIQUE (run_id, dense_index),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sample_observations (
    observation_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    phase               TEXT NOT NULL,
    policy_step         INTEGER NOT NULL,
    generation_round    INTEGER NOT NULL,
    attempt_id          TEXT NOT NULL,
    sample_id           TEXT NOT NULL,
    dense_index         INTEGER NOT NULL,
    n                   INTEGER NOT NULL,
    reward_vector_json  TEXT NOT NULL,
    correct_vector_json TEXT NOT NULL,
    format_vector_json  TEXT NOT NULL,
    p                   REAL NOT NULL,
    q                   REAL NOT NULL,
    mu                  REAL NOT NULL,
    sigma               REAL NOT NULL,
    zero_variance       INTEGER NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, phase, policy_step, generation_round, attempt_id, sample_id),
    FOREIGN KEY (run_id, sample_id) REFERENCES samples(run_id, sample_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rollout_observations (
    rollout_observation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id          INTEGER NOT NULL,
    rollout_index           INTEGER NOT NULL,
    reward                  REAL NOT NULL,
    correct                 INTEGER NOT NULL,
    format                  INTEGER NOT NULL,
    status                  TEXT NOT NULL,
    response                TEXT,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (observation_id, rollout_index),
    FOREIGN KEY (observation_id) REFERENCES sample_observations(observation_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sample_latest (
    run_id                  TEXT NOT NULL,
    sample_id               TEXT NOT NULL,
    latest_observation_id   INTEGER NOT NULL,
    latest_sigma            REAL NOT NULL,
    ema_sigma               REAL NOT NULL,
    last_observed_step      INTEGER NOT NULL,
    observation_count       INTEGER NOT NULL,
    PRIMARY KEY (run_id, sample_id),
    FOREIGN KEY (run_id, sample_id) REFERENCES samples(run_id, sample_id) ON DELETE CASCADE,
    FOREIGN KEY (latest_observation_id) REFERENCES sample_observations(observation_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS checkpoint_snapshots (
    run_id                  TEXT NOT NULL,
    checkpoint_kind         TEXT NOT NULL,
    checkpoint_step         INTEGER NOT NULL,
    sample_id               TEXT NOT NULL,
    latest_observation_id   INTEGER NOT NULL,
    latest_sigma            REAL NOT NULL,
    ema_sigma               REAL NOT NULL,
    last_observed_step      INTEGER NOT NULL,
    observation_count       INTEGER NOT NULL,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, checkpoint_kind, checkpoint_step, sample_id),
    FOREIGN KEY (run_id, sample_id) REFERENCES samples(run_id, sample_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sample_observations_run_phase
ON sample_observations(run_id, phase, sample_id);

CREATE INDEX IF NOT EXISTS idx_checkpoint_snapshots_run_kind_step
ON checkpoint_snapshots(run_id, checkpoint_kind, checkpoint_step);
"""


class SQLiteOnlineStateStore:
    """Single-controller SQLite persistence for online sampling state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(_SCHEMA)
        self._initialize_schema_version()

    def _initialize_schema_version(self) -> None:
        row = self._connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            self._connection.commit()
            return
        if int(row["value"]) != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported online-sampling SQLite schema version {row['value']}; expected {_SCHEMA_VERSION}"
            )

    def ensure_run(self, run_id: str, fingerprint: Mapping[str, Any]) -> None:
        fingerprint_json = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        row = self._connection.execute("SELECT fingerprint_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO runs(run_id, fingerprint_json) VALUES(?, ?)",
                (run_id, fingerprint_json),
            )
            self._connection.commit()
            return
        if row["fingerprint_json"] != fingerprint_json:
            raise ValueError(f"Run fingerprint mismatch for online-sampling run_id={run_id}")

    def register_samples(
        self,
        run_id: str,
        samples: Iterable[tuple[str, int, Mapping[str, Any] | None, bool]],
    ) -> None:
        rows = [
            (
                run_id,
                sample_id,
                dense_index,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                int(eligible),
            )
            for sample_id, dense_index, metadata, eligible in samples
        ]
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO samples(run_id, sample_id, dense_index, metadata_json, eligible)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(run_id, sample_id) DO UPDATE SET
                    dense_index=excluded.dense_index,
                    metadata_json=excluded.metadata_json,
                    eligible=excluded.eligible
                """,
                rows,
            )

    def set_run_value(self, run_id: str, key: str, value: Any) -> None:
        metadata_key = f"run:{run_id}:{key}"
        value_json = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (metadata_key, value_json),
            )

    def get_run_value(self, run_id: str, key: str, default: Any = None) -> Any:
        metadata_key = f"run:{run_id}:{key}"
        row = self._connection.execute("SELECT value FROM metadata WHERE key=?", (metadata_key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def append_observation(
        self,
        run_id: str,
        observation: SampleObservation,
        *,
        ema_alpha: float,
        zero_variance_epsilon: float,
        save_responses: bool,
    ) -> dict[str, float | int]:
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")

        unique_key = (
            run_id,
            observation.phase,
            observation.policy_step,
            observation.generation_round,
            observation.attempt_id,
            observation.sample_id,
        )
        existing = self._connection.execute(
            """
            SELECT observation_id, sigma, policy_step FROM sample_observations
            WHERE run_id=? AND phase=? AND policy_step=? AND generation_round=?
              AND attempt_id=? AND sample_id=?
            """,
            unique_key,
        ).fetchone()
        if existing is not None:
            latest = self._connection.execute(
                """
                SELECT latest_observation_id, ema_sigma, observation_count
                FROM sample_latest
                WHERE run_id=? AND sample_id=?
                """,
                (run_id, observation.sample_id),
            ).fetchone()
            if latest is not None and int(latest["latest_observation_id"]) == int(existing["observation_id"]):
                return self.get_latest_state(run_id, observation.sample_id)
            previous_count = 0 if latest is None else int(latest["observation_count"])
            previous_ema = float(existing["sigma"]) if latest is None else float(latest["ema_sigma"])
            ema_sigma = (
                float(existing["sigma"])
                if previous_count == 0
                else ema_alpha * float(existing["sigma"]) + (1 - ema_alpha) * previous_ema
            )
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO sample_latest(
                        run_id, sample_id, latest_observation_id, latest_sigma, ema_sigma,
                        last_observed_step, observation_count
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, sample_id) DO UPDATE SET
                        latest_observation_id=excluded.latest_observation_id,
                        latest_sigma=excluded.latest_sigma,
                        ema_sigma=excluded.ema_sigma,
                        last_observed_step=excluded.last_observed_step,
                        observation_count=excluded.observation_count
                    """,
                    (
                        run_id,
                        observation.sample_id,
                        int(existing["observation_id"]),
                        float(existing["sigma"]),
                        ema_sigma,
                        int(existing["policy_step"]),
                        previous_count + 1,
                    ),
                )
            return {
                "latest_sigma": float(existing["sigma"]),
                "ema_sigma": ema_sigma,
                "last_observed_step": int(existing["policy_step"]),
                "observation_count": previous_count + 1,
            }

        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO sample_observations(
                    run_id, phase, policy_step, generation_round, attempt_id,
                    sample_id, dense_index, n,
                    reward_vector_json, correct_vector_json, format_vector_json,
                    p, q, mu, sigma, zero_variance
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    observation.phase,
                    observation.policy_step,
                    observation.generation_round,
                    observation.attempt_id,
                    observation.sample_id,
                    observation.dense_index,
                    observation.n,
                    json.dumps(observation.rewards),
                    json.dumps(observation.corrects),
                    json.dumps(observation.formats),
                    observation.p,
                    observation.q,
                    observation.mu,
                    observation.sigma,
                    int(observation.is_zero_variance(zero_variance_epsilon)),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an observation_id")
            observation_id = int(cursor.lastrowid)
            self._connection.executemany(
                """
                INSERT INTO rollout_observations(
                    observation_id, rollout_index, reward, correct, format, status, response
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        observation_id,
                        item.rollout_index,
                        item.reward,
                        item.correct,
                        item.format,
                        item.status,
                        item.response if save_responses else None,
                    )
                    for item in observation.rollouts
                ],
            )

            previous = self._connection.execute(
                """
                SELECT ema_sigma, observation_count
                FROM sample_latest
                WHERE run_id=? AND sample_id=?
                """,
                (run_id, observation.sample_id),
            ).fetchone()
            previous_count = 0 if previous is None else int(previous["observation_count"])
            previous_ema = observation.sigma if previous is None else float(previous["ema_sigma"])
            ema_sigma = (
                observation.sigma
                if previous_count == 0
                else ema_alpha * observation.sigma + (1 - ema_alpha) * previous_ema
            )
            self._connection.execute(
                """
                INSERT INTO sample_latest(
                    run_id, sample_id, latest_observation_id, latest_sigma, ema_sigma,
                    last_observed_step, observation_count
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, sample_id) DO UPDATE SET
                    latest_observation_id=excluded.latest_observation_id,
                    latest_sigma=excluded.latest_sigma,
                    ema_sigma=excluded.ema_sigma,
                    last_observed_step=excluded.last_observed_step,
                    observation_count=excluded.observation_count
                """,
                (
                    run_id,
                    observation.sample_id,
                    observation_id,
                    observation.sigma,
                    ema_sigma,
                    observation.policy_step,
                    previous_count + 1,
                ),
            )

        return {
            "latest_sigma": observation.sigma,
            "ema_sigma": ema_sigma,
            "last_observed_step": observation.policy_step,
            "observation_count": previous_count + 1,
        }

    def get_latest_state(self, run_id: str, sample_id: str) -> dict[str, float | int]:
        row = self._connection.execute(
            """
            SELECT latest_sigma, ema_sigma, last_observed_step, observation_count
            FROM sample_latest
            WHERE run_id=? AND sample_id=?
            """,
            (run_id, sample_id),
        ).fetchone()
        if row is None:
            return {
                "latest_sigma": 0.0,
                "ema_sigma": 0.0,
                "last_observed_step": 0,
                "observation_count": 0,
            }
        return dict(row)

    def load_latest_states(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT s.sample_id, s.dense_index,
                   COALESCE(l.latest_sigma, 0.0) AS latest_sigma,
                   COALESCE(l.ema_sigma, 0.0) AS ema_sigma,
                   COALESCE(l.last_observed_step, 0) AS last_observed_step,
                   COALESCE(l.observation_count, 0) AS observation_count
            FROM samples AS s
            LEFT JOIN sample_latest AS l
              ON s.run_id=l.run_id AND s.sample_id=l.sample_id
            WHERE s.run_id=? AND s.eligible=1
            ORDER BY s.dense_index
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def pending_step0_sample_ids(self, run_id: str) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT s.sample_id
            FROM samples AS s
            WHERE s.run_id=? AND s.eligible=1
              AND NOT EXISTS (
                  SELECT 1
                  FROM sample_observations AS o
                  WHERE o.run_id=s.run_id AND o.sample_id=s.sample_id AND o.phase='step0'
              )
            ORDER BY s.dense_index
            """,
            (run_id,),
        ).fetchall()
        return [str(row["sample_id"]) for row in rows]

    def save_checkpoint_snapshot(self, run_id: str, checkpoint_kind: str, checkpoint_step: int) -> None:
        if checkpoint_step < 0:
            raise ValueError("checkpoint_step must be non-negative")
        if not checkpoint_kind:
            raise ValueError("checkpoint_kind must not be empty")
        with self._connection:
            self._connection.execute(
                """
                DELETE FROM checkpoint_snapshots
                WHERE run_id=? AND checkpoint_kind=? AND checkpoint_step=?
                """,
                (run_id, checkpoint_kind, checkpoint_step),
            )
            self._connection.execute(
                """
                INSERT INTO checkpoint_snapshots(
                    run_id, checkpoint_kind, checkpoint_step, sample_id,
                    latest_observation_id, latest_sigma, ema_sigma,
                    last_observed_step, observation_count
                )
                SELECT run_id, ?, ?, sample_id, latest_observation_id,
                       latest_sigma, ema_sigma, last_observed_step, observation_count
                FROM sample_latest
                WHERE run_id=?
                """,
                (checkpoint_kind, checkpoint_step, run_id),
            )

    def load_checkpoint_snapshot(
        self,
        run_id: str,
        checkpoint_kind: str,
        checkpoint_step: int,
    ) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT s.sample_id, s.dense_index, c.latest_observation_id,
                   COALESCE(c.latest_sigma, 0.0) AS latest_sigma,
                   COALESCE(c.ema_sigma, 0.0) AS ema_sigma,
                   COALESCE(c.last_observed_step, 0) AS last_observed_step,
                   COALESCE(c.observation_count, 0) AS observation_count
            FROM samples AS s
            LEFT JOIN checkpoint_snapshots AS c
              ON s.run_id=c.run_id AND s.sample_id=c.sample_id
             AND c.checkpoint_kind=? AND c.checkpoint_step=?
            WHERE s.run_id=? AND s.eligible=1
            ORDER BY s.dense_index
            """,
            (checkpoint_kind, checkpoint_step, run_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def restore_checkpoint_snapshot(self, run_id: str, checkpoint_kind: str, checkpoint_step: int) -> None:
        if not self.checkpoint_snapshot_exists(run_id, checkpoint_kind, checkpoint_step):
            raise FileNotFoundError(f"Missing online-sampling snapshot kind={checkpoint_kind!r} step={checkpoint_step}")
        with self._connection:
            self._connection.execute("DELETE FROM sample_latest WHERE run_id=?", (run_id,))
            self._connection.execute(
                """
                INSERT INTO sample_latest(
                    run_id, sample_id, latest_observation_id, latest_sigma, ema_sigma,
                    last_observed_step, observation_count
                )
                SELECT run_id, sample_id, latest_observation_id, latest_sigma, ema_sigma,
                       last_observed_step, observation_count
                FROM checkpoint_snapshots
                WHERE run_id=? AND checkpoint_kind=? AND checkpoint_step=?
                """,
                (run_id, checkpoint_kind, checkpoint_step),
            )

    def checkpoint_snapshot_exists(self, run_id: str, checkpoint_kind: str, checkpoint_step: int) -> bool:
        row = self._connection.execute(
            """
            SELECT 1
            FROM checkpoint_snapshots
            WHERE run_id=? AND checkpoint_kind=? AND checkpoint_step=?
            LIMIT 1
            """,
            (run_id, checkpoint_kind, checkpoint_step),
        ).fetchone()
        return row is not None

    def delete_checkpoint_snapshot(self, run_id: str, checkpoint_kind: str, checkpoint_step: int) -> None:
        with self._connection:
            self._connection.execute(
                """
                DELETE FROM checkpoint_snapshots
                WHERE run_id=? AND checkpoint_kind=? AND checkpoint_step=?
                """,
                (run_id, checkpoint_kind, checkpoint_step),
            )

    def prune_checkpoint_snapshots(self, run_id: str, checkpoint_kind: str, keep: int) -> None:
        if keep <= 0:
            return
        steps = [
            int(row["checkpoint_step"])
            for row in self._connection.execute(
                """
                SELECT DISTINCT checkpoint_step
                FROM checkpoint_snapshots
                WHERE run_id=? AND checkpoint_kind=?
                ORDER BY checkpoint_step DESC
                """,
                (run_id, checkpoint_kind),
            ).fetchall()
        ]
        remove_steps = steps[keep:]
        if not remove_steps:
            return
        with self._connection:
            self._connection.executemany(
                """
                DELETE FROM checkpoint_snapshots
                WHERE run_id=? AND checkpoint_kind=? AND checkpoint_step=?
                """,
                [(run_id, checkpoint_kind, step) for step in remove_steps],
            )

    def observation_count(self, run_id: str, phase: str | None = None) -> int:
        if phase is None:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM sample_observations WHERE run_id=?",
                (run_id,),
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM sample_observations WHERE run_id=? AND phase=?",
                (run_id, phase),
            ).fetchone()
        return int(row["count"])

    def integrity_check(self) -> tuple[str, list[dict[str, Any]]]:
        integrity = str(self._connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = [dict(row) for row in self._connection.execute("PRAGMA foreign_key_check").fetchall()]
        return integrity, foreign_keys

    def close(self) -> None:
        self._connection.close()
