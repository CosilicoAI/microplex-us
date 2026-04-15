"""Derived DuckDB index for querying saved US microplex artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from microplex_us.pipelines.registry import (
    FrontierMetric,
    USMicroplexRunRegistryEntry,
    load_us_microplex_run_registry,
)

RUN_INDEX_FILENAME = "run_index.duckdb"


def resolve_us_microplex_run_index_path(path: str | Path) -> Path:
    """Resolve a root directory or explicit DuckDB path to the run-index file."""
    candidate_path = Path(path)
    if candidate_path.suffix == ".duckdb":
        return candidate_path
    return candidate_path / RUN_INDEX_FILENAME


def append_us_microplex_run_index_entry(
    path: str | Path,
    entry: USMicroplexRunRegistryEntry,
    *,
    policyengine_harness_payload: dict[str, Any] | None = None,
) -> Path:
    """Upsert one saved run and its harness detail into the derived DuckDB index."""
    index_path = resolve_us_microplex_run_index_path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(index_path)) as conn:
        _ensure_schema(conn)
        _delete_artifact_rows(conn, entry.artifact_id)
        conn.execute(
            """
            INSERT INTO runs (
                artifact_id,
                created_at,
                artifact_dir,
                manifest_path,
                policyengine_harness_path,
                config_hash,
                synthesis_backend,
                calibration_backend,
                source_names_json,
                rows_seed,
                rows_synthetic,
            rows_calibrated,
            weights_nonzero,
            weights_total,
            full_oracle_capped_mean_abs_relative_error,
            full_oracle_mean_abs_relative_error,
            candidate_mean_abs_relative_error,
            baseline_mean_abs_relative_error,
            mean_abs_relative_error_delta,
                candidate_composite_parity_loss,
                baseline_composite_parity_loss,
                composite_parity_loss_delta,
                slice_win_rate,
                target_win_rate,
                supported_target_rate,
                tag_summaries_json,
                parity_scorecard_json,
                baseline_dataset,
                targets_db,
                target_period,
                target_variables_json,
                target_domains_json,
                target_geo_levels_json,
                target_reform_id,
                policyengine_us_runtime_version,
                improved_candidate_frontier,
                improved_delta_frontier,
                improved_composite_frontier,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _run_row(entry),
        )
        if policyengine_harness_payload is not None:
            slice_rows = _slice_rows(entry.artifact_id, policyengine_harness_payload)
            if slice_rows:
                conn.executemany(
                    """
                    INSERT INTO slice_metrics (
                        artifact_id,
                        slice_name,
                        description,
                        tags_json,
                        query_json,
                        candidate_supported_target_count,
                        candidate_unsupported_target_count,
                        candidate_mean_abs_relative_error,
                        candidate_max_abs_relative_error,
                        baseline_supported_target_count,
                        baseline_unsupported_target_count,
                        baseline_mean_abs_relative_error,
                        baseline_max_abs_relative_error,
                        mean_abs_relative_error_delta,
                        candidate_beats_baseline
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    slice_rows,
                )
            target_rows = _target_metric_rows(entry.artifact_id, policyengine_harness_payload)
            if target_rows:
                conn.executemany(
                    """
                    INSERT INTO target_metrics (
                        artifact_id,
                        slice_name,
                        target_key,
                        target_name,
                        entity,
                        period,
                        measure,
                        aggregation,
                        target_value,
                        tolerance,
                        source,
                        units,
                        description,
                        geo_level,
                        geographic_id,
                        domain_variable,
                        target_metadata_json,
                        filters_json,
                        candidate_actual_value,
                        candidate_absolute_error,
                        candidate_relative_error,
                        baseline_actual_value,
                        baseline_absolute_error,
                        baseline_relative_error,
                        candidate_beats_baseline
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    target_rows,
                )
    return index_path


def rebuild_us_microplex_run_index(
    path: str | Path,
    *,
    registry_path: str | Path,
) -> Path:
    """Rebuild the derived DuckDB index from canonical artifacts and registry entries."""
    index_path = resolve_us_microplex_run_index_path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(index_path)) as conn:
        _ensure_schema(conn)
        conn.execute("DELETE FROM target_metrics")
        conn.execute("DELETE FROM slice_metrics")
        conn.execute("DELETE FROM runs")
    for entry in load_us_microplex_run_registry(registry_path):
        append_us_microplex_run_index_entry(
            index_path,
            entry,
            policyengine_harness_payload=_load_harness_payload(entry.policyengine_harness_path),
        )
    return index_path


def select_us_microplex_frontier_index_row(
    path: str | Path,
    *,
    metric: FrontierMetric = "candidate_composite_parity_loss",
) -> dict[str, Any] | None:
    """Select the best indexed run by one frontier metric."""
    metric_column = {
        "full_oracle_capped_mean_abs_relative_error": "full_oracle_capped_mean_abs_relative_error",
        "full_oracle_mean_abs_relative_error": "full_oracle_mean_abs_relative_error",
        "candidate_composite_parity_loss": "candidate_composite_parity_loss",
        "candidate_mean_abs_relative_error": "candidate_mean_abs_relative_error",
        "mean_abs_relative_error_delta": "mean_abs_relative_error_delta",
    }[metric]
    index_path = resolve_us_microplex_run_index_path(path)
    if not index_path.exists():
        return None
    with duckdb.connect(str(index_path), read_only=True) as conn:
        row = conn.execute(
            f"""
            SELECT *
            FROM runs
            WHERE {metric_column} IS NOT NULL
            ORDER BY {metric_column} ASC, created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        columns = [column[0] for column in conn.description]
    return dict(zip(columns, row, strict=True))


def list_us_microplex_target_delta_rows(
    path: str | Path,
    *,
    artifact_id: str | None = None,
    slice_name: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List indexed per-target deltas, ordered by candidate-vs-baseline improvement."""
    index_path = resolve_us_microplex_run_index_path(path)
    if not index_path.exists():
        return []
    conditions: list[str] = []
    parameters: list[Any] = []
    if artifact_id is not None:
        conditions.append("artifact_id = ?")
        parameters.append(artifact_id)
    if slice_name is not None:
        conditions.append("slice_name = ?")
        parameters.append(slice_name)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
    with duckdb.connect(str(index_path), read_only=True) as conn:
        rows = conn.execute(
            f"""
            SELECT
                artifact_id,
                slice_name,
                target_name,
                geo_level,
                domain_variable,
                target_value,
                candidate_relative_error,
                baseline_relative_error,
                ABS(candidate_relative_error) - ABS(baseline_relative_error) AS abs_relative_error_delta_vs_baseline,
                candidate_beats_baseline
            FROM target_metrics
            {where_clause}
            ORDER BY abs_relative_error_delta_vs_baseline ASC NULLS LAST, target_name ASC
            {limit_clause}
            """,
            parameters,
        ).fetchall()
        columns = [column[0] for column in conn.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def compare_us_microplex_target_delta_rows(
    path: str | Path,
    *,
    artifact_id: str,
    baseline_artifact_id: str,
    slice_name: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Compare per-target candidate error between two saved artifacts."""
    index_path = resolve_us_microplex_run_index_path(path)
    if not index_path.exists():
        return []
    conditions = [
        "current.artifact_id = ?",
        "baseline.artifact_id = ?",
        "current.target_key = baseline.target_key",
        "current.slice_name = baseline.slice_name",
    ]
    parameters: list[Any] = [artifact_id, baseline_artifact_id]
    if slice_name is not None:
        conditions.append("current.slice_name = ?")
        parameters.append(slice_name)
    where_clause = f"WHERE {' AND '.join(conditions)}"
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
    with duckdb.connect(str(index_path), read_only=True) as conn:
        rows = conn.execute(
            f"""
            SELECT
                current.artifact_id,
                baseline.artifact_id AS baseline_artifact_id,
                current.slice_name,
                current.target_key,
                current.target_name,
                current.geo_level,
                current.domain_variable,
                current.target_value,
                current.candidate_relative_error,
                baseline.candidate_relative_error AS baseline_candidate_relative_error,
                ABS(current.candidate_relative_error) - ABS(baseline.candidate_relative_error)
                    AS abs_relative_error_delta_vs_other,
                current.candidate_beats_baseline,
                baseline.candidate_beats_baseline AS baseline_candidate_beats_baseline
            FROM target_metrics AS current
            JOIN target_metrics AS baseline
              ON current.target_key = baseline.target_key
             AND current.slice_name = baseline.slice_name
            {where_clause}
            ORDER BY abs_relative_error_delta_vs_other ASC NULLS LAST, current.target_name ASC
            {limit_clause}
            """,
            parameters,
        ).fetchall()
        columns = [column[0] for column in conn.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            artifact_id TEXT PRIMARY KEY,
            created_at TEXT,
            artifact_dir TEXT,
            manifest_path TEXT,
            policyengine_harness_path TEXT,
            config_hash TEXT,
            synthesis_backend TEXT,
            calibration_backend TEXT,
            source_names_json TEXT,
            rows_seed BIGINT,
            rows_synthetic BIGINT,
            rows_calibrated BIGINT,
            weights_nonzero DOUBLE,
            weights_total DOUBLE,
            full_oracle_capped_mean_abs_relative_error DOUBLE,
            full_oracle_mean_abs_relative_error DOUBLE,
            candidate_mean_abs_relative_error DOUBLE,
            baseline_mean_abs_relative_error DOUBLE,
            mean_abs_relative_error_delta DOUBLE,
            candidate_composite_parity_loss DOUBLE,
            baseline_composite_parity_loss DOUBLE,
            composite_parity_loss_delta DOUBLE,
            slice_win_rate DOUBLE,
            target_win_rate DOUBLE,
            supported_target_rate DOUBLE,
            tag_summaries_json TEXT,
            parity_scorecard_json TEXT,
            baseline_dataset TEXT,
            targets_db TEXT,
            target_period BIGINT,
            target_variables_json TEXT,
            target_domains_json TEXT,
            target_geo_levels_json TEXT,
            target_reform_id BIGINT,
            policyengine_us_runtime_version TEXT,
            improved_candidate_frontier BOOLEAN,
            improved_delta_frontier BOOLEAN,
            improved_composite_frontier BOOLEAN,
            metadata_json TEXT
        )
        """
    )
    _ensure_column(
        conn,
        "runs",
        "full_oracle_capped_mean_abs_relative_error",
        "DOUBLE",
    )
    _ensure_column(
        conn,
        "runs",
        "full_oracle_mean_abs_relative_error",
        "DOUBLE",
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS slice_metrics (
            artifact_id TEXT,
            slice_name TEXT,
            description TEXT,
            tags_json TEXT,
            query_json TEXT,
            candidate_supported_target_count BIGINT,
            candidate_unsupported_target_count BIGINT,
            candidate_mean_abs_relative_error DOUBLE,
            candidate_max_abs_relative_error DOUBLE,
            baseline_supported_target_count BIGINT,
            baseline_unsupported_target_count BIGINT,
            baseline_mean_abs_relative_error DOUBLE,
            baseline_max_abs_relative_error DOUBLE,
            mean_abs_relative_error_delta DOUBLE,
            candidate_beats_baseline BOOLEAN
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS target_metrics (
            artifact_id TEXT,
            slice_name TEXT,
            target_key TEXT,
            target_name TEXT,
            entity TEXT,
            period BIGINT,
            measure TEXT,
            aggregation TEXT,
            target_value DOUBLE,
            tolerance DOUBLE,
            source TEXT,
            units TEXT,
            description TEXT,
            geo_level TEXT,
            geographic_id TEXT,
            domain_variable TEXT,
            target_metadata_json TEXT,
            filters_json TEXT,
            candidate_actual_value DOUBLE,
            candidate_absolute_error DOUBLE,
            candidate_relative_error DOUBLE,
            baseline_actual_value DOUBLE,
            baseline_absolute_error DOUBLE,
            baseline_relative_error DOUBLE,
            candidate_beats_baseline BOOLEAN
        )
        """
    )


def _delete_artifact_rows(conn: duckdb.DuckDBPyConnection, artifact_id: str) -> None:
    conn.execute("DELETE FROM target_metrics WHERE artifact_id = ?", [artifact_id])
    conn.execute("DELETE FROM slice_metrics WHERE artifact_id = ?", [artifact_id])
    conn.execute("DELETE FROM runs WHERE artifact_id = ?", [artifact_id])


def _run_row(entry: USMicroplexRunRegistryEntry) -> tuple[Any, ...]:
    return (
        entry.artifact_id,
        entry.created_at,
        entry.artifact_dir,
        entry.manifest_path,
        entry.policyengine_harness_path,
        entry.config_hash,
        entry.synthesis_backend,
        entry.calibration_backend,
        _json_text(entry.source_names),
        _int_or_none(entry.rows.get("seed")),
        _int_or_none(entry.rows.get("synthetic")),
        _int_or_none(entry.rows.get("calibrated")),
        _float_or_none(entry.weights.get("nonzero")),
        _float_or_none(entry.weights.get("total")),
        _float_or_none(entry.full_oracle_capped_mean_abs_relative_error),
        _float_or_none(entry.full_oracle_mean_abs_relative_error),
        _float_or_none(entry.candidate_mean_abs_relative_error),
        _float_or_none(entry.baseline_mean_abs_relative_error),
        _float_or_none(entry.mean_abs_relative_error_delta),
        _float_or_none(entry.candidate_composite_parity_loss),
        _float_or_none(entry.baseline_composite_parity_loss),
        _float_or_none(entry.composite_parity_loss_delta),
        _float_or_none(entry.slice_win_rate),
        _float_or_none(entry.target_win_rate),
        _float_or_none(entry.supported_target_rate),
        _json_text(entry.tag_summaries),
        _json_text(entry.parity_scorecard),
        entry.baseline_dataset,
        entry.targets_db,
        _int_or_none(entry.target_period),
        _json_text(entry.target_variables),
        _json_text(entry.target_domains),
        _json_text(entry.target_geo_levels),
        _int_or_none(entry.target_reform_id),
        entry.policyengine_us_runtime_version,
        entry.improved_candidate_frontier,
        entry.improved_delta_frontier,
        entry.improved_composite_frontier,
        _json_text(entry.metadata),
    )


def _ensure_column(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    existing = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    }
    if column_name in existing:
        return
    conn.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
    )


def _slice_rows(
    artifact_id: str,
    harness_payload: dict[str, Any],
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for slice_payload in harness_payload.get("slices", []):
        summary = dict(slice_payload.get("summary", {}))
        delta = summary.get("mean_abs_relative_error_delta")
        rows.append(
            (
                artifact_id,
                slice_payload["name"],
                slice_payload.get("description"),
                _json_text(slice_payload.get("tags", [])),
                _json_text(slice_payload.get("query", {})),
                _int_or_none(summary.get("candidate_supported_target_count")),
                _int_or_none(summary.get("candidate_unsupported_target_count")),
                _float_or_none(summary.get("candidate_mean_abs_relative_error")),
                _float_or_none(summary.get("candidate_max_abs_relative_error")),
                _int_or_none(summary.get("baseline_supported_target_count")),
                _int_or_none(summary.get("baseline_unsupported_target_count")),
                _float_or_none(summary.get("baseline_mean_abs_relative_error")),
                _float_or_none(summary.get("baseline_max_abs_relative_error")),
                _float_or_none(delta),
                (delta < 0.0) if delta is not None else None,
            )
        )
    return rows


def _target_metric_rows(
    artifact_id: str,
    harness_payload: dict[str, Any],
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for slice_payload in harness_payload.get("slices", []):
        slice_name = slice_payload["name"]
        candidate_payload = dict(slice_payload.get("candidate", {}))
        baseline_payload = (
            dict(slice_payload.get("baseline", {}))
            if slice_payload.get("baseline") is not None
            else {}
        )
        candidate_by_key = {
            _target_key(item["target"]): item
            for item in candidate_payload.get("evaluations", [])
        }
        baseline_by_key = {
            _target_key(item["target"]): item
            for item in baseline_payload.get("evaluations", [])
        }
        for target_key in sorted(set(candidate_by_key) | set(baseline_by_key)):
            candidate_item = candidate_by_key.get(target_key)
            baseline_item = baseline_by_key.get(target_key)
            target_payload = (
                dict(candidate_item["target"])
                if candidate_item is not None
                else dict(baseline_item["target"])
            )
            metadata = dict(target_payload.get("metadata", {}))
            candidate_relative_error = _float_or_none(
                candidate_item.get("relative_error")
                if candidate_item is not None
                else None
            )
            baseline_relative_error = _float_or_none(
                baseline_item.get("relative_error")
                if baseline_item is not None
                else None
            )
            candidate_beats_baseline = None
            if (
                candidate_relative_error is not None
                and baseline_relative_error is not None
            ):
                candidate_beats_baseline = abs(candidate_relative_error) < abs(
                    baseline_relative_error
                )
            rows.append(
                (
                    artifact_id,
                    slice_name,
                    target_key,
                    target_payload["name"],
                    target_payload["entity"],
                    _int_or_none(target_payload.get("period")),
                    target_payload.get("measure"),
                    target_payload.get("aggregation"),
                    _float_or_none(target_payload.get("value")),
                    _float_or_none(target_payload.get("tolerance")),
                    target_payload.get("source"),
                    target_payload.get("units"),
                    target_payload.get("description"),
                    metadata.get("geo_level"),
                    metadata.get("geographic_id"),
                    metadata.get("domain_variable"),
                    _json_text(metadata),
                    _json_text(target_payload.get("filters", [])),
                    _float_or_none(
                        candidate_item.get("actual_value")
                        if candidate_item is not None
                        else None
                    ),
                    _float_or_none(
                        candidate_item.get("absolute_error")
                        if candidate_item is not None
                        else None
                    ),
                    candidate_relative_error,
                    _float_or_none(
                        baseline_item.get("actual_value")
                        if baseline_item is not None
                        else None
                    ),
                    _float_or_none(
                        baseline_item.get("absolute_error")
                        if baseline_item is not None
                        else None
                    ),
                    baseline_relative_error,
                    candidate_beats_baseline,
                )
            )
    return rows


def _load_harness_payload(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    harness_path = Path(path)
    if not harness_path.exists():
        return None
    return json.loads(harness_path.read_text())


def _target_key(target_payload: dict[str, Any]) -> str:
    return json.dumps(target_payload, sort_keys=True, separators=(",", ":"))


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
