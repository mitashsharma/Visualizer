"""Flask application: JSON API + static multi-page frontend.

Backend contract for the frontend (see ``static/js/api.js``):

``POST /api/datasets``                     upload a CSV/TSV/TXT/XLSX (multipart)
``POST /api/datasets/sample``              load the bundled demo dataset
``GET  /api/session``                      active dataset for this browser
``GET  /api/datasets/<id>``                dataset summary
``GET  /api/datasets/<id>/overview``       profile, quality, preview, insights
``GET  /api/datasets/<id>/preview``        paginated rows
``GET  /api/datasets/<id>/charts``         curated chart specs + column catalog
``POST /api/datasets/<id>/charts``         manually configured chart spec
``POST /api/datasets/<id>/cleaning/preview`` preview a cleaning plan
``POST /api/datasets/<id>/clean``          apply plan, create a new revision
``POST /api/datasets/<id>/restore``        return to the original revision
``GET  /api/datasets/<id>/model/options``  targets, models, limits
``GET  /api/datasets/<id>/target/<name>``  target analysis for one column
``POST /api/datasets/<id>/train``          start a training job (or return cache)
``GET  /api/jobs/<job_id>``                job progress/result
``GET  /api/datasets/<id>/models``         trained model history
``GET  /api/datasets/<id>/insights``       findings + latest model review
``POST /api/datasets/<id>/report``         download HTML/PDF report
``GET  /api/datasets/<id>/export.csv``     download the current revision
``DELETE /api/datasets/<id>``              forget the dataset
"""

from __future__ import annotations

import io
import math
import os
import secrets
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_file, send_from_directory, session
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from src.charts import ChartError, build_chart, chart_catalog, default_charts
from src.config import SETTINGS, Settings
from src.data_cleaner import apply_cleaning_plan, normalise_plan, recommended_plan
from src.data_loader import DatasetLoadError, LoadedDataset, load_bytes, load_file
from src.jobs import JobManager, JobNotFound
from src.ml_model import (
    MIN_ROWS,
    TrainingError,
    analyse_target,
    available_models,
    cache_key,
    default_model_id,
    detect_problem_type,
    public_result,
    reusable,
    select_features,
    train_model,
)
from src.profile import build_profile
from src.store import Dataset, DatasetStore, DatasetNotFound
from src.summarizer import build_insights

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
SAMPLE_PATH = ROOT / "data" / "sample_customers.csv"

PAGES = {
    "/": "index.html",
    "/overview": "overview.html",
    "/prepare": "prepare.html",
    "/explore": "explore.html",
    "/model": "model.html",
    "/insights": "insights.html",
}


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def to_jsonable(value):
    """Recursively convert pandas/NumPy values into JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [to_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _rows_payload(dataset: Dataset, limit: int, offset: int = 0) -> dict:
    frame = dataset.dataframe
    total = len(frame)
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    window = frame.iloc[offset : offset + limit]

    records = []
    for row in window.itertuples(index=False, name=None):
        records.append([to_jsonable(value) for value in row])

    return {
        "columns": [
            {"name": str(name), "dtype": str(dtype)}
            for name, dtype in zip(window.columns, window.dtypes)
        ],
        "rows": records,
        "total": int(total),
        "offset": int(offset),
        "limit": int(limit),
        "has_more": offset + len(records) < total,
    }


def _latest_model(dataset: Dataset) -> dict | None:
    if not dataset.model_results:
        return None
    return max(dataset.model_results.values(), key=lambda item: item.get("created_at", ""))


def _certificate(dataset: Dataset) -> dict:
    return {"dataset": dataset.summary()}


def _target_option(column: dict, rows: int) -> dict:
    kind = column["kind"]
    eligible = True
    task = None
    if column["is_constant"]:
        eligible = False
    elif column["is_identifier"]:
        eligible = False
    elif kind == "datetime":
        eligible = False
    elif rows - column["missing"] < MIN_ROWS:
        # Too few usable rows for a meaningful train/test split: hide the
        # column from the target picker instead of failing after submission.
        eligible = False
    elif kind != "numeric":
        task = "classification"
    elif column["unique"] <= 1:
        eligible = False
    elif column["unique"] <= 2:
        task = "classification"
    elif column["unique"] <= 10 and column["unique"] / max(rows, 1) < 0.5:
        task = "classification"
    else:
        task = "regression"

    return {
        "name": column["name"],
        "kind": kind,
        "unique": column["unique"],
        "missing_pct": column["missing_pct"],
        "is_identifier": column["is_identifier"],
        "eligible": eligible,
        "suggested_task": task,
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or SETTINGS

    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.config.update(
        SECRET_KEY=os.environ.get("AIVIZ_SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=settings.max_upload_bytes,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_HTTPONLY=True,
        JSON_SORT_KEYS=False,
        SEND_FILE_MAX_AGE_DEFAULT=0,
        SETTINGS=settings,
    )

    store = DatasetStore(settings)
    jobs = JobManager(settings)
    app.extensions["aiviz"] = {"store": store, "jobs": jobs, "settings": settings, "started": time.time()}

    # -- helpers ----------------------------------------------------------
    def owner() -> str:
        if "sid" not in session:
            session["sid"] = secrets.token_hex(16)
        return session["sid"]

    def current_dataset(dataset_id: str) -> Dataset:
        return store.get(owner(), dataset_id)

    def payload(value):
        return jsonify(to_jsonable(value))

    # -- pages ------------------------------------------------------------
    for path, filename in PAGES.items():
        def view(page_file=filename):
            return send_from_directory(STATIC_DIR, page_file)

        endpoint = "page_" + (path.strip("/").replace("/", "_") or "home")
        app.add_url_rule(path, endpoint=endpoint, view_func=view)

    @app.get("/favicon.svg")
    def favicon():
        return send_from_directory(STATIC_DIR, "favicon.svg")

    # -- session ----------------------------------------------------------
    @app.get("/api/health")
    def health():
        return payload({"status": "ok", "uptime_s": round(time.time() - app.extensions["aiviz"]["started"], 1)})

    @app.get("/api/session")
    def session_info():
        dataset = store.latest(owner())
        return payload(
            {
                "dataset": dataset.summary() if dataset else None,
                "limits": {
                    "max_upload_mb": settings.max_upload_mb,
                    "max_rows": settings.max_rows,
                    "ml_max_train_rows": settings.ml_max_train_rows,
                },
            }
        )

    # -- datasets ---------------------------------------------------------
    @app.post("/api/datasets")
    def upload_dataset():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            raise DatasetLoadError("Choose a CSV, TSV, TXT or Excel file to upload.")

        raw = upload.stream.read(settings.max_upload_bytes + 1)
        if len(raw) > settings.max_upload_bytes:
            raise DatasetLoadError(
                f"File is larger than the {settings.max_upload_mb} MB limit."
            )

        loaded = load_bytes(raw, upload.filename, settings=settings)
        dataset = store.create(owner(), loaded)
        profile = dataset.profile()

        warnings = []
        if loaded.truncated:
            warnings.append(
                f"Only the first {settings.max_rows:,} rows were loaded "
                f"(file has ~{loaded.total_rows_in_file or 0:,})."
            )
        for note in loaded.notes:
            warnings.append(note)

        return (
            payload(
                {
                    "dataset": dataset.summary(),
                    "warnings": warnings,
                    "profile": profile["overview"],
                }
            ),
            201,
        )

    @app.post("/api/datasets/sample")
    def sample_dataset():
        if not SAMPLE_PATH.exists():
            raise DatasetLoadError("The bundled sample dataset is not available.")
        loaded = load_file(SAMPLE_PATH, settings=settings)
        dataset = store.create(owner(), loaded, name="Customer churn sample")
        return payload({"dataset": dataset.summary(), "warnings": []}), 201

    @app.get("/api/datasets/<dataset_id>")
    def dataset_detail(dataset_id):
        dataset = current_dataset(dataset_id)
        return payload({"dataset": dataset.summary()})

    @app.delete("/api/datasets/<dataset_id>")
    def dataset_delete(dataset_id):
        store.delete(owner(), dataset_id)
        return payload({"deleted": True})

    @app.get("/api/datasets/<dataset_id>/preview")
    def dataset_preview(dataset_id):
        dataset = current_dataset(dataset_id)
        limit = request.args.get("limit", settings.preview_rows, type=int) or settings.preview_rows
        offset = request.args.get("offset", 0, type=int) or 0
        return payload(_rows_payload(dataset, limit, offset))

    @app.get("/api/datasets/<dataset_id>/overview")
    def dataset_overview(dataset_id):
        dataset = current_dataset(dataset_id)
        profile = dataset.profile()
        insights = build_insights(profile, dataset.dataframe)
        suggestions = recommended_plan(dataset.dataframe, profile)
        preview = _rows_payload(dataset, settings.preview_rows)

        return payload(
            {
                "dataset": dataset.summary(),
                "overview": profile["overview"],
                "columns": profile["columns"],
                "kinds": profile["kinds"],
                "quality": profile["quality"],
                "insights": insights,
                "suggestions": suggestions,
                "preview": preview,
            }
        )

    # -- cleaning ---------------------------------------------------------
    @app.post("/api/datasets/<dataset_id>/cleaning/preview")
    def cleaning_preview(dataset_id):
        dataset = current_dataset(dataset_id)
        body = request.get_json(silent=True) or {}
        plan = body.get("plan") or recommended_plan(dataset.dataframe, dataset.profile())
        cleaned, summary = apply_cleaning_plan(dataset.dataframe, plan)

        preview_frame = cleaned.head(8)
        preview = {
            "columns": [str(name) for name in preview_frame.columns],
            "rows": [[to_jsonable(value) for value in row] for row in preview_frame.itertuples(index=False, name=None)],
        }
        return payload({"summary": to_jsonable(summary), "preview": preview})

    @app.post("/api/datasets/<dataset_id>/clean")
    def cleaning_apply(dataset_id):
        dataset = current_dataset(dataset_id)
        body = request.get_json(silent=True) or {}
        plan = normalise_plan(
            body.get("plan") or recommended_plan(dataset.dataframe, dataset.profile()),
            list(dataset.dataframe.columns),
        )

        cleaned, summary = apply_cleaning_plan(dataset.dataframe, plan)
        derived = store.derive(owner(), dataset, cleaned, summary)
        return payload(
            {
                "dataset": derived.summary(),
                "summary": to_jsonable(summary),
                "profile": derived.profile()["overview"],
            }
        )

    @app.post("/api/datasets/<dataset_id>/restore")
    def cleaning_restore(dataset_id):
        dataset = current_dataset(dataset_id)
        try:
            root = store.get(owner(), dataset.root_id)
        except DatasetNotFound as error:
            raise DatasetNotFound(
                "The original revision is no longer in memory. Re-upload the file."
            ) from error
        store.activate(owner(), root.id)
        return payload({"dataset": root.summary()})

    # -- charts -----------------------------------------------------------
    @app.get("/api/datasets/<dataset_id>/charts")
    def charts_list(dataset_id):
        dataset = current_dataset(dataset_id)
        return payload(
            {
                "dataset": dataset.summary(),
                "charts": dataset.charts(),
                "catalog": chart_catalog(dataset.profile()),
            }
        )

    @app.post("/api/datasets/<dataset_id>/charts")
    def charts_build(dataset_id):
        dataset = current_dataset(dataset_id)
        body = request.get_json(silent=True) or {}
        spec = build_chart(
            dataset.dataframe,
            dataset.profile(),
            body.get("type", ""),
            body.get("x"),
            body.get("y"),
            bins=int(body.get("bins") or 24),
            top_n=int(body.get("top_n") or 10),
            settings=settings,
        )
        return payload({"chart": spec})

    # -- models -----------------------------------------------------------
    @app.get("/api/datasets/<dataset_id>/model/options")
    def model_options(dataset_id):
        dataset = current_dataset(dataset_id)
        profile = dataset.profile()
        rows = profile["overview"]["rows"]
        targets = [_target_option(column, rows) for column in profile["columns"]]

        limit_options = [value for value in (10_000, 50_000, 100_000, 250_000) if value < rows]
        limit_options.append(0)

        return payload(
            {
                "dataset": dataset.summary(),
                "targets": targets,
                "models": {
                    "classification": available_models("classification", rows, settings),
                    "regression": available_models("regression", rows, settings),
                },
                "limits": {
                    "rows": rows,
                    "row_options": limit_options,
                    "default_row_limit": settings.ml_max_train_rows,
                    "max_cardinality": settings.ml_max_cardinality,
                },
                # Convention: the last usable column is usually the outcome.
                "default_target": next(
                    (item["name"] for item in reversed(targets) if item["eligible"]),
                    None,
                ),
            }
        )

    @app.get("/api/datasets/<dataset_id>/target/<path:column>")
    def target_detail(dataset_id, column):
        dataset = current_dataset(dataset_id)
        if column not in dataset.dataframe.columns:
            raise TrainingError(f"Column '{column}' does not exist.")
        analysis = analyse_target(dataset.dataframe, column, dataset.profile())
        task = analysis["suggested_task"] or detect_problem_type(dataset.dataframe, column)
        analysis["models"] = available_models(task, analysis["rows"], settings)
        analysis["default_model"] = default_model_id(task, analysis["rows"], settings)
        analysis["max_rows"] = settings.ml_max_train_rows
        return payload(analysis)

    @app.post("/api/datasets/<dataset_id>/train")
    def train(dataset_id):
        dataset = current_dataset(dataset_id)
        body = request.get_json(silent=True) or {}
        target = body.get("target")
        if not target:
            raise TrainingError("Choose a target column.")

        analysis = analyse_target(dataset.dataframe, target, dataset.profile())
        if not analysis["eligible"]:
            raise TrainingError(analysis["message"] or "This column cannot be used as a target.")

        task = body.get("task") or analysis["suggested_task"]
        if task not in ("classification", "regression"):
            raise TrainingError("Task must be 'classification' or 'regression'.")

        model_id = body.get("model") or default_model_id(task, analysis["rows"], settings)
        row_limit = body.get("max_rows")
        if row_limit is None:
            row_limit = settings.ml_max_train_rows
        row_limit = int(row_limit)
        if row_limit < 0:
            raise TrainingError("Row limit must be zero (all rows) or a positive number.")
        if row_limit and row_limit < 200:
            raise TrainingError("Choose a row limit of at least 200, or 0 for all rows.")

        test_size = float(body.get("test_size") or settings.ml_test_size)
        if not 0.1 <= test_size <= 0.4:
            raise TrainingError("Test size must be between 10% and 40%.")

        key = cache_key(
            dataset.fingerprint,
            target,
            task,
            model_id,
            row_limit,
            test_size,
            settings.ml_random_state,
        )

        cached = dataset.model_results.get(key)
        features, _excluded = select_features(dataset.profile(), target, settings)
        signature = {
            "target": target,
            "task": task,
            "features": features,
            "rows": analysis["rows"],
        }
        if cached and reusable(cached, signature, model_id, task):
            result = public_result(cached)
            result["cached"] = True
            return payload(
                {
                    "cached": True,
                    "job": {
                        "id": None,
                        "status": "done",
                        "stage": "Complete (cached)",
                        "progress": 100,
                        "message": "Reused the cached result for this configuration.",
                        "elapsed_ms": 0,
                        "error": None,
                        "result": result,
                    },
                }
            )

        def run(progress):
            result = train_model(
                dataset.dataframe,
                target,
                problem_type=task,
                test_size=test_size,
                random_state=settings.ml_random_state,
                model_id=model_id,
                max_rows=row_limit,
                settings=settings,
                profile=dataset.profile(),
                progress=progress,
            )
            dataset.model_results[key] = result
            # Keep memory bounded: retain the six most recent models.
            if len(dataset.model_results) > 6:
                oldest = min(
                    dataset.model_results,
                    key=lambda item: dataset.model_results[item].get("created_at", ""),
                )
                dataset.model_results.pop(oldest, None)
            public = public_result(result)
            public["cached"] = False
            return public

        job = jobs.submit(owner(), "train", run, dataset_id=dataset.id)
        return payload({"cached": False, "job": job.public()}), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        job = jobs.get(owner(), job_id)
        return payload({"job": job.public()})

    @app.get("/api/datasets/<dataset_id>/models")
    def model_history(dataset_id):
        dataset = current_dataset(dataset_id)
        results = [
            public_result(result)
            for result in sorted(
                dataset.model_results.values(),
                key=lambda item: item.get("created_at", ""),
                reverse=True,
            )
        ]
        return payload({"models": results})

    # -- insights / reports ----------------------------------------------
    @app.get("/api/datasets/<dataset_id>/insights")
    def insights(dataset_id):
        dataset = current_dataset(dataset_id)
        latest = _latest_model(dataset)
        latest_public = public_result(latest) if latest else None
        payload_data = build_insights(dataset.profile(), dataset.dataframe, latest_public)
        return payload(
            {
                "dataset": dataset.summary(),
                "insights": payload_data,
                "model": latest_public,
                "charts": dataset.charts(),
            }
        )

    @app.post("/api/datasets/<dataset_id>/report")
    def report(dataset_id):
        dataset = current_dataset(dataset_id)
        body = request.get_json(silent=True) or {}
        fmt = (body.get("format") or "html").lower()
        if fmt not in ("html", "pdf"):
            raise ValueError("Report format must be 'html' or 'pdf'.")

        latest = _latest_model(dataset)
        latest_public = public_result(latest) if latest else None
        context = {
            "dataset": dataset.summary(),
            "profile": dataset.profile(),
            "insights": build_insights(dataset.profile(), dataset.dataframe, latest_public),
            "ml": latest_public,
            "cleaning": dataset.cleaning,
            "charts": dataset.charts(),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        from src.report_generator import render_report

        file_bytes, mimetype, filename = render_report(
            context,
            fmt,
            include_charts=bool(body.get("include_charts", True)),
            chart_limit=settings.report_chart_limit,
        )
        # ``mimetype`` may carry a charset; ``send_file`` adds one for text
        # types, which previously produced "text/html; charset=utf-8; charset=utf-8".
        base_mimetype = (mimetype or "application/octet-stream").split(";")[0].strip()
        return send_file(
            io.BytesIO(file_bytes),
            mimetype=base_mimetype,
            as_attachment=True,
            download_name=filename,
        )

    @app.get("/api/datasets/<dataset_id>/export.csv")
    def export_csv(dataset_id):
        dataset = current_dataset(dataset_id)
        buffer = io.StringIO()
        dataset.dataframe.to_csv(buffer, index=False)
        filename = f"{dataset.name.lower().replace(' ', '-')}-revision-{dataset.revision}.csv"
        return send_file(
            io.BytesIO(buffer.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=filename,
        )

    # -- error handling ---------------------------------------------------
    @app.errorhandler(DatasetNotFound)
    @app.errorhandler(JobNotFound)
    def handle_not_found(error):
        return payload({"error": "Not found.", "detail": str(error)}), 404

    @app.errorhandler(DatasetLoadError)
    @app.errorhandler(ChartError)
    @app.errorhandler(TrainingError)
    def handle_bad_input(error):
        return payload({"error": str(error)}), 400

    @app.errorhandler(ValueError)
    def handle_value_error(error):
        return payload({"error": str(error)}), 400

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(error):
        return (
            payload(
                {
                    "error": f"File is larger than the {settings.max_upload_mb} MB limit.",
                }
            ),
            413,
        )

    @app.errorhandler(Exception)
    def handle_unexpected(error):
        if isinstance(error, HTTPException):
            return error
        if request.path.startswith("/api/"):
            app.logger.exception("Unhandled API error")
            return payload({"error": "Unexpected server error.", "detail": str(error)}), 500
        raise error

    return app
