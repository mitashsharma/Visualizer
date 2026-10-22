"""Benchmark the new pipeline end to end.

Each dataset size runs in its own subprocess so peak RSS is not shared, exactly
like ``bench_original.py``. Timings come from the public HTTP API, which is what
the browser actually waits for.

    python benchmarks/bench_new.py --all
    python benchmarks/bench_new.py --rows 20000 --ml --json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "benchmarks" / "data"
RESULTS = ROOT / "benchmarks" / "results_new.json"


def peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux and bytes on macOS.
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor, 1)


def timed(client, label, method, path, **kwargs):
    started = time.perf_counter()
    response = getattr(client, method)(path, **kwargs)
    elapsed = time.perf_counter() - started
    if response.status_code >= 400:
        raise RuntimeError(f"{label} failed: {response.status_code} {response.get_data(as_text=True)[:300]}")
    return elapsed, response


def wait_for_job(client, job_id: str, timeout: float = 600.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").get_json()["job"]
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.05)
    raise RuntimeError("training job timed out")


def run(rows: int, ml: bool, report: bool) -> dict:
    from src.api import create_app

    path = DATA / f"customers_{rows}.csv"
    if not path.exists():
        raise SystemExit(f"missing benchmark data: {path} (run benchmarks/make_data.py)")

    app = create_app()
    client = app.test_client()
    payload = path.read_bytes()
    stages: dict[str, dict] = {}

    upload_time, upload = timed(
        client,
        "upload",
        "post",
        "/api/datasets",
        data={"file": (io.BytesIO(payload), path.name)},
        content_type="multipart/form-data",
    )
    dataset = upload.get_json()["dataset"]
    dataset_id = dataset["id"]
    stages["upload_and_profile"] = {
        "seconds": round(upload_time, 3),
        "rows": dataset["rows"],
        "columns": dataset["columns"],
    }

    overview_time, overview = timed(client, "overview", "get", f"/api/datasets/{dataset_id}/overview")
    stages["overview_first"] = {
        "seconds": round(overview_time, 3),
        "findings": len(overview.get_json()["insights"]["findings"]),
    }

    charts_time, charts = timed(client, "charts", "get", f"/api/datasets/{dataset_id}/charts")
    stages["charts_first"] = {
        "seconds": round(charts_time, 3),
        "chart_count": len(charts.get_json()["charts"]),
    }

    stages["first_view_total"] = {
        "seconds": round(upload_time + overview_time + charts_time, 3)
    }

    overview_cached, _ = timed(client, "overview cache", "get", f"/api/datasets/{dataset_id}/overview")
    charts_cached, _ = timed(client, "charts cache", "get", f"/api/datasets/{dataset_id}/charts")
    stages["revisit_cached"] = {
        "overview_seconds": round(overview_cached, 3),
        "charts_seconds": round(charts_cached, 3),
        "total_seconds": round(overview_cached + charts_cached, 3),
    }

    plan = overview.get_json()["suggestions"]
    cleaning_time, cleaning = timed(
        client,
        "cleaning",
        "post",
        f"/api/datasets/{dataset_id}/cleaning/preview",
        json={"plan": plan},
    )
    stages["cleaning_preview"] = {
        "seconds": round(cleaning_time, 3),
        "rows_after": cleaning.get_json()["summary"]["rows_after"],
    }

    if ml:
        started = time.perf_counter()
        response = client.post(
            f"/api/datasets/{dataset_id}/train",
            json={
                "target": "churned",
                "model": "random_forest",
                "task": "classification",
                "max_rows": 50_000,
                "test_size": 0.2,
            },
        )
        job = wait_for_job(client, response.get_json()["job"]["id"])
        elapsed = time.perf_counter() - started
        if job["status"] != "done":
            raise RuntimeError(f"training failed: {job['error']}")
        result = job["result"]
        stages["train"] = {
            "seconds": round(elapsed, 3),
            "internal_seconds": result["timings"]["total"],
            "rows_used": result["rows"]["used"],
            "sampled": result["rows"]["sampled"],
            "accuracy": round(result["metrics"]["accuracy"], 4),
            "features": len(result["feature_names"]),
        }

        cached_started = time.perf_counter()
        cached = client.post(
            f"/api/datasets/{dataset_id}/train",
            json={
                "target": "churned",
                "model": "random_forest",
                "task": "classification",
                "max_rows": 50_000,
                "test_size": 0.2,
            },
        ).get_json()
        stages["train_cached"] = {
            "seconds": round(time.perf_counter() - cached_started, 3),
            "cached": bool(cached.get("cached")),
        }

    if report:
        report_time, report_response = timed(
            client,
            "report",
            "post",
            f"/api/datasets/{dataset_id}/report",
            json={"format": "html", "include_charts": True},
        )
        stages["report_html"] = {
            "seconds": round(report_time, 3),
            "bytes": len(report_response.data),
        }

    return {
        "rows": rows,
        "stages": stages,
        "peak_rss_mb": peak_rss_mb(),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ml", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--json", action="store_true", help="print this process' result as JSON")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    if args.all:
        sizes = [2_000, 20_000, 100_000]
        results = []
        for size in sizes:
            command = [
                sys.executable,
                str(Path(__file__)),
                "--rows",
                str(size),
                "--ml",
                "--report",
                "--json",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            if completed.returncode != 0:
                results.append(
                    {
                        "rows": size,
                        "status": "error",
                        "error": (completed.stderr or completed.stdout)[-800:],
                    }
                )
                print(f"{size:>8,} rows: FAILED", file=sys.stderr)
                continue
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            results.append(payload)
            first = payload["stages"]["first_view_total"]["seconds"]
            print(
                f"{size:>8,} rows: first view {first:.2f}s, "
                f"peak {payload['peak_rss_mb']:.0f} MB"
            )

        document = {
            "implementation": "new (Flask API + curated charts)",
            "python": platform.python_version(),
            "runs": results,
        }
        args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
        return

    result = run(args.rows, ml=args.ml, report=args.report)
    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
