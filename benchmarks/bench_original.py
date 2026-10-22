"""Baseline benchmark for the ORIGINAL implementation (commit 48a6480).

Emulates what dashboard/app.py did on every Streamlit rerun:
read CSV -> clean -> EDA -> visualize -> train model.

The ML stage is measured separately because the original code cannot complete
it on realistic data (dense one-hot encoding of identifier columns exhausts
memory). That failure is recorded as evidence.

Usage:
    python benchmarks/bench_original.py            # all measurements
    python benchmarks/bench_original.py --size 20000 --skip-ml
"""

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = Path(__file__).parent / "results_baseline.json"


def rss_mb():
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return round(pages * 4096 / 1_048_576, 1)
    except OSError:
        return None


def peak_rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def run_pipeline(size: int) -> dict:
    import matplotlib

    matplotlib.use("Agg")

    import pandas as pd

    from benchmarks.make_data import ensure_datasets
    from src.data_cleaner import clean_data
    from src.eda import perform_eda
    from src.visualizer import generate_visualizations

    path = ensure_datasets((size,))[size]
    run = {"rows": size, "stages": {}, "rss_mb": {}}

    def stage(name, func, *args):
        before = rss_mb()
        started = time.perf_counter()
        output = func(*args)
        elapsed = time.perf_counter() - started
        run["stages"][name] = {"seconds": round(elapsed, 3)}
        run["rss_mb"][name] = {"before": before, "after": rss_mb()}
        return output

    df = stage("load_csv", pd.read_csv, path)
    cleaned = stage("clean", clean_data, df)
    stage("eda", perform_eda, cleaned)
    figures = stage("charts", generate_visualizations, cleaned)

    figure_count = sum(
        len(value) if isinstance(value, dict) else 1
        for value in figures.values()
        if value is not None
    )
    run["stages"]["charts"]["figure_count"] = figure_count
    run["time_to_first_useful_view_s"] = round(
        sum(
            run["stages"][key]["seconds"]
            for key in ("load_csv", "clean", "eda", "charts")
        ),
        3,
    )
    run["peak_rss_mb"] = peak_rss_mb()

    import matplotlib.pyplot as plt

    plt.close("all")
    del figures

    return run


def run_ml(size: int, target: str = "churned") -> dict:
    import pandas as pd  # noqa: F401

    from benchmarks.make_data import ensure_datasets
    from src.data_cleaner import clean_data
    from src.ml_model import get_feature_importance, train_model

    path = ensure_datasets((size,))[size]
    cleaned = clean_data(pd.read_csv(path))

    started = time.perf_counter()
    result = train_model(cleaned, target)
    elapsed = time.perf_counter() - started

    importance = get_feature_importance(result)

    return {
        "rows": size,
        "seconds": round(elapsed, 3),
        "train_rows": result["train_rows"],
        "accuracy": round(result["metrics"]["accuracy"], 4),
        "feature_count": int(len(importance)),
        "peak_rss_mb": peak_rss_mb(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--ml-only", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.size and args.ml_only:
        print(json.dumps(run_ml(args.size), indent=2))
        return

    if args.size:
        result = run_pipeline(args.size)
        if not args.skip_ml:
            result["ml"] = run_ml(args.size)
        print(json.dumps(result, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    import pandas as pd

    report = {
        "implementation": "original (48a6480)",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "runs": [],
        "ml": [],
    }

    for size in (20_000, 100_000):
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, __file__, "--size", str(size), "--skip-ml"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        wall = round(time.perf_counter() - started, 2)

        if completed.returncode != 0:
            report["runs"].append(
                {
                    "rows": size,
                    "status": f"FAILED (exit {completed.returncode})",
                    "wall_seconds": wall,
                    "stderr_tail": completed.stderr.strip().splitlines()[-2:],
                }
            )
            print(f"[original] pipeline rows={size:,} -> FAILED (exit {completed.returncode})")
            continue

        run = json.loads(completed.stdout)
        run["status"] = "ok"
        run["wall_seconds"] = wall
        report["runs"].append(run)
        print(
            f"[original] pipeline rows={size:,} -> {run['time_to_first_useful_view_s']}s first view"
            f" | {run['stages']['charts']['figure_count']} figures"
            f" | peak RSS {run['peak_rss_mb']} MB"
        )

    for size in (2_000, 20_000):
        completed = subprocess.run(
            [sys.executable, __file__, "--size", str(size), "--ml-only"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if completed.returncode != 0:
            tail = completed.stderr.strip().splitlines()[-2:]
            report["ml"].append(
                {
                    "rows": size,
                    "status": f"FAILED (exit {completed.returncode})",
                    "stderr_tail": tail,
                }
            )
            print(f"[original] ML rows={size:,} -> FAILED (exit {completed.returncode})")
            continue

        result = json.loads(completed.stdout)
        result["status"] = "ok"
        report["ml"].append(result)
        print(f"[original] ML rows={size:,} -> {result['seconds']}s, acc {result['accuracy']}")

    RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
