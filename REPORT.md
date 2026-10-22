# Refactor report — AI Data Visualizer

Final report for the six-stage enhancement of the repository at `48a6480`.
All work is on `arena/90769277-visualizer`; nothing was committed to `main`.

Environment used for every measurement below: 2 vCPU / 3.9 GB container,
Python 3.11, pandas 2.3.3, scikit-learn 1.9.1, Flask 3.1.3.

---

## Stage 1 — Audit

The original app was a 656-line Streamlit script that re-ran the whole pipeline
on every widget interaction. The audit (`AUDIT.md`) measured the real cost on a
synthetic 13-column customer dataset:

| Rows | Load | Clean | EDA | Charts | First view | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 0.03 s | 0.04 s | 0.08 s | **23.9 s** | **24.0 s** | 764 MB |
| 100,000 | 0.13 s | 0.19 s | 0.32 s | **112.9 s** | **113.5 s** | **3,244 MB** |

Training a RandomForest took **137 s and 3.3 GB at 20k rows** (17,026 features)
and crashed at 100k with a 48.3 GiB allocation.

Seven root causes were identified:

1. **Cartesian chart generation** — a histogram per numeric column, a bar per
   categorical column and a scatter per numeric pair (36 figures), including
   `customer_id` with 100k bars. Figures were never closed.
2. **Dense one-hot encoding of identifiers** — 81k dummy columns for 100k rows.
3. **No caching** — every Streamlit rerun re-read, re-cleaned, re-profiled.
4. **Duplicated EDA work** and four DataFrame copies inside cleaning.
5. **No real frontend** — one long scroll page, no validation, progress or
   error recovery; CSV-only upload despite the README.
6. **Dead code and unused dependencies** — `report_generator.py`, `seaborn`,
   `scipy`, `streamlit`, duplicated insight generators.
7. **Thin ML reporting** — one algorithm, no baseline, no confusion matrix,
   unreadable one-hot feature names, deep sklearn errors.

## Stage 2 — Plan

The plan kept the useful logic and reorganised it into five modules on six
pages:

```
Upload & Overview → Cleaning & Preparation → EDA & Visualization
        → ML & Evaluation → Insights & Report Export
```

Decisions fixed before coding:

* **Charts:** at most five curated specs per dataset (1–2 distributions, a
  missing-value chart when gaps exist, a correlation heatmap when ≥ 3 numeric
  columns exist, and one category or time chart), plus a manual builder. No
  per-column auto charts; identifiers excluded; categories capped at 12;
  scatter sampled at 2,000; browser-rendered SVG.
* **ML:** three algorithms per task, two recommended by default, sparse one-hot,
  explicit feature selection, a 50k row sample, a single honest train/test split,
  baseline comparison, cached results.
* **State:** immutable dataset revisions; caches attached to a revision, so a
  new revision starts with an empty cache and stale results are impossible.
* **Success criteria:** first view under ~2 s at 100k rows, ≤ 5 charts, bounded
  memory, no fabricated metrics, no `main` changes.

## Stage 3 — Backend

New modules:

| File | Role |
|---|---|
| `src/config.py` | Frozen settings with `AIVIZ_*` environment overrides |
| `src/profile.py` | One vectorised pass: dtypes, missingness, cardinality, stats, quality |
| `src/charts.py` | Curated chart selection + seven spec builders (no matplotlib) |
| `src/store.py` | Session store of immutable revisions with lazy caches and TTL/LRU |
| `src/jobs.py` | Background training jobs with stage/progress reporting |
| `src/api.py` | Flask app factory and the whole JSON API |
| `src/formatting.py` | Shared number/percent formatters |

Rewritten: `data_loader.py` (validation, delimiter/encoding sniffing, type
inference, leading-zero preservation, row caps, duplicate headers),
`data_cleaner.py` (vectorised, plan-based, reversible), `eda.py` (reuses the
profile), `ml_model.py` (feature selection, model registry, sampling, metrics,
caching), `summarizer.py` (data-driven findings), `report_generator.py`
(self-contained HTML + PDF, charts embedded), `visualizer.py` (report images
only), `app.py` (CLI entrypoint).

Key safeguards:

* Uploads are capped at 100 MB / 500k rows / 500 columns, parsed in memory and
  never written to disk; filenames are sanitised.
* Identifier columns (name pattern, uniqueness, sequential integers) are
  excluded from charts and features but remain visible in the overview.
* Cached artefacts are keyed by revision; model cache keys include the dataset
  fingerprint, target, task, model, sample limit and split size.
* Every error path returns a readable message (`400`/`404`/`413`), never a
  traceback or a restart.

## Stage 4 — Frontend

Six pages, plain HTML + CSS + ES modules (no framework, no build step):

| Page | Content |
|---|---|
| `/` | Upload with drag & drop, extension/size validation, XHR progress, sample dataset, resume card |
| `/overview` | KPI tiles, quality issues, findings, type composition, column table, paginated preview |
| `/prepare` | Duplicate/whitespace toggles, per-column missing strategy, column drops, live preview, revision banner |
| `/explore` | Curated chart grid, manual builder (7 types), interpretations, sampled badges |
| `/model` | Target analysis, model cards, row limit, progress with stages, metric tiles, confusion matrix, importance, history |
| `/insights` | Narrative, findings, recommendations, model review, HTML/PDF report, CSV export |

Shared shell (`static/js/ui.js`) injects the sidebar/topbar/theme control on
every page. Design system in `static/css/tokens.css` + `layout.css` +
`components.css` + `pages.css`: light/dark themes, responsive at 1080/960/900/
720/620/480 px (sidebar becomes a drawer below 960), reduced-motion support,
keyboard-focus styles, loading skeletons, empty and error states.

Charts are hand-written SVG (`static/js/charts.js`) that measure their host and
re-render on resize/theme changes; no chart library is loaded.

## Stage 5 — Cleanup

* `requirements.txt` now lists only what is used: pandas, numpy, scikit-learn,
  openpyxl, xlrd, flask, matplotlib, reportlab.
* Deleted `dashboard/app.py` (superseded; Streamlit removed) and stray test
  files.
* Upload acceptance reconciled: CSV, TSV, TXT, XLSX and XLS everywhere —
  messages, file picker and server validation now agree.
* `.gitignore` covers benchmark data and report output.
* README rewritten with architecture, API reference, configuration and limits.

## Stage 6 — Verification

* **188 pytest tests** across 12 modules; all pass in ~10 s, on Linux and
  under a simulated Windows cp1252 locale.
* API matrix: pages, upload (CSV/TSV/XLSX), invalid types, empty files, charts,
  cleaning preview/apply/restore, target analysis, training jobs, caching,
  insights, HTML/PDF reports, CSV export, session isolation, 404s.
* Frontend rendered headlessly with jsdom against the live Flask server: all
  six pages, zero console errors, expected element counts.
* End-to-end: sample → clean (1,212 → 1,200 rows) → train → report
  (HTML 137 KB, PDF 108 KB) → CSV export (1,200 rows).

---

## 1. Original problems and root causes

| # | Problem | Root cause | Evidence |
|---|---|---|---|
| P1 | 36 figures, 23.9 s / 112.9 s, 3.2 GB | per-column/per-pair chart generation, no cardinality or identifier guard, figures never closed | `results_baseline.json` |
| P2 | ML OOM at 100k rows | dense `OneHotEncoder` over identifier columns (81,020 columns) | `_ArrayMemoryError … 48.3 GiB` |
| P3 | UI recomputes everything | Streamlit re-executes the script on each interaction; no revision-level cache | code inspection |
| P4 | duplicated EDA / 4 copies | `perform_eda` recomputes overview/outliers; Python-level `.apply` | code inspection |
| P5 | no usable frontend | single scroll page, no routing/states/progress, CSV-only upload | dashboard source |
| P6 | dead code and deps | unused report generator, seaborn/scipy/streamlit | imports |
| P7 | weak ML reporting | single model, no baseline/confusion matrix, raw one-hot names | dashboard source |

## 2. Final architecture and folder structure

```
Browser (6 HTML pages, CSS design system, vanilla ES modules)
   │  fetch JSON                         ▲ SVG chart specs
   ▼                                     │
Flask (src/api.py, app.py)  ──► DatasetStore (session-scoped, TTL/LRU)
   │                                └── immutable revisions + caches
   ├── profile.py       one pass per revision
   ├── charts.py        curated specs + builder
   ├── ml_model.py      selection, training, metrics
   ├── jobs.py          background execution
   ├── summarizer.py    findings and recommendations
   └── report_generator.py  HTML / PDF export
```

```
Visualizer/
├── app.py
├── src/            (15 modules)
├── static/         css/ (4), js/ (9), 6 HTML pages, favicon
├── tests/          (11 modules, 170 tests)
├── benchmarks/     make_data.py, bench_original.py, bench_new.py, results
├── data/           sample_customers.csv (1,212 × 13)
├── AUDIT.md, REPORT.md, README.md
└── requirements.txt
```

## 3. Features kept, merged, simplified, removed

| Feature | Decision | Result |
|---|---|---|
| CSV/Excel loading | Keep + harden | Validation, encoding/delimiter sniffing, type inference, row/column caps, leading zeros preserved |
| Cleaning | Keep, redesign | Explicit reversible plan; every apply creates a new revision |
| EDA statistics | Keep + optimise | Single profile reused by charts, insights and ML |
| Insights | Merge | One data-driven generator (findings, quality, model review) |
| Auto chart gallery | Remove | ≤ 5 curated charts + builder |
| Scatter/box/line | Simplify | Builder options and report images, not auto-generated |
| Random-forest-only ML | Simplify | 3 models per task, 2 recommended |
| Feature importance | Keep + improve | One-hot groups mapped back to source columns, top 15 |
| Report generator | Keep + wire up | HTML/PDF export in the UI |
| Streamlit dashboard | Remove | Replaced by six served pages |
| seaborn / scipy / streamlit | Remove | No imports remain |
| Dataset export | Add | `export.csv` download |

## 4. Files created, modified, deleted

**Created (41):** `app.py`; `src/{api,charts,config,formatting,jobs,profile,store}.py`;
`static/{index,overview,prepare,explore,model,insights}.html`;
`static/css/{tokens,layout,components,pages}.css`;
`static/js/{api,ui,charts,upload,overview,prepare,explore,model,insights}.js`;
`static/favicon.svg`; `data/sample_customers.csv`; `AUDIT.md`; `REPORT.md`;
`benchmarks/{make_data,bench_original,bench_new}.py` + results;
`tests/{conftest,test_api,test_charts,test_profile,test_reports}.py`.

**Modified (16):** `requirements.txt`, `README.md`, `.gitignore`,
`src/{data_loader,data_cleaner,eda,ml_model,summarizer,visualizer,report_generator}.py`,
`tests/{test_data_loader,test_data_cleaner,test_eda,test_ml_model,test_pipeline}.py`.

**Deleted (1):** `dashboard/app.py`.

Net: 16 tracked files changed (+3,817 / −2,748 lines), 41 new files, 1 deletion.

## 5. Frontend implementation

* Six distinct URLs, sidebar navigation, breadcrumb and dataset pill; active
  step highlighted from the URL.
* Upload: drag & drop or picker, client-side extension/size checks, XHR upload
  progress, server validation messages, "Load sample" shortcut, resume banner.
* Overview: four KPIs, five quality checks, findings, type composition, column
  table with missing bars, paginated raw preview.
* Prepare: recommended plan pre-filled, live dry-run projection (rows, columns,
  missing), apply → new revision, restore original.
* Explore: curated grid with one-sentence interpretations, builder with
  type-aware X/Y selectors, bins/top-N controls, inline errors.
* Model: target analysis (task, imbalance, warnings), model cards, row limit,
  staged progress with elapsed time, metric tiles vs baseline, confusion matrix,
  per-class table, importance bars, history.
* Insights: narrative, findings, numbered recommendations, model review,
  report download with format/chart toggles, CSV export.
* States: skeletons, empty states, toasts, inline alerts, disabled/loading
  buttons, retry actions. Dark mode, print-friendly report styling, reduced
  motion, keyboard focus rings, ARIA labels on charts and controls.

## 6. Backend and ML improvements

* **Single profile pass** feeds overview, charts, insights and ML target
  analysis; results memoised per revision.
* **Vectorised cleaning**: `.str.strip()` instead of per-cell `apply`, single
  copy, duplicate detection before/after trimming as documented.
* **Curated charts** aggregate on the server and ship compact JSON (≤ 160 line
  points, ≤ 2,000 scatter points, ≤ 12 categories + "Other").
* **Model registry**: gradient boosting (default above 20k rows), random
  forest (default below), logistic regression / ridge. 60 trees, 120 boosting
  iterations, `n_jobs=-1`, fixed `random_state`.
* **Feature handling**: identifiers, constants and high-cardinality text
  (> 40 values) excluded with reasons; datetimes converted to numeric; sparse
  one-hot; median/mode imputation; scaling only for linear models.
* **Training**: 50k row cap with stratified sampling and a visible "sampled"
  badge, 80/20 split, accuracy / balanced accuracy / macro F1 / ROC AUC,
  MAE / RMSE / R² / MAPE, majority-class or mean baselines, confusion matrix,
  per-class table, grouped importance, stage timings.
* **Reliability**: background jobs, cached retraining (~5 ms), bounded store
  (4 datasets, 60 min TTL), clear errors for tiny/single-class/invalid targets.

## 7. Before / after measurements

Same machine, same CSV generator, same 80/20 split. Baseline from
`benchmarks/results_baseline.json`; after from `benchmarks/results_new.json`
(Flask test client, full request path).

| Metric | Before | After | Change |
|---|---:|---:|---:|
| 2k rows — first view | — | 0.057 s | — |
| 20k rows — first view | 24.00 s | **0.156 s** | **154× faster** |
| 100k rows — first view | 113.5 s | **0.622 s** | **182× faster** |
| 20k charts | 23.85 s | 0.011 s | ~2,100× |
| 100k charts | 112.9 s | 0.034 s | ~3,300× |
| 20k peak RSS | 763.7 MB | **240.9 MB** | 3.2× less |
| 100k peak RSS | 3,243.5 MB | **291.7 MB** | 11.1× less |
| 20k ML train | 137.4 s | **0.703 s** | 195× faster |
| 20k ML memory | 3,312.7 MB | 240.9 MB | 13.7× less |
| 100k ML train | crash (48.3 GiB) | **1.317 s** (50k sample) | now completes |
| Retrain, same config | full retrain | 0.005 s | cached |
| Overview + charts revisit | full recompute | 0.009–0.024 s | cached |
| Report export | unavailable | 0.99 s HTML / 0.70 s PDF | new |

Quality was not traded for speed: on the 20k dataset the new model reaches
accuracy 0.9985 / AUC 0.99998 (the baseline reached 0.9978 on 17k features),
and on the sample data the random forest reaches accuracy 0.8889, balanced
accuracy 0.8422 and AUC 0.8677 versus a 0.6749 majority-class baseline.

## 8. Tests and limitations

**Automated:** `python -m pytest tests -q` → **188 passed in ~10 s**.
Coverage: loader (formats, encodings, caps, dtypes, leading zeros), cleaning
strategies and validation, profiling and identifier detection, curated charts
and builder validation, EDA payloads, ML tasks/metrics/importance/sampling/
caching/errors, HTTP contract and session isolation, HTML/PDF reports, classic
helper chain, and an end-to-end workflow.

**Frontend:** rendered headlessly with jsdom against the running server — all
six pages, zero console errors, expected KPI/chart/table counts.

**Limitations:**

* State is in memory: a restart clears datasets; the app is single-process and
  not a multi-user deployment (no auth).
* The ML split is a single hold-out, not cross-validation; sampling is capped at
  50k rows and there is no hyper-parameter search (deliberately).
* Identifier detection is heuristic; a genuinely predictive ID would be
  excluded from features.
* PDF charts are raster PNGs; the PDF font subset handles Latin text only
  (other characters are folded to ASCII).
* No automated real-browser test was possible in this environment (no Chromium);
  jsdom verifies DOM/JS behaviour but not pixel layout.
* Benchmarks use synthetic data and a 2-vCPU container; absolute numbers will
  differ on other hardware.

## 9. Install and run

```bash
cd Visualizer
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                        # http://127.0.0.1:8000
```

Options: `--host 127.0.0.1`, `--port 9000`, `--debug`.
Environment overrides: `AIVIZ_MAX_UPLOAD_MB`, `AIVIZ_MAX_ROWS`,
`AIVIZ_ML_MAX_TRAIN_ROWS`, `AIVIZ_MAX_DATASETS`, `AIVIZ_SESSION_TTL`.

Tests and benchmarks:

```bash
python -m pytest tests -q
python benchmarks/make_data.py --rows 2000 20000 100000
python benchmarks/bench_new.py --all
```

## 10. Usage guide

1. Open `/` and drop a CSV/TSV/Excel file (or click **Load sample dataset**).
2. Review rows, columns, missingness, duplicates and findings on `/overview`.
3. Go to `/prepare`; the recommended plan is pre-filled. Adjust missing-value
   strategies or columns, watch the projection, then **Apply cleaning**. A new
   revision is created; **Restore original** switches back.
4. Open `/explore` for the curated charts, or use **Chart builder** to pick a
   type and columns.
5. On `/model`, choose a target (task and warnings appear automatically), pick a
   model, optionally cap rows, and train. Watch stages complete, then read
   metrics against the baseline, the confusion matrix, per-class scores and
   feature importance.
6. On `/insights`, read the summary and recommendations, then download an HTML
   or PDF report (with or without charts) or the cleaned CSV.
