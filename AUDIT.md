# Audit — AI Data Visualizer

Evidence-based review of the codebase at `48a6480` (before refactor).
Measurements: 2 vCPU / 3.9 GB sandbox, Python 3.11, pandas 2.3.3, scikit-learn 1.9.1.
The baseline was captured before the refactor. Reproduce it from a checkout of
`48a6480` with `python benchmarks/bench_original.py` (raw numbers in
`benchmarks/results_baseline.json`); the refactored pipeline is measured by
`python benchmarks/bench_new.py --all` (raw numbers in
`benchmarks/results_new.json`).

## 1. Baseline measurements

Synthetic customer dataset (13 columns: 7 numeric, 5 text, 1 target), with
missing values and ~1% duplicates. "First view" = everything
`dashboard/app.py` did before rendering the page.

| Rows | Load | Clean | EDA | Charts | **First view** | Figures | Peak RSS |
|-----:|-----:|------:|----:|-------:|---------------:|--------:|---------:|
| 20,000 | 0.03 s | 0.04 s | 0.08 s | **23.9 s** | **24.0 s** | **36** | 764 MB |
| 100,000 | 0.13 s | 0.19 s | 0.32 s | **112.9 s** | **113.5 s** | **36** | **3,244 MB** |

| ML run (RandomForest, 100 trees) | Time | Features built | Peak RSS |
|---|---:|---:|---:|
| 2,000 rows | 1.03 s | 2,579 | 215 MB |
| 20,000 rows | **137.4 s** | **17,026** | **3,313 MB** |

Chart generation is **99.3 %** of first-view time. EDA is not the bottleneck —
the visualizer is.

## 2. Problems found

### P1 — Unbounded cartesian chart generation (critical)

`visualizer.generate_visualizations()` builds a histogram for *every* numeric
column, a bar chart for *every* categorical column, a box plot, a heatmap and
a scatter plot for *every pair* of numeric columns:

```
charts = n_num histograms + n_cat bars + 1 box + 1 heatmap + C(n_num, 2) scatters
       = 7 + 6 + 1 + 1 + 21 = 36 figures
```

* Cause: charts are selected by column type only; there is no notion of
  relevance, no column cap, and no cardinality guard.
* Worse: `customer_id` (100,000 unique) and `referral_code` (400 unique) are
  plotted as bar charts with one bar per value, and `signup_date` is treated as
  a category. Rendering those is the bulk of the 112.9 s and 3.2 GB.
* Figures are never closed (`plt.close`), so all 36 canvases stay alive for the
  whole session. Matplotlib itself warns after 20 figures.
* Fix: a curated chart engine that selects ≤ 5 high-value charts by *type +
  variation + completeness*, excludes identifiers, caps categories, aggregates
  server-side, and renders in the browser.

### P2 — Dense one-hot encoding of identifiers (critical, OOM)

`ml_model.prepare_data()` one-hot encodes every text column with
`sparse_output=False`:

* `customer_id` → 100,000 dummy columns, `signup_date` → 600, `referral_code`
  → 400.
* Result: a dense `(80000, 81020)` float64 matrix = **48.3 GiB allocation** and
  a crash at 100k rows; at 20k rows it needs 3.3 GB and 137 s.
* Cause: no feature selection, no identifier detection, no cardinality cap, no
  sparse output, no sampling, `n_jobs` unspecified (single core), and a fixed
  100-tree forest trained on every row.
* Fix: explicit feature selection (drop identifiers/constants/high-cardinality
  text), sparse one-hot with unknown/rare handling, a small model registry,
  configurable row cap with honest sampling, `n_jobs=-1`, and cached results.

### P3 — Repeated computation on every interaction (critical UX cost)

Streamlit re-executes the entire script on *every* widget event. Changing the
target-column selectbox re-reads the CSV, re-cleans, re-runs EDA and rebuilds
all 36 figures before the new selectbox value is even used. Nothing is cached
except one `ml_results` entry — which itself holds a `RandomForest` pipeline
inside `st.session_state`.

* Fix: immutable dataset revisions + per-revision memoisation on the server;
  changing a UI control only re-queries the affected endpoint.

### P4 — Duplicated work inside EDA

`perform_eda()` calls `dataset_overview()` and `detect_outliers()` directly and
then again through `generate_insights()`, so both are computed twice per call.
`handle_missing_values` and `clean_data` make up to four full DataFrame copies.
`clean_text_columns` runs a Python-level `.apply(lambda)` per cell instead of
`.str.strip()`.

### P5 — Frontend: one long scroll page, no routing, no states

* Single 656-line Streamlit script with no navigation; the dataset, EDA,
  visualizations and ML are stacked vertically.
* No upload validation (extension is filtered by the widget only), no file-size
  limit, no error recovery — a parse failure leaves an empty page under a red
  banner.
* No loading/progress indicators; the UI freezes for seconds-to-minutes.
* No report/export button even though `report_generator.py` exists.
* No empty states, no sample data, no mobile layout, no design system.
* Excel support is advertised in the README but the uploader only accepts CSV.

### P6 — Dead and redundant features

* `report_generator.py` (583 lines) is imported nowhere. It imports
  `reportlab.platypus.Image` twice and is untested.
* `seaborn` and `scipy` are declared but never imported.
* `streamlit` is only used by the legacy dashboard.
* `visualizer.plot_bar_chart` / `plot_line_chart` / `plot_scatter_plot` are only
  reachable through the cartesian generator; there is no manual chart builder.
* `eda.generate_insights()` duplicates the summarizer, and both only restate
  row/column counts — no actual findings (correlations, skew, concentration,
  model baseline).
* `summarizer` is never called by the dashboard.

### P7 — ML reporting weaknesses

* Only one algorithm, no task override, no model choice, no timing, no
  baseline comparison, no confusion matrix.
* `classification_report` is presented as a raw table with `accuracy`,
  `macro avg` and `weighted avg` mixed in as if they were classes.
* Feature importance returns one-hot column names (`categorical__city_Delhi`)
  rather than the original feature, and there is no cap/readability.
* A single-class target raises deep inside scikit-learn instead of a clear
  message.

## 3. Keep / merge / simplify / remove

| Feature | Decision | Rationale |
|---|---|---|
| CSV/Excel loading | **Keep + harden** | Core; add validation, type inference, row cap, column-name cleanup. |
| Cleaning (dedupe / missing / trim) | **Keep, redesign** | Core, but as an explicit, reversible plan on an immutable revision. |
| EDA statistics & correlations | **Keep + optimise** | Reuse one profile; stop double computation; add quality issues. |
| Automatic insights | **Merge** | `eda.generate_insights` + `summarizer` → single data-driven insights module. |
| 36-chart auto gallery | **Remove** | Replaced by ≤ 5 curated charts + a manual chart builder. |
| Box/scatter/line chart functions | **Simplify** | Kept as builder options / report renderers, no longer auto-generated. |
| RandomForest-only training | **Simplify** | Small model registry, 2 recommended models, sampling, timing. |
| Feature importance | **Keep + improve** | Group one-hot columns back to source features, cap at 15. |
| Report generator | **Keep + wire up** | Was dead code; now the export step of the workflow (HTML + PDF). |
| Streamlit dashboard | **Remove** | Replaced by the multi-page HTML/CSS/JS frontend; would otherwise be a dead 656-line file. |
| `seaborn`, `scipy`, `streamlit` deps | **Remove** | Unused / superseded. |
| Dataset download | **Add** | Export the prepared dataset as CSV. |

## 4. Target architecture

```
Browser (HTML + CSS + vanilla JS, 6 pages)
        │  JSON over /api/*          ▲ JSON chart specs (client-rendered SVG)
        ▼                            │
Flask API ──► DatasetStore (in-memory, per-session, TTL + LRU)
        │         └── immutable revisions (raw → cleaned → …)
        ├──► profile.py   (one vectorised pass per revision, memoised)
        ├──► charts.py    (curated specs + builder, server-side aggregation)
        ├──► ml_model.py  (feature selection, model registry, sampling, jobs)
        ├──► summarizer.py(data-driven insights, no fabricated AI text)
        └──► report_generator.py (HTML/PDF, in-memory, no temp files)
```

Cache keys are `revision + settings hash` on an immutable revision object, so a
stale result is structurally impossible: cleaning or re-uploading creates a new
revision with an empty cache.

## 5. Success criteria

* First useful view (upload → overview → charts) under ~2 s at 100k rows.
* ≤ 5 charts by default, no identifier/cardinality explosions.
* ML training deterministic, memory-bounded, sampled above 50k rows, with
  visible progress and honest metrics.
* No `main`-branch changes; all work on `arena/90769277-visualizer`.

## 6. Outcome

The plan above was implemented in stages 3–5 and verified in stage 6. The
refactored request path is **154× faster at 20k rows** (24.0 s → 0.156 s) and
**182× faster at 100k rows** (113.5 s → 0.622 s), with peak RSS down from
763.7 MB to 240.9 MB (20k) and from 3,243.5 MB to 291.7 MB (100k). ML training
dropped from 137.4 s / 3,313 MB to 0.703 s / 241 MB at 20k rows, and the 100k
case that previously crashed now completes in 1.3 s on a 50k sample.

The full before/after table, file inventory, test results and limitations are
in [`REPORT.md`](REPORT.md). Automated verification: `python -m pytest tests -q`
→ 170 passed.
