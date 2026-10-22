# AI Data Visualizer

A local, no-build data analysis workspace: upload a dataset, review its profile,
apply a controlled cleaning plan, explore curated charts, optionally train a
small set of models, and export an HTML or PDF report.

The frontend is plain HTML, CSS and vanilla JavaScript served by Flask. There is
no bundler, no framework and no external CDN — everything runs from this repo.

## Quick start

```bash
git clone https://github.com/mitashsharma/Visualizer.git
cd Visualizer
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:8000>. Use `python app.py --port 9000` to change the
port, `--host 127.0.0.1` to bind locally only, and `--debug` to enable the Flask
reloader.

A 1,212-row sample workbook ships with the repo; the upload page has a
**Load sample dataset** button, or run:

```bash
python app.py --port 8000
```

## The five modules

| Page | Purpose |
| --- | --- |
| **Upload & Overview** (`/`) | Validate and parse a file, then show rows, columns, dtypes, missing cells, duplicates and profile findings. |
| **Cleaning & Preparation** (`/prepare`) | Preview a cleaning plan (missing values, duplicates, whitespace, dropped columns) before applying it as a new dataset revision. |
| **EDA & Visualization** (`/explore`) | A curated set of at most five charts plus a manual chart builder. Charts are JSON specs rendered as SVG in the browser. |
| **ML & Evaluation** (`/model`) | Analyse a target column, pick one model, train with limits, and review metrics, confusion matrix / residuals and feature importance. |
| **Insights & Report Export** (`/insights`) | Read the generated narrative, revisit the model review, and download an HTML or PDF report or the cleaned CSV. |

## Project structure

```text
Visualizer/
├── app.py                     # entrypoint: argparse + create_app().run()
├── src/
│   ├── api.py                 # Flask app factory, all routes, static pages
│   ├── config.py              # frozen Settings + AIVIZ_* environment overrides
│   ├── data_loader.py         # delimiter/encoding sniffing, type inference, row caps
│   ├── data_cleaner.py        # legacy helpers + plan normalisation and application
│   ├── profile.py             # single-pass column profiling and kind classification
│   ├── eda.py                 # correlation and outlier payloads
│   ├── charts.py              # chart-spec builders (no matplotlib)
│   ├── visualizer.py          # matplotlib renderers used for report images
│   ├── ml_model.py            # task detection, feature selection, training, caching
│   ├── summarizer.py          # deterministic insights and recommendations
│   ├── report_generator.py    # self-contained HTML and PDF reports
│   ├── formatting.py          # number/percent formatting helpers
│   ├── jobs.py                # background job runner
│   └── store.py               # dataset revisions, lazy caches, model results
├── static/
│   ├── *.html                 # six pages sharing one shell
│   ├── css/                   # tokens, layout, components, page rules
│   ├── js/                    # api/ui/charts helpers + one module per page
│   └── favicon.svg
├── data/sample_customers.csv  # 1,212-row demo dataset
├── tests/                     # pytest suite (does not need a browser)
└── benchmarks/                # dataset generator + reproducible benchmarks
```

## API overview

All routes live under `/api` and return JSON. Errors use `{"error": ...}` with a
matching status code.

```text
GET    /api/health                       service liveness
GET    /api/session                      current dataset + limits
POST   /api/datasets                     multipart upload (csv/tsv/txt/xlsx/xls)
POST   /api/datasets/sample              load the bundled sample
GET    /api/datasets/<id>                dataset summary
DELETE /api/datasets/<id>                drop it from the session
GET    /api/datasets/<id>/preview        paginated rows
GET    /api/datasets/<id>/overview       profile + insights + suggestions
POST   /api/datasets/<id>/cleaning/preview   dry-run a plan
POST   /api/datasets/<id>/clean          apply a plan -> new revision
POST   /api/datasets/<id>/restore        go back to the root revision
GET    /api/datasets/<id>/charts         curated charts + catalog
POST   /api/datasets/<id>/charts         build one chart from the catalog
GET    /api/datasets/<id>/model/options  targets, models, limits
GET    /api/datasets/<id>/target/<col>   analyse one candidate target
POST   /api/datasets/<id>/train          202 + job id, or a cached result
GET    /api/jobs/<id>                    job status/progress/result
GET    /api/datasets/<id>/models         training history
GET    /api/datasets/<id>/insights       narrative + model review
POST   /api/datasets/<id>/report         html or pdf download
GET    /api/datasets/<id>/export.csv     current revision as CSV
```

## Configuration

Every limit can be overridden with an environment variable:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AIVIZ_HOST` / `AIVIZ_PORT` | `0.0.0.0` / `8000` | Bind address |
| `AIVIZ_MAX_UPLOAD_MB` | `100` | Upload size ceiling |
| `AIVIZ_MAX_ROWS` | `500000` | Parsed rows kept per dataset |
| `AIVIZ_MAX_COLUMNS` | `500` | Column ceiling |
| `AIVIZ_ML_MAX_ROWS` | `50000` | Rows used per training run (samples are reported) |
| `AIVIZ_MAX_DATASETS` | `4` | Datasets retained per browser session |
| `AIVIZ_SESSION_TTL` | `3600` | Idle seconds before a session is evicted |
| `AIVIZ_SECRET_KEY` | random per run | Flask session-signing key; set it to keep cookies valid across restarts |

Uploads are parsed in memory; nothing is written to disk. Sessions are
cookie-scoped, so two browsers never see each other's data.

## Tests

```bash
python -m pytest tests -q
```

The suite covers loading and type inference, cleaning plans, profiling, chart
specs, EDA payloads, model training and caching, the HTTP contract, report
rendering and the classic helper chain.

## Benchmarks

```bash
python benchmarks/make_data.py --rows 2000 20000 100000
python benchmarks/bench_new.py --all
```

Results are written to `benchmarks/results_new.json`. See `AUDIT.md` for the
baseline (`benchmarks/results_baseline.json`) and the before/after comparison.

## Notes and limitations

- Charts are drawn as SVG in the browser; report images come from matplotlib.
- Machine learning is deliberately small: gradient boosting, random forest and a
  linear model, with a bounded row sample and no hyper-parameter search.
- Metrics are computed on a held-out split of the uploaded data and are never
  synthesised; undersized or single-class targets are rejected with a message.
- The service keeps state in memory per session. It is a single-process tool, not
  a multi-user deployment; run it behind a real server and store for that.
