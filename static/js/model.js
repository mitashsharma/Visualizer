import { api, loadSession, fmtInt, fmtNum, fmtPct, fmtDateTime } from "./api.js";
import { mountShell, toast, icon, escapeHtml, setBusy, showError } from "./ui.js";
import { renderChart } from "./charts.js";

const state = {
  dataset: null,
  options: null,
  analysis: null,
  result: null,
  history: [],
  timers: { poll: null, elapsed: null },
  startedAt: 0,
  training: false,
};

const targetSelect = document.getElementById("target-select");
const trainButton = document.getElementById("train-button");
const rowLimitSelect = document.getElementById("row-limit");

init();

async function init() {
  let session;
  try {
    session = await loadSession();
  } catch {
    return;
  }

  state.dataset = session.dataset;
  mountShell({ page: "model", dataset: session.dataset });

  targetSelect.addEventListener("change", () => loadTarget(targetSelect.value));
  trainButton.addEventListener("click", () => train());

  try {
    const [options, models] = await Promise.all([
      api.get(`/api/datasets/${state.dataset.id}/model/options`),
      api.get(`/api/datasets/${state.dataset.id}/models`),
    ]);
    state.options = options;
    state.history = models.models || [];

    populateTargets(options);
    renderHistory();
    restoreLatestRun();

    if (options.default_target) {
      targetSelect.value = options.default_target;
      await loadTarget(options.default_target);
    }
  } catch (error) {
    showError(document.getElementById("target-summary"), error, init);
    toast({ type: "error", title: "Could not load model options", detail: error.message });
  }
}

function populateTargets(options) {
  const eligible = (options.targets || []).filter((target) => target.eligible);
  targetSelect.disabled = eligible.length === 0;

  if (!eligible.length) {
    targetSelect.innerHTML = '<option>No eligible target columns</option>';
    document.getElementById("target-summary").innerHTML = `<div class="alert alert--warn">${icon("alert")}
      <div class="alert__body"><strong>No column can be predicted.</strong>
      <div>Every column is constant, an identifier, a date, or has too few known values.</div></div></div>`;
    document.getElementById("train-hint").textContent = "Add more usable columns or rows to train a model.";
    return;
  }

  targetSelect.innerHTML = eligible
    .map(
      (target) => `<option value="${escapeHtml(target.name)}">${escapeHtml(target.name)} · ${
        target.suggested_task === "classification" ? "class" : "numeric"
      }</option>`
    )
    .join("");
}

async function loadTarget(column) {
  if (!column) return;
  trainButton.disabled = true;
  trainButton.textContent = "Train model";

  const summary = document.getElementById("target-summary");
  summary.innerHTML = '<span class="muted">Analysing target…</span>';

  try {
    const analysis = await api.get(
      `/api/datasets/${state.dataset.id}/target/${encodeURIComponent(column)}`
    );
    state.analysis = analysis;
    renderTargetSummary(analysis);
    renderModels(analysis);
    populateRowLimits(analysis.rows);
    trainButton.disabled = !analysis.eligible;
    document.getElementById("train-hint").textContent = analysis.eligible
      ? `Ready to train on ${analysis.suggested_task}.`
      : analysis.message || "This column cannot be used as a target.";
  } catch (error) {
    summary.innerHTML = `<div class="alert alert--danger">${icon("alert")}<div class="alert__body">${escapeHtml(error.message)}</div></div>`;
    trainButton.disabled = true;
  }
}

function renderTargetSummary(analysis) {
  const warnings = (analysis.warnings || [])
    .map(
      (warning) => `<div class="alert alert--warn mt-8">${icon("alert")}
        <div class="alert__body"><div>${escapeHtml(warning)}</div></div></div>`
    )
    .join("");

  if (!analysis.eligible) {
    document.getElementById("target-summary").innerHTML = `<div class="alert alert--danger">${icon("alert")}
      <div class="alert__body"><strong>Not usable as a target.</strong>
      <div>${escapeHtml(analysis.message || "")}</div></div></div>`;
    return;
  }

  document.getElementById("target-summary").innerHTML = `
    <div class="row" style="justify-content:space-between">
      <span class="badge badge--brand">${escapeHtml(analysis.suggested_task)}</span>
      <span class="hint">${fmtInt(analysis.rows)} usable rows · ${fmtInt(analysis.unique)} distinct</span>
    </div>
    <p class="hint mt-8">${escapeHtml(analysis.reason)}</p>
    ${
      analysis.imbalance_pct
        ? `<p class="hint mt-8">Largest class: ${fmtPct(analysis.imbalance_pct)} of rows.</p>`
        : ""
    }
    ${warnings}`;
}

function renderModels(analysis) {
  const models = analysis.models || [];
  const container = document.getElementById("model-cards");
  const defaultModel = analysis.default_model || models[0]?.id;

  container.innerHTML = models
    .map(
      (model) => `<label class="model-option">
        <input type="radio" name="model" value="${escapeHtml(model.id)}" ${
          model.id === defaultModel ? "checked" : ""
        }>
        <span class="model-option__box">
          <span class="model-option__top">
            <span class="model-option__name">${escapeHtml(model.label)}</span>
            ${model.recommended ? '<span class="model-option__flag">recommended</span>' : ""}
            <span class="model-option__speed">${escapeHtml(model.speed)}</span>
          </span>
          <span class="model-option__desc">${escapeHtml(model.description)}</span>
        </span>
      </label>`
    )
    .join("");
}

function populateRowLimits(rows) {
  const limits = state.options.limits;
  const options = limits.row_options || [0];
  const defaultLimit = limits.default_row_limit || 0;

  const usable = options.filter((value) => value === 0 || value < rows);
  if (!usable.length) usable.push(0);

  let selected = usable.find((value) => value === defaultLimit);
  if (selected === undefined) {
    selected = usable.filter((value) => value !== 0 && value <= Math.max(defaultLimit, rows)).pop() ?? 0;
  }
  if (rows <= defaultLimit) selected = 0;

  rowLimitSelect.innerHTML = usable
    .map((value) => {
      const label = value === 0 ? `All ${fmtInt(rows)} rows` : `${fmtInt(value)} rows (sample)`;
      return `<option value="${value}" ${value === selected ? "selected" : ""}>${label}</option>`;
    })
    .join("");
}

function selectedModel() {
  return document.querySelector('input[name="model"]:checked')?.value;
}

async function train() {
  if (state.training || !state.analysis) return;
  state.training = true;

  hide("run-idle");
  hide("run-result");
  hide("run-error");
  show("run-progress");

  setBusy(trainButton, true, "Training…");
  document.getElementById("run-badge").textContent = "running";
  document.getElementById("run-badge").className = "badge badge--warn";
  document.getElementById("run-sub").textContent = `${state.analysis.column} · ${selectedModel()}`;
  document.getElementById("progress-bar").style.width = "2%";
  document.getElementById("progress-stage").textContent = "Submitting…";
  document.getElementById("progress-percent").textContent = "0%";
  document.getElementById("progress-message").textContent = "";
  renderSteps(0);
  startElapsed();

  try {
    const response = await api.post(`/api/datasets/${state.dataset.id}/train`, {
      target: state.analysis.column,
      model: selectedModel(),
      task: state.analysis.suggested_task,
      max_rows: Number(rowLimitSelect.value),
      test_size: Number(document.getElementById("test-size").value),
    });

    const job = response.job;
    if (job.id) {
      pollJob(job.id);
    } else {
      finishTraining(job.result, Boolean(response.cached));
    }
  } catch (error) {
    failTraining(error.message);
  }
}

function pollJob(jobId) {
  clearInterval(state.timers.poll);
  state.timers.poll = setInterval(async () => {
    try {
      const data = await api.get(`/api/jobs/${jobId}`);
      const job = data.job;
      renderProgress(job);

      if (job.status === "done") {
        clearInterval(state.timers.poll);
        finishTraining(job.result, false);
      } else if (job.status === "error") {
        clearInterval(state.timers.poll);
        failTraining(job.error || "Training failed.");
      }
    } catch (error) {
      clearInterval(state.timers.poll);
      failTraining(error.message);
    }
  }, 600);
}

function renderProgress(job) {
  document.getElementById("progress-bar").style.width = `${Math.max(job.progress || 0, 2)}%`;
  document.getElementById("progress-percent").textContent = `${job.progress || 0}%`;
  document.getElementById("progress-stage").textContent = job.stage || "Working…";
  document.getElementById("progress-message").textContent = job.message || "";
  document.getElementById("progress-elapsed").textContent = `${(job.elapsed_ms / 1000).toFixed(1)}s elapsed`;
  renderSteps(job.progress || 0);
}

const TRAINING_STEPS = [
  { label: "Validate target & select features", at: 5 },
  { label: "Sample rows", at: 14 },
  { label: "Prepare features", at: 24 },
  { label: "Fit the model", at: 38 },
  { label: "Evaluate on the test split", at: 76 },
  { label: "Rank feature importance", at: 93 },
];

function renderSteps(progress) {
  const container = document.getElementById("training-steps");
  container.innerHTML = TRAINING_STEPS.map((step, index) => {
    const next = TRAINING_STEPS[index + 1];
    const done = progress >= (next ? next.at : 100);
    const active = !done && progress >= step.at;
    return `<div class="training-step ${done ? "is-done" : active ? "is-active" : ""}">
      <span class="training-step__dot"></span>
      <span>${escapeHtml(step.label)}</span>
      ${done ? `<span style="margin-left:auto;color:var(--success)">${icon("check")}</span>` : ""}
    </div>`;
  }).join("");
}

function startElapsed() {
  state.startedAt = performance.now();
  clearInterval(state.timers.elapsed);
  state.timers.elapsed = setInterval(() => {
    const seconds = (performance.now() - state.startedAt) / 1000;
    document.getElementById("progress-elapsed").textContent = `${seconds.toFixed(1)}s elapsed`;
  }, 200);
}

function stopTimers() {
  clearInterval(state.timers.poll);
  clearInterval(state.timers.elapsed);
}

function finishTraining(result, cached) {
  stopTimers();
  state.training = false;
  setBusy(trainButton, false);
  trainButton.disabled = false;

  hide("run-progress");
  show("run-result");

  document.getElementById("run-badge").textContent = cached ? "reused" : "complete";
  document.getElementById("run-badge").className = "badge badge--success";
  document.getElementById("run-sub").textContent = `${result.model_label} · target ${result.target}`;

  renderResult(result);

  toast({
    type: "success",
    title: cached ? "Reused a cached result" : "Training complete",
    detail: `${result.model_label} · ${(result.timings.total ?? 0).toFixed(1)}s`,
  });

  refreshHistory();
}

function showRecord(record, badge = "loaded") {
  hide("run-idle");
  hide("run-progress");
  hide("run-error");
  show("run-result");

  document.getElementById("run-badge").textContent = badge;
  document.getElementById("run-badge").className =
    badge === "last run" ? "badge badge--brand" : "badge badge--neutral";
  document.getElementById("run-sub").textContent = `${record.model_label} · target ${record.target}`;
  renderResult(record);
}

/* A cached run should be visible immediately, not only after a click. */
function restoreLatestRun() {
  const latest = state.history[0];
  if (!latest) return;
  showRecord(latest, "last run");
  document.querySelectorAll(".history-item").forEach((node) => {
    node.classList.toggle(
      "is-active",
      node.dataset.created === String(latest.created_at || "")
    );
  });
}

function refreshHistory() {
  api
    .get(`/api/datasets/${state.dataset.id}/models`)
    .then((data) => {
      state.history = data.models || [];
      renderHistory();
    })
    .catch(() => {});
}

function failTraining(message) {
  stopTimers();
  state.training = false;
  setBusy(trainButton, false);
  trainButton.disabled = false;

  hide("run-progress");
  show("run-error");
  document.getElementById("run-badge").textContent = "failed";
  document.getElementById("run-badge").className = "badge badge--danger";
  document.getElementById("run-error").innerHTML = `<div class="alert alert--danger">${icon("alert")}
    <div class="alert__body"><strong>Training failed.</strong><div>${escapeHtml(message)}</div></div></div>`;
  toast({ type: "error", title: "Training failed", detail: message });
}

/* ----------------------------- result view ----------------------------- */

function metricTiles(result) {
  const metrics = result.metrics || {};
  const baseline = result.baseline || {};

  if (result.task === "classification") {
    const delta =
      baseline.accuracy !== undefined && metrics.accuracy !== undefined
        ? (metrics.accuracy - baseline.accuracy) * 100
        : null;
    return [
      {
        label: "Accuracy",
        value: fmtPct((metrics.accuracy ?? 0) * 100),
        sub: delta !== null ? `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts vs baseline` : "",
      },
      {
        label: "Balanced accuracy",
        value: fmtPct((metrics.balanced_accuracy ?? 0) * 100),
        sub: "accounts for class size",
      },
      {
        label: "Macro F1",
        value: fmtPct((metrics.f1_macro ?? 0) * 100),
        sub: "equal weight per class",
      },
      {
        label: "ROC AUC",
        value: metrics.roc_auc !== undefined ? Number(metrics.roc_auc).toFixed(3) : "—",
        sub: metrics.roc_auc !== undefined ? "binary ranking quality" : "needs binary classes",
      },
    ];
  }

  return [
    { label: "R²", value: (metrics.r2 ?? 0).toFixed(3), sub: "variance explained" },
    { label: "MAE", value: fmtNum(metrics.mae), sub: "mean absolute error" },
    { label: "RMSE", value: fmtNum(metrics.rmse), sub: "penalises large errors" },
    {
      label: "MAPE",
      value: metrics.mape !== undefined ? fmtPct(metrics.mape) : "—",
      sub: "mean absolute % error",
    },
  ];
}

function renderResult(result) {
  state.result = result;
  const host = document.getElementById("run-result");
  const tiles = metricTiles(result);

  const baselineText =
    result.task === "classification"
      ? `Majority-class baseline: ${fmtPct((result.baseline?.accuracy ?? 0) * 100)} (always predict “${
          result.baseline?.label ?? "?"
        }”).`
      : `Baseline (predict the training mean): MAE ${fmtNum(result.baseline?.mae)}.`;

  const better =
    result.task === "classification"
      ? (result.metrics?.accuracy ?? 0) >= (result.baseline?.accuracy ?? 0)
      : (result.metrics?.mae ?? 0) <= (result.baseline?.mae ?? Infinity);

  const rows = result.rows || {};
  const chips = [
    rows.sampled
      ? `<span class="badge badge--warn">sampled ${fmtInt(rows.used)} / ${fmtInt(rows.total)}</span>`
      : `<span class="badge badge--neutral">${fmtInt(rows.used)} rows</span>`,
    `<span class="badge badge--neutral">${fmtInt(result.train_rows)} train / ${fmtInt(result.test_rows)} test</span>`,
    `<span class="badge badge--neutral">${Number(result.timings?.total ?? 0).toFixed(1)}s total</span>`,
    `<span class="badge badge--neutral">${escapeHtml(result.model_label)}</span>`,
  ];

  const warnings = (result.warnings || [])
    .map(
      (warning) => `<div class="alert alert--warn mt-8">${icon("alert")}
        <div class="alert__body"><div>${escapeHtml(warning)}</div></div></div>`
    )
    .join("");

  host.innerHTML = `
    <div class="result-block">
      <div class="metric-grid">
        ${tiles
          .map(
            (tile) => `<div class="metric">
              <div class="metric__label">${escapeHtml(tile.label)}</div>
              <div class="metric__value">${escapeHtml(tile.value)}</div>
              <div class="metric__sub">${escapeHtml(tile.sub)}</div>
            </div>`
          )
          .join("")}
      </div>
      <div class="alert ${better ? "alert--success" : "alert--warn"} mt-12">
        ${icon(better ? "check" : "alert")}
        <div class="alert__body">
          <strong>${better ? "Beats the baseline." : "Below the baseline."}</strong>
          <div>${escapeHtml(baselineText)}</div>
        </div>
      </div>
      ${warnings}
      <div class="row mt-12">${chips.join("")}</div>
    </div>

    ${
      result.task === "classification"
        ? `<div class="result-block">
            <div class="result-block__title">Confusion matrix</div>
            ${confusionTable(result)}
          </div>
          <div class="result-block">
            <div class="result-block__title">Per-class metrics</div>
            ${perClassTable(result)}
          </div>`
        : `<div class="result-block">
            <div class="result-block__title">Predicted vs actual</div>
            <div id="prediction-chart" style="min-height:280px"></div>
            <p class="hint mt-8">Each point is a held-out row; points close to the diagonal are accurate. Shown up to 400 rows.</p>
          </div>`
    }

    <div class="result-block">
      <div class="result-block__title">Feature importance</div>
      ${importanceRows(result)}
    </div>`;

  if (result.task === "regression" && result.predictions?.length) {
    const hostChart = document.getElementById("prediction-chart");
    renderChart(
      {
        type: "scatter",
        title: "Predicted vs actual",
        insight: "",
        x_label: "Actual",
        y_label: "Predicted",
        data: { points: result.predictions },
      },
      hostChart
    );
  }
}

function confusionTable(result) {
  const matrix = result.confusion_matrix || [];
  const classes = result.classes || [];
  if (!matrix.length) return '<p class="hint">Confusion matrix unavailable.</p>';

  const max = Math.max(...matrix.flat(), 1);
  const head = classes.map((label) => `<th class="num">${escapeHtml(truncate(label, 10))}</th>`).join("");

  const body = matrix
    .map((row, rowIndex) => {
      const cells = row
        .map((count, columnIndex) => {
          const ratio = count / max;
          const background = `rgba(79, 70, 229, ${0.06 + ratio * 0.72})`;
          const color = ratio > 0.5 ? "#ffffff" : "var(--ink)";
          return `<td class="num" style="background:${background};color:${color};font-weight:600"
            title="Actual ${escapeHtml(classes[rowIndex])} → predicted ${escapeHtml(classes[columnIndex])}">${fmtInt(count)}</td>`;
        })
        .join("");
      return `<tr><th>${escapeHtml(truncate(classes[rowIndex], 14))}</th>${cells}</tr>`;
    })
    .join("");

  return `<div class="table-wrap"><table class="data">
    <thead><tr><th>Actual \\ predicted</th>${head}</tr></thead>
    <tbody>${body}</tbody></table></div>
    <p class="hint mt-8">Rows are actual classes; columns are predictions. Diagonal cells are correct.</p>`;
}

function perClassTable(result) {
  const rows = result.per_class || [];
  if (!rows.length) return "";
  const body = rows
    .map(
      (row) => `<tr>
        <td class="strong">${escapeHtml(truncate(row.label, 18))}</td>
        <td class="num">${Number(row.precision ?? 0).toFixed(3)}</td>
        <td class="num">${Number(row.recall ?? 0).toFixed(3)}</td>
        <td class="num">${Number(row.f1 ?? 0).toFixed(3)}</td>
        <td class="num">${fmtInt(row.support)}</td>
      </tr>`
    )
    .join("");
  return `<div class="table-wrap"><table class="data">
    <thead><tr><th>Class</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th></tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

function importanceRows(result) {
  const items = (result.importance || []).slice(0, 10);
  if (!items.length) {
    return '<p class="hint">This model does not expose feature importance.</p>';
  }
  const max = Math.max(...items.map((item) => item.importance), 0.0001);
  return items
    .map(
      (item) => `<div class="importance-row">
        <span class="truncate" title="${escapeHtml(item.feature)}">${escapeHtml(item.feature)}</span>
        <span class="importance-row__track">
          <span class="importance-row__fill" style="width:${(item.importance / max) * 100}%"></span>
        </span>
        <span class="importance-row__value">${fmtPct(item.importance * 100, 1)}</span>
      </div>`
    )
    .join("");
}

function renderHistory() {
  const body = document.getElementById("history-body");
  document.getElementById("history-count").textContent = state.history.length;

  if (!state.history.length) {
    body.innerHTML = '<div class="empty"><div class="empty__text">No runs yet.</div></div>';
    return;
  }

  body.innerHTML = state.history
    .map((result) => {
      const metric =
        result.task === "classification"
          ? `accuracy ${fmtPct((result.metrics?.accuracy ?? 0) * 100)}`
          : `R² ${(result.metrics?.r2 ?? 0).toFixed(3)}`;
      return `<div class="history-item" data-created="${escapeHtml(result.created_at || "")}">
        <div class="history-item__main">
          <div class="history-item__title">${escapeHtml(result.model_label)} · ${escapeHtml(result.target)}</div>
          <div class="history-item__meta">${escapeHtml(result.task)} · ${metric} · ${
            result.rows?.sampled ? "sampled · " : ""
          }${fmtDateTime(result.created_at)}</div>
        </div>
        <span class="badge badge--neutral">view</span>
      </div>`;
    })
    .join("");

  body.querySelectorAll(".history-item").forEach((item) => {
    item.addEventListener("click", () => {
      const record = state.history.find((entry) => entry.created_at === item.dataset.created);
      if (!record) return;
      showRecord(record, "loaded");
      document.querySelectorAll(".history-item").forEach((node) => node.classList.remove("is-active"));
      item.classList.add("is-active");
    });
  });
}

/* ------------------------------- helpers ------------------------------- */

function truncate(value, max) {
  const text = String(value ?? "");
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function hide(id) {
  document.getElementById(id)?.classList.add("hidden");
}

function show(id) {
  document.getElementById(id)?.classList.remove("hidden");
}
