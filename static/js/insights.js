import { api, loadSession, fmtInt, fmtPct } from "./api.js";
import { mountShell, toast, escapeHtml, icon, setBusy, showError } from "./ui.js";

const state = {
  dataset: null,
  data: null,
};

init();

async function init() {
  let session;
  try {
    session = await loadSession();
  } catch {
    return;
  }

  state.dataset = session.dataset;
  mountShell({ page: "insights", dataset: session.dataset });

  document.getElementById("report-button").addEventListener("click", downloadReport);
  document.getElementById("csv-button").addEventListener("click", () => {
    api.downloadCsv(state.dataset.id);
  });

  try {
    const data = await api.get(`/api/datasets/${state.dataset.id}/insights`);
    state.data = data;
    render(data);
  } catch (error) {
    showError(document.getElementById("summary-body"), error, init);
    toast({ type: "error", title: "Could not load insights", detail: error.message });
  }
}

function render(data) {
  const dataset = data.dataset;
  const insights = data.insights || {};

  document.getElementById("dataset-sub").textContent =
    `${dataset.name} · revision ${dataset.revision} · ${fmtInt(dataset.rows)} rows × ${dataset.columns} columns`;
  document.getElementById("revision-badge").textContent = `revision ${dataset.revision}`;
  document.getElementById("summary-meta").textContent =
    `${fmtPct(dataset.missing_pct)} missing · ${fmtInt(dataset.duplicate_rows)} duplicate rows · ${(insights.findings || []).length} findings`;

  document.getElementById("summary-body").innerHTML = `
    <p class="narrative">${escapeHtml(insights.summary || "No summary available.")}</p>
    <div class="row mt-16">
      <span class="chip chip--muted">${icon("database")} ${fmtInt(dataset.rows)} rows</span>
      <span class="chip chip--muted">${icon("overview")} ${dataset.columns} columns</span>
      <span class="chip chip--muted">${icon("shield")} ${fmtPct(dataset.missing_pct)} missing</span>
      ${
        dataset.is_cleaned
          ? `<span class="chip chip--muted">${icon("check")} cleaned revision</span>`
          : `<span class="chip chip--muted">${icon("file")} raw upload</span>`
      }
    </div>`;

  renderFindings(insights.findings || []);
  renderRecommendations(insights.recommendations || []);
  renderModel(insights.model, data.model);

  const chartCount = (data.charts || []).length;
  document.getElementById("report-note").textContent =
    `Reports include up to four of the ${chartCount} curated charts.`;
}

function renderFindings(findings) {
  const body = document.getElementById("findings-body");
  if (!findings.length) {
    body.innerHTML = `<p class="hint">No notable findings were detected in this revision.</p>`;
    return;
  }

  body.innerHTML = findings
    .map((finding) => {
      const iconName =
        finding.kind === "quality"
          ? "shield"
          : finding.kind === "relationship"
          ? "sparkles"
          : finding.kind === "distribution"
          ? "explore"
          : "info";
      return `<div class="finding finding--${escapeHtml(finding.severity || "info")}">
        <span class="finding__icon">${icon(iconName)}</span>
        <div>
          <div class="finding__title">${escapeHtml(finding.title)}</div>
          <div class="finding__detail">${escapeHtml(finding.detail)}</div>
        </div>
      </div>`;
    })
    .join("");
}

function renderRecommendations(recommendations) {
  const body = document.getElementById("rec-body");
  if (!recommendations.length) {
    body.innerHTML = `<p class="hint">Nothing to act on — the dataset looks ready.</p>`;
    return;
  }

  body.innerHTML = recommendations
    .map((item, index) => {
      const href = item.page ? `/${item.page}` : null;
      const content = `<span class="rec__num">${index + 1}</span>
        <span>
          <span class="rec__title">${escapeHtml(item.action || "Next step")}</span>
          <span class="rec__detail">${escapeHtml(item.detail || "")}</span>
        </span>`;
      return href
        ? `<a class="rec" href="${href}">${content}</a>`
        : `<div class="rec">${content}</div>`;
    })
    .join("");
}

function renderModel(review, model) {
  const card = document.getElementById("model-card");
  const body = document.getElementById("model-body");

  if (!model || !review) {
    card.hidden = false;
    body.innerHTML = `<div class="empty" style="padding:26px 12px">
      <div class="empty__icon">${icon("model")}</div>
      <div class="empty__title">No model trained yet</div>
      <p class="empty__text">Train a model to add evaluation metrics, a baseline comparison and feature importance to this report.</p>
      <a class="btn btn--sm mt-12" href="/model">Train a model</a>
    </div>`;
    return;
  }

  card.hidden = false;
  const bullets = (review.bullets || []).map((bullet) => `<li>${escapeHtml(bullet)}</li>`).join("");
  body.innerHTML = `<div class="model-review">
    <div class="model-review__headline">${escapeHtml(review.headline || "")}</div>
    ${bullets ? `<ul>${bullets}</ul>` : ""}
    <div class="row mt-12">
      <span class="badge badge--brand">${escapeHtml(model.model_label || model.model_id)}</span>
      <span class="badge badge--neutral">${escapeHtml(model.task)}</span>
      <span class="badge badge--neutral">target ${escapeHtml(model.target)}</span>
      ${
        model.rows?.sampled
          ? `<span class="badge badge--warn">sampled ${fmtInt(model.rows.used)} / ${fmtInt(model.rows.total)}</span>`
          : ""
      }
    </div>
  </div>`;
}

async function downloadReport() {
  const button = document.getElementById("report-button");
  const format = document.getElementById("report-format").value;
  const includeCharts = document.getElementById("include-charts").checked;

  setBusy(button, true, "Building…");
  try {
    const filename = await api.download(
      `/api/datasets/${state.dataset.id}/report`,
      { format, include_charts: includeCharts },
      `report.${format}`
    );
    toast({ type: "success", title: "Report downloaded", detail: filename });
  } catch (error) {
    toast({ type: "error", title: "Report generation failed", detail: error.message });
  } finally {
    setBusy(button, false);
  }
}
