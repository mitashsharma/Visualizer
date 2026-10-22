import {
  api,
  loadSession,
  fmtInt,
  fmtNum,
  fmtPct,
  formatValue,
  kindLabel,
  kindClass,
} from "./api.js";
import {
  mountShell,
  toast,
  icon,
  escapeHtml,
  confirmDialog,
  showError,
  emptyState,
  datasetBadges,
  skeletonLines,
} from "./ui.js";

const state = {
  dataset: null,
  profile: null,
  insights: null,
  tab: "columns",
  preview: { offset: 0, limit: 25, total: 0, loading: false },
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
  mountShell({ page: "overview", dataset: session.dataset });

  document.getElementById("download-button").addEventListener("click", () => {
    api.downloadCsv(state.dataset.id);
  });

  document.querySelectorAll("#table-tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  try {
    const data = await api.get(`/api/datasets/${state.dataset.id}/overview`);
    state.profile = data;
    state.dataset = data.dataset;
    state.insights = data.insights;
    render(data);
  } catch (error) {
    showError(document.getElementById("kpis"), error, init);
    toast({ type: "error", title: "Could not load the overview", detail: error.message });
  }
}

function render(data) {
  const overview = data.overview;
  const dataset = data.dataset;
  document.body.dataset.datasetId = dataset.id;

  document.getElementById("dataset-name").textContent = dataset.name;
  document.getElementById("dataset-sub").textContent =
    `${fmtInt(overview.rows)} rows × ${overview.columns} columns · source ${dataset.source}` +
    (overview.truncated ? " · truncated at the row limit" : "");

  const badges = [datasetBadges(dataset)];
  if (dataset.is_cleaned) badges.push('<span class="badge badge--success">cleaned</span>');
  if (dataset.truncated) badges.push('<span class="badge badge--warn">row limit applied</span>');
  document.getElementById("dataset-badges").innerHTML = badges.join("");

  document.getElementById("download-button").disabled = false;
  renderRestore(dataset);

  renderKpis(overview);
  renderHealth(data.quality);
  renderFindings(data.insights);
  renderComposition(data.kinds);
  renderColumns(data.columns, data.kinds, overview);
  renderNextStep(data);

  document.getElementById("table-sub").textContent =
    `${overview.columns} columns · ${overview.identifier_columns} identifier(s) · ${overview.constant_columns} constant`;
}

function renderRestore(dataset) {
  const button = document.getElementById("restore-button");
  button.hidden = dataset.revision <= 1;
  button.onclick = async () => {
    const confirmed = await confirmDialog({
      title: "Return to the original data?",
      message: "The analysis switches back to the raw upload. Cleaning revisions are discarded.",
      confirmLabel: "Restore original",
    });
    if (!confirmed) return;
    try {
      await api.post(`/api/datasets/${dataset.id}/restore`);
      toast({ type: "success", title: "Original revision restored" });
      window.location.reload();
    } catch (error) {
      toast({ type: "error", title: "Could not restore", detail: error.message });
    }
  };
}

function renderKpis(overview) {
  const tiles = [
    {
      label: "Rows",
      value: fmtInt(overview.rows),
      hint: overview.truncated ? "row limit applied" : "records in this revision",
    },
    {
      label: "Columns",
      value: fmtInt(overview.columns),
      hint: `${overview.numeric_columns} numeric · ${overview.categorical_columns} text`,
    },
    {
      label: "Missing cells",
      value: fmtPct(overview.missing_pct),
      hint: `${fmtInt(overview.missing_cells)} cells · ${overview.datetime_columns} date column(s)`,
    },
    {
      label: "Duplicate rows",
      value: fmtInt(overview.duplicate_rows),
      hint: `${fmtPct(overview.duplicate_pct)} of rows · ${overview.memory_mb} MB in memory`,
    },
  ];

  document.getElementById("kpis").innerHTML = tiles
    .map(
      (tile) => `<div class="kpi">
        <div class="kpi__label">${escapeHtml(tile.label)}</div>
        <div class="kpi__value">${escapeHtml(tile.value)}</div>
        <div class="kpi__hint">${escapeHtml(tile.hint)}</div>
      </div>`
    )
    .join("");
}

function renderHealth(issues) {
  const body = document.getElementById("health-body");
  const counter = document.getElementById("issue-count");

  if (!issues.length) {
    counter.textContent = "clean";
    body.innerHTML = `<div class="alert alert--success">${icon("check")}
      <div class="alert__body"><strong>No quality issues detected.</strong>
      <div>No missing values, duplicates or constant columns were found.</div></div></div>`;
    return;
  }

  counter.textContent = `${issues.length} issue${issues.length > 1 ? "s" : ""}`;
  body.innerHTML = `<div class="health-list">${issues
    .map((issue) => {
      const iconName = issue.severity === "high" ? "alert" : issue.severity === "medium" ? "info" : "shield";
      return `<div class="health-item health-item--${escapeHtml(issue.severity)}">
        <span class="health-item__icon">${icon(iconName)}</span>
        <div>
          <div class="health-item__title">${escapeHtml(issue.title)}</div>
          <div class="health-item__detail">${escapeHtml(issue.detail)}</div>
        </div>
      </div>`;
    })
    .join("")}</div>`;
}

function renderFindings(insights) {
  const body = document.getElementById("findings-body");
  const findings = (insights?.findings || []).slice(0, 4);

  if (!findings.length) {
    body.innerHTML = `<p class="hint">${escapeHtml(insights?.summary || "No findings available.")}</p>`;
    return;
  }

  body.innerHTML = `<div class="findings">${findings
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
    .join("")}</div>`;
}

function renderComposition(kinds) {
  const rows = [
    { label: "Numeric", count: kinds.numeric.length, color: "var(--brand)" },
    { label: "Text", count: kinds.categorical.length, color: "var(--accent)" },
    { label: "Date", count: kinds.datetime.length, color: "var(--info)" },
    { label: "Boolean", count: kinds.boolean.length, color: "var(--warn)" },
  ].filter((row) => row.count > 0);

  const total = rows.reduce((sum, row) => sum + row.count, 0) || 1;

  document.getElementById("composition").innerHTML = rows
    .map(
      (row) => `<div class="composition__row">
        <span class="composition__label">${row.label}</span>
        <span class="composition__track">
          <span class="composition__fill" style="width:${(row.count / total) * 100}%;background:${row.color}"></span>
        </span>
        <span class="composition__count">${row.count}</span>
      </div>`
    )
    .join("");
}

function columnSummary(column) {
  if (column.kind === "numeric" && column.stats) {
    const stats = column.stats;
    const parts = [
      `median ${fmtNum(stats.median)}`,
      `range ${fmtNum(stats.min)} – ${fmtNum(stats.max)}`,
    ];
    if (stats.outliers) parts.push(`${fmtInt(stats.outliers)} outliers`);
    return parts.join(" · ");
  }
  if (column.kind === "datetime" && column.stats) {
    const stats = column.stats;
    return `${String(stats.min || "").slice(0, 10)} → ${String(stats.max || "").slice(0, 10)}`;
  }
  if (column.top_values?.length) {
    const top = column.top_values[0];
    return `most common “${top.value}” (${fmtPct(top.pct)})`;
  }
  if (column.is_identifier) return "unique identifier — excluded from charts";
  return "no summary available";
}

function renderColumns(columns, kinds, overview) {
  const identifiers = new Set(kinds.identifier);
  const rows = columns
    .map((column) => {
      const missingWidth = Math.min(column.missing_pct, 100);
      const summary = columnSummary(column);
      return `<tr>
        <td>
          <div class="col-name">${escapeHtml(column.name)}
            ${identifiers.has(column.name) ? '<span class="badge badge--neutral">ID</span>' : ""}
          </div>
        </td>
        <td><span class="badge ${kindClass(column.kind)}">${escapeHtml(kindLabel(column.kind))}</span></td>
        <td>
          <div class="missing-cell">
            <span class="missing-cell__track">
              <span class="missing-cell__fill" style="width:${missingWidth}%"></span>
            </span>
            <span class="missing-cell__value">${fmtPct(column.missing_pct)}</span>
          </div>
        </td>
        <td class="num">${fmtInt(column.unique)}</td>
        <td><div class="column-summary">${escapeHtml(summary)}</div></td>
      </tr>`;
    })
    .join("");

  document.getElementById("table-area").innerHTML = `
    <div class="table-scroll">
      <table class="data">
        <thead><tr>
          <th>Column</th><th>Type</th><th>Missing</th>
          <th class="num">Distinct</th><th>Summary</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="card__foot">
      <span class="hint">${overview.columns} columns · ${overview.cells.toLocaleString()} cells profiled</span>
      <span class="hint">${overview.memory_mb} MB</span>
    </div>`;
}

function renderNextStep(data) {
  const issues = data.quality.length;
  const text = document.getElementById("next-step-text");
  if (data.dataset.revision > 1) {
    text.textContent =
      "You are working on a cleaned revision. Explore the charts or train a model on the prepared data.";
    return;
  }
  text.textContent = issues
    ? `${issues} data-quality issue${issues > 1 ? "s" : ""} found. Apply the suggested plan in one controlled step.`
    : "The data is already clean. Move on to charts, or train a model directly.";
}

function switchTab(tab) {
  state.tab = tab;
  document.querySelectorAll("#table-tabs .tab").forEach((button) => {
    const active = button.dataset.tab === tab;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });

  if (tab === "columns") {
    if (state.profile) {
      renderColumns(
        state.profile.columns,
        state.profile.kinds,
        state.profile.overview
      );
    }
    return;
  }

  loadPreview(0);
}

async function loadPreview(offset) {
  if (!state.profile) return;
  state.preview.loading = true;

  const area = document.getElementById("table-area");
  area.innerHTML = `<div class="card__body">${skeletonLines(4)}</div>`;

  try {
    const data = await api.get(
      `/api/datasets/${state.dataset.id}/preview?limit=${state.preview.limit}&offset=${offset}`
    );
    state.preview.offset = data.offset;
    state.preview.total = data.total;
    renderPreview(data);
  } catch (error) {
    showError(area, error, () => loadPreview(offset));
  } finally {
    state.preview.loading = false;
  }
}

function renderPreview(data) {
  const area = document.getElementById("table-area");
  const start = data.total ? data.offset + 1 : 0;
  const end = data.offset + data.rows.length;

  const head = data.columns.map((column) => `<th>${escapeHtml(column.name)}</th>`).join("");
  const body = data.rows
    .map((row) => {
      const cells = row
        .map((value) => {
          if (value === null || value === undefined || value === "") {
            return '<td class="empty-cell">null</td>';
          }
          const text = formatValue(value);
          return `<td title="${escapeHtml(text)}">${escapeHtml(text)}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  area.innerHTML = `
    <div class="card__body" style="padding-bottom:8px">
      <div class="preview-controls">
        <button class="btn btn--sm" id="preview-prev" ${data.offset === 0 ? "disabled" : ""}>${icon("left")} Previous</button>
        <span>Rows ${fmtInt(start)}–${fmtInt(end)} of ${fmtInt(data.total)}</span>
        <button class="btn btn--sm" id="preview-next" ${!data.has_more ? "disabled" : ""}>Next ${icon("right")}</button>
      </div>
    </div>
    <div class="table-scroll">
      <table class="data preview-table">
        <thead><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>`;

  const previous = document.getElementById("preview-prev");
  const next = document.getElementById("preview-next");
  previous?.addEventListener("click", () =>
    loadPreview(Math.max(0, data.offset - data.limit))
  );
  next?.addEventListener("click", () => loadPreview(data.offset + data.limit));
}
