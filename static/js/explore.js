import { api, loadSession, kindLabel } from "./api.js";
import { mountShell, toast, escapeHtml, showError, icon } from "./ui.js";
import { chartCard, renderChart, TYPE_LABELS } from "./charts.js";

const state = {
  dataset: null,
  catalog: null,
  activeType: null,
};

const typeSelect = document.getElementById("chart-type");
const xSelect = document.getElementById("chart-x");
const ySelect = document.getElementById("chart-y");
const topnField = document.getElementById("chart-topn-field");
const binsField = document.getElementById("chart-bins-field");
const yField = document.getElementById("chart-y-field");
const errorNode = document.getElementById("builder-error");

init();

async function init() {
  let session;
  try {
    session = await loadSession();
  } catch {
    return;
  }

  state.dataset = session.dataset;
  mountShell({ page: "explore", dataset: session.dataset });

  document.getElementById("view-tabs").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (tab) switchView(tab.dataset.view);
  });

  document.getElementById("render-button").addEventListener("click", renderBuilderChart);
  typeSelect.addEventListener("change", updateBuilderFields);
  Array.from([xSelect, ySelect]).forEach((select) =>
    select.addEventListener("change", () => {
      errorNode.textContent = "";
    })
  );

  try {
    const data = await api.get(`/api/datasets/${state.dataset.id}/charts`);
    state.catalog = data.catalog;
    renderSuggested(data.charts, data.dataset);
    setupBuilder();
  } catch (error) {
    showError(document.getElementById("chart-grid"), error, init);
    toast({ type: "error", title: "Could not load charts", detail: error.message });
  }
}

function switchView(view) {
  document.querySelectorAll("#view-tabs .tab").forEach((tab) => {
    const active = tab.dataset.view === view;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  document.getElementById("suggested-view").classList.toggle("hidden", view !== "suggested");
  document.getElementById("builder-view").classList.toggle("hidden", view !== "builder");
}

function renderSuggested(charts, dataset) {
  const grid = document.getElementById("chart-grid");
  const count = document.getElementById("chart-count");
  const hint = document.getElementById("chart-hint");

  count.textContent = `${charts.length} suggested chart${charts.length === 1 ? "" : "s"}`;
  hint.textContent = `${dataset.name} · revision ${dataset.revision}`;

  grid.replaceChildren();

  if (!charts.length) {
    grid.innerHTML = `<div class="card" style="grid-column:1/-1">
      <div class="empty">
        <div class="empty__icon">${icon("explore")}</div>
        <div class="empty__title">No chart fits this dataset automatically</div>
        <p class="empty__text">Charts need at least one numeric, categorical or date column. Use the builder to choose columns explicitly, or go back and clean the data first.</p>
        <button class="btn btn--sm mt-12" type="button" id="empty-builder">Open chart builder</button>
      </div>
    </div>`;
    document.getElementById("empty-builder")?.addEventListener("click", () => switchView("builder"));
    return;
  }

  charts.forEach((spec) => {
    grid.appendChild(chartCard(spec, { wide: spec.type === "heatmap" }));
  });
}

function setupBuilder() {
  const types = state.catalog.types || [];
  typeSelect.innerHTML = types
    .map((type) => `<option value="${type.id}">${escapeHtml(type.label)}</option>`)
    .join("");

  state.activeType = types[0]?.id;
  updateBuilderFields();
}

function columnsFor(type, axis) {
  const kinds = axis === "y" ? type.y_kinds : type.x_kinds;
  return (state.catalog.columns || []).filter(
    (column) => kinds.includes(column.kind) && !column.is_constant
  );
}

function fillOptions(select, columns, { includeEmpty = false, emptyLabel = "None" } = {}) {
  const options = columns
    .map(
      (column) =>
        `<option value="${escapeHtml(column.name)}">${escapeHtml(column.name)} · ${escapeHtml(
          kindLabel(column.kind)
        )}</option>`
    )
    .join("");
  select.innerHTML = (includeEmpty ? `<option value="">${emptyLabel}</option>` : "") + options;
  select.disabled = columns.length === 0 && !includeEmpty;
}

function updateBuilderFields() {
  const type = (state.catalog.types || []).find((item) => item.id === typeSelect.value);
  if (!type) return;
  state.activeType = type.id;

  const needsX = type.id !== "heatmap";
  document.getElementById("chart-x").closest(".field").classList.toggle("hidden", !needsX);
  yField.classList.toggle("hidden", !type.requires_y);
  binsField.classList.toggle("hidden", type.id !== "histogram");
  topnField.classList.toggle("hidden", type.id !== "bar");
  document.getElementById("chart-type-hint").textContent = type.description;
  errorNode.textContent = "";

  if (!needsX) {
    xSelect.innerHTML = "";
    return;
  }

  const xColumns = columnsFor(type, "x");
  fillOptions(xSelect, xColumns);

  if (type.requires_y) {
    const yColumns = columnsFor(type, "y").filter((column) => column.name !== xSelect.value);
    fillOptions(ySelect, yColumns);
    const lineWithCount = type.id === "line";
    if (lineWithCount) {
      fillOptions(ySelect, yColumns, { includeEmpty: true, emptyLabel: "Row count (no average)" });
    }
  }
}

async function renderBuilderChart() {
  const type = state.activeType;
  const button = document.getElementById("render-button");
  const chartHost = document.getElementById("builder-chart");
  const placeholder = document.getElementById("builder-placeholder");
  const insight = document.getElementById("builder-insight");

  errorNode.textContent = "";
  button.disabled = true;
  button.textContent = "Rendering…";

  const payload = {
    type,
    x: type === "heatmap" ? null : xSelect.value || null,
    y: ySelect.value || null,
    top_n: Number(document.getElementById("chart-topn").value),
    bins: Number(document.getElementById("chart-bins").value),
  };

  try {
    const data = await api.post(`/api/datasets/${state.dataset.id}/charts`, payload);
    placeholder.classList.add("hidden");
    chartHost.classList.remove("hidden");
    renderChart(data.chart, chartHost);
    insight.classList.remove("hidden");
    insight.innerHTML = `<strong>${escapeHtml(data.chart.title)}.</strong> ${escapeHtml(
      data.chart.insight || ""
    )}`;
  } catch (error) {
    errorNode.textContent = error.message;
    toast({ type: "error", title: "Chart could not be drawn", detail: error.message });
  } finally {
    button.disabled = false;
    button.textContent = "Render chart";
  }
}
