import {
  api,
  loadSession,
  fmtInt,
  fmtPct,
  formatValue,
  kindLabel,
  kindClass,
  debounce,
} from "./api.js";
import {
  mountShell,
  toast,
  icon,
  escapeHtml,
  setBusy,
  confirmDialog,
  showError,
} from "./ui.js";

const STRATEGIES = {
  leave: "Leave as is",
  median: "Fill with median",
  mean: "Fill with mean",
  mode: "Fill with mode",
  zero: "Fill with zero",
  drop_rows: "Drop rows",
};

const state = {
  dataset: null,
  profile: null,
  suggestions: null,
  plan: null,
  previewSeq: 0,
};

const refreshPreview = debounce(() => updatePreview(), 320);

init();

async function init() {
  let session;
  try {
    session = await loadSession();
  } catch {
    return;
  }

  state.dataset = session.dataset;
  mountShell({ page: "prepare", dataset: session.dataset });

  document.getElementById("apply-button").addEventListener("click", applyPlan);
  document.getElementById("restore-button").addEventListener("click", restoreOriginal);

  try {
    const data = await api.get(`/api/datasets/${state.dataset.id}/overview`);
    state.profile = data;
    state.dataset = data.dataset;
    state.suggestions = data.suggestions;

    state.plan = {
      remove_duplicates: Boolean(data.suggestions.remove_duplicates),
      trim_whitespace: Boolean(data.suggestions.trim_whitespace),
      drop_columns: [...(data.suggestions.drop_columns || [])],
      missing: { ...(data.suggestions.missing || {}) },
    };

    renderRevision();
    renderToggles();
    renderMissing();
    renderDropList();
    updatePreview();
  } catch (error) {
    showError(document.getElementById("missing-area"), error, init);
    toast({ type: "error", title: "Could not load the dataset", detail: error.message });
  }
}

function renderRevision() {
  const dataset = state.dataset;
  const cleaned = dataset.revision > 1;

  document.getElementById("restore-button").hidden = !cleaned;
  document.getElementById("revision-badge").textContent = cleaned ? "cleaned" : "raw";
  document.getElementById("revision-title").textContent = `Revision ${dataset.revision}`;
  document.getElementById("revision-hint").textContent = cleaned
    ? `${(dataset.cleaning?.actions || []).length} cleaning action(s) applied`
    : "Original upload";

  const banner = document.getElementById("revision-banner");
  if (!cleaned) {
    banner.innerHTML = "";
    return;
  }

  const actions = dataset.cleaning?.actions || [];
  banner.innerHTML = `<div class="alert alert--success section">${icon("check")}
    <div class="alert__body">
      <strong>Revision ${dataset.revision} is active.</strong>
      <div>${actions.map((action) => escapeHtml(action)).join(" · ") || "Cleaning applied."}</div>
      <div class="mt-8"><button class="btn btn--sm" type="button" id="banner-restore">Restore original</button></div>
    </div>
  </div>`;

  document.getElementById("banner-restore")?.addEventListener("click", restoreOriginal);
}

function renderToggles() {
  const overview = state.profile.overview;
  const duplicates = document.getElementById("toggle-duplicates");
  const whitespace = document.getElementById("toggle-whitespace");

  duplicates.checked = Boolean(state.plan.remove_duplicates);
  duplicates.disabled = overview.duplicate_rows === 0;
  document.getElementById("duplicates-title").textContent =
    overview.duplicate_rows === 0 ? "No duplicate rows found" : "Remove duplicate rows";
  document.getElementById("duplicates-hint").textContent =
    overview.duplicate_rows === 0
      ? "Nothing to remove in this revision."
      : `${fmtInt(overview.duplicate_rows)} exact duplicate rows (${fmtPct(overview.duplicate_pct)}).`;

  const whitespaceColumns = state.suggestions.summary.whitespace_columns || [];
  whitespace.checked = Boolean(state.plan.trim_whitespace);
  whitespace.disabled = whitespaceColumns.length === 0;
  document.getElementById("whitespace-title").textContent =
    whitespaceColumns.length === 0
      ? "Whitespace is already consistent"
      : "Trim whitespace in text columns";
  document.getElementById("whitespace-hint").textContent =
    whitespaceColumns.length === 0
      ? "No leading or trailing spaces were detected."
      : `Affects: ${whitespaceColumns.slice(0, 5).join(", ")}${whitespaceColumns.length > 5 ? "…" : ""}`;

  duplicates.addEventListener("change", () => {
    state.plan.remove_duplicates = duplicates.checked;
    refreshPreview();
  });
  whitespace.addEventListener("change", () => {
    state.plan.trim_whitespace = whitespace.checked;
    refreshPreview();
  });
}

function optionsFor(column) {
  const numeric = column.kind === "numeric";
  return Object.entries(STRATEGIES)
    .filter(([key]) => {
      if (key === "median" || key === "mean" || key === "zero") return numeric;
      return true;
    })
    .map(([key, label]) => ({ key, label }));
}

function renderMissing() {
  const columns = state.profile.columns.filter((column) => column.missing > 0);
  const area = document.getElementById("missing-area");
  const totalMissing = state.profile.overview.missing_cells;
  document.getElementById("missing-total").textContent = `${fmtInt(totalMissing)} cells`;

  if (!columns.length) {
    area.innerHTML = `<div class="empty" style="padding:30px 20px">
      <div class="empty__icon">${icon("check")}</div>
      <div class="empty__title">No missing values</div>
      <p class="empty__text">Every cell in this revision is populated.</p>
    </div>`;
    return;
  }

  const rows = columns
    .map((column) => {
      const selected = state.plan.missing[column.name] || "leave";
      const options = optionsFor(column)
        .map(
          (option) =>
            `<option value="${option.key}" ${option.key === selected ? "selected" : ""}>${option.label}</option>`
        )
        .join("");

      return `<tr data-column="${escapeHtml(column.name)}">
        <td>
          <div class="col-name">${escapeHtml(column.name)}</div>
          <span class="badge ${kindClass(column.kind)}" style="margin-top:3px">${escapeHtml(kindLabel(column.kind))}</span>
        </td>
        <td class="num">${fmtInt(column.missing)}<div class="hint">${fmtPct(column.missing_pct)}</div></td>
        <td><select class="select" data-missing-column="${escapeHtml(column.name)}" aria-label="Strategy for ${escapeHtml(column.name)}">${options}</select></td>
      </tr>`;
    })
    .join("");

  area.innerHTML = `<div class="table-wrap">
    <table class="data strategy-table">
      <thead><tr><th>Column</th><th class="num">Missing</th><th>Strategy</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;

  area.querySelectorAll("select[data-missing-column]").forEach((select) => {
    select.addEventListener("change", () => {
      const column = select.dataset.missingColumn;
      if (select.value === "leave") delete state.plan.missing[column];
      else state.plan.missing[column] = select.value;
      refreshPreview();
    });
  });

  syncDroppedRows();
}

function renderDropList() {
  const list = document.getElementById("drop-list");
  const columns = state.profile.columns;

  list.innerHTML = columns
    .map(
      (column) => `<label class="drop-chip">
        <input type="checkbox" data-drop-column="${escapeHtml(column.name)}" ${
          state.plan.drop_columns.includes(column.name) ? "checked" : ""
        }>
        <span>${escapeHtml(column.name)}</span>
      </label>`
    )
    .join("");

  list.addEventListener("change", (event) => {
    const checkbox = event.target.closest("input[data-drop-column]");
    if (!checkbox) return;
    const column = checkbox.dataset.dropColumn;
    const set = new Set(state.plan.drop_columns);
    if (checkbox.checked) set.add(column);
    else set.delete(column);
    state.plan.drop_columns = [...set];

    if (checkbox.checked) delete state.plan.missing[column];
    syncDroppedRows();
    refreshPreview();
  });
}

function syncDroppedRows() {
  document.querySelectorAll("#missing-area tr[data-column]").forEach((row) => {
    row.hidden = state.plan.drop_columns.includes(row.dataset.column);
  });
}

async function updatePreview() {
  if (!state.profile) return;
  const sequence = ++state.previewSeq;
  const note = document.getElementById("projection-note");

  try {
    const data = await api.post(
      `/api/datasets/${state.dataset.id}/cleaning/preview`,
      { plan: buildPlan() }
    );
    if (sequence !== state.previewSeq) return;

    const summary = data.summary;
    renderProjection(summary);
    renderPreview(data.preview);

    const delta = summary.rows_before - summary.rows_after;
    const parts = [];
    if (delta) parts.push(`${fmtInt(delta)} rows removed`);
    if (summary.filled_cells) parts.push(`${fmtInt(summary.filled_cells)} cells filled`);
    if (summary.columns_before !== summary.columns_after) {
      parts.push(`${summary.columns_before - summary.columns_after} columns dropped`);
    }
    note.textContent = parts.length ? parts.join(" · ") : "No changes selected yet.";
    note.classList.remove("faint");

    // The preview is the only place that knows the plan is valid; enable the
    // apply button once it succeeds and restores its label after an apply.
    const apply = document.getElementById("apply-button");
    if (apply) {
      apply.disabled = false;
      apply.textContent = "Apply cleaning";
    }
  } catch (error) {
    if (sequence !== state.previewSeq) return;
    note.textContent = error.message;
    note.classList.add("faint");
  }
}

function renderProjection(summary) {
  document.getElementById("projection").innerHTML = `
    <div class="projection__item">
      <div class="projection__label">Rows</div>
      <div class="projection__value">${fmtInt(summary.rows_after)}</div>
      <div class="projection__delta">${summary.rows_removed ? `−${fmtInt(summary.rows_removed)}` : "no change"}</div>
    </div>
    <div class="projection__item">
      <div class="projection__label">Columns</div>
      <div class="projection__value">${fmtInt(summary.columns_after)}</div>
      <div class="projection__delta">${summary.columns_before - summary.columns_after ? `−${summary.columns_before - summary.columns_after}` : "no change"}</div>
    </div>
    <div class="projection__item">
      <div class="projection__label">Missing</div>
      <div class="projection__value">${fmtInt(summary.missing_after)}</div>
      <div class="projection__delta">was ${fmtInt(summary.missing_before)}</div>
    </div>`;
}

function renderPreview(preview) {
  const area = document.getElementById("preview-area");
  if (!preview.rows.length) {
    area.innerHTML = `<p class="hint">No rows would remain with this plan.</p>`;
    return;
  }

  const columns = preview.columns.slice(0, 6);
  const head = columns.map((name) => `<th>${escapeHtml(name)}</th>`).join("");
  const body = preview.rows
    .map((row) => {
      const cells = row
        .slice(0, 6)
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
    <div class="row" style="justify-content:space-between">
      <span class="result-block__title" style="margin:0">Preview</span>
      <span class="hint">${columns.length} of ${preview.columns.length} columns</span>
    </div>
    <div class="table-wrap mt-8">
      <table class="data preview-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>
    </div>`;
}

function buildPlan() {
  const missing = {};
  Object.entries(state.plan.missing).forEach(([column, strategy]) => {
    if (!state.plan.drop_columns.includes(column)) missing[column] = strategy;
  });

  return {
    remove_duplicates: state.plan.remove_duplicates,
    trim_whitespace: state.plan.trim_whitespace,
    drop_columns: state.plan.drop_columns,
    missing,
  };
}

async function applyPlan() {
  const button = document.getElementById("apply-button");
  setBusy(button, true, "Applying…");

  try {
    const data = await api.post(`/api/datasets/${state.dataset.id}/clean`, {
      plan: buildPlan(),
    });
    state.dataset = data.dataset;
    renderRevision();

    const summary = data.summary;
    const banner = document.getElementById("revision-banner");
    banner.innerHTML = `<div class="alert alert--success section">${icon("check")}
      <div class="alert__body">
        <strong>Cleaning applied — revision ${data.dataset.revision} is now active.</strong>
        <div>${summary.rows_before.toLocaleString()} → ${summary.rows_after.toLocaleString()} rows ·
             ${fmtInt(summary.missing_after)} missing cells remaining.</div>
        <div class="mt-12 row">
          <a class="btn btn--sm btn--primary" href="/overview">View overview</a>
          <a class="btn btn--sm" href="/explore">Explore charts</a>
          <button class="btn btn--sm" type="button" id="banner-restore">Restore original</button>
        </div>
      </div>
    </div>`;
    document.getElementById("banner-restore")?.addEventListener("click", restoreOriginal);

    setBusy(button, false);
    button.disabled = true;
    button.textContent = "Cleaning applied";
    toast({ type: "success", title: "New revision created", detail: `Revision ${data.dataset.revision}` });
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) {
    setBusy(button, false);
    toast({ type: "error", title: "Could not apply cleaning", detail: error.message });
  }
}

async function restoreOriginal() {
  const confirmed = await confirmDialog({
    title: "Restore the original revision?",
    message: "The active dataset switches back to the raw upload. Your cleaned revision stays in memory until it expires.",
    confirmLabel: "Restore original",
  });
  if (!confirmed) return;

  try {
    await api.post(`/api/datasets/${state.dataset.id}/restore`);
    toast({ type: "success", title: "Original revision restored" });
    window.location.assign("/overview");
  } catch (error) {
    toast({ type: "error", title: "Could not restore", detail: error.message });
  }
}
