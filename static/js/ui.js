/* Shell, icons, toasts, dialogs and small render helpers. */

const ICONS = {
  upload: '<path d="M12 16V4m0 0L7 9m5-5 5 5M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
  overview:
    '<rect x="3" y="3" width="7" height="7" rx="1.6"/><rect x="14" y="3" width="7" height="7" rx="1.6"/><rect x="3" y="14" width="7" height="7" rx="1.6"/><rect x="14" y="14" width="7" height="7" rx="1.6"/>',
  prepare:
    '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  explore: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  model:
    '<rect x="6" y="6" width="12" height="12" rx="2.5"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
  insights:
    '<path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.4 1 2.3h6c0-.9.4-1.8 1-2.3A7 7 0 0 0 12 2Z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5h.01"/>',
  download: '<path d="M12 3v12m0 0 4-4m-4 4-4-4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
  refresh: '<path d="M21 12a9 9 0 1 1-2.6-6.4M21 3v6h-6"/>',
  left: '<path d="M15 18l-6-6 6-6"/>',
  right: '<path d="M9 6l6 6-6 6"/>',
  file: '<path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7Z"/><path d="M14 2v5h5M9 13h6M9 17h4"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  database:
    '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  shield: '<path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5Z"/><path d="m9 12 2 2 4-4"/>',
  trash: '<path d="M3 6h18M8 6V4h8v2M6 6l1 15h10l1-15M10 11v6M14 11v6"/>',
  play: '<path d="M7 4v16l13-8Z"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2"/>',
  sparkles: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8Z"/><path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9Z"/>',
};

export function icon(name, className = "") {
  const body = ICONS[name] || ICONS.info;
  return `<svg class="${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function formatFileSize(bytes) {
  if (!bytes && bytes !== 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function initials(name) {
  const words = String(name || "?").trim().split(/\s+/);
  return words.slice(0, 2).map((word) => word[0] || "").join("").toUpperCase() || "?";
}

/* ------------------------------- theme -------------------------------- */

const THEME_KEY = "aiviz-theme";

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
  document.dispatchEvent(new CustomEvent("aiviz:theme", { detail: { theme } }));
}

export function currentTheme() {
  return document.documentElement.dataset.theme || "light";
}

export function toggleTheme() {
  applyTheme(currentTheme() === "dark" ? "light" : "dark");
}

/* -------------------------------- shell -------------------------------- */

const NAV = [
  { page: "index", href: "/", label: "Upload dataset", icon: "upload", step: "01" },
  { page: "overview", href: "/overview", label: "Overview", icon: "overview", step: "02" },
  { page: "prepare", href: "/prepare", label: "Prepare", icon: "prepare", step: "03" },
  { page: "explore", href: "/explore", label: "Explore charts", icon: "explore", step: "04" },
  { page: "model", href: "/model", label: "Model", icon: "model", step: "05" },
  { page: "insights", href: "/insights", label: "Insights & report", icon: "insights", step: "06" },
];

const PAGE_TITLES = {
  index: "Upload",
  overview: "Overview",
  prepare: "Prepare",
  explore: "Explore",
  model: "Model",
  insights: "Insights",
};

export function mountShell({ page, dataset = null }) {
  const navHtml = NAV.map((item) => {
    const disabled = !dataset && item.page !== "index";
    const active = page === item.page;
    return `<a class="nav__item ${active ? "is-active" : ""} ${disabled ? "is-disabled" : ""}"
      href="${item.href}" ${active ? 'aria-current="page"' : ""} ${disabled ? 'aria-disabled="true" tabindex="-1"' : ""}>
      ${icon(item.icon, "nav__icon")}<span>${item.label}</span>
      <span class="nav__num">${item.step}</span>
    </a>`;
  }).join("");

  const datasetCard = dataset
    ? `<div class="sidebar__dataset" title="${escapeHtml(dataset.name)}">${escapeHtml(dataset.name)}</div>
       <div class="sidebar__dataset-meta">${dataset.rows.toLocaleString()} rows · ${dataset.columns} cols · rev ${dataset.revision}</div>`
    : `<div class="sidebar__dataset">No dataset yet</div>
       <div class="sidebar__dataset-meta">Upload a file to begin</div>`;

  const pill = dataset
    ? `<a class="dataset-pill" href="/overview" title="Open overview">
         <span class="dataset-pill__dot">${escapeHtml(initials(dataset.name))}</span>
         <span class="dataset-pill__name">${escapeHtml(dataset.name)}</span>
         <span class="dataset-pill__meta">${dataset.rows.toLocaleString()} × ${dataset.columns}</span>
       </a>`
    : "";

  const markup = `
    <div class="sidebar-overlay" id="sidebar-overlay"></div>
    <aside class="sidebar" aria-label="Main navigation">
      <div class="sidebar__brand">
        <span class="sidebar__logo">${icon("explore")}</span>
        <div>
          <div class="sidebar__name">Data Visualizer</div>
          <div class="sidebar__tag">Analytics workspace</div>
        </div>
      </div>
      <div class="sidebar__section">Workflow</div>
      <nav class="nav">${navHtml}</nav>
      <div class="sidebar__footer">
        <div class="sidebar__card">
          <div class="sidebar__card-label">Active dataset</div>
          ${datasetCard}
        </div>
        <p class="sidebar__note">Datasets stay in memory for 60 minutes of inactivity. Nothing is written to disk.</p>
      </div>
    </aside>
    <header class="topbar">
      <button class="icon-btn topbar__menu" id="menu-button" aria-label="Open navigation" aria-expanded="false">
        ${icon("menu")}
      </button>
      <div class="topbar__crumb">Workspace <span aria-hidden="true">/</span> <strong>${PAGE_TITLES[page] || ""}</strong></div>
      <div class="topbar__spacer"></div>
      <div class="topbar__actions">
        ${pill}
        <a class="btn btn--sm" href="/">${icon("upload")}<span>New dataset</span></a>
        <button class="icon-btn" id="theme-toggle" aria-label="Toggle colour theme" title="Toggle colour theme">
          ${icon(currentTheme() === "dark" ? "sun" : "moon")}
        </button>
      </div>
    </header>`;

  document.body.insertAdjacentHTML("afterbegin", markup);

  document.getElementById("theme-toggle").addEventListener("click", () => {
    toggleTheme();
    document.getElementById("theme-toggle").innerHTML = icon(currentTheme() === "dark" ? "sun" : "moon");
  });

  const menu = document.getElementById("menu-button");
  const overlay = document.getElementById("sidebar-overlay");

  const closeMenu = () => {
    document.body.classList.remove("sidebar-open");
    menu.setAttribute("aria-expanded", "false");
  };

  menu.addEventListener("click", () => {
    const open = document.body.classList.toggle("sidebar-open");
    menu.setAttribute("aria-expanded", String(open));
  });
  overlay.addEventListener("click", closeMenu);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMenu();
  });
}

/* -------------------------------- toasts ------------------------------- */

let toastRoot = null;

export function toast({ type = "info", title, detail = "", timeout = 4800 }) {
  if (!toastRoot) {
    toastRoot = document.createElement("div");
    toastRoot.className = "toast-root";
    document.body.appendChild(toastRoot);
  }

  const iconName = type === "success" ? "check" : type === "error" ? "alert" : "info";
  const node = document.createElement("div");
  node.className = `toast toast--${type}`;
  node.setAttribute("role", type === "error" ? "alert" : "status");
  node.innerHTML = `${icon(iconName, "toast__icon")}
    <div class="toast__body">
      <div class="toast__title">${escapeHtml(title)}</div>
      ${detail ? `<div class="toast__detail">${escapeHtml(detail)}</div>` : ""}
    </div>`;
  toastRoot.appendChild(node);

  const remove = () => {
    node.classList.add("is-leaving");
    setTimeout(() => node.remove(), 220);
  };
  const timer = setTimeout(remove, timeout);
  node.addEventListener("click", () => {
    clearTimeout(timer);
    remove();
  });
  return remove;
}

/* -------------------------------- dialogs ------------------------------ */

export function confirmDialog({ title, message, confirmLabel = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    const root = document.createElement("div");
    root.className = "modal-root is-open";
    root.innerHTML = `
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="dialog-title">
        <h3 id="dialog-title">${escapeHtml(title)}</h3>
        <p>${escapeHtml(message)}</p>
        <div class="modal__actions">
          <button class="btn" data-action="cancel">Cancel</button>
          <button class="btn ${danger ? "btn--danger" : "btn--primary"}" data-action="ok">${escapeHtml(confirmLabel)}</button>
        </div>
      </div>`;

    const close = (result) => {
      document.removeEventListener("keydown", onKey);
      root.remove();
      resolve(result);
    };
    const onKey = (event) => {
      if (event.key === "Escape") close(false);
    };

    root.addEventListener("click", (event) => {
      if (event.target === root) close(false);
      const action = event.target.closest("[data-action]")?.dataset.action;
      if (action === "cancel") close(false);
      if (action === "ok") close(true);
    });

    document.addEventListener("keydown", onKey);
    document.body.appendChild(root);
    root.querySelector('[data-action="ok"]').focus();
  });
}

/* ------------------------------- helpers ------------------------------- */

export function setBusy(button, busy, label = "Working…") {
  if (!button) return;
  if (busy) {
    button.dataset.originalHtml = button.innerHTML;
    button.innerHTML = `<span class="spinner"></span><span>${escapeHtml(label)}</span>`;
    button.disabled = true;
  } else {
    if (button.dataset.originalHtml) button.innerHTML = button.dataset.originalHtml;
    button.disabled = false;
  }
}

export function skeletonLines(count = 3) {
  return Array.from({ length: count }, () => '<div class="skeleton skeleton--line"></div>').join("");
}

export function skeletonChart() {
  return '<div class="skeleton skeleton--chart"></div>';
}

export function emptyState({ iconName = "database", title, text = "", action = "" }) {
  return `<div class="empty">
    <div class="empty__icon">${icon(iconName)}</div>
    <div class="empty__title">${escapeHtml(title)}</div>
    ${text ? `<p class="empty__text">${escapeHtml(text)}</p>` : ""}
    ${action}
  </div>`;
}

export function showError(container, error, onRetry) {
  if (!container) return;
  container.innerHTML = `<div class="alert alert--danger">
    ${icon("alert")}
    <div class="alert__body">
      <strong>Something went wrong.</strong>
      <div>${escapeHtml(error?.message || "Unexpected error.")}</div>
    </div>
  </div>`;
  if (onRetry) {
    const button = document.createElement("button");
    button.className = "btn btn--sm mt-12";
    button.textContent = "Try again";
    button.addEventListener("click", onRetry);
    container.querySelector(".alert__body").appendChild(button);
  }
}

export function renderTable({ columns, rows, rowClass = () => "", empty = "No rows to show." }) {
  if (!rows.length) {
    return `<div class="empty"><div class="empty__text">${escapeHtml(empty)}</div></div>`;
  }

  const head = columns
    .map((column) => `<th class="${column.className || ""}">${escapeHtml(column.label)}</th>`)
    .join("");
  const body = rows
    .map((row) => {
      const cells = columns
        .map((column) => {
          const value = column.render ? column.render(row) : escapeHtml(row[column.key] ?? "—");
          return `<td class="${column.className || ""}">${value}</td>`;
        })
        .join("");
      return `<tr class="${rowClass(row)}">${cells}</tr>`;
    })
    .join("");

  return `<div class="table-wrap"><table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

export function datasetBadges(dataset) {
  if (!dataset) return "";
  return `<span class="badge badge--neutral">${dataset.rows.toLocaleString()} rows</span>
    <span class="badge badge--neutral">${dataset.columns} columns</span>
    <span class="badge badge--brand">revision ${dataset.revision}</span>`;
}
