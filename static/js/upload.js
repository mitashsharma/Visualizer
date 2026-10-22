import { api, loadSession, fmtInt } from "./api.js";
import {
  mountShell,
  toast,
  icon,
  setBusy,
  escapeHtml,
  formatFileSize,
  confirmDialog,
  showError,
} from "./ui.js";

const ALLOWED = ["csv", "tsv", "tab", "txt", "xlsx", "xls"];
const input = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const status = document.getElementById("upload-status");
const sampleButton = document.getElementById("sample-button");

let maxMb = 100;
let busy = false;

init();

async function init() {
  let sessionInfo = null;
  try {
    sessionInfo = await loadSession({ redirect: false });
  } catch (error) {
    toast({ type: "error", title: "Could not reach the server", detail: error.message });
  }

  maxMb = sessionInfo?.limits?.max_upload_mb || 100;
  mountShell({ page: "index", dataset: sessionInfo?.dataset || null });

  if (sessionInfo?.dataset) renderResume(sessionInfo.dataset);
  wireDropzone();
  wireSample();
}

function renderResume(dataset) {
  const slot = document.getElementById("resume-slot");
  slot.innerHTML = `
    <div class="resume">
      <div class="resume__badge">${icon("database")}</div>
      <div class="resume__text">
        <div class="resume__title">${escapeHtml(dataset.name)}</div>
        <div class="resume__meta">
          ${fmtInt(dataset.rows)} rows · ${dataset.columns} columns · revision ${dataset.revision}
          ${dataset.is_cleaned ? " · cleaned" : ""}
        </div>
      </div>
      <a class="btn btn--primary btn--sm" href="/overview">Continue analysis ${icon("arrow")}</a>
      <button class="btn btn--sm" type="button" id="resume-discard">Start fresh</button>
    </div>`;

  document.getElementById("resume-discard").addEventListener("click", async () => {
    const confirmed = await confirmDialog({
      title: "Forget this dataset?",
      message: "The uploaded data will be removed from memory. You can upload it again at any time.",
      confirmLabel: "Forget dataset",
      danger: true,
    });
    if (!confirmed) return;
    try {
      await api.del(`/api/datasets/${dataset.id}`);
      window.location.reload();
    } catch (error) {
      toast({ type: "error", title: "Could not remove the dataset", detail: error.message });
    }
  });
}

function wireDropzone() {
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) upload(file);
    input.value = "";
  });

  ["dragenter", "dragover"].forEach((type) => {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "dragend"].forEach((type) => {
    dropzone.addEventListener(type, () => dropzone.classList.remove("is-dragover"));
  });

  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragover");
    const file = event.dataTransfer?.files?.[0];
    if (file) upload(file);
  });

  // Prevent the browser from opening a dropped file outside the zone.
  ["dragover", "drop"].forEach((type) => {
    window.addEventListener(type, (event) => {
      if (!event.target.closest("#dropzone")) event.preventDefault();
    });
  });
}

function validate(file) {
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  if (!ALLOWED.includes(extension)) {
    return `“${extension || "unknown"}” files are not supported. Upload a CSV, TSV, TXT or Excel file.`;
  }
  if (file.size === 0) return "That file is empty.";
  if (file.size > maxMb * 1024 * 1024) {
    return `This file is ${formatFileSize(file.size)}. The limit is ${maxMb} MB — try a smaller export.`;
  }
  return null;
}

function fileRow(file) {
  return `<div class="upload-status__file">
    ${icon("file")}
    <span class="name">${escapeHtml(file.name)}</span>
    <span class="size">${formatFileSize(file.size)}</span>
  </div>`;
}

function renderProgress(percent) {
  const value = Math.round(percent * 100);
  return `<div class="progress"><div class="progress__bar" style="width:${value}%"></div></div>
    <p class="hint mt-8" id="progress-label">${
      value >= 100 ? "Profiling the dataset…" : `Uploading ${value}%`
    }</p>`;
}

async function upload(file) {
  if (busy) return;

  const error = validate(file);
  if (error) {
    status.innerHTML = `<div class="alert alert--danger">${icon("alert")}<div class="alert__body"><strong>Upload rejected.</strong><div>${escapeHtml(error)}</div></div></div>`;
    toast({ type: "error", title: "Upload rejected", detail: error });
    return;
  }

  busy = true;
  dropzone.classList.add("is-busy");
  status.innerHTML = `${fileRow(file)}${renderProgress(0)}`;

  try {
    const result = await api.upload(file, (fraction) => {
      const progress = status.querySelector(".progress");
      const label = status.querySelector("#progress-label");
      const value = Math.round(fraction * 100);
      if (progress) progress.querySelector(".progress__bar").style.width = `${value}%`;
      if (label) label.textContent = value >= 100 ? "Profiling the dataset…" : `Uploading ${value}%`;
    });

    dropzone.classList.remove("is-busy");
    status.innerHTML = `${fileRow(file)}<div class="progress"><div class="progress__bar" style="width:100%"></div></div>
      <p class="hint mt-8">Ready — ${fmtInt(result.dataset.rows)} rows, ${result.dataset.columns} columns.</p>`;

    (result.warnings || []).forEach((warning) => {
      toast({ type: "info", title: "Note while loading", detail: warning, timeout: 7000 });
    });

    toast({ type: "success", title: "Dataset loaded", detail: result.dataset.name });
    setTimeout(() => window.location.assign("/overview"), 550);
  } catch (uploadError) {
    busy = false;
    dropzone.classList.remove("is-busy");
    showError(status, uploadError, () => upload(file));
    toast({ type: "error", title: "Upload failed", detail: uploadError.message });
  }
}

function wireSample() {
  sampleButton.addEventListener("click", async () => {
    setBusy(sampleButton, true, "Loading sample…");
    try {
      const result = await api.post("/api/datasets/sample");
      toast({ type: "success", title: "Sample dataset loaded", detail: result.dataset.name });
      window.location.assign("/overview");
    } catch (error) {
      setBusy(sampleButton, false);
      toast({ type: "error", title: "Could not load the sample", detail: error.message });
    }
  });
}
