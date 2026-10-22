/* Tiny fetch wrapper + formatters shared by every page. */

export class ApiError extends Error {
  constructor(message, status = 0, detail = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function parseError(response) {
  let message = `Request failed (${response.status})`;
  let detail = "";
  try {
    const body = await response.json();
    message = body.error || body.detail || message;
    detail = body.detail && body.detail !== message ? body.detail : "";
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(message, response.status, detail);
}

async function request(path, options = {}) {
  const { method = "GET", body, signal } = options;
  const response = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
  });

  if (!response.ok) throw await parseError(response);
  if (response.status === 204) return null;

  const type = response.headers.get("content-type") || "";
  if (type.includes("application/json")) return response.json();
  return response;
}

export const api = {
  get: (path, options) => request(path, options),
  post: (path, body, options) => request(path, { ...options, method: "POST", body: body ?? {} }),
  del: (path) => request(path, { method: "DELETE" }),
  session: () => request("/api/session"),

  upload(file, onProgress) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);

      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/datasets");
      xhr.withCredentials = true;
      xhr.responseType = "json";

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(event.loaded / event.total);
        }
      };

      xhr.onload = () => {
        const body = xhr.response;
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(body);
        } else {
          reject(
            new ApiError(
              (body && (body.error || body.detail)) || `Upload failed (${xhr.status})`,
              xhr.status
            )
          );
        }
      };
      xhr.onerror = () => reject(new ApiError("Network error while uploading the file.", 0));
      xhr.ontimeout = () => reject(new ApiError("The upload timed out.", 0));
      xhr.send(form);
    });
  },

  async download(path, body, fallbackName) {
    const response = await fetch(path, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    if (!response.ok) throw await parseError(response);

    const blob = await response.blob();
    const disposition = response.headers.get("content-disposition") || "";
    let filename = fallbackName;
    const utf8 = /filename\*=UTF-8''([^;]+)/i.exec(disposition);
    const plain = /filename="?([^";]+)"?/i.exec(disposition);
    if (utf8) filename = decodeURIComponent(utf8[1]);
    else if (plain) filename = plain[1];

    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 3000);
    return filename;
  },

  downloadCsv(datasetId) {
    const anchor = document.createElement("a");
    anchor.href = `/api/datasets/${datasetId}/export.csv`;
    anchor.download = "";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  },
};

/* ------------------------------ formatters ------------------------------ */

const integerFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const decimalFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

export function fmtInt(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return integerFormat.format(value);
}

export function fmtNum(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (magnitude >= 1e4) return `${(value / 1e3).toFixed(1)}k`;
  if (magnitude >= 10) return decimalFormat.format(value);
  if (magnitude === 0) return "0";
  return Number(value).toFixed(2).replace(/\.?0+$/, "");
}

export function fmtPct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(digits)}%`;
}

export function fmtDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function kindLabel(kind) {
  return { numeric: "Numeric", categorical: "Text", boolean: "Boolean", datetime: "Date" }[kind] || kind;
}

export function kindClass(kind) {
  return { numeric: "kind-numeric", categorical: "kind-categorical", boolean: "kind-boolean", datetime: "kind-datetime" }[kind] || "badge--neutral";
}

export function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return fmtInt(value);
    return Math.abs(value) >= 1 ? decimalFormat.format(value) : String(value);
  }
  return String(value);
}

export function debounce(fn, wait = 260) {
  let timer;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
  wrapped.cancel = () => clearTimeout(timer);
  return wrapped;
}

export async function loadSession({ redirect = true } = {}) {
  const data = await api.session();
  if (!data.dataset && redirect) {
    window.location.replace("/");
    throw new ApiError("No dataset loaded", 0);
  }
  return data;
}
