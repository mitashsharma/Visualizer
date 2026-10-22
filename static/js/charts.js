/* SVG chart renderer. No chart library — plain DOM + SVG. */

import { escapeHtml, icon } from "./ui.js";
import { fmtInt, fmtNum, fmtPct } from "./api.js";

const NS = "http://www.w3.org/2000/svg";
const DEFAULT_WIDTH = 720;
const MIN_WIDTH = 320;
const MAX_WIDTH = 780;

/* The renderers below build their geometry from WIDTH. renderChart sets it to
   the measured width of the host before every draw, so labels stay legible on
   phones instead of being scaled down with the whole drawing. */
let WIDTH = DEFAULT_WIDTH;

const TYPE_LABELS = {
  histogram: "Distribution",
  bar: "Category",
  missing: "Missing data",
  heatmap: "Correlation",
  line: "Trend",
  scatter: "Relationship",
  box: "Comparison",
};

/* ------------------------------- helpers ------------------------------- */

function cssVar(name, fallback = "#4f46e5") {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function svgEl(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  if (parent) parent.appendChild(node);
  return node;
}

function label(parent, x, y, content, options = {}) {
  const node = svgEl(
    "text",
    {
      x,
      y,
      "text-anchor": options.anchor || "start",
      class: options.className || "",
      transform: options.transform || null,
    },
    parent
  );
  node.textContent = content;
  return node;
}

function niceScale(maxValue, tickCount = 4) {
  if (!Number.isFinite(maxValue) || maxValue <= 0) {
    return { max: 1, ticks: [0, 0.25, 0.5, 0.75, 1] };
  }
  const rough = maxValue / tickCount;
  const power = 10 ** Math.floor(Math.log10(rough));
  const norm = rough / power;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * power;
  const max = Math.ceil(maxValue / step) * step;
  const ticks = [];
  for (let value = 0; value <= max + step / 2; value += step) {
    ticks.push(Math.round(value * 1e6) / 1e6);
  }
  return { max, ticks };
}

function evenTicks(min, max, count = 4) {
  if (min === max) max = min + 1;
  const ticks = [];
  for (let index = 0; index <= count; index += 1) {
    ticks.push(min + ((max - min) * index) / count);
  }
  return ticks;
}

function truncate(text, max = 16) {
  const value = String(text ?? "");
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

/* ------------------------------- tooltip ------------------------------- */

let tip = null;

function tipNode() {
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "chart-tooltip";
    document.body.appendChild(tip);
  }
  return tip;
}

function moveTip(event) {
  const node = tipNode();
  const offset = 14;
  let left = event.clientX + offset;
  let top = event.clientY + offset;
  const rect = node.getBoundingClientRect();
  if (left + rect.width > window.innerWidth - 8) left = event.clientX - rect.width - offset;
  if (top + rect.height > window.innerHeight - 8) top = event.clientY - rect.height - offset;
  node.style.left = `${Math.max(8, left)}px`;
  node.style.top = `${Math.max(8, top)}px`;
}

function tooltipFor(node, title, lines = []) {
  const show = (event) => {
    const element = tipNode();
    element.replaceChildren();
    const strong = document.createElement("strong");
    strong.textContent = title;
    element.appendChild(strong);
    lines.forEach((line) => {
      const row = document.createElement("div");
      row.textContent = line;
      element.appendChild(row);
    });
    element.classList.add("is-visible");
    moveTip(event);
  };
  node.addEventListener("mouseenter", show);
  node.addEventListener("mousemove", moveTip);
  node.addEventListener("mouseleave", () => tipNode().classList.remove("is-visible"));
}

function hideTip() {
  if (tip) tip.classList.remove("is-visible");
}

/* -------------------------------- axes --------------------------------- */

function drawYAxis(svg, plot, scale, labelText) {
  const line = cssVar("--border-strong", "#cbd5e1");
  const grid = cssVar("--grid", "#eef2f7");

  scale.ticks.forEach((tick) => {
    const y = plot.y + plot.h - (tick / scale.max) * plot.h;
    svgEl("line", { x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y, class: "grid-line", stroke: grid }, svg);
    label(svg, plot.x - 8, y + 3.5, fmtNum(tick), { anchor: "end" });
  });

  svgEl("line", { x1: plot.x, x2: plot.x, y1: plot.y, y2: plot.y + plot.h, class: "axis-line", stroke: line }, svg);

  if (labelText) {
    label(svg, 12, plot.y + plot.h / 2, labelText, {
      anchor: "middle",
      transform: `rotate(-90 12 ${plot.y + plot.h / 2})`,
    });
  }
}

function drawXAxis(svg, plot, options) {
  const { ticks, formatter, labelText } = options;
  const line = cssVar("--border-strong", "#cbd5e1");
  const grid = cssVar("--grid", "#eef2f7");
  const bottom = plot.y + plot.h;

  svgEl("line", { x1: plot.x, x2: plot.x + plot.w, y1: bottom, y2: bottom, class: "axis-line", stroke: line }, svg);

  const positions = options.positions || ticks.map((_, index) => plot.x + (plot.w * index) / Math.max(ticks.length - 1, 1));

  ticks.forEach((tick, index) => {
    const x = positions[index];
    if (tick === null) return;
    if (options.gridLines) {
      svgEl("line", { x1: x, x2: x, y1: plot.y, y2: bottom, class: "grid-line", stroke: grid }, svg);
    }
    label(svg, x, bottom + 15, formatter ? formatter(tick) : String(tick), {
      anchor: options.anchor || "middle",
    });
  });

  if (labelText) {
    label(svg, plot.x + plot.w / 2, bottom + 34, labelText, { anchor: "middle" });
  }
}

function baseSvg(height, title) {
  const svg = svgEl("svg", {
    viewBox: `0 0 ${WIDTH} ${height}`,
    role: "img",
    "aria-label": title || "Chart",
    preserveAspectRatio: "xMidYMid meet",
  });
  svg.classList.add("chart-svg");
  return svg;
}

/* ------------------------------- renderers ----------------------------- */

function renderHistogram(spec) {
  const bins = spec.data?.bins || [];
  if (!bins.length) return null;

  const height = 320;
  const plot = { x: 58, y: 18, w: WIDTH - 58 - 22, h: height - 18 - 46 };
  const svg = baseSvg(height, spec.title);

  const maxCount = Math.max(...bins.map((bin) => bin.count), 1);
  const scale = niceScale(maxCount);
  drawYAxis(svg, plot, scale, spec.y_label);

  const min = bins[0].start;
  const max = bins[bins.length - 1].end;
  const span = max - min || 1;
  const scaleX = (value) => plot.x + ((value - min) / span) * plot.w;

  const total = bins.reduce((sum, bin) => sum + bin.count, 0) || 1;
  const color = cssVar("--brand", "#4f46e5");

  bins.forEach((bin) => {
    const x = scaleX(bin.start);
    const width = Math.max(scaleX(bin.end) - x - 1, 1);
    const barHeight = (bin.count / scale.max) * plot.h;
    const rect = svgEl("rect", {
      x,
      y: plot.y + plot.h - barHeight,
      width,
      height: Math.max(barHeight, 0),
      rx: Math.min(2.5, width / 3),
      fill: color,
      opacity: 0.88,
      class: "mark",
    }, svg);
    tooltipFor(rect, `${fmtNum(bin.start)} – ${fmtNum(bin.end)}`, [
      `${fmtInt(bin.count)} rows`,
      `${fmtPct((bin.count / total) * 100)} of plotted values`,
    ]);
  });

  const tickCount = 4;
  const ticks = Array.from({ length: tickCount + 1 }, (_, index) => min + (span * index) / tickCount);
  drawXAxis(svg, plot, {
    ticks,
    formatter: (value) => fmtNum(value),
    labelText: spec.x_label,
  });

  const median = spec.data?.stats?.median;
  if (Number.isFinite(median) && median >= min && median <= max) {
    const x = scaleX(median);
    svgEl("line", {
      x1: x,
      x2: x,
      y1: plot.y,
      y2: plot.y + plot.h,
      stroke: cssVar("--accent", "#0d9488"),
      "stroke-width": 1.4,
      "stroke-dasharray": "4 4",
    }, svg);
    label(svg, x + 4, plot.y + 11, `median ${fmtNum(median)}`, { className: "label-strong" });
  }

  return svg;
}

function horizontalBars(spec, { valueKey, labelKey, formatter, total, tooltipTitle, barColor }) {
  const items = spec.data?.[labelKey === "label" ? "categories" : "columns"] || [];
  if (!items.length) return null;

  const height = Math.max(240, 54 + items.length * 30);
  const left = 132;
  const plot = { x: left, y: 14, w: WIDTH - left - 54, h: height - 14 - 34 };
  const svg = baseSvg(height, spec.title);

  const maxValue = Math.max(...items.map((item) => item[valueKey]), 1);
  const scale = niceScale(maxValue, 4);
  const grid = cssVar("--grid", "#eef2f7");
  const color = barColor || cssVar("--brand", "#4f46e5");

  scale.ticks.forEach((tick) => {
    const x = plot.x + (tick / scale.max) * plot.w;
    svgEl("line", { x1: x, x2: x, y1: plot.y, y2: plot.y + plot.h, stroke: grid, class: "grid-line" }, svg);
    label(svg, x, plot.y + plot.h + 15, formatter(tick), { anchor: "middle" });
  });

  const rowHeight = plot.h / items.length;

  items.forEach((item, index) => {
    const value = item[valueKey];
    const y = plot.y + index * rowHeight + rowHeight * 0.18;
    const barHeight = rowHeight * 0.64;
    const width = Math.max((value / scale.max) * plot.w, 1);
    const rect = svgEl("rect", {
      x: plot.x,
      y,
      width,
      height: barHeight,
      rx: Math.min(4, barHeight / 2),
      fill: color,
      opacity: item.label === "Other" ? 0.45 : 0.88,
      class: "mark",
    }, svg);

    const text = tooltipTitle(item);
    tooltipFor(rect, text.title, text.lines);
    label(svg, plot.x - 9, y + barHeight / 2 + 3.5, truncate(item[labelKey], 17), {
      anchor: "end",
      className: "label-strong",
    });
    label(svg, plot.x + width + 6, y + barHeight / 2 + 3.5, formatter(value), {
      anchor: "start",
      className: "label-strong",
    });
  });

  return svg;
}

function renderBar(spec) {
  return horizontalBars(spec, {
    valueKey: "count",
    labelKey: "label",
    formatter: fmtNum,
    tooltipTitle: (item) => ({
      title: String(item.label),
      lines: [`${fmtInt(item.count)} rows`, `${fmtPct(item.pct)} of non-missing values`],
    }),
  });
}

function renderMissing(spec) {
  const svg = horizontalBars(spec, {
    valueKey: "pct",
    labelKey: "column",
    formatter: (value) => fmtPct(value, 0),
    barColor: "#f59e0b",
    tooltipTitle: (item) => ({
      title: String(item.column),
      lines: [`${fmtInt(item.missing)} missing cells`, `${fmtPct(item.pct)} of ${fmtInt(item.rows)} rows`],
    }),
  });
  return svg;
}

function heatColor(value) {
  if (!Number.isFinite(value)) return cssVar("--surface-3", "#f1f5f9");
  const positive = [220, 38, 38];
  const negative = [37, 99, 235];
  const base = [241, 245, 249];
  const target = value >= 0 ? positive : negative;
  const amount = Math.min(Math.abs(value), 1) * 0.85;
  const mixed = base.map((channel, index) => Math.round(channel + (target[index] - channel) * amount));
  return `rgb(${mixed.join(",")})`;
}

function renderHeatmap(spec) {
  const columns = spec.data?.columns || [];
  const values = spec.data?.values || [];
  if (columns.length < 2) return null;

  const cell = Math.min(54, (WIDTH - 150) / columns.length);
  const size = cell * columns.length;
  const left = 118;
  const top = 12;
  const height = Math.max(300, top + size + 86);
  const svg = baseSvg(height, spec.title);

  values.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      const x = left + columnIndex * cell;
      const y = top + rowIndex * cell;
      const rect = svgEl("rect", {
        x,
        y,
        width: cell - 2,
        height: cell - 2,
        rx: 3,
        fill: heatColor(value),
        class: "mark",
      }, svg);
      tooltipFor(rect, `${columns[rowIndex]} × ${columns[columnIndex]}`, [
        value === null ? "not computable" : `r = ${Number(value).toFixed(2)}`,
      ]);
      if (columns.length <= 8 && value !== null) {
        label(svg, x + (cell - 2) / 2, y + (cell - 2) / 2 + 3.5, Number(value).toFixed(2), {
          anchor: "middle",
          className: Math.abs(value) > 0.6 ? "" : "label-strong",
        }).style.fill = Math.abs(value) > 0.6 ? "#ffffff" : cssVar("--ink", "#0f172a");
      }
    });
  });

  columns.forEach((column, index) => {
    label(svg, left + index * cell + (cell - 2) / 2, top + size + 8, truncate(column, 12), {
      anchor: "end",
      transform: `rotate(-45 ${left + index * cell + (cell - 2) / 2} ${top + size + 8})`,
    });
    label(svg, left - 8, top + index * cell + cell / 2 + 3, truncate(column, 14), { anchor: "end" });
  });

  return svg;
}

function renderLine(spec) {
  const points = spec.data?.points || [];
  if (!points.length) return null;

  const height = 320;
  const plot = { x: 58, y: 20, w: WIDTH - 58 - 24, h: height - 20 - 52 };
  const svg = baseSvg(height, spec.title);

  const values = points.map((point) => Number(point[1]) || 0);
  const maxValue = Math.max(...values);
  const scale = niceScale(maxValue * 1.05);
  drawYAxis(svg, plot, scale, spec.y_label);

  const step = points.length > 1 ? plot.w / (points.length - 1) : 0;
  const xFor = (index) => plot.x + index * step;
  const yFor = (value) => plot.y + plot.h - (value / scale.max) * plot.h;

  const lineColor = cssVar("--brand", "#4f46e5");
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${xFor(index)},${yFor(Number(point[1]) || 0)}`).join(" ");
  const area = `${path} L${xFor(points.length - 1)},${plot.y + plot.h} L${plot.x},${plot.y + plot.h} Z`;

  svgEl("path", { d: area, fill: lineColor, opacity: 0.08 }, svg);
  svgEl("path", { d: path, fill: "none", stroke: lineColor, "stroke-width": 2.2, "stroke-linejoin": "round" }, svg);

  const every = Math.max(Math.ceil(points.length / 8), 1);
  points.forEach((point, index) => {
    const circle = svgEl("circle", {
      cx: xFor(index),
      cy: yFor(Number(point[1]) || 0),
      r: points.length > 60 ? 1.8 : 3,
      fill: lineColor,
      class: "mark",
    }, svg);
    tooltipFor(circle, String(point[0]), [`${spec.y_label}: ${fmtInt(point[1])}`]);
  });

  const tickIndices = [];
  for (let index = 0; index < points.length; index += every) tickIndices.push(index);
  if (tickIndices[tickIndices.length - 1] !== points.length - 1) tickIndices.push(points.length - 1);

  drawXAxis(svg, plot, {
    ticks: tickIndices.map((index) => points[index][0]),
    positions: tickIndices.map((index) => xFor(index)),
    formatter: (value) => String(value),
    labelText: spec.x_label,
  });

  return svg;
}

function renderScatter(spec) {
  const points = spec.data?.points || [];
  if (!points.length) return null;

  const height = 330;
  const plot = { x: 60, y: 18, w: WIDTH - 60 - 22, h: height - 18 - 52 };
  const svg = baseSvg(height, spec.title);

  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xPad = (xMax - xMin) * 0.05 || 1;
  const yPad = (yMax - yMin) * 0.05 || 1;

  const x0 = xMin - xPad;
  const x1 = xMax + xPad;
  const y0 = yMin - yPad;
  const y1 = yMax + yPad;

  const xFor = (value) => plot.x + ((value - x0) / (x1 - x0)) * plot.w;
  const yFor = (value) => plot.y + plot.h - ((value - y0) / (y1 - y0)) * plot.h;

  const grid = cssVar("--grid", "#eef2f7");
  evenTicks(y0, y1, 4).forEach((tick) => {
    const y = yFor(tick);
    svgEl("line", { x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y, stroke: grid, class: "grid-line" }, svg);
    label(svg, plot.x - 8, y + 3.5, fmtNum(tick), { anchor: "end" });
  });

  evenTicks(x0, x1, 4).forEach((tick) => {
    const x = xFor(tick);
    label(svg, x, plot.y + plot.h + 15, fmtNum(tick), { anchor: "middle" });
  });

  const radius = points.length > 1500 ? 1.5 : points.length > 500 ? 2 : 2.6;
  const color = cssVar("--brand", "#4f46e5");

  points.forEach((point) => {
    const circle = svgEl("circle", {
      cx: xFor(point[0]),
      cy: yFor(point[1]),
      r: radius,
      fill: color,
      opacity: 0.45,
      class: "mark",
    }, svg);
    tooltipFor(circle, `${spec.x_label}: ${fmtNum(point[0])}`, [`${spec.y_label}: ${fmtNum(point[1])}`]);
  });

  label(svg, plot.x + plot.w / 2, plot.y + plot.h + 36, spec.x_label, { anchor: "middle" });
  label(svg, 12, plot.y + plot.h / 2, spec.y_label, {
    anchor: "middle",
    transform: `rotate(-90 12 ${plot.y + plot.h / 2})`,
  });

  return svg;
}

function renderBox(spec) {
  const groups = spec.data?.groups || [];
  if (!groups.length) return null;

  const height = 330;
  const plot = { x: 58, y: 18, w: WIDTH - 58 - 22, h: height - 18 - 54 };
  const svg = baseSvg(height, spec.title);

  const lows = groups.map((group) => group.whisker_low ?? group.min);
  const highs = groups.map((group) => group.whisker_high ?? group.max);
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const pad = (max - min) * 0.06 || 1;
  const y0 = min - pad;
  const y1 = max + pad;
  const yFor = (value) => plot.y + plot.h - ((value - y0) / (y1 - y0)) * plot.h;

  const grid = cssVar("--grid", "#eef2f7");
  evenTicks(y0, y1, 4).forEach((tick) => {
    const y = yFor(tick);
    svgEl("line", { x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y, stroke: grid, class: "grid-line" }, svg);
    label(svg, plot.x - 8, y + 3.5, fmtNum(tick), { anchor: "end" });
  });

  const slot = plot.w / groups.length;
  const boxWidth = Math.min(slot * 0.45, 54);
  const brand = cssVar("--brand", "#4f46e5");
  const accent = cssVar("--accent", "#0d9488");

  groups.forEach((group, index) => {
    const cx = plot.x + slot * index + slot / 2;
    const q1 = yFor(group.q1);
    const q3 = yFor(group.q3);
    const median = yFor(group.median);
    const low = yFor(group.whisker_low ?? group.min);
    const high = yFor(group.whisker_high ?? group.max);

    svgEl("line", { x1: cx, x2: cx, y1: q1, y2: low, stroke: brand, "stroke-width": 1.2 }, svg);
    svgEl("line", { x1: cx, x2: cx, y1: q3, y2: high, stroke: brand, "stroke-width": 1.2 }, svg);
    svgEl("line", { x1: cx - boxWidth / 3, x2: cx + boxWidth / 3, y1: low, y2: low, stroke: brand, "stroke-width": 1.2 }, svg);
    svgEl("line", { x1: cx - boxWidth / 3, x2: cx + boxWidth / 3, y1: high, y2: high, stroke: brand, "stroke-width": 1.2 }, svg);

    const rect = svgEl("rect", {
      x: cx - boxWidth / 2,
      y: Math.min(q1, q3),
      width: boxWidth,
      height: Math.max(Math.abs(q3 - q1), 2),
      rx: 3,
      fill: brand,
      opacity: 0.16,
      stroke: brand,
      class: "mark",
    }, svg);
    tooltipFor(rect, String(group.label), [
      `median ${fmtNum(group.median)}`,
      `Q1 ${fmtNum(group.q1)} · Q3 ${fmtNum(group.q3)}`,
      `range ${fmtNum(group.whisker_low ?? group.min)} – ${fmtNum(group.whisker_high ?? group.max)}`,
      `${fmtInt(group.outliers)} outliers · ${fmtInt(group.count)} rows`,
    ]);

    svgEl("line", { x1: cx - boxWidth / 2, x2: cx + boxWidth / 2, y1: median, y2: median, stroke: accent, "stroke-width": 2.4 }, svg);
    label(svg, cx, plot.y + plot.h + 16, truncate(group.label, 13), { anchor: "middle" });
  });

  label(svg, plot.x + plot.w / 2, plot.y + plot.h + 38, spec.x_label, { anchor: "middle" });

  return svg;
}

const RENDERERS = {
  histogram: renderHistogram,
  bar: renderBar,
  missing: renderMissing,
  heatmap: renderHeatmap,
  line: renderLine,
  scatter: renderScatter,
  box: renderBox,
};

/* -------------------------------- public ------------------------------- */

function measureWidth(container) {
  const parentWidth = container.parentElement?.clientWidth || 0;
  const raw = container.clientWidth || parentWidth || window.innerWidth - 48 || DEFAULT_WIDTH;
  return Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, Math.round(raw)));
}

function draw(spec, container, width) {
  container.replaceChildren();
  container.classList.add("chart-host");
  container._renderedWidth = width;
  WIDTH = width;

  let svg = null;
  try {
    svg = RENDERERS[spec.type]?.(spec);
  } catch (error) {
    console.error("Chart render failed", spec.type, error);
  }

  if (!svg) {
    container.innerHTML = `<div class="empty" style="padding:28px 10px">
      <div class="empty__text">This chart could not be drawn from the available values.</div>
    </div>`;
    container._chartSpec = null;
    return null;
  }

  container.appendChild(svg);
  container._chartSpec = spec;
  return svg;
}

let observer = null;

function observe(container) {
  if (typeof ResizeObserver === "undefined") return;
  if (!observer) {
    observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const host = entry.target;
        const spec = host._chartSpec;
        if (!spec) continue;
        const width = measureWidth(host);
        if (Math.abs(width - (host._renderedWidth || 0)) < 12) continue;
        draw(spec, host, width);
      }
    });
  }
  if (!container._observed) {
    observer.observe(container);
    container._observed = true;
  }
}

export function renderChart(spec, container) {
  if (!spec || !container) return null;
  const svg = draw(spec, container, measureWidth(container));
  observe(container);
  return svg;
}

export function chartCard(spec, { wide = false } = {}) {
  const card = document.createElement("article");
  card.className = `chart-card ${wide ? "chart-card--wide" : ""}`;

  const meta = spec.meta || {};
  const chips = [];
  if (meta.sampled) chips.push('<span class="badge badge--warn">sampled</span>');
  if (meta.rows) chips.push(`<span class="badge badge--neutral">${fmtInt(meta.rows)} rows</span>`);
  if (meta.missing) chips.push(`<span class="badge badge--neutral">${fmtInt(meta.missing)} missing</span>`);

  card.innerHTML = `
    <div class="chart-card__head">
      <div class="chart-card__title">${escapeHtml(spec.title)}</div>
      <span class="chart-card__type">${escapeHtml(TYPE_LABELS[spec.type] || spec.type)}</span>
    </div>
    <div class="chart-card__body"></div>
    <div class="chart-card__foot">
      <p class="chart-card__insight">${escapeHtml(spec.insight || "")}</p>
      <div class="chart-card__meta">${chips.join("")}</div>
    </div>`;

  renderChart(spec, card.querySelector(".chart-card__body"));
  return card;
}

export function chartPlaceholder(title, text = "") {
  return `<div class="empty">
    <div class="empty__icon">${icon("explore")}</div>
    <div class="empty__title">${escapeHtml(title)}</div>
    <p class="empty__text">${escapeHtml(text)}</p>
  </div>`;
}

/* Re-render on theme change so colours follow the CSS variables. */
document.addEventListener("aiviz:theme", () => {
  hideTip();
  document.querySelectorAll(".chart-host").forEach((container) => {
    if (container._chartSpec) renderChart(container._chartSpec, container);
  });
});

export { TYPE_LABELS };
