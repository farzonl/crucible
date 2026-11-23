"use strict";

// Binary partitioning preserves area proportional to selected test count.
function layoutTiles(nodes, x, y, width, height) {
  if (!nodes.length) return [];
  if (nodes.length === 1) return [{ node: nodes[0], x, y, width, height }];
  const total = nodes.reduce((sum, node) => sum + node.total, 0);
  let split = 1, sum = nodes[0].total;
  while (split < nodes.length - 1 &&
         Math.abs(total / 2 - sum - nodes[split].total) < Math.abs(total / 2 - sum)) {
    sum += nodes[split++].total;
  }
  const fraction = sum / total;
  if (width >= height) {
    return [
      ...layoutTiles(nodes.slice(0, split), x, y, width * fraction, height),
      ...layoutTiles(nodes.slice(split), x + width * fraction, y, width * (1 - fraction), height),
    ];
  }
  return [
    ...layoutTiles(nodes.slice(0, split), x, y, width, height * fraction),
    ...layoutTiles(nodes.slice(split), x, y + height * fraction, width, height * (1 - fraction)),
  ];
}

if (typeof module !== "undefined") module.exports = { layoutTiles };
if (typeof document !== "undefined") initializeCTS();

async function initializeCTS() {
  const byId = id => document.getElementById(id);
  const failureCodes = new Set(["Fail", "InternalError", "ResourceError", "Crash", "Timeout", "CompilerError", "MetadataError"]);
  let latest, history, current, selected = "", sortField = "name", ascending = true;
  const paths = new Map();
  function theme(value) {
    document.documentElement.dataset.theme = value;
    byId("theme").textContent = value === "dark" ? "Light theme" : "Dark theme";
  }
  theme(matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  try { const saved = localStorage.getItem("crucible-theme"); if (["light", "dark"].includes(saved)) theme(saved); }
  catch { /* Theme persistence is optional when browser storage is unavailable. */ }
  byId("theme").onclick = () => {
    const value = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    theme(value);
    try { localStorage.setItem("crucible-theme", value); }
    catch { /* Rendering does not depend on storage permissions. */ }
  };
  function textElement(tag, text) {
    const element = document.createElement(tag);
    element.textContent = text;
    return element;
  }
  function safeLink(label, value) {
    const link = textElement("a", label);
    try {
      const url = new URL(value);
      if (url.protocol === "https:" && url.hostname === "github.com") link.href = url.href;
    } catch { /* Invalid metadata remains visible text, never an executable URL. */ }
    return link;
  }
  try {
    const responses = await Promise.all(["latest", "history"].map(name => fetch(`data/vulkan-cts/${name}.json`)));
    if (responses.some(response => response.status === 404)) {
      byId("load-status").textContent = "No CTS baseline has been published yet. The first macOS CTS workflow run will populate this dashboard.";
      return;
    }
    for (const response of responses) if (!response.ok) throw new Error(`HTTP ${response.status}`);
    [latest, history] = await Promise.all(responses.map(response => response.json()));
    if (latest.measurementMode !== "crucible" || !latest.tree ||
        !Array.isArray(latest.tests) || !Array.isArray(history) || latest.total < 1) {
      throw new Error("Invalid CTS data contract");
    }
    const primary = latest;
    function index(node) { paths.set(node.path, node); (node.children || []).forEach(index); }
    index(latest.tree);
    byId("load-status").textContent = latest.timedOut ? "Time budget exceeded: this run is incomplete; missing results are marked NotRun." : "";
    byId("report").hidden = false;
    function renderStats() {
      byId("stats").replaceChildren();
      for (const [label, value] of [
      [latest === primary ? "Crucible pass rate" : "Native reference rate", `${latest.passRate}%`], ["Selected", latest.total],
      ["Passed", latest.passed], ["Failed / errors", latest.failed],
      ["Unsupported", latest.unsupported], ["Warnings", latest.warnings], ["Unrun", latest.notRun],
      ["Native reference", primary.reference ? `${primary.reference.passRate}%` : "Not available"],
    ]) {
      const card = document.createElement("div");
      card.append(textElement("strong", String(value)), textElement("span", label));
      byId("stats").append(card);
      }
    }
    renderStats();
    const metadata = byId("metadata");
    metadata.append(
      safeLink(`Commit ${latest.shortSha}`, `https://github.com/farzonl/crucible/commit/${latest.sha}`),
      document.createTextNode(" | "), safeLink("Workflow run", latest.runUrl),
      textElement("div", `${latest.timestamp} | ${latest.platform} | ${latest.runtimeVersion}`),
      textElement("div", `Baseline: ${latest.baseline}`),
      safeLink(`CTS ${latest.ctsRevision}`, `https://github.com/KhronosGroup/VK-GL-CTS/commit/${latest.ctsRevision}`),
      textElement("div", `Caselist SHA-256: ${latest.caselistSha256}`),
      textElement("div", latest.measurement),
      textElement("div", `Exit: ${latest.exitCode ?? "timeout"} | Budget: ${latest.timeoutSeconds}s`),
    );
    const invocation = document.createElement("details");
    invocation.append(textElement("summary", "Exact invocation"), textElement("code", latest.invocation));
    metadata.append(invocation);
    for (const code of Object.keys(latest.counts)) {
      const option = textElement("option", `${code} (${latest.counts[code]})`);
      option.value = code;
      byId("status-filter").append(option);
    }
    function describe(node) {
      if (node.code) return `${node.path}: ${node.code}${node.referenceCode ? ` (native ${node.referenceCode})` : ""}. ${node.details}`;
      return `${node.path}: ${node.passRate}% passed (${node.passed}/${node.total}), ${node.failed} failures, ${node.unsupported} unsupported, ${node.notRun} unrun`;
    }
    function selectTest(node) {
      selected = node.path;
      byId("search").value = node.path;
      byId("status-filter").value = "";
      renderTests();
      byId("selection").textContent = describe(node);
      byId("tests-heading").scrollIntoView({ block: "start", behavior: "auto" });
    }
    function navigate(node, focus = false) {
      if (node.code) { selectTest(node); return; }
      current = node;
      selected = "";
      byId("search").value = "";
      byId("breadcrumbs").replaceChildren();
      const parts = node.path.split(".");
      parts.forEach((part, index) => {
        const path = parts.slice(0, index + 1).join(".");
        const button = textElement("button", part);
        button.type = "button";
        button.disabled = path === current.path;
        button.onclick = () => navigate(paths.get(path), true);
        byId("breadcrumbs").append(button);
      });
      byId("selection").textContent = describe(node);
      renderMap();
      renderTests();
      if (focus) byId("treemap").querySelector("button")?.focus({ preventScroll: true });
    }
    function renderMap() {
      const map = byId("treemap");
      map.replaceChildren();
      const nodes = [...(current.children || [])].sort((a, b) => b.total - a.total || a.path.localeCompare(b.path));
      for (const tile of layoutTiles(nodes, 0, 0, map.clientWidth, map.clientHeight)) {
        const node = tile.node;
        const button = document.createElement("button");
        button.type = "button"; button.className = "tile"; button.dataset.path = node.path;
        Object.assign(button.style, {
          left: `${tile.x}px`, top: `${tile.y}px`, width: `${tile.width}px`, height: `${tile.height}px`,
          background: node.unsupported + node.notRun === node.total ? "#475569" : color(node.passRate),
        });
        button.title = describe(node);
        button.setAttribute("aria-label", describe(node));
        button.append(textElement("span", node.code ? node.name.split(".").at(-1) : node.name),
                      textElement("small", node.code || `${node.passRate}% | ${node.total} tests`));
        button.onclick = () => navigate(node, true);
        button.onmouseenter = () => { byId("selection").textContent = describe(node); };
        button.onfocus = () => { byId("selection").textContent = describe(node); };
        map.append(button);
      }
    }
    function color(rate) {
      const start = rate < 50 ? [153, 27, 27] : [133, 77, 14];
      const end = rate < 50 ? [133, 77, 14] : [22, 101, 52];
      const fraction = rate < 50 ? rate / 50 : (rate - 50) / 50;
      return `rgb(${start.map((value, index) => Math.round(value + (end[index] - value) * fraction)).join(",")})`;
    }
    function renderTests() {
      const query = byId("search").value.toLowerCase();
      const status = byId("status-filter").value;
      const tests = latest.tests.filter(test =>
        test.name.startsWith(`${current.path}.`) &&
        `${test.name} ${test.details}`.toLowerCase().includes(query) &&
        (!status || (status === "failures" ? failureCodes.has(test.code) : test.code === status)));
      tests.sort((a, b) => (ascending ? 1 : -1) * a[sortField].localeCompare(b[sortField]));
      byId("test-count").textContent = `${tests.length} tests in ${current.path}`;
      byId("test-rows").replaceChildren();
      for (const test of tests) {
        const row = document.createElement("tr");
        row.dataset.selected = String(selected === test.name);
        const cell = document.createElement("td");
        const button = textElement("button", test.name);
        button.type = "button";
        button.onclick = () => { selected = test.name; byId("selection").textContent = `${test.name}: ${test.code}. ${test.details}`; renderTests(); };
        cell.append(button);
        const compiler = test.compiler ?
          ` Crucible shaders: ${test.compiler.compiled || 0}/${test.compiler.attempt || 0}.` : "";
        row.append(cell, textElement("td", test.code),
                   textElement("td", test.referenceCode || test.code),
                   textElement("td", test.details + compiler));
        byId("test-rows").append(row);
      }
    }
    byId("search").oninput = renderTests;
    byId("status-filter").onchange = renderTests;
    byId("all-tests").onclick = () => navigate(latest.tree, true);
    for (const [id, field] of [["sort-name", "name"], ["sort-status", "code"]]) {
      byId(id).onclick = () => {
        ascending = field === sortField ? !ascending : true;
        sortField = field;
        byId("sort-name").textContent = "Test path";
        byId("sort-status").textContent = "Status";
        byId(id).textContent += ascending ? " (ascending)" : " (descending)";
        renderTests();
      };
    }
    let lastZoom = 0;
    byId("treemap").addEventListener("wheel", event => {
      if (event.ctrlKey || event.deltaY === 0) return;
      event.preventDefault();
      if (Date.now() - lastZoom < 400) return;
      lastZoom = Date.now();
      if (event.deltaY < 0) {
        const parent = paths.get(current.path.split(".").slice(0, -1).join("."));
        if (parent) navigate(parent);
      } else {
        const path = event.target.closest("button")?.dataset.path;
        if (path) navigate(paths.get(path));
      }
    }, { passive: false });
    new ResizeObserver(renderMap).observe(byId("treemap"));
    let initial = latest.tree;
    while (initial.children?.length === 1 && !initial.children[0].code) {
      initial = initial.children[0];
    }
    navigate(initial);
    byId("measurement").disabled = !primary.reference;
    byId("measurement").onchange = () => {
      latest = byId("measurement").value === "reference" ?
        { ...primary, ...primary.reference } : primary;
      paths.clear();
      index(latest.tree);
      byId("status-filter").replaceChildren(textElement("option", "All statuses"));
      byId("status-filter").firstChild.value = "";
      const failures = textElement("option", "Failures only");
      failures.value = "failures";
      byId("status-filter").append(failures);
      for (const [code, count] of Object.entries(latest.counts)) {
        const option = textElement("option", `${code} (${count})`);
        option.value = code;
        byId("status-filter").append(option);
      }
      renderStats();
      navigate(paths.get(current.path) || latest.tree);
    };
    renderHistory();
    function renderHistory() {
      const sameBaseline = history.filter(entry => entry.measurementMode === "crucible" &&
        entry.caselistSha256 === primary.caselistSha256 && entry.adapterVersion === primary.adapterVersion &&
        entry.ctsRevision === primary.ctsRevision && entry.runtimeVersion === primary.runtimeVersion);
      const svg = byId("trend"), ns = "http://www.w3.org/2000/svg";
      function svgElement(tag, attributes, text = "") {
        const element = document.createElementNS(ns, tag);
        for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
        element.textContent = text;
        return element;
      }
      for (const rate of [0, 50, 100]) {
        const y = 130 - rate;
        svg.append(svgElement("line", { x1: 45, x2: 975, y1: y, y2: y, stroke: "var(--border)" }),
                   svgElement("text", { x: 0, y: y + 4 }, `${rate}%`));
      }
      const points = sameBaseline.map((entry, index) => ({
        entry, x: sameBaseline.length === 1 ? 510 : 50 + 920 * index / (sameBaseline.length - 1),
        y: 130 - entry.passRate,
      }));
      svg.append(svgElement("polyline", { points: points.map(p => `${p.x},${p.y}`).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
      for (const point of points) {
        const circle = svgElement("circle", { cx: point.x, cy: point.y, r: 5, fill: "var(--accent)" });
        circle.append(svgElement("title", {}, `${point.entry.shortSha}: ${point.entry.passRate}%`));
        svg.append(circle);
      }
      for (const entry of [...history].reverse()) {
        const row = document.createElement("tr"), cell = document.createElement("td");
        cell.append(safeLink(entry.shortSha, `https://github.com/farzonl/crucible/commit/${entry.sha}`),
                    document.createTextNode(" / "), safeLink("run", entry.runUrl));
        row.append(cell, textElement("td", entry.timestamp), textElement("td", `${entry.passRate}%`),
                   textElement("td", entry.reference ? `${entry.reference.passRate}%` : "Not available"),
                   textElement("td", String(entry.total)), textElement("td", `${entry.baseline} | ${entry.runtimeVersion}`));
        byId("history-rows").append(row);
      }
    }
  } catch (error) {
    byId("report").hidden = true;
    byId("load-status").textContent = `Unable to load CTS results: ${error.message}`;
  }
}
