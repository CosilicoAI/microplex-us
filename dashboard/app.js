"use strict";

const DEFAULT_DATA_URL = "../artifacts/pe_native_target_diagnostics_current.json";
const TABLE_LIMIT = 500;

const state = {
  data: null,
  search: "",
  family: "all",
  scope: "all",
  winner: "all",
  dbMatch: "all",
  sort: "weighted_term_delta:asc",
};

const el = {
  dashboard: document.getElementById("dashboard"),
  emptyState: document.getElementById("emptyState"),
  fileInput: document.getElementById("fileInput"),
  loadStatus: document.getElementById("loadStatus"),
  kpiTargets: document.getElementById("kpiTargets"),
  kpiToWinLabel: document.getElementById("kpiToWinLabel"),
  kpiWinRate: document.getElementById("kpiWinRate"),
  kpiLossDelta: document.getElementById("kpiLossDelta"),
  kpiLossPair: document.getElementById("kpiLossPair"),
  kpiDbMatch: document.getElementById("kpiDbMatch"),
  kpiDbDetail: document.getElementById("kpiDbDetail"),
  scopeSummary: document.getElementById("scopeSummary"),
  familySummary: document.getElementById("familySummary"),
  topImprovements: document.getElementById("topImprovements"),
  topRegressions: document.getElementById("topRegressions"),
  tableCount: document.getElementById("tableCount"),
  searchInput: document.getElementById("searchInput"),
  familyFilter: document.getElementById("familyFilter"),
  scopeFilter: document.getElementById("scopeFilter"),
  winnerFilter: document.getElementById("winnerFilter"),
  dbFilter: document.getElementById("dbFilter"),
  sortSelect: document.getElementById("sortSelect"),
  targetTable: document.getElementById("targetTable"),
};

function labels() {
  const datasetLabels = state.data?.dataset_labels || {};
  return {
    from: datasetLabels.from || "baseline",
    to: datasetLabels.to || "candidate",
  };
}

function numberOrNull(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatNumber(value, options = {}) {
  const numeric = numberOrNull(value);
  if (numeric === null) {
    return "-";
  }
  const abs = Math.abs(numeric);
  if (abs >= 1_000_000 || (abs > 0 && abs < 0.001)) {
    return numeric.toExponential(2);
  }
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: options.maximumFractionDigits ?? 3,
    minimumFractionDigits: options.minimumFractionDigits ?? 0,
  }).format(numeric);
}

function formatCompact(value) {
  const numeric = numberOrNull(value);
  if (numeric === null) {
    return "-";
  }
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(numeric);
}

function formatPercent(value) {
  const numeric = numberOrNull(value);
  if (numeric === null) {
    return "-";
  }
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(numeric);
}

function formatSigned(value) {
  const numeric = numberOrNull(value);
  if (numeric === null) {
    return "-";
  }
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${formatNumber(numeric, { maximumFractionDigits: 4 })}`;
}

function formatError(value) {
  const numeric = numberOrNull(value);
  if (numeric === null) {
    return "-";
  }
  return `${formatNumber(numeric, { maximumFractionDigits: 2 })}%`;
}

function classForDelta(value) {
  const numeric = numberOrNull(value) || 0;
  if (numeric < 0) {
    return "good";
  }
  if (numeric > 0) {
    return "bad";
  }
  return "";
}

function winnerLabel(winner) {
  const currentLabels = labels();
  if (winner === "to") {
    return currentLabels.to;
  }
  if (winner === "from") {
    return currentLabels.from;
  }
  return "tie";
}

function dbMatchLabel(row) {
  const status = row.policyengine_target_match || "unparsed";
  if (status === "matched") {
    return row.policyengine_target_id ? `#${row.policyengine_target_id}` : "matched";
  }
  if (status === "legacy_only") {
    return "legacy only";
  }
  if (status === "db_unavailable") {
    return "db unavailable";
  }
  return status.replaceAll("_", " ");
}

function summarizeRows(rows) {
  const nTargets = rows.length;
  const fromWins = rows.filter((row) => row.winner === "from").length;
  const toWins = rows.filter((row) => row.winner === "to").length;
  const ties = nTargets - fromWins - toWins;
  const fromLoss = mean(rows.map((row) => row.from_weighted_term));
  const toLoss = mean(rows.map((row) => row.to_weighted_term));
  return {
    n_targets: nTargets,
    from_wins: fromWins,
    to_wins: toWins,
    ties,
    from_win_rate: nTargets ? fromWins / nTargets : null,
    to_win_rate: nTargets ? toWins / nTargets : null,
    from_loss: fromLoss,
    to_loss: toLoss,
    loss_delta: toLoss - fromLoss,
    mean_weighted_term_delta: mean(rows.map((row) => row.weighted_term_delta)),
  };
}

function mean(values) {
  const numbers = values.map(Number).filter(Number.isFinite);
  if (!numbers.length) {
    return null;
  }
  return numbers.reduce((sum, value) => sum + value, 0) / numbers.length;
}

function groupSummary(rows, field) {
  const grouped = new Map();
  for (const row of rows) {
    const key = row[field] || "other";
    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key).push(row);
  }
  return Array.from(grouped.entries()).map(([key, groupRows]) => ({
    [field]: key,
    ...summarizeRows(groupRows),
  }));
}

function normalizePayload(payload) {
  const rows = Array.isArray(payload.targets) ? payload.targets : [];
  return {
    ...payload,
    summary: payload.summary || summarizeRows(rows),
    family_summaries: Array.isArray(payload.family_summaries)
      ? payload.family_summaries
      : groupSummary(rows, "target_family"),
    scope_summaries: Array.isArray(payload.scope_summaries)
      ? payload.scope_summaries
      : groupSummary(rows, "target_scope"),
    top_improvements: Array.isArray(payload.top_improvements)
      ? payload.top_improvements
      : [...rows]
          .sort((a, b) => Number(a.weighted_term_delta) - Number(b.weighted_term_delta))
          .slice(0, 25),
    top_regressions: Array.isArray(payload.top_regressions)
      ? payload.top_regressions
      : [...rows]
          .sort((a, b) => Number(b.weighted_term_delta) - Number(a.weighted_term_delta))
          .slice(0, 25),
  };
}

function setData(payload, sourceLabel) {
  state.data = normalizePayload(payload);
  el.dashboard.hidden = false;
  el.emptyState.hidden = true;
  el.loadStatus.textContent = sourceLabel;
  populateFilters();
  render();
}

function showEmpty(message) {
  state.data = null;
  el.dashboard.hidden = true;
  el.emptyState.hidden = false;
  el.loadStatus.textContent = message;
}

async function loadDefault() {
  try {
    const response = await fetch(`${DEFAULT_DATA_URL}?v=${Date.now()}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    setData(await response.json(), "Default artifact loaded");
  } catch (_error) {
    showEmpty("Default artifact unavailable");
  }
}

function loadFile(file) {
  const reader = new FileReader();
  reader.addEventListener("load", () => {
    try {
      setData(JSON.parse(String(reader.result)), file.name);
    } catch (error) {
      showEmpty(`Invalid JSON: ${error.message}`);
    }
  });
  reader.readAsText(file);
}

function populateSelect(select, label, values) {
  const current = select.value || "all";
  select.replaceChildren();
  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = label;
  select.append(allOption);
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
  select.value = values.includes(current) ? current : "all";
}

function populateFilters() {
  const rows = state.data.targets || [];
  const families = [...new Set(rows.map((row) => row.target_family || "other"))].sort();
  const scopes = [...new Set(rows.map((row) => row.target_scope || "other"))].sort();
  const dbStatuses = [...new Set(rows.map((row) => row.policyengine_target_match || "unparsed"))].sort();
  populateSelect(el.familyFilter, "All families", families);
  populateSelect(el.scopeFilter, "All scopes", scopes);
  populateSelect(el.dbFilter, "All DB statuses", dbStatuses);

  const currentLabels = labels();
  el.winnerFilter.replaceChildren();
  for (const [value, label] of [
    ["all", "All winners"],
    ["to", currentLabels.to],
    ["from", currentLabels.from],
    ["tie", "Ties"],
  ]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    el.winnerFilter.append(option);
  }
}

function filteredRows() {
  const query = state.search.trim().toLowerCase();
  const rows = state.data?.targets || [];
  return rows
    .filter((row) => {
      if (state.family !== "all" && row.target_family !== state.family) {
        return false;
      }
      if (state.scope !== "all" && row.target_scope !== state.scope) {
        return false;
      }
      if (state.winner !== "all" && row.winner !== state.winner) {
        return false;
      }
      if (
        state.dbMatch !== "all" &&
        (row.policyengine_target_match || "unparsed") !== state.dbMatch
      ) {
        return false;
      }
      if (!query) {
        return true;
      }
      return [
        row.target_name,
        row.target_family,
        row.target_scope,
        row.policyengine_target_match,
        row.policyengine_target_id,
        row.policyengine_target_source,
        row.policyengine_target_domain_variable,
      ]
        .join(" ")
        .toLowerCase()
        .includes(query);
    })
    .sort((a, b) => {
      const [field, direction] = state.sort.split(":");
      const av = Number(a[field]);
      const bv = Number(b[field]);
      const result = Number.isFinite(av) && Number.isFinite(bv)
        ? av - bv
        : String(a[field] || "").localeCompare(String(b[field] || ""));
      return direction === "desc" ? -result : result;
    });
}

function render() {
  if (!state.data) {
    return;
  }
  renderKpis();
  renderSummaries();
  renderTargetList(el.topImprovements, state.data.top_improvements || [], true);
  renderTargetList(el.topRegressions, state.data.top_regressions || [], false);
  renderTable(filteredRows());
}

function renderKpis() {
  const currentLabels = labels();
  const summary = state.data.summary || {};
  el.kpiTargets.textContent = formatNumber(summary.n_targets);
  el.kpiToWinLabel.textContent = `${currentLabels.to} Wins`;
  el.kpiWinRate.textContent = formatPercent(summary.to_win_rate);
  el.kpiLossDelta.textContent = formatSigned(summary.loss_delta);
  el.kpiLossDelta.className = classForDelta(summary.loss_delta);
  el.kpiLossPair.textContent = `${formatNumber(summary.from_loss)} -> ${formatNumber(summary.to_loss)}`;
  const dbSummary = state.data.target_db_summary || {};
  el.kpiDbMatch.textContent = dbSummary.match_rate === null || dbSummary.match_rate === undefined
    ? formatNumber(dbSummary.matched)
    : formatPercent(dbSummary.match_rate);
  el.kpiDbDetail.textContent = `${formatNumber(dbSummary.matched)} matched / ${formatNumber(dbSummary.legacy_only)} legacy`;
}

function renderSummaries() {
  const familyRows = [...(state.data.family_summaries || [])].sort(
    (a, b) => Number(a.loss_delta) - Number(b.loss_delta),
  );
  const scopeRows = [...(state.data.scope_summaries || [])].sort(
    (a, b) => String(a.target_scope).localeCompare(String(b.target_scope)),
  );
  renderSummaryList(el.scopeSummary, scopeRows, "target_scope");
  renderSummaryList(el.familySummary, familyRows, "target_family");
}

function renderSummaryList(container, rows, field) {
  container.replaceChildren();
  for (const row of rows) {
    const wrapper = document.createElement("div");
    wrapper.className = "summary-row";

    const left = document.createElement("div");
    const name = document.createElement("div");
    name.className = "summary-name";
    name.textContent = row[field] || "other";
    const meta = document.createElement("div");
    meta.className = "summary-meta";
    meta.textContent = `${formatNumber(row.n_targets)} targets - ${formatPercent(row.to_win_rate)} wins`;
    left.append(name, meta);

    const value = document.createElement("div");
    value.className = `summary-value ${classForDelta(row.loss_delta)}`;
    value.textContent = formatSigned(row.loss_delta);
    wrapper.append(left, value);
    container.append(wrapper);
  }
}

function renderTargetList(container, rows, improvementList) {
  container.replaceChildren();
  const displayRows = rows.slice(0, 12);
  for (const row of displayRows) {
    const wrapper = document.createElement("div");
    wrapper.className = "target-row";
    wrapper.title = row.target_name || "";

    const left = document.createElement("div");
    const name = document.createElement("div");
    name.className = "target-name";
    name.textContent = row.target_name || "-";
    const meta = document.createElement("div");
    meta.className = "target-meta";
    meta.textContent = `${row.target_family || "other"} - ${winnerLabel(row.winner)} - ${dbMatchLabel(row)}`;
    left.append(name, meta);

    const delta = document.createElement("div");
    delta.className = `delta ${classForDelta(row.weighted_term_delta)}`;
    delta.textContent = formatSigned(row.weighted_term_delta);
    if (improvementList && Number(row.weighted_term_delta) > 0) {
      delta.classList.add("bad");
    }
    wrapper.append(left, delta);
    container.append(wrapper);
  }
}

function renderTable(rows) {
  el.targetTable.replaceChildren();
  const visibleRows = rows.slice(0, TABLE_LIMIT);
  el.tableCount.textContent = rows.length > TABLE_LIMIT
    ? `${formatNumber(TABLE_LIMIT)} of ${formatNumber(rows.length)} rows`
    : `${formatNumber(rows.length)} rows`;

  const fragment = document.createDocumentFragment();
  for (const row of visibleRows) {
    const tr = document.createElement("tr");
    tr.title = row.target_name || "";
    appendCell(tr, row.target_name || "-");
    appendCell(tr, row.target_family || "other");
    appendCell(tr, row.target_scope || "other");
    appendCell(tr, winnerLabel(row.winner), `winner ${row.winner || "tie"}`);
    appendCell(tr, formatSigned(row.weighted_term_delta), `mono ${classForDelta(row.weighted_term_delta)}`);
    appendCell(tr, formatError(row.from_abs_pct_error), "mono");
    appendCell(tr, formatError(row.to_abs_pct_error), "mono");
    appendCell(tr, formatCompact(row.target_value), "mono");
    appendCell(
      tr,
      dbMatchLabel(row),
      `db-status ${row.policyengine_target_match || "unparsed"}`,
    );
    fragment.append(tr);
  }
  el.targetTable.append(fragment);
}

function appendCell(row, text, className = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) {
    cell.className = className;
  }
  row.append(cell);
}

el.fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files || [];
  if (file) {
    loadFile(file);
  }
});

el.searchInput.addEventListener("input", (event) => {
  state.search = event.target.value;
  render();
});

el.familyFilter.addEventListener("change", (event) => {
  state.family = event.target.value;
  render();
});

el.scopeFilter.addEventListener("change", (event) => {
  state.scope = event.target.value;
  render();
});

el.winnerFilter.addEventListener("change", (event) => {
  state.winner = event.target.value;
  render();
});

el.dbFilter.addEventListener("change", (event) => {
  state.dbMatch = event.target.value;
  render();
});

el.sortSelect.addEventListener("change", (event) => {
  state.sort = event.target.value;
  render();
});

loadDefault();
