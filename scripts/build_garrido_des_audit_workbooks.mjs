#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const HEADER_FILL = "#1F4E78";
const SUBHEADER_FILL = "#D9EAF7";
const OK_FILL = "#E2F0D9";
const WARN_FILL = "#FCE4D6";

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const part = argv[i];
    if (part.startsWith("--")) {
      args[part.slice(2)] = argv[i + 1];
      i += 1;
    }
  }
  if (!args["data-json"] || !args["output-dir"]) {
    throw new Error("Usage: build_garrido_des_audit_workbooks.mjs --data-json PATH --output-dir DIR");
  }
  return args;
}

function asCell(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value !== "string") return String(value);
  const trimmed = value.trim();
  if (trimmed === "") return null;
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(trimmed)) {
    const n = Number(trimmed);
    if (Number.isFinite(n)) return n;
  }
  return value;
}

function asTextCell(value) {
  if (value === undefined || value === null) return null;
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  if (/^\d{4}-\d{2}-\d{2}T/.test(text)) return `'${text}`;
  return text.startsWith("=") ? `'${text}` : text;
}

function colName(indexZeroBased) {
  let n = indexZeroBased + 1;
  let s = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    s = String.fromCharCode(65 + rem) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function rangeAddress(startRow, startCol, rowCount, colCount) {
  const a = `${colName(startCol)}${startRow + 1}`;
  const b = `${colName(startCol + colCount - 1)}${startRow + rowCount}`;
  return `${a}:${b}`;
}

function unionHeaders(rows, preferred = []) {
  const seen = new Set(preferred);
  const headers = [...preferred];
  for (const row of rows || []) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        seen.add(key);
        headers.push(key);
      }
    }
  }
  return headers;
}

function applyHeaderStyle(range) {
  try {
    range.format.fill = { color: HEADER_FILL };
    range.format.font = { color: "#FFFFFF", bold: true };
    range.format.borders = { preset: "outside", style: "thin", color: "#9EADBA" };
  } catch {
    // Formatting support can vary by artifact-tool build; values remain intact.
  }
}

function applySubheaderStyle(range) {
  try {
    range.format.fill = { color: SUBHEADER_FILL };
    range.format.font = { bold: true };
  } catch {
    // Best-effort formatting.
  }
}

function finishSheet(sheet, usedRows, usedCols) {
  sheet.showGridLines = false;
  try {
    sheet.freezePanes.freezeRows(1);
  } catch {
    // Best-effort usability.
  }
  try {
    sheet.getRange(rangeAddress(0, 0, Math.max(1, usedRows), Math.max(1, usedCols))).format.autofitColumns();
  } catch {
    // Best-effort layout.
  }
}

function writeRows(sheet, startRow, startCol, rows, preferredHeaders = []) {
  const headers = unionHeaders(rows, preferredHeaders);
  const matrix = [headers, ...(rows || []).map((row) => headers.map((h) => asCell(row[h])))];
  if (matrix.length === 0 || headers.length === 0) return { rows: 0, cols: 0 };
  const address = rangeAddress(startRow, startCol, matrix.length, headers.length);
  sheet.getRange(address).values = matrix;
  applyHeaderStyle(sheet.getRange(rangeAddress(startRow, startCol, 1, headers.length)));
  try {
    sheet.tables.add(address, true, `${sheet.name.replace(/[^A-Za-z0-9]/g, "")}Table`);
  } catch {
    // Tables are nice, not required.
  }
  return { rows: matrix.length, cols: headers.length };
}

function writeKeyValues(sheet, startRow, startCol, obj) {
  const rows = Object.entries(obj || {}).map(([key, value]) => [
    key,
    asTextCell(value),
  ]);
  const address = rangeAddress(startRow, startCol, Math.max(1, rows.length + 1), 2);
  sheet.getRange(address).values = [["Field", "Value"], ...rows];
  applyHeaderStyle(sheet.getRange(rangeAddress(startRow, startCol, 1, 2)));
  return { rows: rows.length + 1, cols: 2 };
}

function manifestOverview(data) {
  const statusLines = String(data.manifest.git?.status_short || "")
    .split("\n")
    .filter(Boolean);
  return {
    generated_at_utc: data.manifest.generated_at_utc,
    replication_status: data.replication.replication_status,
    formula_rows_checked: data.replication.formula_audit?.total_rows,
    formula_mismatches: data.replication.formula_audit?.total_mismatches,
    mean_abs_ret_gap: data.replication.best_summary?.mean_abs_ret_gap,
    max_abs_ret_gap: data.replication.best_summary?.max_abs_ret_gap,
    best_demand_source: data.replication.best_config?.demand_source,
    best_risk_occurrence: data.replication.best_config?.risk_occurrence_mode,
    best_risk_attribution: data.replication.best_config?.risk_attribution_source,
    best_seed_stream: data.replication.best_config?.seed_stream_mode,
    git_branch: data.manifest.git?.branch,
    git_commit: data.manifest.git?.commit,
    dirty_worktree_entries: statusLines.length,
    replication_dir: data.manifest.replication_dir,
  };
}

function addTitle(sheet, title, subtitle = "") {
  sheet.getRange("A1").values = [[title]];
  try {
    sheet.getRange("A1:H1").merge();
    sheet.getRange("A1").format.font = { bold: true, size: 16, color: "#1F4E78" };
  } catch {
    // Best-effort.
  }
  if (subtitle) {
    sheet.getRange("A2").values = [[subtitle]];
    try {
      sheet.getRange("A2:H2").merge();
      sheet.getRange("A2").format.font = { italic: true, color: "#555555" };
    } catch {
      // Best-effort.
    }
  }
}

function buildSummaryWorkbook(data) {
  const workbook = Workbook.create();
  const readme = workbook.worksheets.add("README");
  addTitle(readme, "Garrido DES-vs-Excel Audit", "Formula-faithful replication package");
  readme.getRange("A4").values = [["Purpose"]];
  applySubheaderStyle(readme.getRange("A4"));
  readme.getRange("A5").values = [[
    "Compare the Python DES order ledger against Garrido's original Excel workbooks using the same order-level ReT formula and CF layout.",
  ]];
  readme.getRange("A7").values = [["Excel ReT formula"]];
  applySubheaderStyle(readme.getRange("A7"));
  readme.getRange("A8").values = [[data.replication.formula || ""]];
  writeKeyValues(readme, 10, 0, manifestOverview(data));
  try {
    readme.getRange("A:A").format.columnWidth = 28;
    readme.getRange("B:B").format.columnWidth = 90;
    readme.getRange("B:B").format.wrapText = true;
  } catch {
    // Best-effort layout.
  }
  finishSheet(readme, 28, 4);

  const formula = workbook.worksheets.add("FormulaGate");
  writeKeyValues(formula, 0, 0, {
    replication_status: data.replication.replication_status,
    ...data.replication.formula_audit,
    gates: data.replication.gates,
    best_config: data.replication.best_config,
    best_summary: data.replication.best_summary,
  });
  finishSheet(formula, 20, 3);

  const cf = workbook.worksheets.add("CF_Summary");
  writeRows(cf, 0, 0, data.cf_summary, [
    "CF",
    "family",
    "target_ret",
    "des_ret",
    "signed_gap",
    "abs_gap",
    "target_orders",
    "des_orders",
    "order_gap",
    "branch_gap_pct",
  ]);
  finishSheet(cf, (data.cf_summary || []).length + 1, 14);

  const risk = workbook.worksheets.add("RiskAttribution");
  writeRows(risk, 0, 0, data.risk_attribution, [
    "CF",
    "family",
    "n_orders",
    "risk_columns",
    "risk_active_share",
    "fill_rate_branch_share",
    "autotomy_branch_share",
    "recovery_branch_share",
    "unfulfilled_branch_share",
    "top_risk_columns",
  ]);
  finishSheet(risk, (data.risk_attribution || []).length + 1, 10);

  const deltas = workbook.worksheets.add("Deltas");
  writeRows(deltas, 0, 0, data.deltas, [
    "CF",
    "family",
    "signed_gap",
    "abs_gap",
    "order_gap",
    "q_max_abs_gap",
    "optj_max_abs_gap",
    "branch_gap_pct",
    "audit_flags",
  ]);
  finishSheet(deltas, (data.deltas || []).length + 1, 12);

  const ledgers = workbook.worksheets.add("SelectedLedgers");
  writeRows(ledgers, 0, 0, data.selected_ledgers || []);
  finishSheet(ledgers, (data.selected_ledgers || []).length + 1, 25);

  return workbook;
}

function buildLedgerWorkbook(data) {
  const workbook = Workbook.create();
  const readme = workbook.worksheets.add("README");
  addTitle(readme, "Garrido-style DES ledgers", "One sheet per CF, generated from des_order_exports");
  writeKeyValues(readme, 4, 0, {
    generated_at_utc: data.manifest.generated_at_utc,
    replication_dir: data.manifest.replication_dir,
    formula: data.replication.formula,
    note: "These are DES exports in Garrido-style order-ledger columns. Original Excel ledgers remain the source target workbooks.",
  });
  try {
    readme.getRange("A:A").format.columnWidth = 28;
    readme.getRange("B:B").format.columnWidth = 110;
    readme.getRange("B:B").format.wrapText = true;
  } catch {
    // Best-effort layout.
  }
  finishSheet(readme, 15, 4);

  const byCf = data.ledger_rows_by_cf || {};
  for (const cf of Object.keys(byCf).sort()) {
    const sheet = workbook.worksheets.add(cf);
    writeRows(sheet, 0, 0, byCf[cf] || []);
    finishSheet(sheet, (byCf[cf] || []).length + 1, 25);
  }
  return workbook;
}

async function saveWorkbook(workbook, outputPath) {
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(outputPath);
}

async function verifyWorkbook(workbook, outputDir, basename) {
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 50 },
    summary: "formula error scan",
  });
  await fs.writeFile(path.join(outputDir, `${basename}_formula_error_scan.ndjson`), errors.ndjson, "utf8");
  const preview = await workbook.render({ sheetName: "README", autoCrop: "all", scale: 1, format: "png" });
  const bytes = new Uint8Array(await preview.arrayBuffer());
  await fs.writeFile(path.join(outputDir, `${basename}_README_preview.png`), bytes);
}

async function main() {
  const args = parseArgs(process.argv);
  const data = JSON.parse(await fs.readFile(args["data-json"], "utf8"));
  const outputDir = args["output-dir"];
  await fs.mkdir(outputDir, { recursive: true });

  const summary = buildSummaryWorkbook(data);
  const ledger = buildLedgerWorkbook(data);
  await verifyWorkbook(summary, outputDir, "garrido_des_audit_summary");
  await verifyWorkbook(ledger, outputDir, "garrido_des_ledgers");
  await saveWorkbook(summary, path.join(outputDir, "garrido_des_audit_summary.xlsx"));
  await saveWorkbook(ledger, path.join(outputDir, "garrido_des_ledgers.xlsx"));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
