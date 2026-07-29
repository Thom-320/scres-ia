#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [sweepDirArg, workbookPath] = process.argv.slice(2);
if (!sweepDirArg || !workbookPath) {
  console.error("usage: build_track_b_sweep_workbook.mjs sweep_dir output.xlsx");
  process.exit(2);
}

const sweepDir = path.resolve(sweepDirArg);

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (ch === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  if (rows.length === 0) return [];
  const headers = rows[0];
  return rows.slice(1).filter((r) => r.some((v) => v !== "")).map((r) => {
    const out = {};
    headers.forEach((h, i) => {
      const raw = r[i] ?? "";
      const num = Number(raw);
      out[h] = raw !== "" && Number.isFinite(num) ? num : raw;
    });
    return out;
  });
}

async function readCsvMaybe(filePath) {
  try {
    return parseCsv(await fs.readFile(filePath, "utf8"));
  } catch {
    return [];
  }
}

function rowsToMatrix(rows) {
  if (!rows || rows.length === 0) return [["(empty)"]];
  const headers = Object.keys(rows[0]);
  return [headers, ...rows.map((row) => headers.map((h) => row[h] ?? null))];
}

function writeTable(sheet, startRow, startCol, rows, color = "#1F4E79") {
  const matrix = rowsToMatrix(rows);
  const range = sheet.getRangeByIndexes(startRow, startCol, matrix.length, matrix[0].length);
  range.values = matrix;
  const header = sheet.getRangeByIndexes(startRow, startCol, 1, matrix[0].length);
  header.format.fill.color = color;
  header.format.font.color = "#FFFFFF";
  header.format.font.bold = true;
  range.format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  range.format.autofitColumns();
  return range;
}

function writeKV(sheet, startRow, startCol, rows) {
  const range = sheet.getRangeByIndexes(startRow, startCol, rows.length, 2);
  range.values = rows;
  sheet.getRangeByIndexes(startRow, startCol, rows.length, 1).format.font.bold = true;
  range.format.borders = { preset: "outside", style: "thin", color: "#D9E2F3" };
  range.format.autofitColumns();
}

function title(sheet, text, row = 0, col = 0, width = 10) {
  const r = sheet.getRangeByIndexes(row, col, 1, width);
  r.values = [[text, ...Array(width - 1).fill(null)]];
  r.merge();
  r.format.fill.color = "#D9EAF7";
  r.format.font.bold = true;
  r.format.font.size = 14;
}

function metricRows(summaryRows) {
  const metrics = [
    "order_ret_excel",
    "order_ret_excel_cvar05",
    "order_ret_excel_p05",
    "order_ret_excel_p50",
    "order_ret_excel_p95",
    "order_ret_excel_rolling_4w_mean",
    "order_ctj_p99",
    "order_rpj_p99",
    "order_dpj_p99",
    "order_service_loss_auc_per_order",
    "flow_fill_rate",
    "ret_garrido2024_sigmoid_mean",
    "assembly_cost_index",
    "op10_multiplier_step_mean",
    "op12_multiplier_step_mean",
  ];
  const rows = [];
  for (const row of summaryRows) {
    for (const metric of metrics) {
      rows.push({
        cell: row.cell,
        reward_mode: row.reward_mode,
        observation_version: row.observation_version,
        metric,
        learned: row[`learned_${metric}`] ?? null,
        best_static: row[`best_static_${metric}`] ?? null,
        delta: row[`delta_${metric}`] ?? null,
        win: row[`win_${metric}`] ?? null,
      });
    }
  }
  return rows;
}

const summaryRows = await readCsvMaybe(path.join(sweepDir, "sweep_summary.csv"));
const promotedRows = await readCsvMaybe(path.join(sweepDir, "promotion_decision.csv"));
let promotionJson = {};
try {
  promotionJson = JSON.parse(await fs.readFile(path.join(sweepDir, "promotion_decision.json"), "utf8"));
} catch {}

const rankedRows = [...summaryRows].sort((a, b) => (
  Number(b.excel_ret_delta_vs_best_static ?? -Infinity)
  - Number(a.excel_ret_delta_vs_best_static ?? -Infinity)
));

const wb = Workbook.create();

const dashboard = wb.worksheets.add("Dashboard");
dashboard.showGridLines = false;
title(dashboard, "Track B Adaptive Sweep Audit", 0, 0, 10);
const best = rankedRows[0] || {};
writeKV(dashboard, 2, 0, [
  ["Generated UTC", new Date().toISOString()],
  ["Sweep directory", sweepDir],
  ["Target metric", "Garrido Excel ReT: order_ret_excel / ledger ReTj"],
  ["Configs found", summaryRows.length],
  ["Promoted configs", (promotionJson.promoted || []).length ?? ""],
  ["Best config by Excel ReT delta", best.cell ?? ""],
  ["Best delta order_ret_excel", best.excel_ret_delta_vs_best_static ?? ""],
  ["Best delta CVaR05", best.cvar05_delta_vs_best_static ?? ""],
  ["Best learned cost index", best.learned_cost_index ?? ""],
]);
writeTable(dashboard, 14, 0, rankedRows.slice(0, 10), "#305496");
dashboard.freezePanes.freezeRows(1);

const ranking = wb.worksheets.add("Ranking");
ranking.showGridLines = false;
title(ranking, "Full Config Ranking", 0, 0, 10);
writeTable(ranking, 2, 0, rankedRows);
ranking.freezePanes.freezeRows(3);

const metrics = wb.worksheets.add("Metric Deltas");
metrics.showGridLines = false;
title(metrics, "Garrido-Style Metric Deltas", 0, 0, 9);
writeTable(metrics, 2, 0, metricRows(rankedRows), "#548235");
metrics.freezePanes.freezeRows(3);

const promotion = wb.worksheets.add("Promotion");
promotion.showGridLines = false;
title(promotion, "Promotion Gate", 0, 0, 8);
writeKV(promotion, 2, 0, [
  ["Excel ReT delta gate", promotionJson.promotion_rule?.excel_ret_delta_gt ?? ""],
  ["CVaR05 delta gate", promotionJson.promotion_rule?.cvar05_delta_gt ?? ""],
  ["Cost cap", promotionJson.promotion_rule?.cost_cap ?? ""],
]);
writeTable(promotion, 8, 0, promotionJson.promoted || promotedRows || []);
promotion.freezePanes.freezeRows(3);

for (const row of rankedRows.slice(0, 3)) {
  const safeName = String(row.cell).slice(0, 28).replace(/[\[\]:*?/\\]/g, "_");
  const sheet = wb.worksheets.add(safeName || "Config");
  sheet.showGridLines = false;
  title(sheet, `Config Detail: ${row.cell}`, 0, 0, 9);
  writeKV(sheet, 2, 0, Object.entries(row));
  const policyRows = await readCsvMaybe(path.join(sweepDir, String(row.cell), "policy_summary.csv"));
  writeTable(sheet, 2, 4, policyRows.slice(0, 20), "#7030A0");
  sheet.freezePanes.freezeRows(3);
}

const inspect = await wb.inspect({
  kind: "sheet,region",
  sheetId: "Dashboard",
  range: "A1:J25",
  maxChars: 4000,
});
await fs.writeFile(`${workbookPath}.inspect.ndjson`, inspect.ndjson);
const preview = await wb.render({ sheetName: "Dashboard", range: "A1:J25", scale: 1 });
await fs.writeFile(`${workbookPath}.dashboard.png`, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(workbookPath);
console.log(`WROTE ${workbookPath}`);
