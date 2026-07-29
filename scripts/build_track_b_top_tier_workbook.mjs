#!/usr/bin/env node
import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [payloadPath, workbookPath] = process.argv.slice(2);
if (!payloadPath || !workbookPath) {
  console.error("usage: build_track_b_top_tier_workbook.mjs payload.json output.xlsx");
  process.exit(2);
}

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));

function rowsToMatrix(rows) {
  if (!rows || rows.length === 0) return [["(empty)"]];
  const headers = Object.keys(rows[0]);
  return [headers, ...rows.map((row) => headers.map((h) => row[h] ?? null))];
}

function writeTable(sheet, startRow, startCol, rows) {
  const matrix = rowsToMatrix(rows);
  const range = sheet.getRangeByIndexes(startRow, startCol, matrix.length, matrix[0].length);
  range.values = matrix;
  const header = sheet.getRangeByIndexes(startRow, startCol, 1, matrix[0].length);
  header.format.fill.color = "#1F4E79";
  header.format.font.color = "#FFFFFF";
  header.format.font.bold = true;
  range.format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  range.format.autofitColumns();
  return range;
}

function writeKV(sheet, startRow, startCol, rows) {
  const matrix = rows.map(([k, v]) => [k, v]);
  const range = sheet.getRangeByIndexes(startRow, startCol, matrix.length, 2);
  range.values = matrix;
  sheet.getRangeByIndexes(startRow, startCol, matrix.length, 1).format.font.bold = true;
  range.format.borders = { preset: "outside", style: "thin", color: "#D9E2F3" };
  range.format.autofitColumns();
}

function title(sheet, text, row = 0, col = 0, width = 8) {
  const r = sheet.getRangeByIndexes(row, col, 1, width);
  r.values = [[text, ...Array(width - 1).fill(null)]];
  r.merge();
  r.format.fill.color = "#D9EAF7";
  r.format.font.bold = true;
  r.format.font.size = 14;
}

const wb = Workbook.create();

const dashboard = wb.worksheets.add("Dashboard");
dashboard.showGridLines = false;
title(dashboard, "Track B Top-Tier Claim Audit", 0, 0, 8);
writeKV(dashboard, 2, 0, [
  ["Generated UTC", payload.generated_at_utc],
  ["Run directory", payload.run_dir],
  ["Learned policy", payload.learned_policy],
  ["Best static by ReT", payload.best_static_policy],
  ["Seeds", (payload.config?.seeds || []).join(",")],
  ["Timesteps", payload.config?.train_timesteps ?? null],
  ["Eval episodes", payload.config?.eval_episodes ?? null],
  ["Risk level", payload.config?.risk_level ?? null],
  ["Reward mode", payload.config?.reward_mode ?? null],
]);
writeTable(dashboard, 13, 0, payload.verdicts || []);
dashboard.freezePanes.freezeRows(1);

const metric = wb.worksheets.add("Metric Panel");
metric.showGridLines = false;
title(metric, "Dynamic vs Best Static: Full Metric Panel", 0, 0, 9);
writeTable(metric, 2, 0, payload.metric_panel || []);
metric.freezePanes.freezeRows(3);

const ledgerMetrics = wb.worksheets.add("Ledger Metrics");
ledgerMetrics.showGridLines = false;
title(ledgerMetrics, "Garrido-Style Per-Order Metrics", 0, 0, 9);
writeTable(ledgerMetrics, 2, 0, payload.ledger_metric_panel || []);
ledgerMetrics.freezePanes.freezeRows(3);

const tails = wb.worksheets.add("Tail CVaR");
tails.showGridLines = false;
title(tails, "Tail Metrics / CVaR", 0, 0, 8);
writeTable(tails, 2, 0, payload.tail_panel || []);
tails.freezePanes.freezeRows(3);

const seeds = wb.worksheets.add("Seed Deltas");
seeds.showGridLines = false;
title(seeds, "Paired Seed Deltas and CI95", 0, 0, 8);
writeTable(seeds, 2, 0, payload.seed_delta_panel || []);
seeds.freezePanes.freezeRows(3);

const frontier = wb.worksheets.add("Static Frontier");
frontier.showGridLines = false;
title(frontier, "Dense Static Frontier / Static Policies", 0, 0, 8);
writeTable(frontier, 2, 0, payload.static_frontier || []);
frontier.freezePanes.freezeRows(3);

const externalFrontier = wb.worksheets.add("External Dense Frontier");
externalFrontier.showGridLines = false;
title(externalFrontier, "External Dense Static Frontier", 0, 0, 8);
writeTable(externalFrontier, 2, 0, payload.external_static_frontier || []);
externalFrontier.freezePanes.freezeRows(3);

const policy = wb.worksheets.add("Policy Summary");
policy.showGridLines = false;
title(policy, "Policy Summary From Source Run", 0, 0, 8);
writeTable(policy, 2, 0, payload.policy_summary || []);
policy.freezePanes.freezeRows(3);

const seedMetrics = wb.worksheets.add("Seed Metrics");
seedMetrics.showGridLines = false;
title(seedMetrics, "Seed Metrics From Source Run", 0, 0, 8);
writeTable(seedMetrics, 2, 0, payload.seed_metrics || []);
seedMetrics.freezePanes.freezeRows(3);

const gaps = wb.worksheets.add("Ledger Gaps");
gaps.showGridLines = false;
title(gaps, "Garrido-Ledger Coverage / Missing Metrics", 0, 0, 8);
writeTable(gaps, 2, 0, payload.data_gaps || []);
gaps.freezePanes.freezeRows(3);

const ledgerSummary = wb.worksheets.add("Ledger Summary");
ledgerSummary.showGridLines = false;
title(ledgerSummary, "Per-Order Ledger Summary", 0, 0, 8);
writeTable(ledgerSummary, 2, 0, payload.ledger_summary || []);
ledgerSummary.freezePanes.freezeRows(3);

const ledgerSample = wb.worksheets.add("Order Ledger Sample");
ledgerSample.showGridLines = false;
title(ledgerSample, "Garrido-Style Per-Order Ledger Sample", 0, 0, 8);
writeTable(ledgerSample, 2, 0, payload.order_ledger_sample || []);
ledgerSample.freezePanes.freezeRows(3);

const inspect = await wb.inspect({
  kind: "sheet,region",
  sheetId: "Dashboard",
  range: "A1:H25",
  maxChars: 4000,
});
await fs.writeFile(`${workbookPath}.inspect.ndjson`, inspect.ndjson);

const preview = await wb.render({ sheetName: "Dashboard", range: "A1:H25", scale: 1 });
await fs.writeFile(`${workbookPath}.dashboard.png`, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(workbookPath);
console.log(`WROTE ${workbookPath}`);
