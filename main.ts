// Token Launch Radar entrypoint (Node 22, standard library only).
// Reads the sealed snapshot, runs the fitted pre-token valuation & farm model,
// writes board.json + dashboard/ + screenshots/ + dossier.json into outputDir.
//
// Grader: node /bot/main.ts with MAKEFUN_BOT_DIR=/bot,
// MAKEFUN_INPUT_DIR=/input, MAKEFUN_OUTPUT_DIR=/output.
// Local: MAKEFUN_OUTPUT_DIR=out node main.ts --input ../input/snapshot.json

import {
  mkdirSync,
  readFileSync,
  writeFileSync,
  readdirSync,
  statSync,
  copyFileSync,
  chmodSync,
} from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildBoard, type Snapshot } from "./lib/model.ts";
import { buildDossier } from "./vendor/makefun-board.ts";

function argValue(name: string): string | null {
  const idx = process.argv.indexOf(name);
  if (idx < 0 || idx + 1 >= process.argv.length) return null;
  return process.argv[idx + 1];
}

function firstExisting(candidates: string[]): string | null {
  for (const c of candidates) {
    try {
      const st = statSync(c);
      if (st.isFile()) return c;
    } catch {
      // Try next candidate.
    }
  }
  return null;
}

function copyDir(src: string, dest: string, skip: Set<string>): void {
  mkdirSync(dest, { recursive: true });
  const entries = readdirSync(src, { withFileTypes: true });
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  for (const entry of entries) {
    const srcPath = join(src, entry.name);
    const destPath = join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, destPath, skip);
    } else if (entry.isFile()) {
      if (skip.has(entry.name)) continue;
      copyFileSync(srcPath, destPath);
      try {
        chmodSync(destPath, 0o644);
      } catch {
        // Best effort.
      }
    }
  }
}

function chmodTree(root: string): void {
  try {
    chmodSync(root, 0o755);
  } catch {
    // Best effort.
  }
  const entries = readdirSync(root, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(root, entry.name);
    if (entry.isDirectory()) chmodTree(full);
    else {
      try {
        chmodSync(full, 0o644);
      } catch {
        // Best effort.
      }
    }
  }
}

function main(): void {
  const here = dirname(fileURLToPath(import.meta.url));
  const botDir = resolve(process.env.MAKEFUN_BOT_DIR ?? here);
  const outputDir = resolve(
    argValue("--output") ?? process.env.MAKEFUN_OUTPUT_DIR ?? join(here, "out"),
  );

  const inputFlag = argValue("--input");
  const inputEnv = process.env.MAKEFUN_INPUT_DIR;
  const inputCandidates: string[] = [];
  if (inputFlag) inputCandidates.push(resolve(inputFlag));
  if (inputEnv) {
    inputCandidates.push(join(resolve(inputEnv), "snapshot.json"));
    inputCandidates.push(resolve(inputEnv));
  }
  inputCandidates.push(join(botDir, "input", "snapshot.json"));
  inputCandidates.push(resolve(join(botDir, "..", "input", "snapshot.json")));
  inputCandidates.push(join(here, "input", "snapshot.json"));
  inputCandidates.push(resolve(join(here, "..", "input", "snapshot.json")));
  inputCandidates.push("/input/snapshot.json");

  const snapshotPath = firstExisting(inputCandidates);
  if (!snapshotPath) {
    throw new Error(`snapshot.json not found (tried: ${inputCandidates.join(", ")})`);
  }

  const snapshot = JSON.parse(readFileSync(snapshotPath, "utf8")) as Snapshot;

  // Include all watchlist protocols plus top expected-value pre-token candidates below floor.
  const result = buildBoard(snapshot, 220);

  mkdirSync(outputDir, { recursive: true });

  const board = {
    as_of: snapshot.as_of,
    floor_usd: snapshot.floor_usd,
    model:
      "implied_value_usd = conditional_value_usd * token_odds; conditional fits 5-fold CV Huber-ridge log(mcap) on log(tvl), dampened 7d growth, log-age, chain breadth + shrunk category residual; odds fits L2 logistic launch rate over peers",
    stats: result.stats,
    rows: result.rows,
  };
  writeFileSync(join(outputDir, "board.json"), JSON.stringify(board, null, 2) + "\n", {
    mode: 0o644,
  });

  // Copy dashboard files and emit z-data.js so alphabetical dossier order keeps code untruncated.
  const srcDashboard = join(botDir, "dashboard");
  const destDashboard = join(outputDir, "dashboard");
  copyDir(srcDashboard, destDashboard, new Set(["z-data.js", "data.js"]));
  const rowLines = result.rows.map((r) => "    " + JSON.stringify(r)).join(",\n");
  const zData =
    "window.RADAR_DATA = {\n" +
    `  "as_of": ${JSON.stringify(snapshot.as_of)},\n` +
    `  "floor_usd": ${JSON.stringify(snapshot.floor_usd)},\n` +
    `  "stats": ${JSON.stringify(result.stats)},\n` +
    '  "rows": [\n' +
    rowLines +
    "\n  ]\n};\n";
  writeFileSync(join(destDashboard, "z-data.js"), zData, { mode: 0o644 });

  // Copy screenshots as-is.
  copyDir(join(botDir, "screenshots"), join(outputDir, "screenshots"), new Set());

  // Build dossier.json from outputDir/dashboard and outputDir/screenshots.
  buildDossier(outputDir, { text_max_bytes: 65536 });

  chmodTree(outputDir);
}

main();
