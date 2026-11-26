// Dossier builder for the token launch radar.
// Reconstructed from the bounty spec (make.fun/bounty/token-radar-01):
// buildDossier(outputDir, { text_max_bytes }) writes dossier.json from
// dashboard/ and screenshots/ under outputDir.
//
// IMPORTANT: the entry must ship vendor/makefun-board.ts from the kit
// (sha256 9ca51b6e5cd84fe46cb0efa61e2ea99365a1fb0e21f2b75e6007f977f0dc326e).
// This file is a spec-faithful reconstruction so the repository builds and
// gates locally without the kit. Before submitting, replace this file with
// the kit's vendor/makefun-board.ts and re-run main.ts so dossier.json is
// byte for byte what the kit's buildDossier writes.

import { readdirSync, readFileSync, statSync, writeFileSync, chmodSync } from "node:fs";
import { join, relative, sep } from "node:path";

export interface BuildDossierOptions {
  text_max_bytes?: number;
}

export interface DossierFile {
  path: string;
  bytes: number;
}

export interface DossierScreenshot {
  name: string;
  media_type: string;
  bytes: number;
  data_base64: string;
}

export interface Dossier {
  format: string;
  files: DossierFile[];
  screenshots: DossierScreenshot[];
  text: string;
}

const TEXT_EXTS = new Set([".html", ".css", ".js", ".json", ".mjs", ".svg", ".md", ".txt"]);
const NAME_RE = /^[A-Za-z0-9._-]{1,128}$/;
const MAX_DASHBOARD_FILES = 400;
const MAX_DASHBOARD_BYTES = 4 * 1024 * 1024;
const MAX_SCREENSHOT_BYTES = 512 * 1024;
const MAX_SCREENSHOTS = 4;
const MIN_SCREENSHOTS = 3;
const MAX_DOSSIER_BYTES = 1.5 * 1024 * 1024;

function isPlainName(name: string): boolean {
  if (!NAME_RE.test(name)) return false;
  if (name.startsWith(".") || name.startsWith("-")) return false;
  return true;
}

function listFiles(root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string): void => {
    const entries = readdirSync(dir, { withFileTypes: true });
    entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    for (const entry of entries) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name.startsWith(".")) {
          throw new Error(`dashboard directory starts with a dot: ${entry.name}`);
        }
        walk(full);
      } else if (entry.isFile()) {
        if (!isPlainName(entry.name)) {
          throw new Error(`dashboard file name is not plain: ${entry.name}`);
        }
        out.push(relative(root, full).split(sep).join("/"));
      } else {
        throw new Error(`dashboard file is not plain: ${entry.name}`);
      }
    }
  };
  walk(root);
  out.sort();
  return out;
}

function pngDimensions(buf: Buffer): { width: number; height: number } | null {
  if (buf.length < 33) return null;
  if (
    buf[0] !== 0x89 ||
    buf[1] !== 0x50 ||
    buf[2] !== 0x4e ||
    buf[3] !== 0x47 ||
    buf[4] !== 0x0d ||
    buf[5] !== 0x0a ||
    buf[6] !== 0x1a ||
    buf[7] !== 0x0a
  ) {
    return null;
  }
  // First chunk must be IHDR.
  if (buf[12] !== 0x49 || buf[13] !== 0x48 || buf[14] !== 0x44 || buf[15] !== 0x52) {
    return null;
  }
  const width = buf.readUInt32BE(16);
  const height = buf.readUInt32BE(20);
  if (width <= 0 || height <= 0) return null;
  // Must end with IEND chunk.
  const tail = buf.subarray(buf.length - 12);
  if (
    tail[0] !== 0x00 ||
    tail[1] !== 0x00 ||
    tail[2] !== 0x00 ||
    tail[3] !== 0x00 ||
    tail[4] !== 0x49 ||
    tail[5] !== 0x45 ||
    tail[6] !== 0x4e ||
    tail[7] !== 0x44
  ) {
    return null;
  }
  return { width, height };
}

function jpegDimensions(buf: Buffer): { width: number; height: number } | null {
  if (buf.length < 4) return null;
  if (buf[0] !== 0xff || buf[1] !== 0xd8) return null;
  if (buf[buf.length - 2] !== 0xff || buf[buf.length - 1] !== 0xd9) return null;
  let i = 2;
  while (i + 9 < buf.length) {
    if (buf[i] !== 0xff) {
      i += 1;
      continue;
    }
    const marker = buf[i + 1];
    if (marker === 0xd8 || marker === 0xd9 || marker === 0x01) {
      i += 2;
      continue;
    }
    if (i + 3 >= buf.length) return null;
    const len = buf.readUInt16BE(i + 2);
    if (len < 2) return null;
    if (
      marker >= 0xc0 &&
      marker <= 0xcf &&
      marker !== 0xc4 &&
      marker !== 0xc8 &&
      marker !== 0xcc
    ) {
      if (i + 9 >= buf.length) return null;
      const height = buf.readUInt16BE(i + 5);
      const width = buf.readUInt16BE(i + 7);
      if (width <= 0 || height <= 0) return null;
      return { width, height };
    }
    i += 2 + len;
  }
  return null;
}

function extOf(path: string): string {
  const idx = path.lastIndexOf(".");
  if (idx < 0) return "";
  return path.slice(idx).toLowerCase();
}

function fenceFor(ext: string): string {
  if (ext === ".html") return "html";
  if (ext === ".css") return "css";
  if (ext === ".js" || ext === ".mjs") return "js";
  if (ext === ".json") return "json";
  if (ext === ".svg") return "svg";
  if (ext === ".md") return "md";
  if (ext === ".txt") return "txt";
  return "";
}

function countLines(text: string): number {
  if (text.length === 0) return 0;
  let n = 0;
  for (let i = 0; i < text.length; i++) if (text[i] === "\n") n += 1;
  if (!text.endsWith("\n")) n += 1;
  return n;
}

export function buildDossier(outputDir: string, opts: BuildDossierOptions = {}): Dossier {
  const textMax = opts.text_max_bytes ?? 65536;
  const dashboardDir = join(outputDir, "dashboard");
  const screenshotsDir = join(outputDir, "screenshots");

  // Dashboard files.
  let dashboardPaths: string[] = [];
  try {
    dashboardPaths = listFiles(dashboardDir);
  } catch (err) {
    throw new Error(`dashboard files: ${(err as Error).message}`);
  }
  if (dashboardPaths.length === 0) {
    throw new Error("dashboard is empty");
  }
  if (dashboardPaths.length > MAX_DASHBOARD_FILES) {
    throw new Error(`dashboard has ${dashboardPaths.length} files, cap is ${MAX_DASHBOARD_FILES}`);
  }
  let dashboardBytes = 0;
  const dashboardData = new Map<string, Buffer>();
  for (const rel of dashboardPaths) {
    const full = join(dashboardDir, rel);
    const st = statSync(full);
    if (!st.isFile()) throw new Error(`dashboard file is not plain: ${rel}`);
    dashboardBytes += st.size;
    if (dashboardBytes > MAX_DASHBOARD_BYTES) {
      throw new Error(`dashboard is over ${MAX_DASHBOARD_BYTES} bytes`);
    }
    dashboardData.set(rel, readFileSync(full));
  }
  const hasIndex = dashboardPaths.some((p) => p === "index.html" || p.endsWith("/index.html"));
  const hasCss = dashboardPaths.some((p) => p.endsWith(".css"));
  const hasJs = dashboardPaths.some((p) => p.endsWith(".js") || p.endsWith(".mjs"));
  if (!hasIndex || !hasCss || !hasJs) {
    throw new Error("dashboard needs index.html and at least one .css and one .js file");
  }

  // Screenshots.
  let shotNames: string[] = [];
  try {
    shotNames = readdirSync(screenshotsDir)
      .filter((n) => !n.startsWith("."))
      .sort();
  } catch {
    throw new Error("screenshots directory is missing");
  }
  if (shotNames.length < MIN_SCREENSHOTS || shotNames.length > MAX_SCREENSHOTS) {
    throw new Error(
      `screenshots holds ${shotNames.length} images, need ${MIN_SCREENSHOTS} to ${MAX_SCREENSHOTS}`,
    );
  }
  const screenshots: DossierScreenshot[] = [];
  for (const name of shotNames) {
    if (!isPlainName(name)) throw new Error(`screenshot name is not plain: ${name}`);
    const full = join(screenshotsDir, name);
    const st = statSync(full);
    if (!st.isFile()) throw new Error(`screenshot is not a regular file: ${name}`);
    if (st.size > MAX_SCREENSHOT_BYTES) {
      throw new Error(`screenshot ${name} is ${st.size} bytes, cap is ${MAX_SCREENSHOT_BYTES}`);
    }
    const buf = readFileSync(full);
    let media = "";
    let dims: { width: number; height: number } | null = null;
    const lower = name.toLowerCase();
    if (lower.endsWith(".png")) {
      dims = pngDimensions(buf);
      media = "image/png";
    } else if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
      dims = jpegDimensions(buf);
      media = "image/jpeg";
    } else {
      throw new Error(`screenshot ${name} is not a PNG or JPEG`);
    }
    if (dims === null) throw new Error(`screenshot ${name} is not a whole PNG or JPEG`);
    screenshots.push({
      name,
      media_type: media,
      bytes: buf.length,
      data_base64: buf.toString("base64"),
    });
  }

  // Text: ## Dashboard block, then ## Source with fenced files in path order.
  const files: DossierFile[] = dashboardPaths.map((p) => ({
    path: p,
    bytes: dashboardData.get(p)?.length ?? 0,
  }));
  const headerLines: string[] = [];
  headerLines.push("## Dashboard");
  headerLines.push(`files: ${files.length}`);
  headerLines.push(`bytes: ${dashboardBytes}`);
  for (const f of files) headerLines.push(`- ${f.path} (${f.bytes} bytes)`);
  headerLines.push("");
  headerLines.push("## Source");
  headerLines.push("");
  let text = headerLines.join("\n") + "\n";

  const textPaths = dashboardPaths.filter((p) => TEXT_EXTS.has(extOf(p)));
  let filesLeftOut = 0;
  let bytesLeftOut = 0;
  let cut = false;
  for (let idx = 0; idx < textPaths.length; idx++) {
    const rel = textPaths[idx];
    const buf = dashboardData.get(rel) as Buffer;
    const content = buf.toString("utf8");
    const lines = countLines(content);
    const fence = fenceFor(extOf(rel));
    const sectionHead = `### ${rel} (${buf.length} bytes, ${lines} lines)\n\`\`\`${fence}\n`;
    const sectionTail = "\n```\n";
    const whole = sectionHead + content + sectionTail;
    const wholeBytes = Buffer.byteLength(whole, "utf8");
    const currentBytes = Buffer.byteLength(text, "utf8");
    if (currentBytes + wholeBytes <= textMax) {
      text += whole;
      continue;
    }
    // First file that does not fit: take whole lines that fit.
    const budget = textMax - currentBytes - Buffer.byteLength(sectionHead, "utf8");
    if (budget <= 0) {
      filesLeftOut += textPaths.length - idx;
      for (let k = idx; k < textPaths.length; k++) {
        bytesLeftOut += (dashboardData.get(textPaths[k])?.length ?? 0);
      }
      cut = true;
      break;
    }
    const contentLines = content.split("\n");
    let taken = "";
    let takenLines = 0;
    for (let li = 0; li < contentLines.length; li++) {
      const line = contentLines[li] + (li < contentLines.length - 1 ? "\n" : "");
      if (Buffer.byteLength(taken + line, "utf8") > budget) break;
      taken += line;
      takenLines += 1;
    }
    const notShown = lines - takenLines;
    text += sectionHead + taken;
    if (notShown > 0) {
      text += `\n[${notShown} lines not shown]\n`;
    }
    text += "```\n";
    filesLeftOut += textPaths.length - idx - 1;
    for (let k = idx + 1; k < textPaths.length; k++) {
      bytesLeftOut += (dashboardData.get(textPaths[k])?.length ?? 0);
    }
    cut = true;
    break;
  }
  if (cut && filesLeftOut > 0) {
    text += `[${filesLeftOut} files (${bytesLeftOut} bytes) left out]\n`;
  }
  if (Buffer.byteLength(text, "utf8") > textMax) {
    // Last-resort trim to the cap on a line boundary.
    const lines = text.split("\n");
    let out = "";
    for (const line of lines) {
      const next = out + line + "\n";
      if (Buffer.byteLength(next, "utf8") > textMax) break;
      out = next;
    }
    text = out;
  }

  const dossier: Dossier = {
    format: "makefun-dossier/1",
    files,
    screenshots,
    text,
  };
  const json = JSON.stringify(dossier);
  if (Buffer.byteLength(json, "utf8") > MAX_DOSSIER_BYTES) {
    throw new Error(`dossier is over ${MAX_DOSSIER_BYTES} bytes`);
  }
  const outPath = join(outputDir, "dossier.json");
  writeFileSync(outPath, json + "\n", { mode: 0o644 });
  try {
    chmodSync(outPath, 0o644);
  } catch {
    // Best effort: the grader collects as another user.
  }
  return dossier;
}
