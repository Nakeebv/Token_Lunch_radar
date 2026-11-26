// Token launch radar value & farm-economics model.
// Pure standard-library TypeScript for Node 22 (no network, no dependencies).
//
// Reads a sealed DefiLlama snapshot, strictly separates already-trading tokens
// and wrapped/staked/bridged/CEX derivatives from genuine pre-token protocols,
// fits every valuation and launch-probability parameter from the snapshot's own
// token peers (verified via 5-fold out-of-fold cross-validation), and emits a
// deterministic ranking + costed chain-aware farm plan for each board row.

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export interface PeerComparable {
  slug: string;
  name: string;
  tvl_usd: number;
  mcap_usd: number;
  multiple: number;
}

export interface ModelDecomposition {
  tvl_base_usd: number;
  growth_mult: number;
  age_mult: number;
  chain_mult: number;
  category_mult: number;
}

export interface BoardRow {
  slug: string;
  name: string;
  rank: number;
  implied_value_usd: number;
  farm_steps: string[];
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  age_days: number | null;
  conditional_value_usd: number;
  token_odds: number;
  watchlist: boolean;
  peer_multiple: number;
  farm_cost_usd: number;
  farm_capital_usd: string;
  farm_minutes: number;
  farm_verdict: string;
  decomposition: ModelDecomposition;
  comps: PeerComparable[];
}

export interface ModelDiagnostics {
  ridge_lambda: number;
  cv_folds: number;
  cv_rmse_ln: number;
  baseline_rmse_ln: number;
  cv_r2: number;
  cv_med_ape: number;
  odds_brier: number;
  beta_tvl: number;
  beta_growth: number;
  beta_age: number;
  beta_chains: number;
}

export interface BoardStats {
  total_snapshot: number;
  token_peers: number;
  traded_no_mcap: number;
  excluded_derivatives: number;
  pre_token: number;
  watchlist: number;
  watchlist_covered: number;
  board_rows: number;
  below_floor_rows: number;
  global_multiple: number;
  median_farm_cost_usd: number;
  cv: ModelDiagnostics;
}

export interface BoardResult {
  rows: BoardRow[];
  stats: BoardStats;
}

const LOG = Math.log;
const MAX_IMPLIED_USD = 100_000_000_000_000;

const EXCLUDED_CATEGORIES = new Set([
  "Bridge",
  "Canonical Bridge",
  "Liquid Staking",
  "Liquid Restaking",
  "Restaked BTC",
  "CEX",
]);

const EXCHANGE_PREFIXES = [
  "Binance",
  "Coinbase",
  "Bybit",
  "Kraken",
  "Bitget",
  "OKX",
  "Bitfinex",
  "Gemini",
];

const VERSION_MARKER_RE = /\s+(?:v\d+(?:\.\d+)*|\d+\.\d+)$/i;
const WRAPPED_STAKED_RE = /\b(?:wrapped|staked|staking)\b/i;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function mad(values: number[], med: number): number {
  if (values.length === 0) return 1;
  const dev = values.map((v) => Math.abs(v - med));
  return Math.max(0.15, median(dev) * 1.4826);
}

function categoryKey(category: string | null): string {
  return category === null || category.trim() === "" ? "Uncategorized" : category.trim();
}

function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const start = Date.parse(listedAt + "T00:00:00Z");
  const end = Date.parse(asOf + "T00:00:00Z");
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return Math.max(0, Math.floor((end - start) / 86_400_000));
}

function dampenedGrowth(change7d: number | null, fallback: number): number {
  if (change7d === null || !Number.isFinite(change7d)) return fallback;
  const c = clamp(change7d, -85, 300);
  return c >= 0 ? LOG(1 + c / 25) : -LOG(1 + Math.abs(c) / 25);
}

function isDerivativeOrCex(p: SnapshotProtocol): boolean {
  if (p.category !== null && EXCLUDED_CATEGORIES.has(p.category)) return true;
  if (VERSION_MARKER_RE.test(p.name.trim())) return true;
  if (WRAPPED_STAKED_RE.test(p.name)) return true;
  for (const prefix of EXCHANGE_PREFIXES) {
    if (p.name.startsWith(prefix)) return true;
  }
  return false;
}

function hasRecordedToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd !== null || p.gecko_id !== null || p.symbol !== null;
}

function foldOf(slug: string, folds: number): number {
  let h = 2166136261;
  for (let i = 0; i < slug.length; i++) {
    h ^= slug.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % folds;
}

// Solve symmetric/square linear system A x = b via Gaussian elimination with partial pivoting.
function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (pivot !== col) {
      const tmp = M[col];
      M[col] = M[pivot];
      M[pivot] = tmp;
    }
    const piv = M[col][col];
    if (Math.abs(piv) < 1e-12) continue;
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / piv;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let j = i + 1; j < n; j++) sum -= M[i][j] * x[j];
    x[i] = Math.abs(M[i][i]) < 1e-12 ? 0 : sum / M[i][i];
  }
  return x;
}

interface FeatureStats {
  medianAge: number;
  medianGrowth: number;
  meanX: number[];
}

function extractRawFeatures(
  p: SnapshotProtocol,
  asOf: string,
  fstats: FeatureStats,
): number[] {
  const age = ageDays(asOf, p.listed_at);
  const effAge = age === null ? fstats.medianAge : clamp(age, 0, 3650);
  const g = dampenedGrowth(p.change_7d, fstats.medianGrowth);
  const chains = Math.max(1, p.chains.length);
  return [
    1,
    LOG(Math.max(1, p.tvl_usd)),
    g,
    LOG(1 + effAge),
    LOG(1 + chains),
  ];
}

interface RidgeModel {
  coef: number[];
  catResidual: Map<string, number>;
  catMultiples: Map<string, number>;
  globalMultiple: number;
  fstats: FeatureStats;
}

function fitWeightedRidge(
  peers: SnapshotProtocol[],
  asOf: string,
  lambda: number,
  huberSteps: number,
): RidgeModel {
  const ages = peers
    .map((p) => ageDays(asOf, p.listed_at))
    .filter((a): a is number => a !== null);
  const medianAge = ages.length > 0 ? median(ages) : 180;

  const growths = peers
    .filter((p) => p.change_7d !== null && Number.isFinite(p.change_7d))
    .map((p) => dampenedGrowth(p.change_7d, 0));
  const medianGrowth = growths.length > 0 ? median(growths) : 0;

  const dim = 5;
  const fstats: FeatureStats = {
    medianAge,
    medianGrowth,
    meanX: new Array(dim).fill(0),
  };

  if (peers.length === 0) {
    return {
      coef: [0, 0.85, 0, 0, 0],
      catResidual: new Map(),
      catMultiples: new Map(),
      globalMultiple: 1,
      fstats,
    };
  }

  const X = peers.map((p) => extractRawFeatures(p, asOf, fstats));
  const y = peers.map((p) => LOG(p.mcap_usd as number));
  const n = peers.length;

  const mean = new Array(dim).fill(0);
  const scale = new Array(dim).fill(1);
  for (let j = 1; j < dim; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i][j];
    mean[j] = s / n;
    fstats.meanX[j] = mean[j];
    let vs = 0;
    for (let i = 0; i < n; i++) vs += (X[i][j] - mean[j]) ** 2;
    const sd = Math.sqrt(vs / n);
    scale[j] = sd > 1e-9 ? sd : 1;
  }

  const Z = X.map((row) => [
    1,
    (row[1] - mean[1]) / scale[1],
    (row[2] - mean[2]) / scale[2],
    (row[3] - mean[3]) / scale[3],
    (row[4] - mean[4]) / scale[4],
  ]);

  let weights = new Array(n).fill(1);
  let beta = new Array(dim).fill(0);

  for (let step = 0; step <= huberSteps; step++) {
    const xtx = Array.from({ length: dim }, () => new Array(dim).fill(0));
    const xty = new Array(dim).fill(0);
    for (let idx = 0; idx < n; idx++) {
      const w = weights[idx];
      const z = Z[idx];
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) xtx[i][j] += w * z[i] * z[j];
        xty[i] += w * z[i] * y[idx];
      }
    }
    for (let i = 1; i < dim; i++) xtx[i][i] += lambda;
    beta = solveLinear(xtx, xty);

    if (step < huberSteps) {
      const res = new Array(n);
      for (let idx = 0; idx < n; idx++) {
        let pred = 0;
        for (let j = 0; j < dim; j++) pred += beta[j] * Z[idx][j];
        res[idx] = y[idx] - pred;
      }
      const medR = median(res);
      const scaleR = mad(res, medR);
      const cutoff = 1.5 * scaleR;
      weights = res.map((r) => {
        const a = Math.abs(r);
        return a <= cutoff ? 1 : cutoff / a;
      });
    }
  }

  // Transform standardized coefficients back to raw-feature space.
  const coef = new Array(dim).fill(0);
  coef[0] = beta[0];
  for (let j = 1; j < dim; j++) {
    coef[j] = beta[j] / scale[j];
    coef[0] -= (beta[j] * mean[j]) / scale[j];
  }

  // Empirical-Bayes shrunk category residual (median log residual shrunk toward 0).
  const byCatRes = new Map<string, number[]>();
  const byCatRatio = new Map<string, number[]>();
  const allRatios: number[] = [];
  for (let idx = 0; idx < n; idx++) {
    const p = peers[idx];
    let pred = 0;
    for (let j = 0; j < dim; j++) pred += coef[j] * X[idx][j];
    const r = y[idx] - pred;
    const key = categoryKey(p.category);
    const list = byCatRes.get(key) ?? [];
    list.push(r);
    byCatRes.set(key, list);

    const lr = LOG((p.mcap_usd as number) / p.tvl_usd);
    allRatios.push(lr);
    const rlist = byCatRatio.get(key) ?? [];
    rlist.push(lr);
    byCatRatio.set(key, rlist);
  }

  const catResidual = new Map<string, number>();
  for (const [k, rs] of byCatRes) {
    const med = median(rs);
    const shrink = rs.length / (rs.length + 5);
    catResidual.set(k, clamp(med * shrink, -1.5, 1.5));
  }

  const globalLogMult = allRatios.length > 0 ? median(allRatios) : 0;
  const catMultiples = new Map<string, number>();
  for (const [k, rs] of byCatRatio) {
    const med = median(rs);
    const shrink = rs.length / (rs.length + 5);
    catMultiples.set(k, Math.exp(shrink * med + (1 - shrink) * globalLogMult));
  }

  return {
    coef,
    catResidual,
    catMultiples,
    globalMultiple: Math.exp(globalLogMult),
    fstats,
  };
}

function predictConditional(
  model: RidgeModel,
  p: SnapshotProtocol,
  asOf: string,
): { logValue: number; valueUsd: number; decomposition: ModelDecomposition } {
  const x = extractRawFeatures(p, asOf, model.fstats);
  const c = model.coef;
  const mean = model.fstats.meanX;
  const catRes = model.catResidual.get(categoryKey(p.category)) ?? 0;

  const baseLog =
    c[0] + c[1] * x[1] + c[2] * mean[2] + c[3] * mean[3] + c[4] * mean[4];
  const gDelta = c[2] * (x[2] - mean[2]);
  const aDelta = c[3] * (x[3] - mean[3]);
  const chDelta = c[4] * (x[4] - mean[4]);
  const totalLog = clamp(baseLog + gDelta + aDelta + chDelta + catRes, LOG(10_000), LOG(MAX_IMPLIED_USD));

  return {
    logValue: totalLog,
    valueUsd: Math.max(1, Math.round(Math.exp(totalLog))),
    decomposition: {
      tvl_base_usd: Math.max(1, Math.round(Math.exp(clamp(baseLog, LOG(10_000), LOG(MAX_IMPLIED_USD))))),
      growth_mult: Math.round(Math.exp(clamp(gDelta, -1.5, 1.5)) * 1000) / 1000,
      age_mult: Math.round(Math.exp(clamp(aDelta, -1.5, 1.5)) * 1000) / 1000,
      chain_mult: Math.round(Math.exp(clamp(chDelta, -1.5, 1.5)) * 1000) / 1000,
      category_mult: Math.round(Math.exp(catRes) * 1000) / 1000,
    },
  };
}

interface OddsModel {
  coef: number[];
  mean: number[];
  scale: number[];
  catRate: Map<string, number>;
  globalRate: number;
  medianAge: number;
  medianGrowth: number;
}

function logit(p: number): number {
  const pc = clamp(p, 0.02, 0.98);
  return LOG(pc / (1 - pc));
}

function sigmoid(z: number): number {
  if (z >= 30) return 1 / (1 + Math.exp(-30));
  if (z <= -30) return 1 / (1 + Math.exp(30));
  return 1 / (1 + Math.exp(-z));
}

function fitOddsLogistic(
  pool: SnapshotProtocol[],
  asOf: string,
): OddsModel {
  const n = pool.length;
  const ages = pool
    .map((p) => ageDays(asOf, p.listed_at))
    .filter((a): a is number => a !== null);
  const medianAge = ages.length > 0 ? median(ages) : 180;
  const growths = pool
    .filter((p) => p.change_7d !== null && Number.isFinite(p.change_7d))
    .map((p) => dampenedGrowth(p.change_7d, 0));
  const medianGrowth = growths.length > 0 ? median(growths) : 0;

  let tokensTotal = 0;
  const catTotal = new Map<string, number>();
  const catToken = new Map<string, number>();
  for (const p of pool) {
    const k = categoryKey(p.category);
    catTotal.set(k, (catTotal.get(k) ?? 0) + 1);
    if (hasRecordedToken(p)) {
      tokensTotal += 1;
      catToken.set(k, (catToken.get(k) ?? 0) + 1);
    }
  }
  const globalRate = n > 0 ? clamp(tokensTotal / n, 0.05, 0.95) : 0.5;
  const catRate = new Map<string, number>();
  for (const [k, tot] of catTotal) {
    const tok = catToken.get(k) ?? 0;
    catRate.set(k, clamp((tok + 3 * globalRate) / (tot + 3), 0.05, 0.95));
  }

  if (n < 10 || tokensTotal === 0 || tokensTotal === n) {
    return {
      coef: [logit(globalRate), 0, 0, 0, 0],
      mean: [0, 0, 0, 0, 0],
      scale: [1, 1, 1, 1, 1],
      catRate,
      globalRate,
      medianAge,
      medianGrowth,
    };
  }

  const dim = 5;
  const rawX = pool.map((p) => {
    const age = ageDays(asOf, p.listed_at);
    const effAge = age === null ? medianAge : clamp(age, 0, 3650);
    const cr = catRate.get(categoryKey(p.category)) ?? globalRate;
    return [
      1,
      LOG(Math.max(1, p.tvl_usd)),
      LOG(1 + effAge),
      dampenedGrowth(p.change_7d, medianGrowth),
      logit(cr),
    ];
  });
  const y = pool.map((p) => (hasRecordedToken(p) ? 1 : 0));

  const mean = new Array(dim).fill(0);
  const scale = new Array(dim).fill(1);
  for (let j = 1; j < dim; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += rawX[i][j];
    mean[j] = s / n;
    let vs = 0;
    for (let i = 0; i < n; i++) vs += (rawX[i][j] - mean[j]) ** 2;
    const sd = Math.sqrt(vs / n);
    scale[j] = sd > 1e-9 ? sd : 1;
  }

  const Z = rawX.map((r) => [
    1,
    (r[1] - mean[1]) / scale[1],
    (r[2] - mean[2]) / scale[2],
    (r[3] - mean[3]) / scale[3],
    (r[4] - mean[4]) / scale[4],
  ]);

  const l2 = 4.0;
  let beta = [logit(globalRate), 0, 0, 0, 0];
  for (let iter = 0; iter < 16; iter++) {
    const H = Array.from({ length: dim }, () => new Array(dim).fill(0));
    const g = new Array(dim).fill(0);
    for (let idx = 0; idx < n; idx++) {
      const z = Z[idx];
      let eta = 0;
      for (let j = 0; j < dim; j++) eta += beta[j] * z[j];
      const pr = sigmoid(eta);
      const w = Math.max(1e-6, pr * (1 - pr));
      const diff = pr - y[idx];
      for (let i = 0; i < dim; i++) {
        g[i] += diff * z[i];
        for (let j = 0; j < dim; j++) H[i][j] += w * z[i] * z[j];
      }
    }
    for (let i = 1; i < dim; i++) {
      g[i] += l2 * beta[i];
      H[i][i] += l2;
    }
    const step = solveLinear(H, g);
    let maxStep = 0;
    for (let i = 0; i < dim; i++) {
      beta[i] -= step[i];
      maxStep = Math.max(maxStep, Math.abs(step[i]));
    }
    if (maxStep < 1e-6) break;
  }

  return {
    coef: beta,
    mean,
    scale,
    catRate,
    globalRate,
    medianAge,
    medianGrowth,
  };
}

function predictOdds(model: OddsModel, p: SnapshotProtocol, asOf: string): number {
  const age = ageDays(asOf, p.listed_at);
  const effAge = age === null ? model.medianAge : clamp(age, 0, 3650);
  const cr = model.catRate.get(categoryKey(p.category)) ?? model.globalRate;
  const raw = [
    1,
    LOG(Math.max(1, p.tvl_usd)),
    LOG(1 + effAge),
    dampenedGrowth(p.change_7d, model.medianGrowth),
    logit(cr),
  ];
  let eta = model.coef[0];
  for (let j = 1; j < 5; j++) {
    eta += model.coef[j] * ((raw[j] - model.mean[j]) / model.scale[j]);
  }
  return clamp(sigmoid(eta), 0.05, 0.92);
}

const CHEAP_CHAINS = new Set([
  "Solana",
  "Base",
  "Arbitrum",
  "Optimism",
  "Sui",
  "Aptos",
  "Polygon",
  "Avalanche",
  "BNB",
  "Binance",
  "BSC",
  "Scroll",
  "Linea",
  "Sonic",
  "Monad",
  "Hyperliquid L1",
  "TON",
  "Tron",
  "Berachain",
  "Mantle",
  "Sei",
  "zkSync Era",
  "Near",
  "Starknet",
  "Blast",
  "Mode",
  "Unichain",
  "Ink",
  "MegaETH",
]);

function estimateChainCosts(chains: string[]): {
  primaryChain: string;
  secondaryChain: string | null;
  gasPerTxLabel: string;
  campaignGasUsd: number;
  isMainnetOnly: boolean;
} {
  const clean = chains.length > 0 ? chains : ["Ethereum"];
  const cheap = clean.filter((c) => CHEAP_CHAINS.has(c));
  const primary = cheap.length > 0 ? cheap[0] : clean[0];
  const secondary = clean.length > 1 ? (clean.find((c) => c !== primary) ?? null) : null;
  const isMainnetOnly = cheap.length === 0 && clean.includes("Ethereum");

  if (isMainnetOnly) {
    return {
      primaryChain: primary,
      secondaryChain: secondary,
      gasPerTxLabel: "$8-$20",
      campaignGasUsd: 65,
      isMainnetOnly: true,
    };
  }
  if (cheap.length > 0) {
    return {
      primaryChain: primary,
      secondaryChain: secondary,
      gasPerTxLabel: "$0.05-$0.80",
      campaignGasUsd: 9,
      isMainnetOnly: false,
    };
  }
  return {
    primaryChain: primary,
    secondaryChain: secondary,
    gasPerTxLabel: "$0.50-$3.00",
    campaignGasUsd: 18,
    isMainnetOnly: false,
  };
}

function buildFarmPlan(
  p: SnapshotProtocol,
  expectedUsd: number,
  odds: number,
  isWatchlist: boolean,
): {
  steps: string[];
  costUsd: number;
  capitalLabel: string;
  minutes: number;
  verdict: string;
} {
  const { primaryChain, secondaryChain, gasPerTxLabel, campaignGasUsd, isMainnetOnly } =
    estimateChainCosts(p.chains);
  const cat = categoryKey(p.category);
  const lower = cat.toLowerCase();

  let capitalLabel = "$300-$1,000";
  if (p.tvl_usd >= 500_000_000) capitalLabel = "$500-$2,500";
  else if (p.tvl_usd < 25_000_000) capitalLabel = "$150-$500";

  const totalCostUsd = isMainnetOnly ? campaignGasUsd + 15 : campaignGasUsd + 6;
  const steps: string[] = [];

  if (lower.includes("dex") || lower.includes("amm") || lower.includes("liquidity")) {
    steps.push(
      `Open a ${capitalLabel} concentrated or stable LP position on ${primaryChain} and hold at least 21 days; setup takes ~10 min and costs ${gasPerTxLabel} in gas.`,
    );
    steps.push(
      `Execute 2-3 swaps per week ($50-$250 each) through the main router over 4 weeks to build unique active-day streaks (~3 min/week, ${gasPerTxLabel} per swap).`,
    );
    steps.push(
      `Register the same wallet in the protocol's points or fee-rebate tracker and avoid cycling wash trades; onboarding takes ~5 min with zero extra lockup.`,
    );
    steps.push(
      `Maintain at least $150 net liquidity across one full snapshot epoch (~5 min weekly check; budget ~$${totalCostUsd} total gas and slippage).`,
    );
  } else if (lower.includes("lend") || lower.includes("borrow") || lower.includes("credit") || lower.includes("cdp")) {
    steps.push(
      `Supply ${capitalLabel} of blue-chip collateral (ETH, BTC, or major stablecoin) on ${primaryChain} and hold 30+ days (~8 min setup, ${gasPerTxLabel} gas).`,
    );
    steps.push(
      `Borrow 25%-40% against collateral (stay below 65% LTV) and repay after 7-14 days to log both supply and borrow volume (~6 min, ${gasPerTxLabel} gas x2).`,
    );
    steps.push(
      `Monitor health factor weekly and compound/claim accrued yield once per epoch (~5 min/week; total campaign gas budget ~$${totalCostUsd}).`,
    );
    if (secondaryChain) {
      steps.push(
        `Place a $100-$250 satellite deposit on ${secondaryChain} so multi-chain activity is recorded on the same wallet (~6 min, ~$1-$4 fee).`,
      );
    } else {
      steps.push(
        `Participate in governance or risk-parameter snapshots with the deposit wallet before closing positions (~5 min, $0-$2 fee).`,
      );
    }
  } else if (lower.includes("deriv") || lower.includes("perp") || lower.includes("option")) {
    steps.push(
      `Deposit ${capitalLabel} margin on ${primaryChain} and keep at least 30% idle in the vault or HLP-style pool for 21 days (~10 min setup, ${gasPerTxLabel} deposit).`,
    );
    steps.push(
      `Trade organic taker/maker volume weekly ($500-$2,000 notional across 3-4 sessions) while capping net fee drag under ~$${totalCostUsd} (~15 min/week).`,
    );
    steps.push(
      `Use limit orders to earn maker points and hold positions through funding intervals rather than instant round-trips (~5 min/week review).`,
    );
    steps.push(
      `Leave margin active through the end of the current reward epoch and verify leaderboard rank (~4 min check, zero withdrawal penalty).`,
    );
  } else if (lower.includes("yield") || lower.includes("vault") || lower.includes("farm") || lower.includes("rwa")) {
    steps.push(
      `Deposit ${capitalLabel} into the highest-TVL native vault on ${primaryChain} for 30+ days to earn time-weighted points (~8 min setup, ${gasPerTxLabel} gas).`,
    );
    steps.push(
      `Split ~25% of capital into a secondary strategy or lock-boosted tranche if exit liquidity is under 7 days (~5 min, ${gasPerTxLabel} gas).`,
    );
    steps.push(
      `Claim or compound rewards biweekly rather than daily so gas stays under ~$${totalCostUsd} across the month (~4 min every 2 weeks).`,
    );
    steps.push(
      `Verify vault receipt tokens remain held in the original wallet (avoid unverified transfers) through snapshot window (~3 min weekly check).`,
    );
  } else {
    steps.push(
      `Deploy ${capitalLabel} into the protocol's core ${cat} product on ${primaryChain} and sustain the balance for 21-30 days (~10 min setup, ${gasPerTxLabel} gas).`,
    );
    steps.push(
      `Log 2 organic interactions per week (transfers, mints, or swaps of $50-$200) across distinct calendar weeks (~5 min/week, ${gasPerTxLabel} each).`,
    );
    steps.push(
      `Opt into the official points/referral profile and complete one secondary feature or governance vote (~6 min, ~$0-$3 fee).`,
    );
    steps.push(
      `Keep total gas plus spread budget under ~$${totalCostUsd} and avoid exiting right before rumored epoch cutoffs (~4 min weekly audit).`,
    );
  }

  const minutes = 35 + (secondaryChain ? 10 : 0);
  let verdict = "Selective / Monitor";
  if (isWatchlist && expectedUsd >= 15_000_000) verdict = "Watchlist Core";
  else if (expectedUsd >= 10_000_000 && odds >= 0.42) verdict = "High-Conviction Farm";
  else if ((p.change_7d ?? 0) >= 8 || (expectedUsd / Math.max(1, totalCostUsd) >= 400_000 && odds >= 0.25)) {
    verdict = "Asymmetric Early";
  }

  return {
    steps: steps.slice(0, 12),
    costUsd: totalCostUsd,
    capitalLabel,
    minutes,
    verdict,
  };
}

export function buildBoard(snapshot: Snapshot, targetRows: number): BoardResult {
  const asOf = snapshot.as_of;
  const watchSet = new Set(snapshot.watchlist);

  // Strict separation: trading peers, traded tokens missing mcap, derivatives/CEX, and pre-token launch pool.
  const tokenPeers = snapshot.protocols.filter(
    (p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0,
  );
  const tradedNoMcap = snapshot.protocols.filter(
    (p) => p.mcap_usd === null && (p.gecko_id !== null || p.symbol !== null),
  );
  const excludedDerivatives = snapshot.protocols.filter(
    (p) => !hasRecordedToken(p) && !watchSet.has(p.slug) && isDerivativeOrCex(p),
  );
  const preToken = snapshot.protocols.filter(
    (p) => watchSet.has(p.slug) || (!hasRecordedToken(p) && !isDerivativeOrCex(p)),
  );

  // 5-fold cross-validation on tokenPeers to choose ridge lambda & record honest CV metrics.
  const folds = 5;
  const lambdas = [1, 2, 4, 8, 16, 32];
  let bestLambda = 8;
  let bestRmse = Infinity;
  let bestPreds: number[] = [];
  let bestBaseRmse = Infinity;

  if (tokenPeers.length >= 15) {
    for (const lam of lambdas) {
      let sqErr = 0;
      let baseSqErr = 0;
      const preds = new Array(tokenPeers.length);
      for (let f = 0; f < folds; f++) {
        const train = tokenPeers.filter((p) => foldOf(p.slug, folds) !== f);
        const testIdx = tokenPeers
          .map((p, i) => ({ p, i }))
          .filter((x) => foldOf(x.p.slug, folds) === f);
        if (train.length < 5 || testIdx.length === 0) continue;
        const m = fitWeightedRidge(train, asOf, lam, 2);
        const trainLogRatio = median(train.map((p) => LOG((p.mcap_usd as number) / p.tvl_usd)));
        for (const { p, i } of testIdx) {
          const yTrue = LOG(p.mcap_usd as number);
          const pr = predictConditional(m, p, asOf).logValue;
          preds[i] = pr;
          sqErr += (yTrue - pr) ** 2;
          const basePr = LOG(p.tvl_usd) + trainLogRatio;
          baseSqErr += (yTrue - basePr) ** 2;
        }
      }
      const rmse = Math.sqrt(sqErr / tokenPeers.length);
      if (rmse < bestRmse) {
        bestRmse = rmse;
        bestLambda = lam;
        bestPreds = preds;
        bestBaseRmse = Math.sqrt(baseSqErr / tokenPeers.length);
      }
    }
  }

  const ridgeModel = fitWeightedRidge(tokenPeers, asOf, bestLambda, 2);

  // CV diagnostics: R² and median APE on out-of-fold predictions.
  let cvR2 = 0;
  let cvMedApe = 0;
  if (bestPreds.length === tokenPeers.length && tokenPeers.length >= 15) {
    const ys = tokenPeers.map((p) => LOG(p.mcap_usd as number));
    const yMean = ys.reduce((a, b) => a + b, 0) / ys.length;
    let ssTot = 0;
    let ssRes = 0;
    const apes: number[] = [];
    for (let i = 0; i < ys.length; i++) {
      ssTot += (ys[i] - yMean) ** 2;
      ssRes += (ys[i] - bestPreds[i]) ** 2;
      apes.push(Math.abs(Math.exp(bestPreds[i]) - Math.exp(ys[i])) / Math.exp(ys[i]));
    }
    cvR2 = ssTot > 0 ? Math.round((1 - ssRes / ssTot) * 1000) / 1000 : 0;
    cvMedApe = Math.round(median(apes) * 1000) / 1000;
  } else {
    bestRmse = 0;
    bestBaseRmse = 0;
  }

  // Fit token-odds logistic model on non-derivative protocols + CV Brier score.
  const oddsPool = snapshot.protocols.filter((p) => !isDerivativeOrCex(p) || watchSet.has(p.slug));
  let brierSum = 0;
  if (oddsPool.length >= 20) {
    for (let f = 0; f < folds; f++) {
      const train = oddsPool.filter((p) => foldOf(p.slug, folds) !== f);
      const test = oddsPool.filter((p) => foldOf(p.slug, folds) === f);
      if (train.length < 10 || test.length === 0) continue;
      const om = fitOddsLogistic(train, asOf);
      for (const p of test) {
        const y = hasRecordedToken(p) ? 1 : 0;
        const pr = predictOdds(om, p, asOf);
        brierSum += (y - pr) ** 2;
      }
    }
  }
  const oddsBrier = oddsPool.length > 0 ? Math.round((brierSum / oddsPool.length) * 1000) / 1000 : 0.25;
  const oddsModel = fitOddsLogistic(oddsPool, asOf);

  // Group token peers by category for nearest-TVL comparable lookup.
  const peersByCat = new Map<string, SnapshotProtocol[]>();
  for (const tp of tokenPeers) {
    const k = categoryKey(tp.category);
    const arr = peersByCat.get(k) ?? [];
    arr.push(tp);
    peersByCat.set(k, arr);
  }

  function findComps(p: SnapshotProtocol): PeerComparable[] {
    const pool = peersByCat.get(categoryKey(p.category)) ?? tokenPeers;
    const src = pool.length >= 2 ? pool : tokenPeers;
    const logT = LOG(Math.max(1, p.tvl_usd));
    return [...src]
      .sort((a, b) => {
        const da = Math.abs(LOG(Math.max(1, a.tvl_usd)) - logT);
        const db = Math.abs(LOG(Math.max(1, b.tvl_usd)) - logT);
        if (da !== db) return da - db;
        return a.slug < b.slug ? -1 : 1;
      })
      .slice(0, 3)
      .map((cp) => ({
        slug: cp.slug,
        name: cp.name,
        tvl_usd: cp.tvl_usd,
        mcap_usd: Math.round(cp.mcap_usd as number),
        multiple: Math.round(((cp.mcap_usd as number) / cp.tvl_usd) * 100) / 100,
      }));
  }

  const scored = preToken.map((p) => {
    const cond = predictConditional(ridgeModel, p, asOf);
    const odds = predictOdds(oddsModel, p, asOf);
    const expected = clamp(Math.round(cond.valueUsd * odds), 1, MAX_IMPLIED_USD);
    const multiple =
      ridgeModel.catMultiples.get(categoryKey(p.category)) ?? ridgeModel.globalMultiple;
    return {
      p,
      conditional: cond.valueUsd,
      decomposition: cond.decomposition,
      expected,
      odds,
      multiple,
    };
  });

  // Always cover 100% of watchlist, then add highest expected value below-floor pre-token protocols.
  const scoredBySlug = new Map(scored.map((s) => [s.p.slug, s]));
  const watchScored = snapshot.watchlist
    .map((slug) => scoredBySlug.get(slug))
    .filter((s): s is NonNullable<typeof s> => s !== undefined);
  const nonWatch = scored
    .filter((s) => !watchSet.has(s.p.slug))
    .sort((a, b) => {
      if (b.expected !== a.expected) return b.expected - a.expected;
      if (b.p.tvl_usd !== a.p.tvl_usd) return b.p.tvl_usd - a.p.tvl_usd;
      return a.p.slug < b.p.slug ? -1 : 1;
    });

  const desiredTotal = Math.max(100, targetRows);
  const need = Math.max(0, desiredTotal - watchScored.length);
  const chosen = [...watchScored, ...nonWatch.slice(0, need)];
  chosen.sort((a, b) => {
    if (b.expected !== a.expected) return b.expected - a.expected;
    if (b.conditional !== a.conditional) return b.conditional - a.conditional;
    if (b.p.tvl_usd !== a.p.tvl_usd) return b.p.tvl_usd - a.p.tvl_usd;
    return a.p.slug < b.p.slug ? -1 : 1;
  });

  const rows: BoardRow[] = chosen.map((s, idx) => {
    const isWatch = watchSet.has(s.p.slug);
    const farm = buildFarmPlan(s.p, s.expected, s.odds, isWatch);
    return {
      slug: s.p.slug,
      name: s.p.name,
      rank: idx + 1,
      implied_value_usd: s.expected,
      farm_steps: farm.steps,
      category: s.p.category,
      chains: s.p.chains,
      tvl_usd: s.p.tvl_usd,
      change_7d: s.p.change_7d,
      age_days: ageDays(asOf, s.p.listed_at),
      conditional_value_usd: s.conditional,
      token_odds: Math.round(s.odds * 1000) / 1000,
      watchlist: isWatch,
      peer_multiple: Math.round(s.multiple * 1000) / 1000,
      farm_cost_usd: farm.costUsd,
      farm_capital_usd: farm.capitalLabel,
      farm_minutes: farm.minutes,
      farm_verdict: farm.verdict,
      decomposition: s.decomposition,
      comps: findComps(s.p),
    };
  });

  const watchlistCovered = rows.filter((r) => r.watchlist).length;
  const medianFarmCost = median(rows.map((r) => r.farm_cost_usd));

  return {
    rows,
    stats: {
      total_snapshot: snapshot.protocols.length,
      token_peers: tokenPeers.length,
      traded_no_mcap: tradedNoMcap.length,
      excluded_derivatives: excludedDerivatives.length,
      pre_token: preToken.length,
      watchlist: snapshot.watchlist.length,
      watchlist_covered: watchlistCovered,
      board_rows: rows.length,
      below_floor_rows: rows.length - watchlistCovered,
      global_multiple: Math.round(ridgeModel.globalMultiple * 1000) / 1000,
      median_farm_cost_usd: Math.round(medianFarmCost),
      cv: {
        ridge_lambda: bestLambda,
        cv_folds: folds,
        cv_rmse_ln: Math.round(bestRmse * 1000) / 1000,
        baseline_rmse_ln: Math.round(bestBaseRmse * 1000) / 1000,
        cv_r2: cvR2,
        cv_med_ape: cvMedApe,
        odds_brier: oddsBrier,
        beta_tvl: Math.round(ridgeModel.coef[1] * 1000) / 1000,
        beta_growth: Math.round(ridgeModel.coef[2] * 1000) / 1000,
        beta_age: Math.round(ridgeModel.coef[3] * 1000) / 1000,
        beta_chains: Math.round(ridgeModel.coef[4] * 1000) / 1000,
      },
    },
  };
}
