# Token_Lunch_radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/Nakeebv/Token_Lunch_radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/Nakeebv/Token_Lunch_radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Pre-token launch radar and farm-economics terminal for make.fun's [Build the best token launch radar](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) bounty (`token-radar-01`).

## How it works

1. **Snapshot hygiene & separation (`lib/model.ts`)**: Strictly separates trading token peers (`mcap_usd` present), traded tokens missing market cap (`gecko_id` or `symbol` present), and wrapped/staked/bridged/CEX derivatives from genuine pre-token launch candidates while preserving 100% coverage of the competition watchlist.
2. **5-fold CV conditional launch valuation**: Fits a Huber-reweighted ridge regression of `ln(mcap_usd)` on `ln(tvl_usd)`, dampened 7-day TVL momentum, `ln(1 + age_days)`, and chain breadth across token peers (selecting regularization `lambda` via deterministic 5-fold cross-validation), plus empirical-Bayes shrunk category residuals.
3. **Token launch odds**: Fits an L2-regularized logistic launch-rate model over non-derivative protocols using TVL, age, momentum, and shrunk category base rates.
4. **Implied value & costed farm plans**: Ranks pre-token protocols by `expected_usd = conditional_value_usd * token_odds`, attaches nearest token-trading peers from the snapshot, and builds chain-aware gas estimates, capital bands, active time budgets, farm verdicts, and actionable steps.
5. **Static offline dashboard (`dashboard/`)**: Pure HTML/CSS/JS terminal with integrity funnel cards, CV diagnostics, interactive log-TVL vs log-implied-value SVG map, multi-facet filtering/sorting, and a sticky detail inspector.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input ../input/snapshot.json
```

> **Note on kit completion before submitting on `make.fun`**:
> Copy `starter/vendor/makefun-board.ts` from the unpacked `token-radar-01` kit over `vendor/makefun-board.ts`, run `main.ts` against the sealed `input/snapshot.json`, capture `screenshots/` in Chrome/Firefox from `out/dashboard/index.html`, and re-run `main.ts` before bundling & submitting.

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards
