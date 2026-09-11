# Handoff: utilization dataset and forecaster

> **Port note.** This code now lives standalone in `jackson82a3/Jackson`, extracted
> from the OptionsLab repo referenced below. The dataset, model and every measured
> number in §5 reproduce exactly after the move. Two §3 environment facts no longer
> apply here: there is no Next.js app and no CommonJS config, so `package.json` sets
> `"type": "module"` and the npm scripts do **not** pass
> `--disable-warning=MODULE_TYPELESS_PACKAGE_JSON`. Everything else below stands as
> written — see `README.md` for the current entry point.
>
> **§6 is now built.** The open question below — does this beat forecasting from
> PM hour allocations? — has been answered on simulated allocations, along with a
> sensitivity sweep over how good planners would have to be for the answer to
> flip. See `docs/utilization-plan-headtohead.md`. The §6 caveats were carried
> into it: plans are snapshotted at forecast time and a self-check enforces it,
> and the absence of any self-fulfilment effect is stated as a limitation rather
> than quietly simulated away. §7's other limitations still stand.

Everything needed to continue this work in a fresh session. The code is on
GitHub; this file carries the reasoning, the dead ends, and the open question.

- **Repo**: `https://github.com/jackson82a3/OptionsLab`
- **Branch**: `claude/utilization-forecast-training-myyfax`
- **Commit**: `b2448c6` — *feat: twelve-period utilization dataset and next-period forecaster*
- **No PR opened.**

```bash
git clone https://github.com/jackson82a3/OptionsLab
cd OptionsLab && git checkout claude/utilization-forecast-training-myyfax
npm install
npm run util:generate && npm run util:train && npm run util:test
```

## 1. What this is

Two things, both self-contained and **not wired into the Next.js options-trading
app** that occupies the rest of the repo:

1. A seeded generator for twelve monthly workforce-utilization extracts.
2. A ridge model that forecasts each person's next-period Util %, backtested
   against naive baselines.

The original request: *"Using all these data identifiers, self generate a set of
12 periods data. Use the dataset to train a utilization forecast tool."* The
identifiers were given as five groups (file/org, utilization, billability, core
hours, indirect/nonproductive, leave/fringe) and are reproduced exactly as CSV
columns, in the order given, in `src/lib/utilization/types.ts`.

## 2. Map of the code

| File | Role |
| --- | --- |
| `src/lib/utilization/types.ts` | Record type, CSV column order, `verifyRecord` (accounting identities) |
| `src/lib/utilization/csv.ts` | Parse/serialize; decodes the period out of `Source.Name` |
| `src/lib/utilization/generate.ts` | Seeded simulation of the twelve periods |
| `src/lib/utilization/features.ts` | Feature engineering; `buildTrainingSamples` / `buildForecastSamples` |
| `src/lib/utilization/ridge.ts` | Standardized ridge via normal equations + Gaussian elimination |
| `src/lib/utilization/forecast.ts` | Rolling-origin CV, penalty selection, baselines, forecast, rollup |
| `scripts/generate-utilization-data.ts` | `npm run util:generate` |
| `scripts/train-utilization-forecast.ts` | `npm run util:train` |
| `scripts/test-utilization.ts` | `npm run util:test` (17 checks) |
| `data/utilization.csv` | The dataset, 720 rows |
| `data/utilization-model.json` | Trained coefficients + all metrics |
| `data/utilization-forecast.csv` | P13 forecast per person with 80% intervals |
| `docs/utilization-forecast.md` | Full write-up: schema, identities, method, results |

## 3. Environment facts that will bite you

- **Node runs the `.ts` files directly** (type stripping, Node >= 22.6). There is
  no ts-node, tsx, or build step for these scripts. Consequences:
  - Imports between these modules **must carry the `.ts` extension** — Node ESM
    does no extension resolution. `tsconfig.json` has
    `allowImportingTsExtensions: true` (safe: the config is `noEmit`).
  - **Erasable syntax only** — no `enum`, no `namespace`, no parameter
    properties. Use `import type` for type-only imports or they fail at runtime.
  - The npm scripts pass `--disable-warning=MODULE_TYPELESS_PACKAGE_JSON`.
    Do **not** add `"type": "module"` to `package.json` to silence it — that
    would break `next.config.js` and `postcss.config.js`, which are CommonJS.
- `npx tsc --noEmit` and `npm run build` were both clean at `b2448c6`.

## 4. Design decisions, and the two that actually mattered

Both were forced by backtest results, not chosen up front. **Do not undo them
without re-running the backtest.**

### 4.1 Fit the change, and penalize the intercept

The model predicts `forecast = last Util % + ridge(features)`, and the intercept
is penalized alongside the coefficients (`penalizeIntercept` in `ridge.ts`).

- Fitting the **level** instead of the change made shrinkage pull toward the firm
  mean, and the model **lost to the naive baseline**: MAE 5.00 vs 4.63. Anchoring
  on last period makes shrinkage fall back to that baseline instead.
- Leaving the intercept **unpenalized** left a drift term (+1.2 to +1.4pp,
  fitted on past periods) that did not carry forward: 4.63 → 4.56 MAE when
  penalized. With standardized (centered) columns the penalized intercept is
  exactly `mean(y) / (1 + lambda)`.

### 4.2 No absolute cost-center/firm levels, no month dummies

With one period per calendar month, `firm_util_lag0` is effectively a period
label. The first version leaned on it (largest coefficient in the model) and
produced a **fictitious ~8pp firm-wide collapse** for P13 — pure extrapolation.
Only *relative* terms survive: person vs cost center, cost center vs firm, and
momentum differences.

Month sin/cos was dropped for a related reason: under rolling-origin CV on a
single fiscal year, **every validation month is one the model has never seen**,
so month effects can be neither learned nor validated. This is the main thing a
second year of data would unlock.

### 4.3 Things already tried that did *not* help

- Larger feature set (34 features incl. job-level and target-type dummies,
  billability terms, absolute levels): no better than the pruned 18, and noisier.
- Hierarchical shrinkage of the person mean toward a cost-center × job-level mean
  (tested offline, `k` = 0/1/2/4/8): MAE 4.515 → 4.496. Not worth the complexity.
- Sweeping the mean-reversion coefficient to 0.75 looks like it gives MAE 4.465,
  **but that was tuned on the validation set** — fitted honestly on training
  folds the coefficient comes out near 0.9 and the gain mostly disappears. Don't
  be fooled by this if you re-derive it.

## 5. Measured results (at `b2448c6`)

Rolling-origin CV, folds = target periods 8–12, 300 person-periods, Util % in pp:

| model | MAE | RMSE | bias | R² | within 5pp |
| --- | --- | --- | --- | --- | --- |
| **ridge** | **4.56** | **6.07** | −0.13 | 0.899 | 65.3% |
| last period | 4.63 | 6.10 | −0.33 | 0.898 | 62.0% |
| 3-period MA | 4.97 | 6.28 | −0.05 | 0.892 | 57.7% |
| person mean | 6.19 | 7.92 | −1.18 | 0.828 | 48.3% |
| target | 9.62 | 13.42 | 3.12 | 0.506 | 36.0% |
| cost-center mean | 14.61 | 18.29 | −0.28 | 0.082 | 21.7% |

Penalty `lambda = 3` (interior optimum on a 0.01–100 grid). Hours-weighted
cost-center rollup: MAE 3.22pp over 30 cost-center periods. 80% interval ±7.8pp,
realised coverage 83.0%. Worst fold is P10 (July, MAE 6.73) — the vacation trough
is a month effect one year cannot teach.

**State the gain honestly**: 1.5% MAE over "same as last period". The bias
reduction, the 5pp hit rate, and the cost-center rollup are the stronger claims.

## 6. The open question (this is where the work goes next)

The user's real process forecasts utilization from **PM hour allocations to
projects**. They asked whether this model beats that. The answer given, and it
should not be softened:

> No, not as it stands, and there is no measurement claiming otherwise. The
> backtest compared against naive statistical baselines, not against PM plans —
> the dataset has no planned-hours column.

PM allocations carry forward information the model structurally cannot see
(project end dates, new awards, client slips, planned leave). The model carries
what PMs lack (optimism bias correction, consistency, calibrated intervals). The
model is also **one period ahead by construction** — beyond 2–3 periods it decays
toward the person-mean baseline, 6.19pp MAE here.

### Proposed next build (offered, not yet started)

Add planned hours to the schema and run a **three-way head-to-head**:

1. Extend `generate.ts` to emit a PM allocation per person-period, simulated with
   realistic **optimism bias** (plans book toward target) and **staleness**
   (allocations not updated after a slip).
2. Add a `planned_util`-style feature block, and a model variant fitted on
   `(PM plan, history) → actual` — i.e. a bias-correction layer on the plan.
3. Score **PM plan vs model vs blend** on identical rows with the existing
   rolling-origin protocol.

Expected shape of the result: the plan wins on level, the model wins on bias, the
blend beats both. Two caveats to carry into it:

- On real data, allocations must be **snapshotted at forecast time**. Scoring
  later-revised allocations invents accuracy that never existed.
- Allocations are partly **self-fulfilling** — people charge to what they are
  booked to — so a plan can look accurate because it caused the outcome.

## 7. Other known limitations

- Static roster: no joiners, leavers or transfers mid-year, so the forecaster is
  never asked about a person with no history. Real data will have this;
  `buildForecastSamples` currently skips anyone with < 3 periods or who is absent
  from the last period.
- Linear, person-level. No project pipeline or backlog data — which is what
  actually drives future direct hours.
- The data is synthetic: calibrated to be plausible and internally consistent,
  not to match any real firm. Every accuracy number above is a number *about this
  simulation*. On real data the model-vs-naive gap could go either way.
- No UI. Nothing under `src/app` or `src/components` imports any of this. Wiring a
  tab into the options-trading dashboard was deliberately not done — different
  domain, and it would have put the Next build at risk for no gain.
