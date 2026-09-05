# Utilization dataset and next-period forecaster

A seeded generator for twelve monthly workforce-utilization extracts, and a ridge
model that forecasts each person's next-period Util %, backtested against naive
baselines.

```bash
npm install
npm run util:generate   # writes data/utilization.csv (720 rows, deterministic)
npm run util:train      # writes data/utilization-model.json + utilization-forecast.csv
npm run util:test       # 17 self-checks
npm run typecheck
```

## Layout

| File | Role |
| --- | --- |
| `src/lib/utilization/types.ts` | Record type, CSV column order, `verifyRecord` (accounting identities) |
| `src/lib/utilization/csv.ts` | Parse/serialize; decodes the period out of `Source.Name` |
| `src/lib/utilization/generate.ts` | Seeded simulation of the twelve periods |
| `src/lib/utilization/features.ts` | Feature engineering; `buildTrainingSamples` / `buildForecastSamples` |
| `src/lib/utilization/ridge.ts` | Standardized ridge via normal equations + Gaussian elimination |
| `src/lib/utilization/forecast.ts` | Rolling-origin CV, penalty selection, baselines, forecast, rollup |
| `data/utilization.csv` | The dataset, 720 rows |
| `data/utilization-model.json` | Trained coefficients + all metrics |
| `data/utilization-forecast.csv` | P13 forecast per person with 80% intervals |
| `docs/utilization-forecast.md` | Full write-up: schema, identities, method, results |
| `docs/HANDOFF-utilization-forecast.md` | Design rationale, dead ends, and the open question |

## Results

Rolling-origin CV, folds = target periods 8-12, 300 person-periods, Util % in pp:

| model | MAE | RMSE | bias | R² | within 5pp |
| --- | --- | --- | --- | --- | --- |
| **ridge** | **4.56** | **6.07** | -0.13 | 0.899 | 65.3% |
| last period | 4.63 | 6.10 | -0.33 | 0.898 | 62.0% |
| 3-period MA | 4.97 | 6.28 | -0.05 | 0.892 | 57.7% |
| person mean | 6.19 | 7.92 | -1.18 | 0.828 | 48.3% |
| target | 9.62 | 13.42 | 3.12 | 0.506 | 36.0% |
| cost-center mean | 14.61 | 18.29 | -0.28 | 0.082 | 21.7% |

Penalty `lambda = 3` (interior optimum on a 0.01-100 grid). Hours-weighted
cost-center rollup: MAE 3.22pp over 30 cost-center periods. 80% interval ±7.8pp,
realised coverage 83.0%.

The honest statement of the gain is **1.5% MAE over "same as last period"**. The
bias reduction, the 5pp hit rate, and the cost-center rollup are the stronger
claims. The data is synthetic, so every number above is a number *about this
simulation*.

Read `docs/HANDOFF-utilization-forecast.md` before changing the model — it records
which design decisions were forced by backtest results and which alternatives were
already tried and rejected.

## Environment notes

Node runs the `.ts` files directly via type stripping, so **Node >= 22.6 is
required**. There is no ts-node, tsx, or build step.

- Imports between these modules **must carry the `.ts` extension** — Node ESM does
  no extension resolution. `allowImportingTsExtensions` is on (safe: `noEmit`).
- **Erasable syntax only** — no `enum`, no `namespace`, no parameter properties.
  Use `import type` for type-only imports or they fail at runtime;
  `verbatimModuleSyntax` enforces this at typecheck time.
