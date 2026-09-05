# Utilization dataset and next-period forecaster

A seeded generator for twelve monthly workforce-utilization extracts, and a ridge
model that forecasts each person's next-period Util %, backtested against naive
baselines.

```bash
npm install
npm run util:generate   # writes data/utilization.csv (720 rows, deterministic)
npm run util:train      # writes data/utilization-model.json + utilization-forecast.csv
npm run util:test       # 33 self-checks
npm run typecheck
```

The PM-plan head-to-head, which answers whether the model beats forecasting from
PM hour allocations:

```bash
npm run util:plans      # writes data/utilization-plan.csv (660 simulated allocations)
npm run util:compare    # PM plan vs model vs blend, on identical rows
npm run util:sweep      # how good would PM plans have to be? (30-cell sensitivity grid)
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
| `src/lib/utilization/plan.ts` | PM allocation schema, seeded planner simulation, CSV round-trip |
| `src/lib/utilization/headtohead.ts` | Plan feature block, the four forecasters, walk-forward blend |
| `data/utilization-plan.csv` | Simulated PM allocations, 660 rows, snapshotted at forecast time |
| `data/utilization-headtohead.json` | Head-to-head metrics + the 12-draw robustness study |
| `data/utilization-sweep.json` | Sensitivity of the head-to-head to assumed planner quality |
| `docs/utilization-forecast.md` | Full write-up: schema, identities, method, results |
| `docs/utilization-plan-headtohead.md` | PM plan vs model vs blend: protocol, results, caveats |
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

## PM plan vs model vs blend

The forecaster's open question was whether it beats forecasting utilization from
PM hour allocations. With allocations added to the schema, on the same 300
person-periods:

| forecaster | MAE | RMSE | bias | R² | within 5pp |
| --- | --- | --- | --- | --- | --- |
| **blend (plan + model)** | **4.26** | **5.76** | +0.55 | 0.909 | 65.3% |
| model on plan + history | 4.47 | 5.83 | +0.86 | 0.907 | 64.0% |
| model (history only) | 4.56 | 6.07 | -0.13 | 0.899 | 65.3% |
| PM plan (as-is) | 5.17 | 7.30 | +1.90 | 0.854 | 64.7% |

The plan carries information the model does not have, and using both beats
either: adding plans is worth -6.5% MAE, more than four times the model's own
-1.5% edge over "same as last period". But the top two are **not
distinguishable** - over 12 draws of the allocations the blend averages 4.25pp
(sd 0.05) and the plan-augmented model 4.27pp (sd 0.12), each best in 6 of 12.
What survives re-drawing is that using the plan beat the history-only model in
12/12 draws and the raw plan beat it in 0/12.

"The plan loses" is a statement about *assumed planner quality*, so `util:sweep`
varies it. The raw plan wins in 10 of 30 cells, needs foresight >= 0.50, and once
plans are 35% pulled toward target it never wins at any foresight - an unbiased
mediocre plan beats a sharp optimistic one. The blend is best in 29 of 30 cells
and never meaningfully worse than ignoring plans; the plan-*anchored* model is
worse than ignoring plans in 11 cells. Full protocol and caveats, including why
this simulation understates real plan accuracy, in
`docs/utilization-plan-headtohead.md`.

Read `docs/HANDOFF-utilization-forecast.md` before changing the model — it records
which design decisions were forced by backtest results and which alternatives were
already tried and rejected.

## Environment notes

Node runs the `.ts` files directly via type stripping, so **Node >= 22.18 is
required** - that is the release where type stripping stopped needing a flag.
On 22.6-22.17 the same scripts work but must be run with
`--experimental-strip-types`. There is no ts-node, tsx, or build step.

- Imports between these modules **must carry the `.ts` extension** — Node ESM does
  no extension resolution. `allowImportingTsExtensions` is on (safe: `noEmit`).
- **Erasable syntax only** — no `enum`, no `namespace`, no parameter properties.
  Use `import type` for type-only imports or they fail at runtime;
  `verbatimModuleSyntax` enforces this at typecheck time.
