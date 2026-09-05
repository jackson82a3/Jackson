# Utilization forecaster: operating guide

Everything needed to run this, read its output, judge whether to trust it, and
change it without breaking the claims it makes. Written for someone who did not
build it.

If you read only one section, read [§2 What the numbers mean](#2-what-the-numbers-actually-mean)
and [§8 When not to trust it](#8-when-not-to-trust-it).

---

## Contents

1. [What this is, in one page](#1-what-this-is-in-one-page)
2. [What the numbers actually mean](#2-what-the-numbers-actually-mean)
3. [Running it](#3-running-it)
4. [The monthly routine](#4-the-monthly-routine)
5. [Reading the output](#5-reading-the-output)
6. [How it works](#6-how-it-works)
7. [Changing it safely](#7-changing-it-safely)
8. [When not to trust it](#8-when-not-to-trust-it)
9. [Troubleshooting](#9-troubleshooting)
10. [Glossary](#10-glossary)

---

## 1. What this is, in one page

A forecaster for **next period's Util %**, per person, rolled up to cost centre.

It takes a monthly timesheet extract — one row per person per period, with hours
split into direct, indirect and fringe — and predicts each person's utilization
in the period that has not happened yet. It also ships a simulated dataset, so
the whole thing runs end to end with no access to real data.

**The honest summary of its accuracy**: on the panel every published number is
measured on, it is about 1% better than assuming next period equals last period.
**Re-generate the world from a different random seed and that edge disappears.**
Across 20 seeds the model beats the naive baseline on 7, and averages 0.55%
*worse*. The published panel turns out to be the most favourable of the 20.

The one thing that does survive re-drawing is **blending in PM hour
allocations**: that beats the naive baseline on 14 of 20 seeds. The value is in
the extra data, not in the estimator.

So the short version: if you have PM allocations, use the blend. If you do not,
"same as last period" is as good as this model and far simpler. §2.1 has the
numbers.

```bash
npm run util:train   -- --plans allocations.csv   # fits and stores the blend weight
npm run util:predict -- --plans allocations.csv   # forecasts using it
```

The plan file must contain allocations for the **period being forecast**, not
only for periods already closed; `util:predict` refuses rather than quietly
forecasting from history alone if it does not.

| | |
| --- | --- |
| Predicts | Util % for period t+1, per person |
| Trained on | 12 monthly periods, 60 people, 720 rows (synthetic) |
| Out-of-sample MAE | 4.58pp vs 4.63pp for "same as last period" — **on one seed; see §2.1** |
| Across 20 seeds | model beats the baseline on **7/20**; blend with plans on **14/20** |
| Usable horizon | **t+1 only** — it loses to naive baselines at t+2, see §8 |
| Interval | 80% nominal, **74.6% realised** — see §2 |
| Cost-centre rollup | 3.26pp MAE, hours-weighted |
| Dependencies | none (Node's standard library only) |
| Node | >= 22.18 |

---

## 2. What the numbers actually mean

This is the section that stops the model being misused.

### 2.1 The edge over the naive baseline is not reliable

Run `npm run util:seedstudy`. It regenerates the whole world from 20 seeds and
scores the model identically each time. This is the most important table in the
project:

| claim | model | baseline | model better on |
| --- | --- | --- | --- |
| MAE (pp) | 4.033 | 4.011 | 7/20 seeds |
| absolute bias (pp) | 0.669 | 0.664 | 5/20 seeds |
| within 5pp | 0.709 | 0.709 | 10/20 seeds |
| cost-centre rollup MAE (pp) | 2.355 | 2.324 | 8/20 seeds |
| **blend with PM plans (pp)** | **3.939** | 4.011 | **14/20 seeds** |

**None of the history-only model's claims survive re-drawing the world.** Not
level accuracy, not the bias correction, not the 5pp hit rate, not the
cost-centre rollup — every one of them is a coin flip or worse across seeds. The
shipped panel gives +1.0%; the other 19 average −0.63%, and +1.0% is the best of
all 20 draws.

The one claim that does survive is the blend with PM allocations, at 14/20 seeds
and a mean 4.011 → 3.939pp. It is a modest, real effect, and it comes from having
more information rather than from the model being clever.

That is not a reason to distrust the numbers below — they are correctly measured
on the panel they describe. It is a reason not to generalise them. Everything in
the rest of §2 is *about this panel*.

### 2.1.1 What the published panel says, correctly measured

The model is scored by **rolling-origin cross-validation**: to score period 10,
it is fitted only on periods that closed before 10, then asked to predict 10.
That is exactly how it gets used, so the number is not a fit statistic.

| model | MAE | RMSE | bias | R² | within 5pp |
| --- | --- | --- | --- | --- | --- |
| **ridge (this model)** | **4.58** | 6.09 | −0.08 | 0.898 | 65.0% |
| same as last period | 4.63 | 6.10 | −0.33 | 0.898 | 62.0% |
| 3-period moving average | 4.97 | 6.28 | −0.05 | 0.892 | 57.7% |
| person's own mean | 6.19 | 7.92 | −1.18 | 0.828 | 48.3% |
| their target | 9.62 | 13.42 | +3.12 | 0.506 | 36.0% |
| cost-centre mean | 14.61 | 18.29 | −0.28 | 0.082 | 21.7% |

**Read that first column honestly: 4.58 against 4.63 is a 1% improvement** — on
this panel, and per §2.1 not on most others. The bias (−0.08 vs −0.33), the 5pp
hit rate (65.0% vs 62.0%) and the 3.26pp cost-centre rollup look like stronger
claims here, and were presented as such before the seed study existed; they do
not hold up across seeds either.

The penalty is chosen *inside each fold's own past*, not on the folds being
reported. If it is chosen the usual sloppy way — scan the grid, report the best
— the number improves to 4.56pp. That 0.02pp gap is the size of the optimism,
and it is reported in every run so nobody has to wonder.

### 2.2 The 80% interval covers about 75%, and you should know why

The forecast ships an 80% prediction interval of roughly ±7.8pp. **Measured
honestly it covers 74.6%, not 80%.**

"Measured honestly" matters here. Calibrating an interval on a set of residuals
and then measuring coverage on those same residuals reports the fit, not the
coverage. Done that way this interval "covers" 83.7%. Done walk-forward — each
fold's interval calibrated only on folds before it — it covers 74.6%.

The shortfall is almost entirely **one period**:

| fold | half width | coverage |
| --- | --- | --- |
| P09 | 6.0pp | 81.7% |
| **P10** | 5.8pp | **56.7%** |
| P11 | 7.7pp | 83.3% |
| P12 | 7.7pp | 76.7% |

P10 is July — the vacation trough. It is the fold the model is worst on
(6.79pp MAE against ~4 elsewhere), and an interval calibrated on quiet months
under-covers a volatile one. **This is not fixable with one fiscal year of
data**, because with one period per calendar month every validation month is a
month the model has never seen. A second year is the fix.

Three alternatives were tried and are worse:

| method | interval | realised coverage |
| --- | --- | --- |
| Student-t width | ±9.0pp | 87.9% |
| **Gaussian (shipped)** | **±7.8pp** | **74.6%** |
| conformal, symmetric | ±7.3pp | 72.5% |
| conformal, asymmetric | −7.4/+7.0pp | 65.0% |

Conformal is worse. The t-width over-covers with intervals up to ±14pp in early
folds, which is too wide to be useful. Gaussian is kept because it is the
sharpest thing that is not badly wrong. Every run prints all four.

**Practical guidance**: treat the interval as roughly a 75% interval, and widen
your own judgement in months with unusual leave patterns.

### 2.3 The model is at the ceiling of this data

Things tried that did **not** help, all measured through the shipped protocol:

| tried | result |
| --- | --- |
| Huber loss (targets MAE directly) | 4.61 vs 4.58 — worse, wins 2/5 folds |
| Recency weighting (half-life 3) | 4.67 — worse, wins 0/5 folds |
| Both together | 4.66 — worse |
| 34 features incl. job-level and target-type dummies | no better, noisier |
| Hierarchical shrinkage to cost-centre × job-level | 4.515 → 4.496, not worth it |
| Absolute cost-centre / firm levels | produced a fictitious 8pp firm-wide collapse |
| Month sin/cos | cannot be validated on one fiscal year |
| Tuning mean reversion to 0.75 | illusory — it was tuned on the validation set |

Run `npm run util:experiment` to re-check the first three at any time. **The
useful conclusion is that further estimator tuning is not where the gains are** —
and §2.1 sharpens that: the estimator has no reliable edge to tune. A second year
does not help either (`npm run util:secondyear`: neither sin/cos nor month
dummies improve anything, and dummies make it worse). The one input that moves
the number is PM allocations (§2.4).

### 2.4 PM plans beat estimator tuning

Adding PM hour allocations as an input improves MAE by 6.5% — more than six
times the model's own edge over the naive baseline.

| forecaster | MAE |
| --- | --- |
| **blend (plan + model)** | **4.28** |
| model on plan + history | 4.49 |
| model (history only) | 4.58 |
| PM plan used as-is | 5.17 |

The raw plan is *worse* than the model, but it carries information the model
structurally cannot have, so combining them beats either.

**This is the only claim in the project that survives the seed study**, and even
it is more modest than this table suggests: across 20 seeds the blend averages
3.939pp against the baseline's 4.011pp and wins on 14 of 20, an edge of about
1.8% rather than the 6.5% this panel shows. Full protocol, sensitivity analysis,
and the caveats in
[`utilization-plan-headtohead.md`](utilization-plan-headtohead.md).

---

## 3. Running it

```bash
npm install       # no runtime dependencies; TypeScript for typechecking only
npm run util:test # 101 self-checks - run this first
```

| Command | What it does | Writes |
| --- | --- | --- |
| `util:generate` | Regenerates the synthetic dataset from its seed | `data/utilization.csv` |
| `util:train` | Fits, backtests, and writes the model + forecast | `utilization-model.json`, `utilization-forecast.csv` |
| `util:predict` | Forecasts from a **saved** model, no retraining | `utilization-forecast.csv` |
| `util:monitor` | Checks a deployed model for drift; **exits non-zero on breach** | `utilization-monitor.json` |
| `util:experiment` | Scores model variants through the shipped protocol | `utilization-experiments.json` |
| `util:horizon` | Measures accuracy at t+1..t+4 against baselines | `utilization-horizons.json` |
| `util:seedstudy` | **Re-runs everything on 20 seeds. Read this before trusting any number.** | `utilization-seedstudy.json` |
| `util:secondyear` | Tests whether a second year unlocks seasonality (it does not) | `utilization-secondyear.json` |
| `util:plans` | Generates simulated PM allocations | `utilization-plan.csv` |
| `util:compare` | PM plan vs model vs blend | `utilization-headtohead.json` |
| `util:sweep` | How good would PM plans have to be? | `utilization-sweep.json` |
| `util:test` | 101 self-checks | — |
| `typecheck` | `tsc --noEmit` | — |

Every command takes `--data <path>` to point at a different extract. `util:train`
takes `--out-dir`; the others take `--out`.

### Pointing it at real data

The CSV must have the 32 columns listed in `src/lib/utilization/types.ts`, in
any order, with `Source.Name` encoding the period as
`FY26_P03_Utilization_2025-12.csv`. Then:

```bash
npm run util:train -- --data /path/to/real-extract.csv
```

Validation runs first and will tell you what is wrong with the file before
anything is fitted to it. **Fix the extract, not the tolerance.**

---

## 4. The monthly routine

Once a period closes:

```bash
# 1. Has the deployed model held up? Exits non-zero if not.
npm run util:monitor -- --data new-extract.csv

# 2. Forecast the coming period from the reviewed model. Add --plans if you have
#    allocations for the coming period - that is the version with evidence
#    behind it - and --roster to cover joiners.
npm run util:predict -- --data new-extract.csv --plans allocations.csv

# 3. Only if monitoring flagged drift, or on a fixed schedule (see below):
npm run util:train -- --data new-extract.csv --plans allocations.csv
```

**Why predict and train are separate.** The model is a reviewed artifact. If
every run refitted it, the model would change silently whenever new data
arrived, and a forecast nobody reviewed would go out under the authority of one
that was reviewed. `util:predict` cannot change the model; `util:train` is a
deliberate act.

**When to retrain.** Retrain when monitoring flags drift, when the roster changes
materially (a reorganisation, an acquisition), or on a fixed cadence — quarterly
is reasonable — so the model does not drift far from current conditions between
incidents. Always read the training report before shipping the new artifact:
if MAE has moved materially, find out why before deploying it.

### What monitoring tells you

Two signals, deliberately, because they fail at different times:

- **Accuracy drift** compares predictions against outcomes that have landed. It
  is the real answer and it always lags by at least one period.
- **Feature drift** compares today's inputs against the distribution the model
  was standardised on. It needs no outcomes, so it fires first — a cost-centre
  reorganisation or a leave-policy change moves this before any accuracy number.

A worked example, from the self-checks: a model fitted on P1–P9 claims 3.42pp
MAE. Checked against the P10–P12 it never saw, it delivers **5.29pp** — 55%
worse — with interval coverage falling from 82% to 66%. `firm_util_momentum` was
already 1.02 sd adrift before those outcomes landed.

A window that overlaps the training range never raises drift, because in-sample
accuracy cannot evidence it. The tool says so rather than quietly flattering the
model.

---

## 5. Reading the output

### `utilization-forecast.csv`

One row per person on the roster. The columns that need explanation:

| Column | Meaning |
| --- | --- |
| `Forecast Util %` | The point forecast for the coming period |
| `Forecast Low 80` / `High 80` | The interval — realistically ~75%, see §2.2 |
| `Forecast Variance` | Forecast minus target. Negative = expected to miss |
| `Method` | `blend`, `model`, `short_history`, or `cold_start` — **read this** |
| `Periods Of History` | Closed periods behind the row |
| `Basis` | For a fallback row, what it was derived from |

`Last Util %` is empty on a `cold_start` row, because there is no last
observation. It is not zero, and it is not `NaN`.

**`Method` is the column people skip and shouldn't.** `blend` is the model
combined with that person's PM allocation, and is the only method with evidence
behind it across panels (§2.1); the `Basis` column gives the mix. `model` is the
history-only forecast, used where no allocation was supplied for that person.
`short_history` means the person has one or two periods and the row is
just their last observation carried forward. `cold_start` means there was no
usable history at all and the row is a peer-group median — barely a forecast, and
the `Basis` column says which peer group and how many people it covered.

`cold_start` only appears when you supply a **roster** (`--roster`), because the
timesheet contains only people who have already charged time — a joiner starting
next period is invisible to it, and they are exactly who a resourcing question is
about:

```bash
npm run util:predict -- --data extract.csv --roster roster.csv
```

The roster needs `Cost Center` and `Person Name`; `Job Level`, `Target Type`,
`Util % Target` and `Expected Avail Hours` sharpen the row if present. The peer
group widens until it describes at least three distinct people, so a cost centre
holding one person at a given level does not have that person's recent luck
reported as a cohort.

Anyone missing from the final period is **excluded and listed by name**, not
silently dropped. That is deliberate: a forecast that quietly covers 54 of 60
people produces a headcount discrepancy nobody can explain.

### The cost-centre rollup

Hours-weighted, and **more accurate than the person-level numbers** (3.26pp vs
4.58pp) because individual errors partly cancel within a cost centre. If you are
making a staffing decision at cost-centre level, use the rollup — do not average
the person-level numbers yourself and do not read too much into any one person's
row.

---

## 6. How it works

### The prediction

```
forecast = (that person's last Util %) + ridge(features)
```

The model predicts the **change** from last period, not the level. This is not a
stylistic choice — it was forced by the backtest. Fitting the level made
shrinkage pull toward the firm mean, and the model **lost** to the naive
baseline (5.00 vs 4.63). Anchored on last period, shrinkage falls back to the
strongest naive baseline instead. The intercept is penalised for the same
reason: leaving it free left a +1.2 to +1.4pp drift term that did not carry
forward.

### The features

18, all computed strictly from periods at or before the origin. Four groups:

- **Mean reversion** — how far this person sits from their own run rate.
- **Where their non-billable time went** — bench, training, leadership,
  opportunity, fringe, overtime, as shares of available hours.
- **Cross-sectional position** — person vs cost centre, cost centre vs firm.
- **Momentum** — differences, never levels.

**Absolute cost-centre and firm levels are deliberately excluded.** With one
period per calendar month they act as a period label; the first version leaned
on `firm_util_lag0` and produced a fictitious ~8pp firm-wide collapse — pure
extrapolation. Only *relative* terms survive.

### The evaluation

- **Rolling origin.** Fold k trains on everything before k, predicts k.
- **Nested penalty selection.** Fold k's penalty comes from an inner rolling
  origin over the periods before k, so nothing about a fold touches the period
  it is scored on.
- **Walk-forward interval calibration.** Fold k's interval is calibrated only on
  folds before it.
- **Leakage tests.** Feature rows are rebuilt from data truncated to the origin
  and required to come back identical. Three separate invariants are tested this
  way (features, plan features, and the plans themselves).

### The files

| File | Role |
| --- | --- |
| `types.ts` | Record type, CSV columns, the accounting identities |
| `csv.ts` | Parse/serialize; decodes the period from `Source.Name` |
| `generate.ts` | Seeded simulation of the twelve periods |
| `features.ts` | Feature engineering, sample construction |
| `ridge.ts` | Standardized ridge, weighted least squares, Huber IRLS |
| `forecast.ts` | Rolling origin, penalty selection, intervals, forecasting |
| `validate.ts` | Input validation with actionable errors |
| `model-io.ts` | Versioned artifact read/write with compatibility refusal |
| `monitor.ts` | Accuracy and feature drift |
| `plan.ts` | PM allocation schema and simulation |
| `headtohead.ts` | Plan vs model vs blend |

---

## 7. Changing it safely

The claims in §2 are only true because of specific protocol decisions. These are
the ones that are easy to break without noticing.

**Run `npm run util:test` before and after any change.** 101 checks, and the ones
that matter most are the leakage invariants.

### Rules that are not style preferences

1. **Never select anything on the rows you report.** Penalties, blend weights,
   interval widths, feature sets — all chosen from data that closed before the
   fold being scored. The handoff records a case where tuning mean reversion on
   the validation set produced an apparent 4.465pp that evaporated when fitted
   honestly.
2. **Never let a feature see its own future.** Add a feature, and add it to the
   truncation test. A feature that quietly uses the target period will look
   fantastic and be worthless.
3. **Do not undo the change-anchoring or the penalised intercept** without
   re-running the backtest. Both were forced by measurement; see §6.
4. **Do not add absolute period-level features.** See §6.
5. **Regenerate committed data if the generator changes**, and expect CI to fail
   if you forget — every documented number is a number about those files.

### Adding a model variant

Add an entry to `ESTIMATORS` in `forecast.ts`, then `npm run util:experiment`.
It runs candidates through the *same* protocol as the shipped model rather than
a copy of it, which matters: the easiest way to invent an improvement is to
evaluate the candidate slightly differently from the incumbent.

Ship a variant only if it wins on the honest number **and** on most folds. The
harness prints both, plus the worst single-fold regression, because a variant
that wins on average while losing on most folds has won a lottery.

### Changing the feature set

The model artifact records its feature list, and loading **refuses** an artifact
whose list differs from what the code builds. This is deliberate: a ridge model
will happily multiply the wrong coefficient by the wrong column and return a
believable number. So change features, then retrain — `util:predict` will stop
you if you forget.

### Changing the artifact format

Bump `MODEL_FORMAT_VERSION` in `model-io.ts` when the shape changes in a way
older readers cannot handle, and note what changed in the comment above it.

---

## 8. When not to trust it

Read this before any decision that affects a person.

- **The model has no reliable edge over "same as last period".** Across 20
  simulated panels it wins on 7 (§2.1). Do not deploy it on the strength of the
  headline figure. If you have PM allocations, the blend is the defensible
  choice; if you do not, the naive baseline is as good and far simpler to
  explain.
- **Every accuracy number is a number about a simulation.** The dataset is
  synthetic — calibrated to be plausible and internally consistent, not to match
  any real firm. On real data the model-vs-naive gap could go either way. Nothing
  here has been validated against a real workforce.
- **One period ahead only — and this is now measured, not assumed.** Run
  `npm run util:horizon`. The model beats the best naive baseline at t+1 by 1.0%
  and **loses at t+2 and t+3**:

  | horizon | model MAE | best baseline | margin |
  | --- | --- | --- | --- |
  | t+1 | 4.58 | last period, 4.63 | **+1.0%** |
  | t+2 | 5.95 | 3-period MA, 5.80 | −2.7% |
  | t+3 | 6.30 | last period, 6.16 | −2.3% |
  | t+4 | 6.35 | last period, 6.81 | +6.7% |

  **Use it for t+1 only.** At t+2 and beyond a naive baseline is at least as
  good, so the model adds complexity and no accuracy. The t+4 figure comes after
  the model has already lost at a shorter horizon — with 300 rows per horizon
  these margins move a couple of points either way, so read it as noise rather
  than as range. Do not use this for quarterly or annual planning.
- **One fiscal year.** Seasonal effects cannot be learned or validated. The July
  vacation trough is the clearest case, and it is exactly where the model and its
  intervals are worst.
- **Not for individual performance management.** A ±7.8pp interval that really
  covers ~75% is not a basis for a conversation about one person's performance.
  It is a planning tool for capacity at cost-centre level. The person-level rows
  exist to be aggregated.
- **No project pipeline.** The model has no visibility of backlog, project end
  dates, or new awards — which is what actually drives future direct hours. This
  is precisely the information PM allocations carry, and why the blend helps.
- **Fallback rows are not forecasts.** `short_history` and `cold_start` rows get
  the naive baseline's error width, which is measured on people who *have* full
  history. A person with two periods behind them is genuinely less predictable
  than that, so those intervals are likely optimistic.
- **The roster is static in this data.** No joiners, leavers or transfers, so the
  fallback paths are exercised by tests rather than by evidence. A `cold_start`
  row in particular is a peer-group median wearing a forecast's clothes; treat it
  as a placeholder until that person has three periods of their own.

---

## 9. Troubleshooting

**`Refusing to train on data with errors`**
Validation found something that makes training unsound. The report names each
issue with a code and examples. Common ones: `duplicate_person_period` (usually a
mid-period transfer appearing twice), `identity_violation` (the hours do not add
up — a real extract problem, not a tolerance problem), `missing_periods` (lagged
features would compare across the gap as though adjacent).

**`The saved model was trained on a different feature set`**
The code's features have changed since the artifact was written. Retrain:
`npm run util:train`. This is the guard working.

**`Model format 1 cannot be read`**
An artifact from before versioning. Retrain.

**`No sample reaches period 8; not enough history`**
Training needs at least 9 periods so the rolling origin has folds to score.
Forecasting from an existing model does not — use `util:predict`.

**Monitoring exits 1**
Either a threshold was breached, or neither signal could be measured. Check
which. Accuracy drift means retrain and re-review. Feature drift alone means
something changed in the inputs — find out what before assuming the model is
still valid. Tune with `--mae-ratio` and `--drift-sds` if the defaults are wrong
for your data.

**`Feature drift: NOT MEASURED`**
There was not enough contiguous history to build a feature row (it needs four
periods). This is reported loudly and exits non-zero on purpose: a monitor that
quietly reports "no drift" when it checked nothing is worse than no monitor.

**CI fails on "Committed data is reproducible from its seed"**
The generator changed and the committed dataset no longer matches. Run
`npm run util:generate && npm run util:plans` and commit the result — and
re-check every number in the docs, because they are numbers about those files.

**`SyntaxError` running a script**
Node is below 22.18. Upgrade, or run with `--experimental-strip-types` on
22.6–22.17.

---

## 10. Glossary

| Term | Meaning |
| --- | --- |
| **Util %** | Direct hours ÷ available hours × 100 |
| **Available hours** | Total hours minus fringe (leave, statutory, wellness) |
| **Direct hours** | Hours charged to billable project work |
| **Indirect hours** | Paid, non-billable: admin, bench, business development, management, opportunity, training |
| **Fringe** | Leave, statutory and discretionary holiday, wellness |
| **Billability %** | Direct ÷ *total* hours — differs from Util % by the leave in the denominator |
| **Period** | One fiscal month; P1 is October in this FY26 calendar |
| **Origin** | The last period a forecast is allowed to see |
| **Rolling origin** | Fitting on everything before period k and predicting k, walking k forward |
| **Nested selection** | Choosing a hyperparameter inside the training window, never on the reported fold |
| **Walk-forward calibration** | Calibrating an interval only on folds before the one it is measured on |
| **MAE** | Mean absolute error, in percentage points |
| **Bias** | Mean signed error; positive means the model forecasts too high |
| **Anchor** | The value a prediction is a correction to — here, last period's Util % |
| **Shrinkage** | The ridge penalty pulling coefficients toward zero, and so the forecast toward its anchor |
