# Utilization forecasting

A self-contained pipeline that (1) generates twelve monthly periods of workforce
utilization data in the shape of a combined timesheet extract, and (2) trains a
next-period Util % forecaster on it.

```bash
npm run util:generate   # writes data/utilization.csv          (720 rows)
npm run util:train      # trains, backtests, writes model + forecast
npm run util:test       # self-checks incl. a look-ahead leakage test
```

Everything is plain TypeScript run directly by Node (>= 22.6, type stripping);
there is no extra toolchain and nothing here is imported by the Next.js app.

## 1. The dataset

`data/utilization.csv` is 60 people x 12 periods (FY26, Oct 2025 - Sep 2026)
across six cost centers, written as one flat table with the columns of a
"combine files" extract. The period lives only in `Source.Name`
(`FY26_P03_Utilization_2025-12.csv`), exactly as it would after combining one
workbook per month; `parseCsv` decodes it back into a period index and month.

| Group | Columns |
| --- | --- |
| File and organizational | Source.Name, Cost Center, Cost Center Name, Person Name, Target Type, Job Level |
| Utilization | Util % YTD, Util %, Util % Target, Util % Variance |
| Billability | Billability % YTD, Billability %, Billability % Target, Billability % Variance |
| Core hours | Total, Avail Total, Direct Total, Indirect Total, OT Hours |
| Indirect / nonproductive | Admin, Bench, Bus Dev, Mngmt, Opportunity, Training, Other, Unpaid OT |
| Leave and fringe | Fringe Total, Stat & Disc, Vacation, Wellness, Unpaid Regular |

Every row satisfies these identities, and both CLIs refuse to run on data that
does not (`verifyRecord` in `src/lib/utilization/types.ts`):

```
Total          = Direct Total + Indirect Total + Fringe Total
Avail Total    = Total - Fringe Total
Indirect Total = Admin + Bench + Bus Dev + Mngmt + Opportunity + Training + Other
Fringe Total   = Stat & Disc + Vacation + Wellness
Util %         = Direct Total / Avail Total * 100
Billability %  = Direct Total / Total * 100
Variance       = actual - target        (both metric families)
YTD            = cumulative Direct / cumulative Avail (resp. Total), fiscal year to date
```

`OT Hours` are paid overtime already inside Direct Total. `Unpaid OT` and
`Unpaid Regular` are memo hours: unpaid, so outside Total.

### How the numbers are generated

`generateDataset` is a seeded simulation (mulberry32, default seed `20260901`),
so the twelve files are reproducible. Utilization is drawn first and the hour
split is derived from it:

- **Level and cost-center effects** - base utilization by job level (L1-L5) and a
  per-cost-center offset; a small share of staff sit on Overhead targets.
- **Person effect** - a persistent individual offset, plus per-person volatility.
- **Persistence** - an AR(1) shock (rho 0.55) so a good or bad period carries over.
- **Firm-wide demand** - one shared AR(1) factor (rho 0.7) that moves everyone
  together, which is what makes cost-center context informative.
- **Seasonality** - a December dip, a July vacation trough, a September year-end push.
- **Bench events** - occasional multi-period slumps, heavier in Digital Solutions
  and at junior levels.
- **New-hire ramps** - a few people open the year training-heavy and below target.
- **Leave** - vacation drawn against an annual budget with monthly weights,
  statutory holidays by month, occasional wellness days and unpaid leave.

Direct hours follow from the drawn utilization; indirect hours are then split
across the categories (admin, management, business development, opportunity,
training, other), with **Bench absorbing whatever slack is left** - the same
mechanism that makes bench hours the tell-tale of a weak period in real data.

The result: mean Util % 70.3 (sd 18.7), 55% of rows below target, within-person
sd 6.3pp against between-person sd 17.3pp, and a visible seasonal profile
(P10/July is the trough at 67.0, P02/November the peak at 72.1).

## 2. The forecaster

**Task.** For each person, predict their Util % in the next period from history
up to and including the current one; roll individual forecasts up to cost-center
and firm level, weighted by available hours.

**Model.** Ridge regression over 18 engineered features, fitted on the *change*
from the person's last observed Util %:

```
forecast = last Util % + ridge(features)
```

Anchoring on last period matters: it makes the penalty shrink the forecast back
toward "same as last period" (the strongest naive baseline on this panel) rather
than toward the firm average. The intercept is penalized alongside the
coefficients, because on a twelve-period panel the mean drift is a period effect
that does not carry forward - leaving it unpenalized cost about 1.5% MAE.

**Features** (all computed strictly from periods <= the origin):

- *Mean reversion*: Util % vs the person's running mean, MA3 vs that mean,
  momentum, the person's own volatility, gap to target, YTD vs current.
- *Where non-billable time went*: bench share (current and 3-period mean),
  training, leadership (Bus Dev + Mngmt), opportunity, fringe share, avail ratio,
  OT share.
- *Cross-sectional position*: person vs cost center, cost center vs firm,
  cost-center momentum relative to the firm, firm momentum.

Absolute cost-center and firm utilization levels are deliberately **excluded**:
with one period per calendar month they act as a period label, and an early
version of the model that used them simply refit the period mean and lost to the
naive baseline out of sample. Calendar-month dummies are excluded for the same
reason - with a single fiscal year, every validation month is one the model has
never seen, so month effects cannot be learned or validated here.

**Protocol.** Rolling-origin cross-validation: fold *k* trains on every sample
whose target period is before *k* and validates on period *k*, for periods 8-12
(5 folds, 60 people each). The ridge penalty is chosen on pooled fold MAE from a
grid spanning 0.01-100; the final model refits on all 540 samples. Baselines are
scored on the identical validation rows. `npm run util:test` asserts the
no-look-ahead property directly: rebuilding a feature row from data truncated at
the origin must reproduce it bit for bit.

## 3. Results

Out-of-sample, 300 person-periods, Util % in percentage points:

| model | MAE | RMSE | bias | R2 | within 5pp |
| --- | --- | --- | --- | --- | --- |
| **ridge (this model)** | **4.58** | **6.09** | -0.08 | 0.898 | 65.0% |
| baseline: last period | 4.63 | 6.10 | -0.33 | 0.898 | 62.0% |
| baseline: 3-period moving average | 4.97 | 6.28 | -0.05 | 0.892 | 57.7% |
| baseline: person mean | 6.19 | 7.92 | -1.18 | 0.828 | 48.3% |
| baseline: target | 9.62 | 13.42 | 3.12 | 0.506 | 36.0% |
| baseline: cost-center mean | 14.61 | 18.29 | -0.28 | 0.082 | 21.7% |

These come from **nested** penalty selection: each fold's penalty is chosen by an
inner rolling origin over the periods that closed before it, so nothing about a
fold touches the period it is scored on. Choosing the penalty on the folds being
reported - the usual shortcut, and what this model originally did - says 4.56pp
instead. That 0.02pp is the size of the optimism, and both are printed each run.

The honest summary: on a twelve-period panel the gain over "same as last period"
is real but small (1.0% MAE, and a larger gain on hit rate within 5pp), while
bias drops to roughly zero and the hours-weighted **cost-center rollup lands
within 3.3pp MAE**, which is the number a resourcing conversation actually runs
on. The weakest fold is P10 (July, MAE 6.8): the vacation trough is a month
effect that one year of history cannot teach.

The 80% interval is +/-7.8pp. Measured on the residuals that defined it, it
covers 83.7%; measured **walk-forward**, with each fold's interval calibrated
only on folds before it, it covers **74.6%** - and nearly all of the shortfall is
P10 at 56.7%. Treat it as roughly a 75% interval. Conformal and Student-t widths
were tried as replacements and are worse or uselessly wide; every run reports all
four. See `OPERATING-GUIDE.md` section 2.2.

Top standardized drivers: `training_share` (+0.28 - a training-heavy period is
followed by recovery), `gap_to_target` (+0.17), `util_vs_person_mean` (-0.16 -
mean reversion), `ytd_vs_util` (+0.16).

## 4. Outputs

- `data/utilization-model.json` - a versioned artifact: coefficients,
  standardization stats, the penalty each fold chose and the full grid,
  fold-by-fold and baseline metrics, interval calibration for every method
  tried, and the training period range. Loading refuses an artifact whose
  feature list does not match the code's.
- `data/utilization-forecast.csv` - per person: last Util %, forecast, 80%
  interval, target, forecast variance, expected available hours.
- The training report also prints the cost-center rollup for the next period and
  the largest forecast shortfalls against target.

## 5. Limitations

- One fiscal year means **no learnable seasonality**; a second year would let
  month effects into the model and should take a visible bite out of the July fold.
- The roster is static - no joiners, leavers or transfers mid-year - so the
  forecaster is never asked about a person with no history.
- Linear and person-level by construction. Project pipeline or backlog data,
  which is what actually drives future direct hours, is not in this schema.
- The data is synthetic. It is calibrated to be *plausible* and internally
  consistent, not to match any real firm.
