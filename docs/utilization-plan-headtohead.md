# PM plan vs model vs blend

The forecaster's open question was whether it beats the firm's real process,
which forecasts utilization from PM hour allocations to projects. It could not be
answered, because the dataset had no planned-hours column and the backtest only
ever compared against naive statistical baselines.

This adds the missing column and runs the three-way comparison.

```bash
npm run util:plans     # data/utilization-plan.csv     - simulated allocations
npm run util:compare   # data/utilization-headtohead.json - plan vs model vs blend
npm run util:sweep     # data/utilization-sweep.json   - sensitivity to planner quality
```

## 1. The short answer

**With plans as good as the ones simulated here, the raw PM plan loses to the
model, but the plan carries information the model does not have, and using both
beats either.** The gain from adding plans (4.58 → 4.28pp MAE, −6.5%) is more
than six times the gain the model itself had over "same as last period" (−1.0%).

| forecaster | MAE | RMSE | bias | R² | within 5pp |
| --- | --- | --- | --- | --- | --- |
| **blend (plan + model)** | **4.28** | **5.79** | +0.59 | 0.908 | 65.0% |
| model on plan + history | 4.49 | 5.86 | +0.68 | 0.906 | 64.0% |
| model (history only) | 4.58 | 6.09 | −0.08 | 0.898 | 65.0% |
| PM plan (as-is) | 5.17 | 7.30 | +1.90 | 0.854 | 64.7% |

Rolling-origin CV, folds = target periods 8–12, the same 300 person-periods the
history-only model was always scored on. Util % in percentage points.

Three qualifications belong with that table, and none of them are small:

1. **The blend is the robust choice, and this only became clear under honest
   penalty selection.** Over 12 independent draws of the allocations, the blend
   averages 4.28pp (sd 0.05) and is best in 10 of 12; the plan-augmented model
   averages 4.33pp (sd 0.12) and is best in 2. Under the earlier protocol, where
   each family's penalty was tuned on the folds being reported, the two looked
   like a coin flip at 6-6 — the plan-augmented model was picking a near-zero
   penalty with knowledge of the answer, which flattered it. Fixing the
   selection changed the conclusion, which is the argument for fixing it.
2. **"The plan loses" is a statement about assumed planner quality**, not a
   finding about PM plans. §4 sweeps it, and the plan wins in a third of the grid.
3. **The dataset is still synthetic.** Everything here is a number about this
   simulation.

What *does* survive re-drawing: using the plan beat the history-only model in
**12 of 12 draws**, and the raw plan beat it in **0 of 12**.

## 2. The allocation data

`data/utilization-plan.csv`, 660 rows — one per person per period from P2 on.
It is a **separate file** from the timesheet extract, because in a real firm the
allocations live in the project system, not the timesheet system, and because it
means `data/utilization.csv` is untouched: every number previously measured on
the actuals still stands, unchanged.

| Column | Meaning |
| --- | --- |
| `Snapshot Source.Name` | The extract the plan was frozen against — period t |
| `Cost Center`, `Cost Center Name`, `Person Name` | Who the plan is for |
| `Plan Period`, `Plan Month` | The period the plan is *about* — always t+1 |
| `Planned Avail Hours` | Capacity the planner expected, net of approved leave |
| `Planned Direct Hours` | Hours booked to projects |
| `Planned Util %` | `Planned Direct / Planned Avail × 100` |
| `Plan Age Periods` | Periods since the allocation was last refreshed |

`verifyPlan` enforces the identity in *hours* rather than percentage points,
because planned direct hours are the rounded quantity — checking it the other way
round flags a tenth of an hour of rounding as a violation.

### The simulated planner

Actuals are generated **first** and are never a function of the plan, so nothing
in the planner can feed back into the dataset. The planner then gets three
properties, each of which shows up in the results:

- **Foresight.** It sees a noisy read on the coming period — project end dates,
  new awards, approved leave — which the history-only model structurally cannot
  see. This is the only reason a plan can beat a model at all. Default 0.55,
  meaning the planner closes 55% of the gap between a naive prior and the truth.
- **Optimism, asymmetric.** Plans sitting below target are pulled up toward it
  much harder (0.35) than plans above target are pulled down (0.10), because
  people get booked to fill their available hours. This produces the +1.9pp bias
  in the table above, which is the order of bias professional-services planning
  tends to carry.
- **Staleness.** About a quarter of allocations are carried forward rather than
  rebuilt, so they describe a period they were never written for. Fresh plans
  score 4.93pp MAE; allocations one period stale score 6.26pp.

Each person gets their **own PRNG stream**, seeded from their name and the run
seed. That is not cosmetic: it means truncating the data to period p reproduces
the first p−1 plan rows exactly, which is what makes the leakage invariant in §3
testable rather than merely asserted.

## 3. Protocol, and the three ways this could have been rigged

The comparison is only worth anything if it is hard to cheat. Three specific
traps, and what was done about each:

**Plans revised after the fact.** A plan is frozen at t and scored on t+1, never
revised. `planSnapshotPeriod` is always exactly one before `planPeriod`, and a
self-check regenerates the allocations from data truncated at period p and
asserts the first p−1 rows come back byte-identical — a plan may see the period
it is *about* (that is the foresight), but nothing after it.

**Features that see the future.** The plan feature block gets the same treatment
the history features already had: rebuild every feature row from data truncated
to what was knowable at the origin, and require an exact match.

**A blend weight fitted on the rows it is scored on.** This is the trap the
handoff's mean-reversion sweep fell into, and it would have been easy to repeat.
The weight for fold k is fitted by least squares on out-of-sample rows from folds
*before* k only; the first fold, having no earlier fold to learn from, uses a flat
0.5. The weights actually used were 0.50, 0.27, 0.23, 0.26, 0.28 — after the
first fold the process settles on trusting the plan about a quarter.

Both ridge families pick their penalty by **nested selection**: fold k's penalty
comes from an inner rolling origin over the periods that closed before k, so
nothing about a fold — fit or penalty — touches the period it is scored on.
`history_ridge` therefore reproduces the shipped model exactly, which a
self-check asserts; if the two ever drift apart the comparison has stopped being
like for like. The deployed fits land on λ=3 and λ=0.3.

This was not always so, and fixing it mattered: under the earlier pooled
selection the plan-augmented model chose λ=0.01 — the very edge of the grid —
with knowledge of the folds it was reported on, and looked equal to the blend.
It is not.

The four forecasters:

- `plan` — the allocation, used as-is.
- `history_ridge` — the shipped model, features and anchor unchanged. It
  reproduces its 4.5780pp exactly, which a self-check asserts.
- `plan_ridge` — the same history features plus a plan block, fitted on the
  *plan's* error so shrinkage falls back to the raw plan. Same anchoring logic
  that makes the history model fall back to "same as last period".
- `blend` — a convex combination of `plan` and `history_ridge`.

The plan block is 7 features: `plan_vs_last`, `plan_vs_person_mean`,
`plan_vs_target`, `plan_age`, `plan_error_ma` (this person's own past plan
error — the bias-correction signal), `plan_avail_change`, `cc_plan_vs_cc_last`.
Three of the top five drivers of the correction are plan features, led by
`plan_vs_last` (−1.38) and `plan_error_ma` (+1.00): the model is mostly learning
how far to discount a plan that departs from the person's run rate, and how
optimistic this particular person's plans have been.

## 4. How good would PM plans have to be?

The single-point answer above is a statement about assumed planner quality, so
the assumptions get swept: foresight 0.20–0.80 against optimism 0.00–0.50,
regenerating the allocations at each of the 30 cells and re-running the identical
protocol. The history-only model never sees a plan, and its MAE is 4.58pp in
every cell with 0.0000pp drift — a useful check that the sweep only moves what it
means to move.

**Where the raw plan beats the model** (blank = model wins):

| foresight | opt 0.00 | 0.10 | 0.20 | 0.35 | 0.50 |
| --- | --- | --- | --- | --- | --- |
| 0.20 | | | | | |
| 0.35 | | | | | |
| 0.50 | plan | plan | | | |
| 0.55 | plan | plan | | | |
| 0.65 | plan | plan | plan | | |
| 0.80 | plan | plan | plan | plan | |

The raw plan wins in 11 of 30 cells. It needs foresight ≥ 0.50 to win at all, and
at optimism 0.35 it wins only from the top of the foresight range — **an unbiased
mediocre plan beats a sharp optimistic one.** That is the practically useful
finding: de-biasing allocations is worth more than improving them.

The sweep also exposes a failure mode the single run hid:

| | best in | worse than ignoring plans |
| --- | --- | --- |
| blend | 29 of 30 cells | 1 cell, by 0.033pp |
| plan+history model | 0 of 30 cells | 11 cells, all at foresight ≤ 0.55 |
| history-only model | 1 of 30 cells | — |
| PM plan | 0 of 30 cells | — |

**Anchoring on the plan is only safe when the plan is good.** When plans are
poor, `plan_ridge` inherits their errors and does worse than ignoring plans
entirely. The blend, which learns how far to trust the plan from past folds
instead of assuming it, degrades gracefully — it is never meaningfully worse than
the history-only model anywhere on the grid. If one of these goes to production,
it is the blend, and not because it won the point estimate.

## 5. What this does not answer

- **It is a simulation of PM plans, not PM plans.** The honest reading is not
  "the model beats PM plans" but "here is what it would take for either to win,
  and here is which combination is robust". On real allocations the crossover
  could sit anywhere in §4's grid; the way to find out is to run the same
  protocol on real snapshotted plans, and the code takes them as a CSV.
- **No self-fulfilment.** Real allocations partly *cause* the outcome — people
  charge to what they are booked to — so a real plan can look accurate because it
  made itself true. Nothing here models that, which means this simulation
  **understates** how accurate real plans would look, and a real head-to-head
  would need to separate a plan that predicted the outcome from one that caused
  it. That is not a measurement problem the backtest can solve on its own.
- **Snapshots are the whole ballgame on real data.** If a firm's project system
  overwrites allocations in place rather than versioning them, this comparison
  cannot be run honestly at all — scoring today's copy of a past allocation
  invents accuracy that never existed. Check for history before trusting any
  number from this protocol.
- **Still one period ahead, still one fiscal year.** Nothing here changes either
  limitation. There is deliberately no plan for P13: the simulated planner's
  foresight is a read on the actual, and past the last extract there is no actual
  to read, so a P13 plan would be structurally unlike every row the model was
  fitted on. `npm run util:train` remains the P13 forecast.
- **The roster is still static**, so no plan is ever written for someone with no
  history.
