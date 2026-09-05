import { buildTrainingSamples, FEATURE_NAMES, periodAggregates } from './features.ts';
import type { Sample } from './features.ts';
import {
  evaluate,
  FIRST_VALIDATION_PERIOD,
  LAMBDA_GRID,
  selectLambdaBefore,
} from './forecast.ts';
import type { Metrics } from './forecast.ts';
import { indexPlans, planKey } from './plan.ts';
import type { SnapshottedPlan } from './plan.ts';
import { fitRidge, predictOne } from './ridge.ts';
import type { RidgeModel } from './ridge.ts';
import type { PeriodedRecord } from './types.ts';

/**
 * PM plan vs model vs blend, on identical rows.
 *
 * The open question the dataset could not answer was whether the forecaster
 * beats the firm's real process, which forecasts utilization from PM hour
 * allocations. It could not be answered because the extract had no planned
 * hours in it. With `plan.ts` supplying a snapshotted allocation per
 * person-period, all three can be scored on exactly the same rows under exactly
 * the same rolling-origin protocol.
 *
 * Four forecasters are scored:
 *
 *   - `plan`          - the PM allocation, used as-is.
 *   - `history_ridge` - the existing model, features and anchor unchanged.
 *   - `plan_ridge`    - a bias-correction layer on the plan: the same history
 *                       features plus a plan block, fitted on the *plan's*
 *                       error, so shrinking the coefficients walks it back to
 *                       the raw plan rather than to the firm mean. This is the
 *                       same anchoring lesson as the history model, which falls
 *                       back to "same as last period".
 *   - `blend`         - a convex combination of `plan` and `history_ridge`.
 *
 * The blend weight is the part most easily got wrong. Fitting it on the
 * validation rows would report a number nobody could have achieved in advance,
 * which is the same trap the mean-reversion sweep fell into. Here the weight for
 * fold k is fitted only on out-of-sample rows from folds *before* k, and the
 * first fold - which has no earlier fold to learn from - uses a flat 0.5. The
 * weights actually used are reported so the choice can be inspected.
 *
 * Both ridge families pick their penalty by nested selection - fold k's penalty
 * comes from an inner rolling origin over the periods that closed before k - so
 * nothing about a fold, the fit or the penalty, touches the period it is scored
 * on. `history_ridge` therefore reproduces the shipped model exactly, which a
 * self-check asserts; if the two ever drift apart, the comparison has stopped
 * being like for like.
 */

export const PLAN_FEATURE_NAMES = [
  /** How far the plan departs from the strongest naive baseline. */
  'plan_vs_last',
  'plan_vs_person_mean',
  'plan_vs_target',
  /** Periods since the allocation was refreshed; a stale plan is worth less. */
  'plan_age',
  /** This person's own past plan error - the bias-correction signal. */
  'plan_error_ma',
  /** Planned capacity change, which is mostly approved leave the model cannot see. */
  'plan_avail_change',
  /** The same relative framing the history block uses, applied to the plan. */
  'cc_plan_vs_cc_last',
];

export interface PlanSample {
  /** The history-only sample, exactly as the existing model builds it. */
  base: Sample;
  plan: SnapshottedPlan;
  /** The plan feature block, aligned with `PLAN_FEATURE_NAMES`. */
  planX: number[];
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function clampUtil(value: number): number {
  return Math.min(100, Math.max(0, value));
}

/**
 * Joins plans onto the existing training samples and adds the plan features.
 *
 * Samples without a plan for their target period are dropped, so every
 * forecaster below is scored on an identical row set.
 */
export function buildPlanSamples(
  records: PeriodedRecord[],
  plans: SnapshottedPlan[],
): PlanSample[] {
  const samples = buildTrainingSamples(records);
  const planIndex = indexPlans(plans);
  const aggregates = periodAggregates(records);

  // Per-person plan errors, keyed by the period the plan was about. Only pairs
  // whose actual has already landed are ever read, below.
  const actualByKey = new Map<string, number>();
  for (const record of records) {
    actualByKey.set(planKey(record.costCenter, record.personName, record.periodIndex), record.utilPct);
  }
  const errorsByPerson = new Map<string, { period: number; error: number }[]>();
  for (const plan of plans) {
    const actual = actualByKey.get(planKey(plan.costCenter, plan.personName, plan.planPeriod));
    if (actual === undefined) continue;
    const person = `${plan.costCenter}|${plan.personName}`;
    const list = errorsByPerson.get(person) ?? [];
    list.push({ period: plan.planPeriod, error: actual - plan.plannedUtilPct });
    errorsByPerson.set(person, list);
  }
  for (const list of errorsByPerson.values()) list.sort((a, b) => a.period - b.period);

  // Cost-center mean planned Util % per period, weighted by planned capacity.
  // Every plan for period p is frozen at p-1, so this is known at forecast time.
  const ccSums = new Map<number, Map<string, { d: number; a: number }>>();
  for (const plan of plans) {
    const byCc = ccSums.get(plan.planPeriod) ?? new Map<string, { d: number; a: number }>();
    const entry = byCc.get(plan.costCenter) ?? { d: 0, a: 0 };
    entry.d += plan.plannedDirectHours;
    entry.a += plan.plannedAvailHours;
    byCc.set(plan.costCenter, entry);
    ccSums.set(plan.planPeriod, byCc);
  }
  const ccPlanUtil = new Map<number, Map<string, number>>();
  for (const [period, byCc] of ccSums) {
    const out = new Map<string, number>();
    for (const [cc, entry] of byCc) out.set(cc, (entry.d / entry.a) * 100);
    ccPlanUtil.set(period, out);
  }

  const panelLastAvail = new Map<string, number>();
  for (const record of records) {
    panelLastAvail.set(
      planKey(record.costCenter, record.personName, record.periodIndex),
      record.availTotal,
    );
  }

  const out: PlanSample[] = [];
  for (const base of samples) {
    const plan = planIndex.get(planKey(base.costCenter, base.personName, base.targetPeriod));
    if (!plan) continue;

    const person = `${base.costCenter}|${base.personName}`;
    // Strictly before the target period: at the origin these actuals are closed.
    const past = (errorsByPerson.get(person) ?? []).filter(e => e.period <= base.originPeriod);
    const planErrorMa = past.length > 0 ? mean(past.slice(-4).map(e => e.error)) : 0;

    const lastAvail =
      panelLastAvail.get(planKey(base.costCenter, base.personName, base.originPeriod)) ??
      plan.plannedAvailHours;
    const ccPlan = ccPlanUtil.get(base.targetPeriod)?.get(base.costCenter);
    const ccLast =
      aggregates.get(base.originPeriod)?.byCostCenter.get(base.costCenter) ?? base.baselines.ccLast;

    const planX = [
      plan.plannedUtilPct - base.baselines.last,
      plan.plannedUtilPct - base.baselines.personMean,
      plan.plannedUtilPct - base.utilTarget,
      plan.planAgePeriods,
      planErrorMa,
      (plan.plannedAvailHours / lastAvail) * 100 - 100,
      (ccPlan ?? ccLast) - ccLast,
    ];

    if (planX.length !== PLAN_FEATURE_NAMES.length) {
      throw new Error(
        `Plan feature row has ${planX.length} values, expected ${PLAN_FEATURE_NAMES.length}`,
      );
    }
    out.push({ base, plan, planX });
  }
  return out;
}

export type ForecasterName = 'plan' | 'history_ridge' | 'plan_ridge' | 'blend';

export const FORECASTERS: ForecasterName[] = ['plan', 'history_ridge', 'plan_ridge', 'blend'];

export interface HeadToHeadPrediction {
  sample: PlanSample;
  targetPeriod: number;
  actual: number;
  byForecaster: Record<ForecasterName, number>;
}

export interface FoldSummary {
  targetPeriod: number;
  n: number;
  /** MAE per forecaster on this fold. */
  mae: Record<ForecasterName, number>;
  /** Weight on the plan used by the blend, fitted on earlier folds only. */
  blendWeight: number;
}

export interface HeadToHead {
  trainedAt: string;
  /** Penalties for the deployed fit; folds each chose their own, see `lambdaByFold`. */
  lambdaHistory: number;
  lambdaPlan: number;
  /** What each fold selected from its own past. */
  lambdaByFold: { targetPeriod: number; history: number; plan: number }[];
  rows: number;
  validationRows: number;
  metrics: Record<ForecasterName, Metrics>;
  byFold: FoldSummary[];
  /** Final plan-augmented model, refit on every sample. */
  planModel: RidgeModel;
  /** Plan accuracy split by how stale the allocation was. */
  planByAge: { age: string; n: number; mae: number; bias: number }[];
  /** Paired comparisons against the plan, on the identical rows. */
  versusPlan: {
    forecaster: ForecasterName;
    maeDelta: number;
    /** Share of rows where this forecaster was closer than the plan. */
    winRate: number;
  }[];
}

const augmentedNames = [...FEATURE_NAMES, ...PLAN_FEATURE_NAMES];

function augmentedX(sample: PlanSample): number[] {
  return [...sample.base.x, ...sample.planX];
}

function validationPeriods(samples: PlanSample[]): number[] {
  return [...new Set(samples.map(s => s.base.targetPeriod))]
    .filter(p => p >= FIRST_VALIDATION_PERIOD)
    .sort((a, b) => a - b);
}

/**
 * Least-squares weight on the plan in `w * plan + (1 - w) * model`, clamped to a
 * genuine convex combination. Returns undefined when there is nothing to fit on.
 */
function fitBlendWeight(
  rows: { actual: number; plan: number; model: number }[],
): number | undefined {
  if (rows.length === 0) return undefined;
  let numerator = 0;
  let denominator = 0;
  for (const row of rows) {
    const spread = row.plan - row.model;
    numerator += (row.actual - row.model) * spread;
    denominator += spread * spread;
  }
  if (denominator < 1e-9) return undefined;
  return Math.min(1, Math.max(0, numerator / denominator));
}

interface RollingPass {
  predictions: HeadToHeadPrediction[];
  byFold: FoldSummary[];
  lambdaByFold: { targetPeriod: number; history: number; plan: number }[];
}

function rollingOrigin(samples: PlanSample[]): RollingPass {
  const predictions: HeadToHeadPrediction[] = [];
  const byFold: FoldSummary[] = [];
  const lambdaByFold: { targetPeriod: number; history: number; plan: number }[] = [];
  // Out-of-sample rows from folds already scored, which is all the blend weight
  // is ever allowed to see.
  const seen: { actual: number; plan: number; model: number }[] = [];

  for (const period of validationPeriods(samples)) {
    const train = samples.filter(s => s.base.targetPeriod < period);
    const validate = samples.filter(s => s.base.targetPeriod === period);
    if (train.length === 0 || validate.length === 0) continue;

    const lambdaHistory = selectLambdaBefore(samples.map(s => s.base), period);
    const lambdaPlan = selectPlanLambdaBefore(samples, period);
    lambdaByFold.push({ targetPeriod: period, history: lambdaHistory, plan: lambdaPlan });

    const historyModel = fitRidge(
      train.map(s => s.base.x),
      train.map(s => s.base.y - s.base.anchor),
      lambdaHistory,
      FEATURE_NAMES,
      true,
    );
    const planModel = fitRidge(
      train.map(augmentedX),
      train.map(s => s.base.y - s.plan.plannedUtilPct),
      lambdaPlan,
      augmentedNames,
      true,
    );

    const blendWeight = fitBlendWeight(seen) ?? 0.5;
    const foldRows: HeadToHeadPrediction[] = [];

    for (const sample of validate) {
      const plan = sample.plan.plannedUtilPct;
      const history = clampUtil(sample.base.anchor + predictOne(historyModel, sample.base.x));
      const corrected = clampUtil(plan + predictOne(planModel, augmentedX(sample)));
      const blend = clampUtil(blendWeight * plan + (1 - blendWeight) * history);
      foldRows.push({
        sample,
        targetPeriod: period,
        actual: sample.base.y,
        byForecaster: { plan, history_ridge: history, plan_ridge: corrected, blend },
      });
    }

    const actual = foldRows.map(r => r.actual);
    const mae = {} as Record<ForecasterName, number>;
    for (const name of FORECASTERS) {
      mae[name] = evaluate(actual, foldRows.map(r => r.byForecaster[name])).mae;
    }
    byFold.push({ targetPeriod: period, n: foldRows.length, mae, blendWeight });

    predictions.push(...foldRows);
    for (const row of foldRows) {
      seen.push({
        actual: row.actual,
        plan: row.byForecaster.plan,
        model: row.byForecaster.history_ridge,
      });
    }
  }

  return { predictions, byFold, lambdaByFold };
}

/**
 * Penalty for the plan-augmented family, chosen from periods before
 * `beforePeriod` only - the same nested rule the history model uses, so neither
 * family gets to see the fold it will be scored on.
 */
function selectPlanLambdaBefore(samples: PlanSample[], beforePeriod: number): number {
  const inner = samples.filter(s => s.base.targetPeriod < beforePeriod);
  const middle = LAMBDA_GRID[Math.floor(LAMBDA_GRID.length / 2)];
  if (inner.length === 0) return middle;

  const minTarget = Math.min(...inner.map(s => s.base.targetPeriod));
  const innerPeriods = [...new Set(inner.map(s => s.base.targetPeriod))]
    .filter(p => p >= minTarget + 2)
    .sort((a, b) => a - b);
  if (innerPeriods.length === 0) return middle;

  let best = { lambda: LAMBDA_GRID[0], mae: Number.POSITIVE_INFINITY };
  for (const lambda of LAMBDA_GRID) {
    const actual: number[] = [];
    const predicted: number[] = [];
    for (const period of innerPeriods) {
      const train = inner.filter(s => s.base.targetPeriod < period);
      const validate = inner.filter(s => s.base.targetPeriod === period);
      if (train.length === 0 || validate.length === 0) continue;
      const model = fitRidge(
        train.map(augmentedX),
        train.map(s => s.base.y - s.plan.plannedUtilPct),
        lambda,
        augmentedNames,
        true,
      );
      for (const sample of validate) {
        actual.push(sample.base.y);
        predicted.push(clampUtil(sample.plan.plannedUtilPct + predictOne(model, augmentedX(sample))));
      }
    }
    if (actual.length === 0) continue;
    const mae = evaluate(actual, predicted).mae;
    if (mae < best.mae) best = { lambda, mae };
  }
  return best.lambda;
}

export function runHeadToHead(
  records: PeriodedRecord[],
  plans: SnapshottedPlan[],
): HeadToHead {
  const samples = buildPlanSamples(records, plans);
  if (samples.length === 0) throw new Error('No plan-joined samples could be built');

  const pass = rollingOrigin(samples);
  // Penalties for the deployed fit, chosen from everything that has closed -
  // the same rule a fold applies, with the whole panel as its past.
  const lastPeriod = Math.max(...samples.map(s => s.base.targetPeriod));
  const lambdaHistory = selectLambdaBefore(samples.map(s => s.base), lastPeriod + 1);
  const lambdaPlan = selectPlanLambdaBefore(samples, lastPeriod + 1);

  const actual = pass.predictions.map(p => p.actual);

  const metrics = {} as Record<ForecasterName, Metrics>;
  for (const name of FORECASTERS) {
    metrics[name] = evaluate(actual, pass.predictions.map(p => p.byForecaster[name]));
  }

  // Plan accuracy by staleness, which is the mechanism the generator puts in.
  const ageBuckets = new Map<string, { errors: number[] }>();
  for (const p of pass.predictions) {
    const age = p.sample.plan.planAgePeriods;
    const label = age === 0 ? 'fresh (age 0)' : `stale (age ${Math.min(age, 3)})`;
    const bucket = ageBuckets.get(label) ?? { errors: [] };
    bucket.errors.push(p.byForecaster.plan - p.actual);
    ageBuckets.set(label, bucket);
  }
  const planByAge = [...ageBuckets.entries()]
    .map(([age, bucket]) => ({
      age,
      n: bucket.errors.length,
      mae: mean(bucket.errors.map(Math.abs)),
      bias: mean(bucket.errors),
    }))
    .sort((a, b) => a.age.localeCompare(b.age));

  const versusPlan = FORECASTERS.filter(name => name !== 'plan').map(name => {
    let wins = 0;
    for (const p of pass.predictions) {
      const mine = Math.abs(p.byForecaster[name] - p.actual);
      const theirs = Math.abs(p.byForecaster.plan - p.actual);
      if (mine < theirs) wins++;
    }
    return {
      forecaster: name,
      maeDelta: metrics[name].mae - metrics.plan.mae,
      winRate: wins / pass.predictions.length,
    };
  });

  const planModel = fitRidge(
    samples.map(augmentedX),
    samples.map(s => s.base.y - s.plan.plannedUtilPct),
    lambdaPlan,
    augmentedNames,
    true,
  );

  return {
    trainedAt: new Date().toISOString(),
    lambdaHistory,
    lambdaPlan,
    rows: samples.length,
    validationRows: pass.predictions.length,
    metrics,
    byFold: pass.byFold,
    lambdaByFold: pass.lambdaByFold,
    planModel,
    planByAge,
    versusPlan,
  };
}
