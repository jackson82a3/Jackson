import {
  buildForecastSamples,
  buildTrainingSamples,
  FEATURE_NAMES,
  groupByPerson,
} from './features.ts';
import type { Sample } from './features.ts';
import { fitHuberRidge, fitRidge, predictOne } from './ridge.ts';
import type { RidgeModel } from './ridge.ts';
import type { PeriodedRecord } from './types.ts';

/**
 * Next-period Util % forecaster.
 *
 * Ridge regression over lagged personal, cost-center and firm-level features,
 * The response is the *change* from the person's last observed Util %, so the
 * ridge penalty shrinks the forecast back toward "same as last period" - the
 * strongest naive baseline on this panel - instead of toward the firm average.
 *
 * The penalty is chosen by rolling-origin cross-validation: fold k trains on
 * every sample whose target period is before k and validates on period k, which
 * is exactly how the model gets used in production (fit on everything closed to
 * date, predict the period that has not happened yet). The naive baselines are
 * scored on the identical validation rows so the comparison is like for like.
 */

/** First period used as a validation fold; earlier periods are the seed history. */
export const FIRST_VALIDATION_PERIOD = 8;

export const LAMBDA_GRID = [0.01, 0.03, 0.1, 0.3, 1, 2, 3, 5, 10, 30, 100];

/** z for an 80% interval. */
const Z80 = 1.2816;

/**
 * How a fold turns its training samples into a fitted model.
 *
 * Pluggable so that model variants are evaluated through the *same* rolling
 * origin, nested penalty selection and interval calibration as the shipped
 * model, rather than through a copy of that protocol that might quietly differ.
 * `scripts/experiment-utilization.ts` is the harness that uses it.
 */
export interface EstimatorContext {
  train: Sample[];
  lambda: number;
  /** The period being predicted, so an estimator can weight by recency. */
  targetPeriod: number;
}

export type Estimator = (ctx: EstimatorContext) => RidgeModel;

/** Exponential recency weights with the given half-life in periods. */
function recencyWeights(train: Sample[], targetPeriod: number, halfLife: number): number[] {
  return train.map(s => Math.pow(0.5, (targetPeriod - s.targetPeriod) / halfLife));
}

export const ESTIMATORS: Record<string, Estimator> = {
  /** The shipped model: least squares, penalised intercept, anchored on last period. */
  ridge: ({ train, lambda }) =>
    fitRidge(train.map(s => s.x), train.map(s => s.y - s.anchor), lambda, FEATURE_NAMES, true),

  /** Huber loss, which targets MAE rather than squared error. */
  huber: ({ train, lambda }) =>
    fitHuberRidge(train.map(s => s.x), train.map(s => s.y - s.anchor), lambda, FEATURE_NAMES, true),

  /** Least squares with recent periods weighted more heavily. */
  recency: ({ train, lambda, targetPeriod }) =>
    fitRidge(
      train.map(s => s.x),
      train.map(s => s.y - s.anchor),
      lambda,
      FEATURE_NAMES,
      true,
      recencyWeights(train, targetPeriod, 3),
    ),

  /** Both at once. */
  huber_recency: ({ train, lambda, targetPeriod }) => {
    const weights = recencyWeights(train, targetPeriod, 3);
    const X = train.map(s => s.x);
    const y = train.map(s => s.y - s.anchor);
    let model = fitRidge(X, y, lambda, FEATURE_NAMES, true, weights);
    for (let i = 0; i < 8; i++) {
      const residuals = X.map((x, k) => y[k] - predictOne(model, x));
      const sorted = residuals.map(Math.abs).sort((a, b) => a - b);
      const mad = sorted[sorted.length >> 1] || 1e-9;
      const delta = 1.345 * 1.4826 * mad;
      model = fitRidge(
        X,
        y,
        lambda,
        FEATURE_NAMES,
        true,
        residuals.map((r, k) => weights[k] * (Math.abs(r) <= delta ? 1 : delta / Math.abs(r))),
      );
    }
    return model;
  },
};

export const DEFAULT_ESTIMATOR: Estimator = ESTIMATORS.ridge;

export interface Metrics {
  n: number;
  mae: number;
  rmse: number;
  bias: number;
  r2: number;
  /** Share of predictions landing within 5 percentage points of the actual. */
  within5: number;
}

export function evaluate(actual: number[], predicted: number[]): Metrics {
  const n = actual.length;
  if (n === 0) return { n: 0, mae: 0, rmse: 0, bias: 0, r2: 0, within5: 0 };
  let absError = 0;
  let squaredError = 0;
  let error = 0;
  let within = 0;
  for (let i = 0; i < n; i++) {
    const e = predicted[i] - actual[i];
    absError += Math.abs(e);
    squaredError += e * e;
    error += e;
    if (Math.abs(e) <= 5) within++;
  }
  const mean = actual.reduce((a, b) => a + b, 0) / n;
  const totalSquares = actual.reduce((a, b) => a + (b - mean) ** 2, 0);
  return {
    n,
    mae: absError / n,
    rmse: Math.sqrt(squaredError / n),
    bias: error / n,
    r2: totalSquares > 0 ? 1 - squaredError / totalSquares : 0,
    within5: within / n,
  };
}

export interface FoldPrediction {
  sample: Sample;
  predicted: number;
}

export interface CrossValidation {
  lambda: number;
  /** Out-of-sample predictions pooled across folds. */
  predictions: FoldPrediction[];
  byFold: { targetPeriod: number; n: number; mae: number; rmse: number }[];
}

function validationPeriods(samples: Sample[]): number[] {
  const periods = [...new Set(samples.map(s => s.targetPeriod))]
    .filter(p => p >= FIRST_VALIDATION_PERIOD)
    .sort((a, b) => a - b);
  if (periods.length === 0) {
    throw new Error(`No sample reaches period ${FIRST_VALIDATION_PERIOD}; not enough history`);
  }
  return periods;
}

export function crossValidate(samples: Sample[], lambda: number): CrossValidation {
  const predictions: FoldPrediction[] = [];
  const byFold: CrossValidation['byFold'] = [];

  for (const period of validationPeriods(samples)) {
    const train = samples.filter(s => s.targetPeriod < period);
    const validate = samples.filter(s => s.targetPeriod === period);
    if (train.length === 0 || validate.length === 0) continue;
    const model = fitRidge(
      train.map(s => s.x),
      train.map(s => s.y - s.anchor),
      lambda,
      FEATURE_NAMES,
      true,
    );
    const foldPredictions = validate.map(sample => ({
      sample,
      predicted: clampUtil(sample.anchor + predictOne(model, sample.x)),
    }));
    predictions.push(...foldPredictions);
    const metrics = evaluate(
      foldPredictions.map(p => p.sample.y),
      foldPredictions.map(p => p.predicted),
    );
    byFold.push({ targetPeriod: period, n: metrics.n, mae: metrics.mae, rmse: metrics.rmse });
  }

  return { lambda, predictions, byFold };
}

function clampUtil(value: number): number {
  return Math.min(100, Math.max(0, value));
}

/**
 * Picks a penalty using only samples whose target period is before `beforePeriod`.
 *
 * This is the honest version of penalty selection. Scanning the grid on the same
 * folds the result is then reported on - which is what `trainForecaster` used to
 * do, and what most quick backtests do - lets the penalty be chosen with
 * knowledge of the answer, so the reported error is a little better than
 * anything achievable in advance. Here the choice for fold k is made inside the
 * data that closed before k, exactly as it would have to be in production.
 *
 * Returns the grid's smallest-MAE penalty, or the middle of the grid when there
 * is not yet enough history to run an inner comparison at all.
 */
export function selectLambdaBefore(
  samples: Sample[],
  beforePeriod: number,
  estimator: Estimator = DEFAULT_ESTIMATOR,
): number {
  const inner = samples.filter(s => s.targetPeriod < beforePeriod);
  if (inner.length === 0) return LAMBDA_GRID[Math.floor(LAMBDA_GRID.length / 2)];

  const minTarget = Math.min(...inner.map(s => s.targetPeriod));
  // An inner fold needs at least two earlier periods to train on.
  const innerPeriods = [...new Set(inner.map(s => s.targetPeriod))]
    .filter(p => p >= minTarget + 2)
    .sort((a, b) => a - b);
  if (innerPeriods.length === 0) return LAMBDA_GRID[Math.floor(LAMBDA_GRID.length / 2)];

  let best = { lambda: LAMBDA_GRID[0], mae: Number.POSITIVE_INFINITY };
  for (const lambda of LAMBDA_GRID) {
    const actual: number[] = [];
    const predicted: number[] = [];
    for (const period of innerPeriods) {
      const train = inner.filter(s => s.targetPeriod < period);
      const validate = inner.filter(s => s.targetPeriod === period);
      if (train.length === 0 || validate.length === 0) continue;
      const model = estimator({ train, lambda, targetPeriod: period });
      for (const sample of validate) {
        actual.push(sample.y);
        predicted.push(clampUtil(sample.anchor + predictOne(model, sample.x)));
      }
    }
    if (actual.length === 0) continue;
    const mae = evaluate(actual, predicted).mae;
    if (mae < best.mae) best = { lambda, mae };
  }
  return best.lambda;
}

/**
 * Rolling-origin CV where every fold picks its own penalty from its own past.
 *
 * Nothing about fold k - not the fit, not the penalty - touches a period at or
 * after k, so the pooled result is a genuine held-out estimate.
 */
export function crossValidateNested(
  samples: Sample[],
  estimator: Estimator = DEFAULT_ESTIMATOR,
): CrossValidation & { lambdaByFold: { targetPeriod: number; lambda: number }[] } {
  const predictions: FoldPrediction[] = [];
  const byFold: CrossValidation['byFold'] = [];
  const lambdaByFold: { targetPeriod: number; lambda: number }[] = [];

  for (const period of validationPeriods(samples)) {
    const train = samples.filter(s => s.targetPeriod < period);
    const validate = samples.filter(s => s.targetPeriod === period);
    if (train.length === 0 || validate.length === 0) continue;

    const lambda = selectLambdaBefore(samples, period, estimator);
    lambdaByFold.push({ targetPeriod: period, lambda });

    const model = estimator({ train, lambda, targetPeriod: period });
    const foldPredictions = validate.map(sample => ({
      sample,
      predicted: clampUtil(sample.anchor + predictOne(model, sample.x)),
    }));
    predictions.push(...foldPredictions);
    const metrics = evaluate(
      foldPredictions.map(p => p.sample.y),
      foldPredictions.map(p => p.predicted),
    );
    byFold.push({ targetPeriod: period, n: metrics.n, mae: metrics.mae, rmse: metrics.rmse });
  }

  // `lambda` on the returned object is the most recent fold's choice, which is
  // the one a model fitted today would inherit.
  return {
    lambda: lambdaByFold.length > 0 ? lambdaByFold[lambdaByFold.length - 1].lambda : LAMBDA_GRID[0],
    predictions,
    byFold,
    lambdaByFold,
  };
}

export type BaselineName = 'last_period' | 'moving_average_3' | 'person_mean' | 'target' | 'cost_center_last';

export function baselinePrediction(sample: Sample, baseline: BaselineName): number {
  switch (baseline) {
    case 'last_period':
      return sample.baselines.last;
    case 'moving_average_3':
      return sample.baselines.ma3;
    case 'person_mean':
      return sample.baselines.personMean;
    case 'target':
      return sample.baselines.target;
    case 'cost_center_last':
      return sample.baselines.ccLast;
  }
}

export const BASELINES: BaselineName[] = [
  'last_period',
  'moving_average_3',
  'person_mean',
  'target',
  'cost_center_last',
];

/** One way of turning past residuals into an interval, and how it actually did. */
export interface IntervalMethod {
  name: string;
  /** Offsets added to a prediction to get its bounds; `low` is negative. */
  low: number;
  high: number;
  /**
   * Coverage measured the only honest way: each fold's interval is calibrated on
   * earlier folds only, then checked against this fold. Calibrating and
   * measuring on the same residuals reports the fit, not the coverage a future
   * period would see - and on this data the difference is about nine points.
   */
  coverageWalkForward: number;
}

export interface IntervalCalibration {
  /** Residual sd over the pooled out-of-sample folds. */
  sigma: number;
  /** The method `forecastNextPeriod` actually uses. */
  shipped: string;
  /** Every method tried, with its honestly measured coverage. */
  methods: IntervalMethod[];
  /** Coverage of the shipped method fold by fold, which is where it fails. */
  byFold: { targetPeriod: number; n: number; coverage: number; halfWidth: number }[];
  nWalkForward: number;
}

/**
 * Empirical quantile with the conformal (n+1) correction, which is what turns
 * "the 80th percentile of past errors" into an interval with a finite-sample
 * guarantee under exchangeability. Period-to-period data is not exchangeable,
 * so the coverage it achieves is measured rather than assumed.
 */
function conformalQuantile(values: number[], p: number): number {
  if (values.length === 0) return Number.POSITIVE_INFINITY;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil((sorted.length + 1) * p);
  return sorted[Math.min(Math.max(rank, 1), sorted.length) - 1];
}

/** One-sided 0.9 quantile of Student's t, for a two-sided 80% interval. */
function tQuantile90(df: number): number {
  const table: Record<number, number> = {
    1: 3.0777, 2: 1.8856, 3: 1.6377, 4: 1.5332, 5: 1.4759,
    6: 1.4398, 7: 1.4149, 8: 1.3968, 9: 1.383, 10: 1.3722,
  };
  if (df <= 1) return table[1];
  if (df >= 30) return Z80;
  return table[Math.min(Math.round(df), 10)] ?? 1.33;
}

function sdOf(values: number[]): number {
  const m = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / values.length);
}

/**
 * Builds the candidate intervals from past residuals.
 *
 * `actual = predicted - residual`, so the residual's upper tail sets the lower
 * bound - which is why the asymmetric offsets look inverted.
 */
function intervalsFrom(residuals: number[], priorFolds: number): Record<string, [number, number]> {
  const sd = sdOf(residuals);
  const symmetric = conformalQuantile(residuals.map(Math.abs), 0.8);
  return {
    gaussian: [-Z80 * sd, Z80 * sd],
    conformal_symmetric: [-symmetric, symmetric],
    conformal_asymmetric: [
      -conformalQuantile(residuals, 0.9),
      -conformalQuantile(residuals, 0.1),
    ],
    student_t: [-tQuantile90(priorFolds) * sd, tQuantile90(priorFolds) * sd],
  };
}

/** Which method `forecastNextPeriod` uses. See the write-up for why. */
export const SHIPPED_INTERVAL = 'gaussian';

export function calibrateIntervals(predictions: FoldPrediction[]): IntervalCalibration {
  const residuals = predictions.map(p => p.predicted - p.sample.y);
  const sigma = sdOf(residuals);

  const periods = [...new Set(predictions.map(p => p.sample.targetPeriod))].sort((a, b) => a - b);
  const names = Object.keys(intervalsFrom([0, 1], 1));
  const hits: Record<string, number> = {};
  for (const name of names) hits[name] = 0;
  const byFold: IntervalCalibration['byFold'] = [];

  const seen: number[] = [];
  let priorFolds = 0;
  let checked = 0;

  for (const period of periods) {
    const fold = predictions.filter(p => p.sample.targetPeriod === period);
    const foldResiduals = fold.map(p => p.predicted - p.sample.y);
    if (priorFolds > 0) {
      const candidates = intervalsFrom(seen, priorFolds);
      for (const name of names) {
        const [low, high] = candidates[name];
        hits[name] += foldResiduals.filter(r => -r >= low && -r <= high).length;
      }
      const [low, high] = candidates[SHIPPED_INTERVAL];
      const covered = foldResiduals.filter(r => -r >= low && -r <= high).length;
      byFold.push({
        targetPeriod: period,
        n: fold.length,
        coverage: covered / fold.length,
        halfWidth: (high - low) / 2,
      });
      checked += fold.length;
    }
    seen.push(...foldResiduals);
    priorFolds++;
  }

  // Deployed widths use every out-of-sample residual available.
  const final = intervalsFrom(residuals, priorFolds);

  return {
    sigma,
    shipped: SHIPPED_INTERVAL,
    methods: names.map(name => ({
      name,
      low: final[name][0],
      high: final[name][1],
      coverageWalkForward: checked > 0 ? hits[name] / checked : 0,
    })),
    byFold,
    nWalkForward: checked,
  };
}

export interface TrainedForecaster {
  model: RidgeModel;
  trainedAt: string;
  /**
   * Out-of-sample metrics from the rolling-origin folds, with each fold's
   * penalty chosen from its own past. This is the number to quote.
   */
  metrics: Metrics;
  /**
   * The same folds with the penalty chosen on the folds being reported. Strictly
   * optimistic; kept so the size of that optimism stays visible.
   */
  optimisticMetrics: Metrics;
  byFold: CrossValidation['byFold'];
  /** The penalty each fold selected from its own history. */
  lambdaByFold: { targetPeriod: number; lambda: number }[];
  byLambda: { lambda: number; mae: number; rmse: number }[];
  baselines: { name: BaselineName; metrics: Metrics }[];
  /** Residual sd from the pooled folds. Superseded by `interval`; kept for continuity. */
  sigma: number;
  /**
   * Residual sd of the "same as last period" baseline. Fallback rows are not
   * model predictions, so they are given this width rather than the model's.
   *
   * Caveat worth carrying: this is measured on people who *have* full history,
   * because the panel contains no joiners. A person with two periods behind them
   * is genuinely less predictable than that, so this width is likely optimistic
   * for exactly the rows it is used on. Real data with joiners would let it be
   * measured properly.
   */
  fallbackSigma: number;
  /** Walk-forward coverage of the shipped interval. Not the in-sample figure. */
  coverage80: number;
  /** Interval calibration, including honestly measured coverage. */
  interval: IntervalCalibration;
  /** Out-of-sample accuracy of the available-hours-weighted cost-center rollup. */
  costCenterMetrics: Metrics;
  training: {
    rows: number;
    people: number;
    periods: number;
    samples: number;
    validationSamples: number;
    features: number;
    /**
     * Which periods the fit actually saw. Monitoring needs this to tell an
     * honest out-of-sample window from one that overlaps training; it is
     * optional so that an artifact written before it existed still loads.
     */
    periodRange?: { first: number; last: number };
  };
}

function costCenterRollupMetrics(predictions: FoldPrediction[]): Metrics {
  const groups = new Map<string, { actual: number; predicted: number; weight: number }[]>();
  for (const p of predictions) {
    const key = `${p.sample.costCenter}|${p.sample.targetPeriod}`;
    const entry = groups.get(key) ?? [];
    entry.push({ actual: p.sample.y, predicted: p.predicted, weight: p.sample.weight });
    groups.set(key, entry);
  }
  const actual: number[] = [];
  const predicted: number[] = [];
  for (const rows of groups.values()) {
    const weight = rows.reduce((a, r) => a + r.weight, 0);
    actual.push(rows.reduce((a, r) => a + r.actual * r.weight, 0) / weight);
    predicted.push(rows.reduce((a, r) => a + r.predicted * r.weight, 0) / weight);
  }
  return evaluate(actual, predicted);
}

export function trainForecaster(
  records: PeriodedRecord[],
  horizon = 1,
): TrainedForecaster {
  const samples = buildTrainingSamples(records, horizon);
  if (samples.length === 0) throw new Error('No training samples could be built');

  // Headline numbers come from the nested run, where each fold's penalty is
  // chosen from that fold's own past.
  const nested = crossValidateNested(samples);
  const predictions = nested.predictions;
  const actual = predictions.map(p => p.sample.y);
  const metrics = evaluate(actual, predictions.map(p => p.predicted));

  // The same protocol with the penalty picked on the reported folds. Keeping it
  // makes the size of that optimism visible instead of leaving it to be
  // rediscovered later.
  const byLambda = LAMBDA_GRID.map(lambda => {
    const cv = crossValidate(samples, lambda);
    const m = evaluate(cv.predictions.map(p => p.sample.y), cv.predictions.map(p => p.predicted));
    return { lambda, mae: m.mae, rmse: m.rmse, cv };
  });
  const best = byLambda.reduce((a, b) => (b.mae < a.mae ? b : a));
  const optimisticMetrics = evaluate(
    best.cv.predictions.map(p => p.sample.y),
    best.cv.predictions.map(p => p.predicted),
  );

  const intervals = calibrateIntervals(predictions);

  // The deployed model is fitted on every closed period, with the penalty chosen
  // the same way a fold would choose it - from everything that closed before the
  // period being predicted, which here is all of it.
  const lambda = selectLambdaBefore(samples, Math.max(...samples.map(s => s.targetPeriod)) + 1);
  const model = fitRidge(
    samples.map(s => s.x),
    samples.map(s => s.y - s.anchor),
    lambda,
    FEATURE_NAMES,
    true,
  );

  const people = new Set(records.map(r => `${r.costCenter}|${r.personName}`));
  const periods = new Set(records.map(r => r.periodIndex));

  return {
    model,
    trainedAt: new Date().toISOString(),
    metrics,
    optimisticMetrics,
    byFold: nested.byFold,
    lambdaByFold: nested.lambdaByFold,
    byLambda: byLambda.map(({ lambda: l, mae, rmse }) => ({ lambda: l, mae, rmse })),
    baselines: BASELINES.map(name => ({
      name,
      metrics: evaluate(actual, predictions.map(p => baselinePrediction(p.sample, name))),
    })),
    sigma: intervals.sigma,
    fallbackSigma: (() => {
      const errors = predictions.map(p => baselinePrediction(p.sample, 'last_period') - p.sample.y);
      const m = errors.reduce((a, b) => a + b, 0) / errors.length;
      return Math.sqrt(errors.reduce((a, b) => a + (b - m) ** 2, 0) / errors.length);
    })(),
    coverage80:
      intervals.methods.find(m => m.name === intervals.shipped)?.coverageWalkForward ?? 0,
    interval: intervals,
    costCenterMetrics: costCenterRollupMetrics(predictions),
    training: {
      rows: records.length,
      people: people.size,
      periods: periods.size,
      samples: samples.length,
      validationSamples: predictions.length,
      features: FEATURE_NAMES.length,
      periodRange: {
        first: Math.min(...records.map(r => r.periodIndex)),
        last: Math.max(...records.map(r => r.periodIndex)),
      },
    },
  };
}

/**
 * How a person's forecast was produced.
 *
 * The model needs three closed periods of history. Real rosters do not oblige:
 * people join, transfer between cost centres and leave. Silently dropping those
 * people - which is what happens if you only ever build model samples - produces
 * a forecast that quietly covers less of the firm than it appears to, and the
 * gap shows up as a headcount discrepancy nobody can explain. So everyone on the
 * roster gets a row and a label saying how it was made.
 */
export type ForecastMethod =
  /** Full model forecast: >= 3 periods of history, present in the last period. */
  | 'model'
  /** 1-2 periods of history: carried forward from the last observation. */
  | 'short_history'
  /** No usable history at all: the cost-centre and job-level median. */
  | 'cold_start';

export interface PersonForecast {
  personName: string;
  costCenter: string;
  costCenterName: string;
  jobLevel: string;
  targetType: string;
  targetPeriod: number;
  targetMonth: string;
  lastUtil: number;
  utilTarget: number;
  forecastUtil: number;
  low80: number;
  high80: number;
  forecastVariance: number;
  expectedAvailHours: number;
  /** How this row was produced; anything but `model` is a fallback. */
  method: ForecastMethod;
  /** Closed periods of history behind this row. */
  periodsOfHistory: number;
}

/** People on the roster who are deliberately not forecast, and why. */
export interface ExcludedPerson {
  personName: string;
  costCenter: string;
  /** Last period this person appears in. */
  lastSeenPeriod: number;
  reason: 'absent_from_last_period';
}

export interface ForecastResult {
  forecasts: PersonForecast[];
  excluded: ExcludedPerson[];
  /** Row counts per method, so coverage is visible without recomputing it. */
  coverage: { method: ForecastMethod | 'excluded'; people: number }[];
}

function monthLabel(periodIndex: number, firstMonth: string): string {
  const [year, month] = firstMonth.split('-').map(Number);
  const zeroBased = (month - 1) + (periodIndex - 1);
  return `${year + Math.floor(zeroBased / 12)}-${String((zeroBased % 12) + 1).padStart(2, '0')}`;
}

/**
 * Forecasts every person on the roster for the period after the data ends.
 *
 * `forecastAll` returns the fallbacks and exclusions alongside the model rows;
 * `forecastNextPeriod` keeps the original signature and returns just the rows.
 */
export function forecastAll(
  trained: TrainedForecaster,
  records: PeriodedRecord[],
): ForecastResult {
  const firstPeriod = Math.min(...records.map(r => r.periodIndex));
  const lastPeriod = Math.max(...records.map(r => r.periodIndex));
  const firstMonth = records.find(r => r.periodIndex === firstPeriod)?.periodMonth ?? '2025-10';
  const targetPeriod = lastPeriod + 1;
  const targetMonth = monthLabel(targetPeriod - firstPeriod + 1, firstMonth);

  const shippedInterval =
    trained.interval.methods.find(m => m.name === trained.interval.shipped) ??
    { name: 'gaussian', low: -Z80 * trained.sigma, high: Z80 * trained.sigma, coverageWalkForward: 0 };
  // A fallback row is not a model prediction, so it does not get the model's
  // interval. The naive baseline's own error spread is the honest width for it.
  const fallbackHalfWidth = Z80 * trained.fallbackSigma;

  const panel = groupByPerson(records);
  const modelled = new Map<string, Sample>(
    buildForecastSamples(records).map(s => [`${s.costCenter}|${s.personName}`, s]),
  );

  // Cost-centre x job-level medians, for people with nothing else to go on.
  const cohorts = new Map<string, number[]>();
  for (const record of records) {
    if (record.periodIndex < lastPeriod - 2) continue;
    const key = `${record.costCenter}|${record.jobLevel}`;
    const list = cohorts.get(key) ?? [];
    list.push(record.utilPct);
    cohorts.set(key, list);
  }
  const cohortMedian = (costCenter: string, jobLevel: string, fallback: number): number => {
    const values = cohorts.get(`${costCenter}|${jobLevel}`);
    if (!values || values.length === 0) return fallback;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  };

  const forecasts: PersonForecast[] = [];
  const excluded: ExcludedPerson[] = [];

  for (const [key, rows] of panel) {
    const last = rows[rows.length - 1];
    if (last.periodIndex !== lastPeriod) {
      // Left the roster, or the extract is missing them. Either way there is
      // nothing to forecast, and saying so beats a silent omission.
      excluded.push({
        personName: last.personName,
        costCenter: last.costCenter,
        lastSeenPeriod: last.periodIndex,
        reason: 'absent_from_last_period',
      });
      continue;
    }

    const sample = modelled.get(key);
    const base = {
      personName: last.personName,
      costCenter: last.costCenter,
      costCenterName: last.costCenterName,
      jobLevel: last.jobLevel,
      targetType: last.targetType,
      targetPeriod,
      targetMonth,
      utilTarget: last.utilPctTarget,
      periodsOfHistory: rows.length,
    };

    if (sample) {
      const forecastUtil = clampUtil(sample.anchor + predictOne(trained.model, sample.x));
      forecasts.push({
        ...base,
        lastUtil: sample.baselines.last,
        forecastUtil,
        low80: clampUtil(forecastUtil + shippedInterval.low),
        high80: clampUtil(forecastUtil + shippedInterval.high),
        forecastVariance: forecastUtil - sample.utilTarget,
        expectedAvailHours: sample.weight,
        method: 'model',
      });
      continue;
    }

    // One or two closed periods: the model cannot build a feature row, but
    // "same as last period" is still the strongest thing available.
    const forecastUtil = clampUtil(last.utilPct);
    forecasts.push({
      ...base,
      lastUtil: last.utilPct,
      forecastUtil,
      low80: clampUtil(forecastUtil - fallbackHalfWidth),
      high80: clampUtil(forecastUtil + fallbackHalfWidth),
      forecastVariance: forecastUtil - last.utilPctTarget,
      expectedAvailHours: rows.reduce((a, r) => a + r.availTotal, 0) / rows.length,
      method: 'short_history',
    });
  }

  // Anyone in the last period the panel never grouped (no history at all) would
  // land here; with a complete extract this is empty, but real feeds are not
  // always complete.
  const seen = new Set(forecasts.map(f => `${f.costCenter}|${f.personName}`));
  for (const record of records.filter(r => r.periodIndex === lastPeriod)) {
    const key = `${record.costCenter}|${record.personName}`;
    if (seen.has(key)) continue;
    const forecastUtil = clampUtil(cohortMedian(record.costCenter, record.jobLevel, record.utilPctTarget));
    forecasts.push({
      personName: record.personName,
      costCenter: record.costCenter,
      costCenterName: record.costCenterName,
      jobLevel: record.jobLevel,
      targetType: record.targetType,
      targetPeriod,
      targetMonth,
      lastUtil: record.utilPct,
      utilTarget: record.utilPctTarget,
      forecastUtil,
      low80: clampUtil(forecastUtil - fallbackHalfWidth),
      high80: clampUtil(forecastUtil + fallbackHalfWidth),
      forecastVariance: forecastUtil - record.utilPctTarget,
      expectedAvailHours: record.availTotal,
      method: 'cold_start',
      periodsOfHistory: 0,
    });
    seen.add(key);
  }

  forecasts.sort(
    (a, b) => a.costCenter.localeCompare(b.costCenter) || a.personName.localeCompare(b.personName),
  );

  const methods: (ForecastMethod | 'excluded')[] = ['model', 'short_history', 'cold_start'];
  const coverage = methods.map(method => ({
    method,
    people: forecasts.filter(f => f.method === method).length,
  }));
  coverage.push({ method: 'excluded', people: excluded.length });

  return { forecasts, excluded, coverage };
}

export function forecastNextPeriod(
  trained: TrainedForecaster,
  records: PeriodedRecord[],
): PersonForecast[] {
  return forecastAll(trained, records).forecasts;
}

export interface CostCenterForecast {
  costCenter: string;
  costCenterName: string;
  people: number;
  lastUtil: number;
  forecastUtil: number;
  utilTarget: number;
  forecastVariance: number;
  expectedDirectHours: number;
  belowTarget: number;
}

export function rollupByCostCenter(forecasts: PersonForecast[]): CostCenterForecast[] {
  const groups = new Map<string, PersonForecast[]>();
  for (const forecast of forecasts) {
    const rows = groups.get(forecast.costCenter) ?? [];
    rows.push(forecast);
    groups.set(forecast.costCenter, rows);
  }

  const weighted = (rows: PersonForecast[], pick: (f: PersonForecast) => number) => {
    const weight = rows.reduce((a, r) => a + r.expectedAvailHours, 0);
    return rows.reduce((a, r) => a + pick(r) * r.expectedAvailHours, 0) / weight;
  };

  return [...groups.values()]
    .map(rows => {
      const forecastUtil = weighted(rows, r => r.forecastUtil);
      const utilTarget = weighted(rows, r => r.utilTarget);
      return {
        costCenter: rows[0].costCenter,
        costCenterName: rows[0].costCenterName,
        people: rows.length,
        lastUtil: weighted(rows, r => r.lastUtil),
        forecastUtil,
        utilTarget,
        forecastVariance: forecastUtil - utilTarget,
        expectedDirectHours: rows.reduce(
          (a, r) => a + (r.forecastUtil / 100) * r.expectedAvailHours,
          0,
        ),
        belowTarget: rows.filter(r => r.forecastVariance < 0).length,
      };
    })
    .sort((a, b) => a.costCenter.localeCompare(b.costCenter));
}

/** Standardized coefficients ranked by absolute size - the model's drivers. */
export function featureImportance(model: RidgeModel): { feature: string; coefficient: number }[] {
  return model.featureNames
    .map((feature, i) => ({ feature, coefficient: model.coefficients[i] }))
    .sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient));
}
