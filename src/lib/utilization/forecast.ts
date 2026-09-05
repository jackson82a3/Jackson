import { buildForecastSamples, buildTrainingSamples, FEATURE_NAMES } from './features.ts';
import type { Sample } from './features.ts';
import { fitRidge, predictOne } from './ridge.ts';
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

export interface TrainedForecaster {
  model: RidgeModel;
  trainedAt: string;
  /** Out-of-sample metrics from the rolling-origin folds. */
  metrics: Metrics;
  byFold: CrossValidation['byFold'];
  byLambda: { lambda: number; mae: number; rmse: number }[];
  baselines: { name: BaselineName; metrics: Metrics }[];
  /** Residual sd from the pooled folds, used for prediction intervals. */
  sigma: number;
  /** Share of validation rows the 80% interval actually covered. */
  coverage80: number;
  /** Out-of-sample accuracy of the available-hours-weighted cost-center rollup. */
  costCenterMetrics: Metrics;
  training: {
    rows: number;
    people: number;
    periods: number;
    samples: number;
    validationSamples: number;
    features: number;
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

export function trainForecaster(records: PeriodedRecord[]): TrainedForecaster {
  const samples = buildTrainingSamples(records);
  if (samples.length === 0) throw new Error('No training samples could be built');

  const byLambda = LAMBDA_GRID.map(lambda => {
    const cv = crossValidate(samples, lambda);
    const metrics = evaluate(cv.predictions.map(p => p.sample.y), cv.predictions.map(p => p.predicted));
    return { lambda, mae: metrics.mae, rmse: metrics.rmse, cv };
  });
  const best = byLambda.reduce((a, b) => (b.mae < a.mae ? b : a));

  const predictions = best.cv.predictions;
  const actual = predictions.map(p => p.sample.y);
  const fitted = predictions.map(p => p.predicted);
  const metrics = evaluate(actual, fitted);

  const residuals = predictions.map(p => p.predicted - p.sample.y);
  const residualMean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
  const sigma = Math.sqrt(
    residuals.reduce((a, b) => a + (b - residualMean) ** 2, 0) / residuals.length,
  );
  const coverage80 =
    residuals.filter(r => Math.abs(r) <= Z80 * sigma).length / residuals.length;

  // Final model: refit on every sample, including the validation folds.
  const model = fitRidge(
    samples.map(s => s.x),
    samples.map(s => s.y - s.anchor),
    best.lambda,
    FEATURE_NAMES,
    true,
  );

  const people = new Set(records.map(r => `${r.costCenter}|${r.personName}`));
  const periods = new Set(records.map(r => r.periodIndex));

  return {
    model,
    trainedAt: new Date().toISOString(),
    metrics,
    byFold: best.cv.byFold,
    byLambda: byLambda.map(({ lambda, mae, rmse }) => ({ lambda, mae, rmse })),
    baselines: BASELINES.map(name => ({
      name,
      metrics: evaluate(actual, predictions.map(p => baselinePrediction(p.sample, name))),
    })),
    sigma,
    coverage80,
    costCenterMetrics: costCenterRollupMetrics(predictions),
    training: {
      rows: records.length,
      people: people.size,
      periods: periods.size,
      samples: samples.length,
      validationSamples: predictions.length,
      features: FEATURE_NAMES.length,
    },
  };
}

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
}

function monthLabel(periodIndex: number, firstMonth: string): string {
  const [year, month] = firstMonth.split('-').map(Number);
  const zeroBased = (month - 1) + (periodIndex - 1);
  return `${year + Math.floor(zeroBased / 12)}-${String((zeroBased % 12) + 1).padStart(2, '0')}`;
}

export function forecastNextPeriod(
  trained: TrainedForecaster,
  records: PeriodedRecord[],
): PersonForecast[] {
  const firstPeriod = Math.min(...records.map(r => r.periodIndex));
  const firstMonth = records.find(r => r.periodIndex === firstPeriod)?.periodMonth ?? '2025-10';

  return buildForecastSamples(records)
    .map(sample => {
      const forecastUtil = clampUtil(sample.anchor + predictOne(trained.model, sample.x));
      return {
        personName: sample.personName,
        costCenter: sample.costCenter,
        costCenterName: sample.costCenterName,
        jobLevel: sample.jobLevel,
        targetType: sample.targetType,
        targetPeriod: sample.targetPeriod,
        targetMonth: monthLabel(sample.targetPeriod - firstPeriod + 1, firstMonth),
        lastUtil: sample.baselines.last,
        utilTarget: sample.utilTarget,
        forecastUtil,
        low80: clampUtil(forecastUtil - Z80 * trained.sigma),
        high80: clampUtil(forecastUtil + Z80 * trained.sigma),
        forecastVariance: forecastUtil - sample.utilTarget,
        expectedAvailHours: sample.weight,
      };
    })
    .sort((a, b) => a.costCenter.localeCompare(b.costCenter) || a.personName.localeCompare(b.personName));
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
