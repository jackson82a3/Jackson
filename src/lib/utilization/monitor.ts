import { buildTrainingSamples, FEATURE_NAMES } from './features.ts';
import { evaluate } from './forecast.ts';
import type { Metrics } from './forecast.ts';
import type { ModelArtifact } from './model-io.ts';
import { predictOne } from './ridge.ts';
import type { RidgeModel } from './ridge.ts';
import type { PeriodedRecord } from './types.ts';

/**
 * Is the deployed model still working?
 *
 * A backtest describes the past a model was built from. Once it is in use the
 * only question is whether it still holds, and that has two answers with
 * different latencies:
 *
 *   - **Accuracy drift** compares the model's predictions against outcomes that
 *     have since landed. It is the real answer, and it always lags by at least
 *     one period.
 *   - **Feature drift** compares today's inputs against the distribution the
 *     model was standardised on. It needs no outcomes, so it fires immediately -
 *     when a cost centre reorganises or leave patterns shift, this moves before
 *     any accuracy number can.
 *
 * Both are computed here rather than in the script so they can be tested.
 */

export interface AccuracyDrift {
  /** Accuracy over the monitored window. */
  observed: Metrics;
  /** What the artifact claimed when it was trained. */
  claimed: Metrics;
  /** Realised coverage of the shipped interval over the window. */
  observedCoverage: number;
  claimedCoverage: number;
  /** `observed.mae / claimed.mae`; 1 means unchanged. */
  maeRatio: number;
  window: { since: number; samples: number };
  /**
   * True when the window includes periods the model was fitted on, which makes
   * the comparison flattering rather than informative.
   */
  overlapsTraining: boolean;
}

export interface FeatureDrift {
  feature: string;
  /** Mean the model was standardised with. */
  trained: number;
  /** Mean over the most recent period. */
  now: number;
  /** The gap, in training standard deviations. */
  drift: number;
}

function clampUtil(value: number): number {
  return Math.min(100, Math.max(0, value));
}

/**
 * Scores the saved model on every sample whose target period is `since` or later.
 *
 * Returns undefined when no such period has closed yet - which is the normal
 * state right after a retrain, not an error.
 */
export function accuracyDrift(
  artifact: ModelArtifact,
  model: RidgeModel,
  records: PeriodedRecord[],
  since: number,
): AccuracyDrift | undefined {
  const samples = buildTrainingSamples(records).filter(s => s.targetPeriod >= since);
  if (samples.length === 0) return undefined;

  const actual = samples.map(s => s.y);
  const predicted = samples.map(s => clampUtil(s.anchor + predictOne(model, s.x)));
  const observed = evaluate(actual, predicted);

  const shipped = artifact.interval.methods.find(m => m.name === artifact.interval.shipped);
  const observedCoverage = shipped
    ? predicted.filter((p, i) => actual[i] >= p + shipped.low && actual[i] <= p + shipped.high)
        .length / predicted.length
    : 0;

  const trainedThrough = artifact.dataset.periodRange?.last;
  return {
    observed,
    claimed: artifact.metrics,
    observedCoverage,
    claimedCoverage: shipped?.coverageWalkForward ?? 0,
    maeRatio: artifact.metrics.mae > 0 ? observed.mae / artifact.metrics.mae : Number.NaN,
    window: { since, samples: samples.length },
    overlapsTraining: trainedThrough !== undefined && since <= trainedThrough,
  };
}

/**
 * Compares the current feature means against the model's standardisation.
 *
 * Expressed in training standard deviations, because that is the unit the model
 * actually works in: a feature two sds from where it was fitted is being
 * extrapolated on, whatever its raw scale happens to be.
 */
export function featureDrift(
  model: RidgeModel,
  records: PeriodedRecord[],
): FeatureDrift[] {
  const lastPeriod = Math.max(...records.map(r => r.periodIndex));
  const samples = buildTrainingSamples(records);
  const recent = samples.filter(s => s.targetPeriod === lastPeriod);
  const rows = recent.length > 0 ? recent : samples;

  return FEATURE_NAMES.map((feature, j) => {
    const values = rows.map(s => s.x[j]);
    const now =
      values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : Number.NaN;
    return {
      feature,
      trained: model.means[j],
      now,
      drift: (now - model.means[j]) / model.sds[j],
    };
  }).sort((a, b) => Math.abs(b.drift) - Math.abs(a.drift));
}

export interface MonitorThresholds {
  /** How many times the claimed MAE counts as drift. */
  maeRatio: number;
  /** Feature drift threshold, in training standard deviations. */
  featureSds: number;
}

export const DEFAULT_THRESHOLDS: MonitorThresholds = { maeRatio: 1.25, featureSds: 1 };

export interface MonitorVerdict {
  accuracyBreached: boolean;
  driftedFeatures: FeatureDrift[];
  /** True when anything is worth a human looking at it. */
  breached: boolean;
}

export function verdict(
  accuracy: AccuracyDrift | undefined,
  drift: FeatureDrift[],
  thresholds: MonitorThresholds = DEFAULT_THRESHOLDS,
): MonitorVerdict {
  // An in-sample window cannot evidence drift, so it never raises one.
  const accuracyBreached =
    accuracy !== undefined && !accuracy.overlapsTraining && accuracy.maeRatio > thresholds.maeRatio;
  const driftedFeatures = drift.filter(d => Math.abs(d.drift) > thresholds.featureSds);
  return {
    accuracyBreached,
    driftedFeatures,
    breached: accuracyBreached || driftedFeatures.length > 0,
  };
}
