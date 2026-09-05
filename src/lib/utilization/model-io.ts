import { FEATURE_NAMES } from './features.ts';
import type { IntervalCalibration, Metrics, TrainedForecaster } from './forecast.ts';
import type { RidgeModel } from './ridge.ts';

/**
 * Reading and writing the trained model as a file.
 *
 * The point of the checks here is that a model artifact outlives the code that
 * wrote it. Someone re-runs a forecast next quarter against a checkout where the
 * feature list has moved on by one column, and a ridge model will happily
 * multiply the wrong coefficient by the wrong feature and return a plausible
 * number. Nothing crashes and nothing looks wrong. So loading refuses anything
 * it cannot prove is compatible, rather than scoring on a guess.
 */

export const MODEL_KIND = 'utilization-next-period-ridge';

/**
 * Bumped when the artifact's shape changes in a way older readers cannot handle.
 *
 * 1 - original: coefficients, metrics, a Gaussian sigma.
 * 2 - adds nested penalty selection, interval calibration and fallbackSigma;
 *     `metrics` changed meaning, from penalty-tuned-on-reported-folds to the
 *     nested estimate, so a v1 artifact's numbers are not comparable to a v2's.
 */
export const MODEL_FORMAT_VERSION = 2;

export interface ModelArtifact {
  kind: typeof MODEL_KIND;
  formatVersion: number;
  trainedAt: string;
  dataset: { file: string } & TrainedForecaster['training'];
  lambda: number;
  lambdaGrid: TrainedForecaster['byLambda'];
  lambdaByFold: TrainedForecaster['lambdaByFold'];
  /** Nested estimate: each fold's penalty chosen from its own past. */
  metrics: Metrics;
  /** Penalty tuned on the reported folds. Optimistic; kept for comparison. */
  optimisticMetrics: Metrics;
  byFold: TrainedForecaster['byFold'];
  baselines: TrainedForecaster['baselines'];
  costCenterMetrics: Metrics;
  sigma: number;
  fallbackSigma: number;
  coverage80: number;
  /**
   * Deployment weight on the PM plan, present only when the model was trained
   * with allocations. Optional because an artifact trained without them is still
   * perfectly valid - it just cannot blend.
   */
  blendWeight?: number;
  interval: IntervalCalibration;
  model: {
    featureNames: string[];
    coefficients: number[];
    means: number[];
    sds: number[];
    intercept: number;
  };
}

export function serializeModel(
  trained: TrainedForecaster,
  datasetFile: string,
): ModelArtifact {
  return {
    kind: MODEL_KIND,
    formatVersion: MODEL_FORMAT_VERSION,
    trainedAt: trained.trainedAt,
    dataset: { file: datasetFile, ...trained.training },
    lambda: trained.model.lambda,
    lambdaGrid: trained.byLambda,
    lambdaByFold: trained.lambdaByFold,
    metrics: trained.metrics,
    optimisticMetrics: trained.optimisticMetrics,
    byFold: trained.byFold,
    baselines: trained.baselines,
    costCenterMetrics: trained.costCenterMetrics,
    sigma: trained.sigma,
    fallbackSigma: trained.fallbackSigma,
    coverage80: trained.coverage80,
    ...(trained.blendWeight !== undefined ? { blendWeight: trained.blendWeight } : {}),
    interval: trained.interval,
    model: {
      featureNames: trained.model.featureNames,
      coefficients: trained.model.coefficients,
      means: trained.model.means,
      sds: trained.model.sds,
      intercept: trained.model.intercept,
    },
  };
}

export class ModelLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelLoadError';
  }
}

function requireArray(value: unknown, label: string): number[] {
  if (!Array.isArray(value) || value.some(v => typeof v !== 'number' || !Number.isFinite(v))) {
    throw new ModelLoadError(`${label} must be an array of finite numbers.`);
  }
  return value as number[];
}

/**
 * Reads an artifact and returns something safe to score with.
 *
 * Refuses, rather than warns, when the feature contract does not match: a model
 * whose columns mean something different from this code's columns produces
 * numbers that look reasonable and are not.
 */
export function loadModel(raw: unknown): {
  artifact: ModelArtifact;
  model: RidgeModel;
} {
  if (typeof raw !== 'object' || raw === null) {
    throw new ModelLoadError('The model file is not a JSON object.');
  }
  const artifact = raw as Partial<ModelArtifact>;

  if (artifact.kind !== MODEL_KIND) {
    throw new ModelLoadError(
      `Expected a "${MODEL_KIND}" artifact, got "${String(artifact.kind)}".`,
    );
  }
  const version = artifact.formatVersion;
  if (typeof version !== 'number') {
    throw new ModelLoadError(
      'The model file has no formatVersion. It predates versioning (format 1); retrain with ' +
        '`npm run util:train` to produce a readable artifact.',
    );
  }
  if (version !== MODEL_FORMAT_VERSION) {
    throw new ModelLoadError(
      `Model format ${version} cannot be read by this code, which writes format ` +
        `${MODEL_FORMAT_VERSION}. Retrain with \`npm run util:train\`.`,
    );
  }

  const model = artifact.model;
  if (typeof model !== 'object' || model === null) {
    throw new ModelLoadError('The artifact has no `model` block.');
  }

  const names = model.featureNames;
  if (!Array.isArray(names) || names.some(n => typeof n !== 'string')) {
    throw new ModelLoadError('`model.featureNames` must be an array of strings.');
  }
  if (names.length !== FEATURE_NAMES.length || names.some((n, i) => n !== FEATURE_NAMES[i])) {
    const missing = FEATURE_NAMES.filter(n => !names.includes(n));
    const extra = names.filter(n => !FEATURE_NAMES.includes(n));
    throw new ModelLoadError(
      'The saved model was trained on a different feature set than this code builds, so its ' +
        'coefficients do not line up with the columns they would be multiplied by. Refusing to ' +
        'score.' +
        (missing.length > 0 ? ` Missing from the artifact: ${missing.join(', ')}.` : '') +
        (extra.length > 0 ? ` Not built by this code: ${extra.join(', ')}.` : '') +
        ' Retrain with `npm run util:train`.',
    );
  }

  const coefficients = requireArray(model.coefficients, 'model.coefficients');
  const means = requireArray(model.means, 'model.means');
  const sds = requireArray(model.sds, 'model.sds');
  for (const [label, arr] of [
    ['coefficients', coefficients],
    ['means', means],
    ['sds', sds],
  ] as const) {
    if (arr.length !== names.length) {
      throw new ModelLoadError(
        `model.${label} has ${arr.length} entries for ${names.length} features.`,
      );
    }
  }
  if (sds.some(s => s === 0)) {
    throw new ModelLoadError('model.sds contains a zero, which would divide by zero when scoring.');
  }
  if (typeof model.intercept !== 'number' || !Number.isFinite(model.intercept)) {
    throw new ModelLoadError('model.intercept must be a finite number.');
  }
  if (typeof artifact.lambda !== 'number') {
    throw new ModelLoadError('The artifact has no `lambda`.');
  }
  if (typeof artifact.interval !== 'object' || artifact.interval === null) {
    throw new ModelLoadError(
      'The artifact has no interval calibration, so any interval it produced would be invented.',
    );
  }
  if (typeof artifact.fallbackSigma !== 'number' || !Number.isFinite(artifact.fallbackSigma)) {
    throw new ModelLoadError('The artifact has no `fallbackSigma` for fallback rows.');
  }
  if (artifact.blendWeight !== undefined) {
    const w = artifact.blendWeight;
    // A weight outside [0,1] is not a blend; it extrapolates beyond both inputs.
    if (typeof w !== 'number' || !Number.isFinite(w) || w < 0 || w > 1) {
      throw new ModelLoadError(`blendWeight must be a number in [0, 1], got ${String(w)}.`);
    }
  }

  return {
    artifact: artifact as ModelArtifact,
    model: {
      featureNames: names,
      coefficients,
      means,
      sds,
      intercept: model.intercept,
      lambda: artifact.lambda,
    },
  };
}

/** Rebuilds the parts of a `TrainedForecaster` that forecasting actually reads. */
export function forecasterFromArtifact(
  artifact: ModelArtifact,
  model: RidgeModel,
): TrainedForecaster {
  return {
    model,
    trainedAt: artifact.trainedAt,
    metrics: artifact.metrics,
    optimisticMetrics: artifact.optimisticMetrics,
    byFold: artifact.byFold,
    lambdaByFold: artifact.lambdaByFold,
    byLambda: artifact.lambdaGrid,
    baselines: artifact.baselines,
    sigma: artifact.sigma,
    fallbackSigma: artifact.fallbackSigma,
    coverage80: artifact.coverage80,
    blendWeight: artifact.blendWeight,
    interval: artifact.interval,
    costCenterMetrics: artifact.costCenterMetrics,
    training: artifact.dataset,
  };
}
