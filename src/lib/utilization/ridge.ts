/**
 * Ridge regression on standardized features, solved through the normal equations.
 *
 * The design here is small (a few hundred rows, ~30 columns), so an explicit
 * Gram matrix plus Gaussian elimination is both fast and easier to audit than an
 * iterative solver. Features are standardized and the response is centered, so
 * the penalty treats every column alike and the intercept stays unpenalized.
 */

export interface RidgeModel {
  featureNames: string[];
  /** Coefficients in standardized feature space, aligned with `featureNames`. */
  coefficients: number[];
  means: number[];
  sds: number[];
  intercept: number;
  lambda: number;
}

/** Solves `A x = b` for a square, well-conditioned A (partial pivoting). */
export function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const m = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (Math.abs(m[pivot][col]) < 1e-12) {
      throw new Error(`Singular system at column ${col}; increase the ridge penalty`);
    }
    [m[col], m[pivot]] = [m[pivot], m[col]];

    const diag = m[col][col];
    for (let row = col + 1; row < n; row++) {
      const factor = m[row][col] / diag;
      if (factor === 0) continue;
      for (let k = col; k <= n; k++) m[row][k] -= factor * m[col][k];
    }
  }

  const x = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = m[row][n];
    for (let k = row + 1; k < n; k++) sum -= m[row][k] * x[k];
    x[row] = sum / m[row][row];
  }
  return x;
}

export function fitRidge(
  X: number[][],
  y: number[],
  lambda: number,
  featureNames: string[],
  penalizeIntercept = false,
  weights?: number[],
): RidgeModel {
  const n = X.length;
  if (n === 0) throw new Error('Cannot fit a model on an empty design matrix');
  const p = X[0].length;
  const w = weights ?? new Array<number>(n).fill(1);
  if (w.length !== n) throw new Error(`Got ${w.length} weights for ${n} rows`);
  const wSum = w.reduce((a, b) => a + b, 0);
  if (!(wSum > 0)) throw new Error('Weights must sum to something positive');

  const means = new Array<number>(p).fill(0);
  const sds = new Array<number>(p).fill(0);
  for (let j = 0; j < p; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += w[i] * X[i][j];
    means[j] = sum / wSum;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += w[i] * (X[i][j] - means[j]) ** 2;
    // Constant columns (an unused dummy in a fold) get sd 1 so they contribute nothing.
    sds[j] = Math.sqrt(variance / wSum) || 1;
  }

  const yMean = y.reduce((a, b, i) => a + w[i] * b, 0) / wSum;
  const Z: number[][] = X.map(row => row.map((v, j) => (v - means[j]) / sds[j]));
  const yc = y.map(v => v - yMean);

  const gram: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const rhs = new Array<number>(p).fill(0);
  for (let i = 0; i < n; i++) {
    const row = Z[i];
    const wi = w[i];
    if (wi === 0) continue;
    for (let j = 0; j < p; j++) {
      const zj = row[j] * wi;
      if (zj !== 0) {
        for (let k = j; k < p; k++) gram[j][k] += zj * row[k];
        rhs[j] += zj * yc[i];
      }
    }
  }
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) gram[j][k] = gram[k][j];
    // Scaled by the weight mass rather than the row count, so a given lambda
    // means the same thing whether or not rows are downweighted.
    gram[j][j] += lambda * wSum;
  }

  return {
    featureNames,
    coefficients: solveLinearSystem(gram, rhs),
    means,
    sds,
    intercept: penalizeIntercept ? yMean / (1 + lambda) : yMean,
    lambda,
  };
}

/** Median of a copy of `values`; used for the robust scale below. */
function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Ridge under a Huber loss, by iteratively reweighted least squares.
 *
 * Squared error is what ridge minimizes, but MAE is what this forecaster is
 * judged on, and the two disagree about outliers: one person having a terrible
 * month pulls a least-squares fit toward it much harder than it pulls the median
 * error. Huber is quadratic for small residuals and linear beyond `delta`, so
 * the fit stops chasing the tail.
 *
 * `delta` is set from the MAD of the current residuals (1.345 sigma is the
 * standard choice, giving ~95% of least-squares efficiency on clean Gaussian
 * data), so it adapts to each fold rather than being a tuned constant.
 */
export function fitHuberRidge(
  X: number[][],
  y: number[],
  lambda: number,
  featureNames: string[],
  penalizeIntercept = false,
  iterations = 8,
): RidgeModel {
  let model = fitRidge(X, y, lambda, featureNames, penalizeIntercept);
  for (let iteration = 0; iteration < iterations; iteration++) {
    const residuals = X.map((x, i) => y[i] - predictOne(model, x));
    // 1.4826 rescales the MAD to a Gaussian-comparable sd.
    const scale = 1.4826 * median(residuals.map(r => Math.abs(r))) || 1e-9;
    const delta = 1.345 * scale;
    const weights = residuals.map(r => (Math.abs(r) <= delta ? 1 : delta / Math.abs(r)));
    const next = fitRidge(X, y, lambda, featureNames, penalizeIntercept, weights);
    const shift = Math.max(
      ...next.coefficients.map((c, j) => Math.abs(c - model.coefficients[j])),
      Math.abs(next.intercept - model.intercept),
    );
    model = next;
    if (shift < 1e-9) break;
  }
  return model;
}

export function predictOne(model: RidgeModel, x: number[]): number {
  let sum = model.intercept;
  for (let j = 0; j < model.coefficients.length; j++) {
    sum += model.coefficients[j] * ((x[j] - model.means[j]) / model.sds[j]);
  }
  return sum;
}

export function predictAll(model: RidgeModel, X: number[][]): number[] {
  return X.map(x => predictOne(model, x));
}
