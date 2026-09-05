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
): RidgeModel {
  const n = X.length;
  if (n === 0) throw new Error('Cannot fit a model on an empty design matrix');
  const p = X[0].length;

  const means = new Array<number>(p).fill(0);
  const sds = new Array<number>(p).fill(0);
  for (let j = 0; j < p; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += X[i][j];
    means[j] = sum / n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (X[i][j] - means[j]) ** 2;
    // Constant columns (an unused dummy in a fold) get sd 1 so they contribute nothing.
    sds[j] = Math.sqrt(variance / n) || 1;
  }

  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const Z: number[][] = X.map(row => row.map((v, j) => (v - means[j]) / sds[j]));
  const yc = y.map(v => v - yMean);

  const gram: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const rhs = new Array<number>(p).fill(0);
  for (let i = 0; i < n; i++) {
    const row = Z[i];
    for (let j = 0; j < p; j++) {
      const zj = row[j];
      if (zj !== 0) {
        for (let k = j; k < p; k++) gram[j][k] += zj * row[k];
        rhs[j] += zj * yc[i];
      }
    }
  }
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) gram[j][k] = gram[k][j];
    gram[j][j] += lambda * n;
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
