/**
 * Scores model variants through the shipped protocol.
 *
 *   npm run util:experiment
 *
 * Every variant goes through the same rolling origin, the same nested penalty
 * selection and the same interval calibration as the deployed model - not a copy
 * of that protocol, the actual one, via the `Estimator` seam. That matters
 * because the easiest way to invent an improvement is to evaluate the candidate
 * slightly differently from the incumbent.
 *
 * A variant is only worth shipping if it wins on the honest number *and* the win
 * is bigger than the fold-to-fold noise. The per-fold columns are printed so
 * that judgement can be made rather than assumed.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseCsv } from '../src/lib/utilization/csv.ts';
import { buildTrainingSamples } from '../src/lib/utilization/features.ts';
import {
  calibrateIntervals,
  crossValidateNested,
  ESTIMATORS,
  evaluate,
} from '../src/lib/utilization/forecast.ts';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const dataPath = path.resolve(process.cwd(), arg('data', 'data/utilization.csv'));
const outPath = path.resolve(process.cwd(), arg('out', 'data/utilization-experiments.json'));

const records = parseCsv(fs.readFileSync(dataPath, 'utf8'));
const samples = buildTrainingSamples(records);

const results = Object.entries(ESTIMATORS).map(([name, estimator]) => {
  const cv = crossValidateNested(samples, estimator);
  const metrics = evaluate(
    cv.predictions.map(p => p.sample.y),
    cv.predictions.map(p => p.predicted),
  );
  const interval = calibrateIntervals(cv.predictions);
  return {
    name,
    metrics,
    byFold: cv.byFold,
    lambdaByFold: cv.lambdaByFold,
    coverage: interval.methods.find(m => m.name === interval.shipped)?.coverageWalkForward ?? 0,
  };
});

const baseline = results.find(r => r.name === 'ridge');
if (!baseline) throw new Error('The ridge baseline must be present to compare against');

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const thin = '-'.repeat(78);
const folds = baseline.byFold.map(f => f.targetPeriod);

console.log('Model variants, scored through the shipped protocol');
console.log('='.repeat(78));
console.log(`${samples.length} samples | ${baseline.metrics.n} scored out-of-sample | nested penalty selection`);
console.log('');

console.log('variant                    MAE   vs ridge     RMSE     bias  within5  coverage');
console.log(thin);
for (const r of [...results].sort((a, b) => a.metrics.mae - b.metrics.mae)) {
  const delta = r.metrics.mae - baseline.metrics.mae;
  console.log(
    r.name.padEnd(20) +
      r.metrics.mae.toFixed(3).padStart(8) +
      `${delta >= 0 ? '+' : '-'}${Math.abs(delta).toFixed(3)}`.padStart(11) +
      r.metrics.rmse.toFixed(2).padStart(9) +
      `${r.metrics.bias >= 0 ? '+' : '-'}${Math.abs(r.metrics.bias).toFixed(2)}`.padStart(9) +
      pct(r.metrics.within5).padStart(9) +
      pct(r.coverage).padStart(10),
  );
}
console.log('');

console.log('MAE by fold (the win has to be bigger than what moves between these)');
console.log('variant           ' + folds.map(f => `P${String(f).padStart(2, '0')}`.padStart(9)).join(''));
console.log(thin);
for (const r of results) {
  console.log(
    r.name.padEnd(18) +
      folds
        .map(f => (r.byFold.find(b => b.targetPeriod === f)?.mae ?? Number.NaN).toFixed(2).padStart(9))
        .join(''),
  );
}
console.log('');

// A variant that wins on average but loses on most folds has won a lottery, not
// an argument, so count both.
console.log('Fold-level record against ridge');
console.log(thin);
for (const r of results.filter(r => r.name !== 'ridge')) {
  let wins = 0;
  let losses = 0;
  let worstRegression = 0;
  for (const fold of r.byFold) {
    const mine = fold.mae;
    const theirs = baseline.byFold.find(b => b.targetPeriod === fold.targetPeriod)?.mae ?? mine;
    if (mine < theirs) wins++;
    else if (mine > theirs) losses++;
    worstRegression = Math.max(worstRegression, mine - theirs);
  }
  const delta = r.metrics.mae - baseline.metrics.mae;
  const verdict =
    delta < 0 && wins > losses
      ? 'better'
      : delta < 0
        ? 'better on average, not on most folds'
        : 'no better';
  console.log(
    `  ${r.name.padEnd(18)} wins ${wins}/${r.byFold.length} folds, worst single-fold regression ` +
      `+${worstRegression.toFixed(2)}pp - ${verdict}`,
  );
}
console.log('');

const best = [...results].sort((a, b) => a.metrics.mae - b.metrics.mae)[0];
console.log(
  best.name === 'ridge'
    ? 'Nothing beats the shipped model. Keeping ridge.'
    : `Best on MAE is ${best.name} (${(baseline.metrics.mae - best.metrics.mae).toFixed(3)}pp better). ` +
        'Check the fold record above before shipping it.',
);

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(
  outPath,
  JSON.stringify({ generatedAt: new Date().toISOString(), results }, null, 2) + '\n',
);
console.log(`\nWrote ${path.relative(process.cwd(), outPath)}`);
