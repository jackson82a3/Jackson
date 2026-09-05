/**
 * How far ahead is this model still worth using?
 *
 *   npm run util:horizon [-- --max 4]
 *
 * The shipped model predicts one period ahead. It has always been documented as
 * decaying beyond that, toward the person-mean baseline - but that was an
 * assertion, not a measurement, and "decays" is not something a planner can act
 * on. This measures it: the same features, the same nested penalty selection,
 * the same rolling origin, asking a harder question each time.
 *
 * The number that matters is not the model's MAE at each horizon but its margin
 * over the best naive baseline at that horizon. A model that degrades while the
 * baselines degrade just as fast has not stopped working; a model whose margin
 * reaches zero has.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseCsv } from '../src/lib/utilization/csv.ts';
import { buildTrainingSamples } from '../src/lib/utilization/features.ts';
import {
  BASELINES,
  baselinePrediction,
  calibrateIntervals,
  crossValidateNested,
  evaluate,
} from '../src/lib/utilization/forecast.ts';
import { formatValidationReport, validateDataset } from '../src/lib/utilization/validate.ts';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const dataPath = path.resolve(process.cwd(), arg('data', 'data/utilization.csv'));
const outPath = path.resolve(process.cwd(), arg('out', 'data/utilization-horizons.json'));
const maxHorizon = Number(arg('max', '4'));

const records = parseCsv(fs.readFileSync(dataPath, 'utf8'));
const validation = validateDataset(records);
if (!validation.ok) {
  console.log(formatValidationReport(validation));
  console.error('Refusing to measure horizons on data with errors.');
  process.exit(1);
}

const results = [];
for (let horizon = 1; horizon <= maxHorizon; horizon++) {
  const samples = buildTrainingSamples(records, horizon);
  if (samples.length === 0) break;
  let cv;
  try {
    cv = crossValidateNested(samples);
  } catch {
    // Not enough periods left to form a fold at this horizon.
    break;
  }
  if (cv.predictions.length === 0) break;

  const actual = cv.predictions.map(p => p.sample.y);
  const metrics = evaluate(actual, cv.predictions.map(p => p.predicted));
  const baselines = BASELINES.map(name => ({
    name,
    metrics: evaluate(actual, cv.predictions.map(p => baselinePrediction(p.sample, name))),
  }));
  const best = baselines.reduce((a, b) => (b.metrics.mae < a.metrics.mae ? b : a));
  const interval = calibrateIntervals(cv.predictions);
  results.push({
    horizon,
    samples: samples.length,
    scored: metrics.n,
    metrics,
    baselines,
    bestBaseline: best.name,
    bestBaselineMae: best.metrics.mae,
    marginPct: (1 - metrics.mae / best.metrics.mae) * 100,
    coverage: interval.methods.find(m => m.name === interval.shipped)?.coverageWalkForward ?? 0,
  });
}

const thin = '-'.repeat(78);
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

console.log('Forecast accuracy by horizon');
console.log('='.repeat(78));
console.log(`${records.length} rows | nested penalty selection | same features at every horizon`);
console.log('');
console.log(
  'horizon  scored      MAE     RMSE  within5  coverage   best baseline        margin',
);
console.log(thin);
for (const r of results) {
  console.log(
    `t+${r.horizon}`.padEnd(9) +
      String(r.scored).padStart(6) +
      r.metrics.mae.toFixed(2).padStart(9) +
      r.metrics.rmse.toFixed(2).padStart(9) +
      pct(r.metrics.within5).padStart(9) +
      pct(r.coverage).padStart(10) +
      `   ${r.bestBaseline} ${r.bestBaselineMae.toFixed(2)}`.padEnd(24) +
      `${r.marginPct >= 0 ? '+' : ''}${r.marginPct.toFixed(1)}%`.padStart(8),
  );
}
console.log('');

console.log('What each baseline does as the horizon grows (MAE, pp)');
console.log('baseline'.padEnd(24) + results.map(r => `t+${r.horizon}`.padStart(9)).join(''));
console.log(thin);
console.log(
  'ridge (this model)'.padEnd(24) + results.map(r => r.metrics.mae.toFixed(2).padStart(9)).join(''),
);
for (const name of BASELINES) {
  console.log(
    name.padEnd(24) +
      results
        .map(r => (r.baselines.find(b => b.name === name)?.metrics.mae ?? Number.NaN).toFixed(2).padStart(9))
        .join(''),
  );
}
console.log('');

// The usable horizon is the longest *unbroken* run of wins from t+1. A later
// horizon that happens to come out ahead after the model has already lost is
// not a continuation of anything - with 300 rows per horizon these margins are
// worth a couple of points either way - and reading it as one would license
// using the model exactly where it has been shown not to work.
let usable = 0;
for (const r of results) {
  if (r.marginPct > 0 && r.horizon === usable + 1) usable = r.horizon;
  else break;
}
if (usable === 0) {
  console.log('The model does not beat the best naive baseline even at t+1.');
} else {
  console.log(
    `Usable horizon: t+${usable}. ` +
      (usable < results.length
        ? `At t+${usable + 1} a naive baseline is at least as good, so do not use the model past t+${usable}.`
        : 'That is the furthest horizon this panel can score.'),
  );
}
const laterWins = results.filter(r => r.horizon > usable + 1 && r.marginPct > 0);
if (laterWins.length > 0) {
  console.log(
    `  (${laterWins.map(r => `t+${r.horizon}`).join(', ')} also come out ahead, but only after the ` +
      'model has already lost at a shorter horizon. Treat that as noise, not as range.)',
  );
}
const first = results[0];
const last = results[results.length - 1];
if (last && first && last.horizon > first.horizon) {
  console.log(
    `Margin over the best baseline goes ${first.marginPct.toFixed(1)}% at t+${first.horizon} -> ` +
      `${last.marginPct.toFixed(1)}% at t+${last.horizon}, while absolute MAE goes ` +
      `${first.metrics.mae.toFixed(2)} -> ${last.metrics.mae.toFixed(2)}pp.`,
  );
}

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(
  outPath,
  JSON.stringify({ generatedAt: new Date().toISOString(), results }, null, 2) + '\n',
);
console.log(`\nWrote ${path.relative(process.cwd(), outPath)}`);
