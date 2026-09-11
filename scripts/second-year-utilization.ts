/**
 * Does a second year of data unlock the seasonal effect?
 *
 *   npm run util:secondyear
 *
 * This is the project's own biggest open question. With one period per calendar
 * month, every validation month is a month the model has never seen, so month
 * effects can be neither learned nor validated - which is why month sin/cos was
 * dropped from the feature set and why the worst fold is always P10, the July
 * vacation trough. Two years is the smallest panel where that stops being true.
 *
 * Three things are measured, all on generated 24-period data, all through the
 * shipped protocol:
 *
 *   1. **More data, same features.** Does simply having a second year help?
 *   2. **Month features, now that they can be validated.** The same 18 features
 *      plus sin/cos of the target month.
 *   3. **The July fold specifically**, since that is the one the seasonal story
 *      is about. A month effect that helps on average but not there has not
 *      done the thing it was added for.
 *
 * Scoring starts at period 13 so that every validation month has been seen once
 * before. Scoring the first year would ask the month features to predict months
 * that are still novel, which is the situation they were dropped for.
 *
 * The dataset used here is generated in memory. `data/utilization.csv` stays as
 * it is - every published number is a number about that file.
 */
import fs from 'node:fs';
import path from 'node:path';
import { toCsv, parseCsv } from '../src/lib/utilization/csv.ts';
import { generateDataset } from '../src/lib/utilization/generate.ts';
import { buildTrainingSamples, FEATURE_NAMES } from '../src/lib/utilization/features.ts';
import type { Sample } from '../src/lib/utilization/features.ts';
import {
  BASELINES,
  baselinePrediction,
  crossValidateNested,
  evaluate,
} from '../src/lib/utilization/forecast.ts';
import type { Estimator } from '../src/lib/utilization/forecast.ts';
import { fitRidge } from '../src/lib/utilization/ridge.ts';
import { verifyRecord } from '../src/lib/utilization/types.ts';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const seed = Number(arg('seed', '20260901'));
const firstFold = Number(arg('first-fold', '13'));
const outPath = path.resolve(process.cwd(), arg('out', 'data/utilization-secondyear.json'));
const writeCsv = arg('write-csv', '');

const PERIODS_PER_YEAR = 12;

const twoYears = parseCsv(toCsv(generateDataset({ seed, years: 2 })));
const problems = twoYears.flatMap(r => verifyRecord(r));
if (problems.length > 0) {
  console.error(`Two-year data violates its own identities: ${problems[0]}`);
  process.exit(1);
}
if (writeCsv) {
  fs.writeFileSync(path.resolve(process.cwd(), writeCsv), toCsv(twoYears));
}

const monthOfYearFor = (targetPeriod: number) => (targetPeriod - 1) % PERIODS_PER_YEAR;

/**
 * Two ways of writing down "which month is this".
 *
 * sin/cos is the usual choice and is **mis-specified here on purpose**, stated
 * before running: a single sin/cos pair captures only the first annual harmonic,
 * a smooth one-peak-one-trough cycle. The seasonality in the generator is not
 * smooth - December is -2.5, July -1.5, September +2.0 - so a first harmonic
 * cannot represent it however much data it is given. Month dummies can represent
 * an arbitrary monthly pattern, at the cost of eleven parameters.
 *
 * Both are reported, because "seasonality does not help" and "this encoding of
 * seasonality does not help" are different claims and only the second would be
 * supported by testing sin/cos alone.
 */
const SIN_COS_NAMES = ['month_sin', 'month_cos'];
const withSinCos = (samples: Sample[]): Sample[] =>
  samples.map(s => {
    const angle = (2 * Math.PI * monthOfYearFor(s.targetPeriod)) / PERIODS_PER_YEAR;
    return { ...s, x: [...s.x, Math.sin(angle), Math.cos(angle)] };
  });

// Eleven dummies, not twelve: the omitted month is absorbed by the intercept.
const DUMMY_NAMES = Array.from({ length: PERIODS_PER_YEAR - 1 }, (_, i) => `month_${i + 1}`);
const withDummies = (samples: Sample[]): Sample[] =>
  samples.map(s => {
    const monthOfYear = monthOfYearFor(s.targetPeriod);
    const dummies = Array.from({ length: PERIODS_PER_YEAR - 1 }, (_, i) =>
      monthOfYear === i + 1 ? 1 : 0,
    );
    return { ...s, x: [...s.x, ...dummies] };
  });

const estimatorFor = (names: string[]): Estimator => ({ train, lambda }) =>
  fitRidge(
    train.map(s => s.x),
    train.map(s => s.y - s.anchor),
    lambda,
    names,
    true,
  );

const base = buildTrainingSamples(twoYears);
const runs = [
  { name: 'base 18 features', samples: base, estimator: undefined },
  {
    name: '+ month sin/cos',
    samples: withSinCos(base),
    estimator: estimatorFor([...FEATURE_NAMES, ...SIN_COS_NAMES]),
  },
  {
    name: '+ month dummies',
    samples: withDummies(base),
    estimator: estimatorFor([...FEATURE_NAMES, ...DUMMY_NAMES]),
  },
] as const;

const results = runs.map(run => {
  const cv = crossValidateNested(run.samples, run.estimator, firstFold);
  const actual = cv.predictions.map(p => p.sample.y);
  const metrics = evaluate(actual, cv.predictions.map(p => p.predicted));
  const baselines = BASELINES.map(name => ({
    name,
    metrics: evaluate(actual, cv.predictions.map(p => baselinePrediction(p.sample, name))),
  }));
  const bestBaseline = baselines.reduce((a, b) => (b.metrics.mae < a.metrics.mae ? b : a));
  return { name: run.name, metrics, byFold: cv.byFold, bestBaseline };
});

// The one-year figure, scored the way it is published, for context only. It is
// NOT a like-for-like comparison: different folds, different rows.
const oneYear = buildTrainingSamples(parseCsv(toCsv(generateDataset({ seed, years: 1 }))));
const oneYearCv = crossValidateNested(oneYear);
const oneYearMetrics = evaluate(
  oneYearCv.predictions.map(p => p.sample.y),
  oneYearCv.predictions.map(p => p.predicted),
);

const thin = '-'.repeat(78);
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const signed = (x: number, d = 2) => `${x >= 0 ? '+' : '-'}${Math.abs(x).toFixed(d)}`;

console.log('Does a second year unlock the seasonal effect?');
console.log('='.repeat(78));
console.log(`${twoYears.length} rows over 24 periods | folds from P${firstFold} | seed ${seed}`);
console.log('');

console.log('Out-of-sample accuracy on the second year');
console.log(
  'variant'.padEnd(22) + 'MAE'.padStart(8) + 'RMSE'.padStart(8) + 'bias'.padStart(8) +
    'within5'.padStart(9) + 'best baseline'.padStart(21) + 'margin'.padStart(9),
);
console.log(thin);
for (const r of results) {
  const margin = (1 - r.metrics.mae / r.bestBaseline.metrics.mae) * 100;
  console.log(
    r.name.padEnd(22) +
      r.metrics.mae.toFixed(2).padStart(8) +
      r.metrics.rmse.toFixed(2).padStart(8) +
      signed(r.metrics.bias).padStart(8) +
      pct(r.metrics.within5).padStart(9) +
      `${r.bestBaseline.name} ${r.bestBaseline.metrics.mae.toFixed(2)}`.padStart(21) +
      `${margin >= 0 ? '+' : ''}${margin.toFixed(1)}%`.padStart(9),
  );
}
const [baseRun, ...seasonalRuns] = results;
console.log('');
for (const run of seasonalRuns) {
  const d = run.metrics.mae - baseRun.metrics.mae;
  console.log(
    `  ${run.name.padEnd(20)} changes MAE by ${signed(d, 3)}pp ` +
      `(${((d / baseRun.metrics.mae) * 100).toFixed(1)}%)`,
  );
}
const bestSeasonal = seasonalRuns.reduce((a, b) => (b.metrics.mae < a.metrics.mae ? b : a));
const delta = bestSeasonal.metrics.mae - baseRun.metrics.mae;
console.log('');

// July is the fold the seasonal story is about: month index 9 in this calendar.
const julyPeriods = baseRun.byFold
  .map(f => f.targetPeriod)
  .filter(p => (p - 1) % PERIODS_PER_YEAR === 9);
console.log('The July folds specifically (the vacation trough this was added for)');
console.log('period'.padEnd(20) + 'base'.padStart(10) + seasonalRuns.map(r => r.name.padStart(18)).join(''));
console.log(thin);
for (const period of julyPeriods) {
  const b = baseRun.byFold.find(f => f.targetPeriod === period)?.mae ?? Number.NaN;
  console.log(
    `P${period} (July)`.padEnd(20) +
      b.toFixed(3).padStart(10) +
      seasonalRuns
        .map(r => {
          const m = r.byFold.find(f => f.targetPeriod === period)?.mae ?? Number.NaN;
          return `${m.toFixed(3)} (${signed(m - b, 3)})`.padStart(18);
        })
        .join(''),
  );
}
console.log('');

const record = (run: (typeof results)[number]) => {
  let wins = 0;
  let losses = 0;
  for (const fold of baseRun.byFold) {
    const m = run.byFold.find(f => f.targetPeriod === fold.targetPeriod)?.mae;
    if (m === undefined) continue;
    if (m < fold.mae) wins++;
    else if (m > fold.mae) losses++;
  }
  return { wins, losses };
};
for (const run of seasonalRuns) {
  const { wins, losses } = record(run);
  console.log(`  ${run.name.padEnd(20)} wins ${wins} of ${baseRun.byFold.length} folds, loses ${losses}.`);
}
const { wins, losses } = record(bestSeasonal);
console.log('');

console.log('For context, not a like-for-like comparison:');
console.log(`  one year, folds P8-P12   MAE ${oneYearMetrics.mae.toFixed(2)}pp over ${oneYearMetrics.n} rows`);
console.log(`  two years, folds P${firstFold}-P24  MAE ${baseRun.metrics.mae.toFixed(2)}pp over ${baseRun.metrics.n} rows`);
console.log('  Different folds and different rows, so the gap is not a measure of');
console.log('  what a second year is worth. Only the month comparison above is like for like.');
console.log('');

const verdict =
  delta < -0.05 && wins > losses
    ? `Seasonality helps, best as "${bestSeasonal.name}" (${signed(delta, 3)}pp). A second year ` +
      'is worth collecting, and this encoding should be added once real data covers two years.'
    : delta > 0.05
      ? 'No encoding of seasonality helps even with a second year, on this simulation. Adding ' +
        'month effects to the shipped model is not supported by this evidence.'
      : 'Seasonality makes no material difference on this simulation, in either encoding. ' +
        'Not worth the complexity on this evidence.';
console.log(verdict);
console.log('');

// The second, larger finding: on this panel the model does not beat the naive
// baseline at all. Reporting only the seasonal comparison would bury it.
const baseMargin = (1 - baseRun.metrics.mae / baseRun.bestBaseline.metrics.mae) * 100;
if (baseMargin <= 0) {
  console.log('A bigger result than the seasonal one, and it should not be buried:');
  console.log(
    `  On this two-year panel the model does not beat "${baseRun.bestBaseline.name}" at all - ` +
      `${baseRun.metrics.mae.toFixed(2)}pp against ${baseRun.bestBaseline.metrics.mae.toFixed(2)}pp,`,
  );
  console.log(
    `  a margin of ${baseMargin.toFixed(1)}%. The published +1.0% edge comes from a different`,
  );
  console.log('  simulation and a different fold range, so it does not carry over. Read the');
  console.log('  published edge as "about as good as the naive baseline", not as a reliable gain.');
} else {
  console.log(
    `The model beats "${baseRun.bestBaseline.name}" by ${baseMargin.toFixed(1)}% on this panel.`,
  );
}
console.log('');
console.log('Note this is a *simulated* second year, so it tests whether the protocol can');
console.log('detect a seasonal effect that the generator genuinely puts in - not whether a');
console.log('real firm has one. On real data the answer could differ.');

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(
  outPath,
  JSON.stringify(
    { generatedAt: new Date().toISOString(), seed, firstFold, results, oneYearMetrics },
    null,
    2,
  ) + '\n',
);
console.log(`\nWrote ${path.relative(process.cwd(), outPath)}`);
