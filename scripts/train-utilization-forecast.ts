/**
 * Trains the next-period utilization forecaster on data/utilization.csv, scores
 * it out-of-sample against naive baselines, and writes the model plus the
 * next-period forecast.
 *
 *   npm run util:train [-- --data data/utilization.csv --out-dir data]
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseCsv } from '../src/lib/utilization/csv.ts';
import { verifyRecord } from '../src/lib/utilization/types.ts';
import {
  featureImportance,
  forecastAll,
  rollupByCostCenter,
  trainForecaster,
} from '../src/lib/utilization/forecast.ts';
import type { Metrics } from '../src/lib/utilization/forecast.ts';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const dataPath = path.resolve(process.cwd(), arg('data', 'data/utilization.csv'));
const outDir = path.resolve(process.cwd(), arg('out-dir', 'data'));

if (!fs.existsSync(dataPath)) {
  console.error(`No dataset at ${dataPath}. Run \`npm run util:generate\` first.`);
  process.exit(1);
}

const records = parseCsv(fs.readFileSync(dataPath, 'utf8'));
const badRows = records.filter(r => verifyRecord(r).length > 0);
if (badRows.length > 0) {
  console.error(`${badRows.length} rows fail the accounting identities; refusing to train.`);
  console.error(`  e.g. ${badRows[0].personName} in ${badRows[0].sourceName}: ${verifyRecord(badRows[0])[0]}`);
  process.exit(1);
}

const trained = trainForecaster(records);
const result = forecastAll(trained, records);
const forecasts = result.forecasts;
const rollup = rollupByCostCenter(forecasts);

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const row = (cells: (string | number)[], widths: number[]) =>
  cells
    .map((cell, i) => (i === 0 ? String(cell).padEnd(widths[i]) : String(cell).padStart(widths[i])))
    .join('  ');

const metricLine = (label: string, m: Metrics, widths: number[]) =>
  row(
    [label, m.mae.toFixed(2), m.rmse.toFixed(2), m.bias.toFixed(2), m.r2.toFixed(3), pct(m.within5)],
    widths,
  );

const t = trained.training;
console.log('Utilization forecast - training report');
console.log('='.repeat(78));
console.log(
  `Data      ${t.rows} rows | ${t.people} people | ${t.periods} periods | ${path.relative(process.cwd(), dataPath)}`,
);
console.log(
  `Samples   ${t.samples} person-period pairs | ${t.features} features | ${t.validationSamples} scored out-of-sample`,
);
console.log(
  `Model     ridge, lambda=${trained.model.lambda}, chosen inside each fold's own past (nested selection)`,
);
console.log(
  `          per-fold penalties: ${trained.lambdaByFold.map(l => `P${l.targetPeriod}:${l.lambda}`).join(' ')}`,
);
console.log('');

const widths = [28, 7, 7, 7, 7, 8];
console.log('Out-of-sample accuracy (Util %, percentage points)');
console.log(row(['model', 'MAE', 'RMSE', 'bias', 'R2', 'within5'], widths));
console.log('-'.repeat(78));
console.log(metricLine('ridge (this model)', trained.metrics, widths));
for (const baseline of trained.baselines) {
  console.log(metricLine(`baseline: ${baseline.name}`, baseline.metrics, widths));
}
console.log(
  metricLine('  (penalty tuned on these folds)', trained.optimisticMetrics, widths),
);
const bestBaseline = trained.baselines.reduce((a, b) => (b.metrics.mae < a.metrics.mae ? b : a));
const lift = (1 - trained.metrics.mae / bestBaseline.metrics.mae) * 100;
console.log('');
console.log(`MAE is ${lift.toFixed(1)}% below the best baseline (${bestBaseline.name}).`);
console.log(
  `Cost-center rollup (hours-weighted): MAE ${trained.costCenterMetrics.mae.toFixed(2)}pp, ` +
    `RMSE ${trained.costCenterMetrics.rmse.toFixed(2)}pp over ${trained.costCenterMetrics.n} cost-center periods.`,
);
console.log('');

const iv = trained.interval;
console.log('80% prediction interval - every method tried, coverage measured walk-forward');
console.log(row(['method', 'interval', 'coverage'], [28, 20, 12]));
console.log('-'.repeat(78));
for (const m of [...iv.methods].sort((a, b) => b.coverageWalkForward - a.coverageWalkForward)) {
  const label = m.name === iv.shipped ? `${m.name}  <- shipped` : m.name;
  console.log(
    row(
      [label, `${m.low.toFixed(1)} / +${m.high.toFixed(1)}pp`, pct(m.coverageWalkForward)],
      [28, 20, 12],
    ),
  );
}
console.log(
  `Nominal coverage is 80%; each fold's interval is calibrated only on folds before it ` +
    `(${iv.nWalkForward} rows).`,
);
console.log('');

console.log('Forecast coverage of the roster');
console.log(row(['method', 'people'], [28, 9]));
console.log('-'.repeat(78));
for (const c of result.coverage) console.log(row([c.method, c.people], [28, 9]));
if (result.excluded.length > 0) {
  console.log(
    `Excluded (absent from the last period): ${result.excluded
      .slice(0, 5)
      .map(e => `${e.personName} (last seen P${e.lastSeenPeriod})`)
      .join(', ')}${result.excluded.length > 5 ? ', ...' : ''}`,
  );
}
console.log('');

console.log('Coverage of the shipped interval, by fold');
console.log(row(['period', 'n', 'half width', 'coverage'], [28, 7, 12, 10]));
console.log('-'.repeat(78));
for (const f of iv.byFold) {
  console.log(
    row(
      [`P${String(f.targetPeriod).padStart(2, '0')}`, f.n, `${f.halfWidth.toFixed(1)}pp`, pct(f.coverage)],
      [28, 7, 12, 10],
    ),
  );
}
console.log('');

console.log('Accuracy by validation fold');
console.log(row(['period', 'n', 'MAE', 'RMSE'], [28, 7, 7, 7]));
console.log('-'.repeat(78));
for (const fold of trained.byFold) {
  console.log(
    row(
      [`P${String(fold.targetPeriod).padStart(2, '0')}`, fold.n, fold.mae.toFixed(2), fold.rmse.toFixed(2)],
      [28, 7, 7, 7],
    ),
  );
}
console.log('');

console.log('Top drivers (standardized coefficients)');
for (const { feature, coefficient } of featureImportance(trained.model).slice(0, 10)) {
  console.log(`  ${feature.padEnd(26)} ${coefficient >= 0 ? '+' : ''}${coefficient.toFixed(3)}`);
}
console.log('');

const nextMonth = forecasts[0]?.targetMonth ?? 'next period';
console.log(`Forecast for ${nextMonth} (P${forecasts[0]?.targetPeriod ?? '?'})`);
console.log(row(['cost center', 'people', 'last', 'fcst', 'target', 'var', 'below'], [24, 7, 7, 7, 7, 7, 6]));
console.log('-'.repeat(78));
for (const cc of rollup) {
  console.log(
    row(
      [
        `${cc.costCenter} ${cc.costCenterName}`.slice(0, 24),
        cc.people,
        cc.lastUtil.toFixed(1),
        cc.forecastUtil.toFixed(1),
        cc.utilTarget.toFixed(1),
        `${cc.forecastVariance >= 0 ? '+' : ''}${cc.forecastVariance.toFixed(1)}`,
        cc.belowTarget,
      ],
      [24, 7, 7, 7, 7, 7, 6],
    ),
  );
}
const firmAvail = forecasts.reduce((a, f) => a + f.expectedAvailHours, 0);
const firmForecast =
  forecasts.reduce((a, f) => a + f.forecastUtil * f.expectedAvailHours, 0) / firmAvail;
console.log('-'.repeat(78));
console.log(
  `Firm: ${forecasts.length} people, forecast ${firmForecast.toFixed(1)}% util on ` +
    `${Math.round(firmAvail).toLocaleString('en-US')} available hours.`,
);
console.log('');

const atRisk = forecasts
  .filter(f => f.forecastVariance < -8)
  .sort((a, b) => a.forecastVariance - b.forecastVariance)
  .slice(0, 10);
if (atRisk.length > 0) {
  console.log('Largest forecast shortfalls vs target');
  for (const f of atRisk) {
    console.log(
      `  ${f.personName.padEnd(22)} ${f.costCenter}  last ${f.lastUtil.toFixed(1).padStart(5)}  ` +
        `fcst ${f.forecastUtil.toFixed(1).padStart(5)} (${f.low80.toFixed(1)}-${f.high80.toFixed(1)})  ` +
        `target ${f.utilTarget.toFixed(0)}  ${f.forecastVariance.toFixed(1)}pp`,
    );
  }
  console.log('');
}

fs.mkdirSync(outDir, { recursive: true });

const modelPath = path.join(outDir, 'utilization-model.json');
fs.writeFileSync(
  modelPath,
  JSON.stringify(
    {
      kind: 'utilization-next-period-ridge',
      trainedAt: trained.trainedAt,
      dataset: { file: path.relative(process.cwd(), dataPath), ...trained.training },
      lambda: trained.model.lambda,
      lambdaGrid: trained.byLambda,
      metrics: trained.metrics,
      byFold: trained.byFold,
      baselines: trained.baselines,
      costCenterMetrics: trained.costCenterMetrics,
      sigma: trained.sigma,
      coverage80: trained.coverage80,
      model: {
        featureNames: trained.model.featureNames,
        coefficients: trained.model.coefficients,
        means: trained.model.means,
        sds: trained.model.sds,
        intercept: trained.model.intercept,
      },
    },
    null,
    2,
  ) + '\n',
);

const forecastPath = path.join(outDir, 'utilization-forecast.csv');
const header = [
  'Cost Center',
  'Cost Center Name',
  'Person Name',
  'Target Type',
  'Job Level',
  'Forecast Period',
  'Forecast Month',
  'Last Util %',
  'Forecast Util %',
  'Forecast Low 80',
  'Forecast High 80',
  'Util % Target',
  'Forecast Variance',
  'Expected Avail Hours',
].join(',');
const lines = forecasts.map(f =>
  [
    f.costCenter,
    f.costCenterName,
    f.personName,
    f.targetType,
    f.jobLevel,
    f.targetPeriod,
    f.targetMonth,
    f.lastUtil.toFixed(2),
    f.forecastUtil.toFixed(2),
    f.low80.toFixed(2),
    f.high80.toFixed(2),
    f.utilTarget.toFixed(2),
    f.forecastVariance.toFixed(2),
    f.expectedAvailHours.toFixed(1),
  ].join(','),
);
fs.writeFileSync(forecastPath, [header, ...lines].join('\n') + '\n');

console.log(`Wrote ${path.relative(process.cwd(), modelPath)} and ${path.relative(process.cwd(), forecastPath)}`);
