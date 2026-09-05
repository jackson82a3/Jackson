/**
 * Forecasts the next period from a *saved* model, without retraining.
 *
 *   npm run util:predict [-- --data data/utilization.csv --model data/utilization-model.json]
 *
 * This is the inference path. `util:train` fits and scores; this one only
 * scores, which is what a monthly run actually needs: the model is a reviewed
 * artifact, and the new extract should not silently change it.
 *
 * It exits non-zero if the data fails validation or the artifact does not match
 * the code, because a forecast produced from a mismatched model is worse than no
 * forecast - it looks exactly like a good one.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseCsv } from '../src/lib/utilization/csv.ts';
import { forecastAll, rollupByCostCenter } from '../src/lib/utilization/forecast.ts';
import {
  forecasterFromArtifact,
  loadModel,
  ModelLoadError,
} from '../src/lib/utilization/model-io.ts';
import { formatValidationReport, validateDataset } from '../src/lib/utilization/validate.ts';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const has = (name: string) => process.argv.includes(`--${name}`);

const dataPath = path.resolve(process.cwd(), arg('data', 'data/utilization.csv'));
const modelPath = path.resolve(process.cwd(), arg('model', 'data/utilization-model.json'));
const outPath = path.resolve(process.cwd(), arg('out', 'data/utilization-forecast.csv'));
const strict = has('strict');

for (const [label, file] of [
  ['dataset', dataPath],
  ['model', modelPath],
] as const) {
  if (!fs.existsSync(file)) {
    console.error(`No ${label} at ${path.relative(process.cwd(), file)}.`);
    process.exit(1);
  }
}

const records = parseCsv(fs.readFileSync(dataPath, 'utf8'));

const validation = validateDataset(records);
// Forecasting does not need the full period history that training needs, so an
// insufficient-periods error is not fatal here.
const blocking = validation.errors.filter(e => e.code !== 'insufficient_periods');
if (validation.issues.length > 0) {
  console.log(formatValidationReport(validation));
  console.log('');
}
if (blocking.length > 0) {
  console.error(`Refusing to forecast: ${blocking.map(e => e.code).join(', ')}.`);
  process.exit(1);
}
if (strict && validation.warnings.length > 0) {
  console.error(`--strict: refusing to forecast with ${validation.warnings.length} warning(s).`);
  process.exit(1);
}

let loaded;
try {
  loaded = loadModel(JSON.parse(fs.readFileSync(modelPath, 'utf8')));
} catch (error) {
  if (error instanceof ModelLoadError) {
    console.error(`Cannot use ${path.relative(process.cwd(), modelPath)}:`);
    console.error(`  ${error.message}`);
    process.exit(1);
  }
  throw error;
}

const trained = forecasterFromArtifact(loaded.artifact, loaded.model);
const result = forecastAll(trained, records);
const rollup = rollupByCostCenter(result.forecasts);

const shipped = trained.interval.methods.find(m => m.name === trained.interval.shipped);
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

console.log('Utilization forecast (from a saved model)');
console.log('='.repeat(78));
console.log(`Model     trained ${loaded.artifact.trainedAt} on ${loaded.artifact.dataset.file}`);
console.log(
  `          lambda=${loaded.artifact.lambda}, out-of-sample MAE ${loaded.artifact.metrics.mae.toFixed(2)}pp`,
);
console.log(`Data      ${records.length} rows | ${path.relative(process.cwd(), dataPath)}`);
console.log(
  `Interval  ${trained.interval.shipped}, ` +
    `${shipped ? `${shipped.low.toFixed(1)}/+${shipped.high.toFixed(1)}pp` : 'n/a'}, ` +
    `${shipped ? pct(shipped.coverageWalkForward) : 'n/a'} realised coverage (nominal 80%)`,
);
console.log('');

console.log('Coverage of the roster');
for (const c of result.coverage) {
  console.log(`  ${c.method.padEnd(16)} ${String(c.people).padStart(4)}`);
}
if (result.excluded.length > 0) {
  console.log('');
  console.log('Excluded (absent from the last period):');
  for (const e of result.excluded.slice(0, 10)) {
    console.log(`  ${e.personName} (${e.costCenter}), last seen P${e.lastSeenPeriod}`);
  }
  if (result.excluded.length > 10) console.log(`  ... and ${result.excluded.length - 10} more`);
}
console.log('');

console.log('Cost centre rollup');
console.log(
  'cost center'.padEnd(30) + 'people'.padStart(8) + 'last'.padStart(9) + 'fcst'.padStart(9) +
    'target'.padStart(9) + 'var'.padStart(9),
);
console.log('-'.repeat(78));
for (const r of rollup) {
  console.log(
    `${r.costCenter} ${r.costCenterName}`.slice(0, 29).padEnd(30) +
      String(r.people).padStart(8) +
      r.lastUtil.toFixed(1).padStart(9) +
      r.forecastUtil.toFixed(1).padStart(9) +
      r.utilTarget.toFixed(1).padStart(9) +
      `${r.forecastVariance >= 0 ? '+' : ''}${r.forecastVariance.toFixed(1)}`.padStart(9),
  );
}
console.log('');

const header = [
  'Cost Center', 'Cost Center Name', 'Person Name', 'Target Type', 'Job Level',
  'Forecast Period', 'Forecast Month', 'Last Util %', 'Forecast Util %',
  'Forecast Low 80', 'Forecast High 80', 'Util % Target', 'Forecast Variance',
  'Expected Avail Hours', 'Method', 'Periods Of History',
];
const lines = result.forecasts.map(f =>
  [
    f.costCenter, f.costCenterName, f.personName, f.targetType, f.jobLevel,
    f.targetPeriod, f.targetMonth, f.lastUtil.toFixed(2), f.forecastUtil.toFixed(2),
    f.low80.toFixed(2), f.high80.toFixed(2), f.utilTarget.toFixed(2),
    f.forecastVariance.toFixed(2), f.expectedAvailHours.toFixed(1), f.method,
    f.periodsOfHistory,
  ]
    .map(cell => (/[",\n]/.test(String(cell)) ? `"${String(cell).replace(/"/g, '""')}"` : cell))
    .join(','),
);
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, [header.join(','), ...lines].join('\n') + '\n');
console.log(`Wrote ${result.forecasts.length} forecasts to ${path.relative(process.cwd(), outPath)}`);
