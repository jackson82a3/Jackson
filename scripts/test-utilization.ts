/**
 * Self-checks for the utilization dataset and forecaster.
 *
 *   npm run util:test
 *
 * These cover the claims the pipeline rests on: the extract round-trips through
 * CSV, the generator is deterministic and internally consistent, the ridge
 * solver recovers known coefficients, and - the one that matters most for an
 * honest backtest - no feature row can see a period after its origin.
 */
import { generateDataset } from '../src/lib/utilization/generate.ts';
import { parseCsv, toCsv } from '../src/lib/utilization/csv.ts';
import { verifyRecord, COLUMNS } from '../src/lib/utilization/types.ts';
import { buildTrainingSamples, FEATURE_NAMES } from '../src/lib/utilization/features.ts';
import { fitRidge, predictOne } from '../src/lib/utilization/ridge.ts';
import { forecastNextPeriod, rollupByCostCenter, trainForecaster } from '../src/lib/utilization/forecast.ts';

let failures = 0;

function check(name: string, condition: boolean, detail = ''): void {
  if (condition) {
    console.log(`  ok   ${name}`);
  } else {
    failures++;
    console.log(`  FAIL ${name}${detail ? ` - ${detail}` : ''}`);
  }
}

console.log('utilization pipeline self-checks');

// --- Dataset ---------------------------------------------------------------
const records = generateDataset({ seed: 20260901 });
const parsed = parseCsv(toCsv(records));

check('generator is deterministic', toCsv(generateDataset({ seed: 20260901 })) === toCsv(records));
check('a different seed gives different data', toCsv(generateDataset({ seed: 7 })) !== toCsv(records));
check('12 periods are present', new Set(parsed.map(r => r.periodIndex)).size === 12);
check(
  'every person appears in every period',
  new Set(parsed.map(r => `${r.costCenter}|${r.personName}`)).size * 12 === parsed.length,
  `${parsed.length} rows`,
);

const roundTripMismatch = parsed.find((row, i) =>
  COLUMNS.some(([, key]) => {
    const before = records[i][key];
    const after = row[key];
    return typeof before === 'number'
      ? Math.abs(before - (after as number)) > 1e-9
      : before !== after;
  }),
);
check('CSV round-trips without loss', roundTripMismatch === undefined, roundTripMismatch?.personName);

const identityProblems = parsed.flatMap(r => verifyRecord(r));
check('accounting identities hold on every row', identityProblems.length === 0, identityProblems[0]);
check(
  'period is recoverable from Source.Name',
  parsed.every(r => r.sourceName.includes(r.periodMonth) && r.periodIndex >= 1 && r.periodIndex <= 12),
);

// --- Ridge solver ----------------------------------------------------------
const X = Array.from({ length: 200 }, (_, i) => [i / 20, Math.sin(i), (i % 7) - 3]);
const y = X.map(([a, b, c]) => 5 + 2 * a - 3 * b + 0.5 * c);
const fitted = fitRidge(X, y, 1e-8, ['a', 'b', 'c']);
const recovered = X.map(x => predictOne(fitted, x));
const worstFit = Math.max(...recovered.map((v, i) => Math.abs(v - y[i])));
check('ridge recovers a known linear function', worstFit < 1e-6, `max error ${worstFit}`);

// --- Leakage ---------------------------------------------------------------
const samples = buildTrainingSamples(parsed);
check('feature rows match the declared feature names', samples.every(s => s.x.length === FEATURE_NAMES.length));

let leaks = 0;
for (const origin of [3, 5, 8]) {
  // Rebuilding from data that stops one period after the origin must reproduce
  // exactly the same feature row: anything else means a feature saw the future.
  const truncated = parsed.filter(r => r.periodIndex <= origin + 1);
  const truncatedSamples = new Map(
    buildTrainingSamples(truncated)
      .filter(s => s.originPeriod === origin)
      .map(s => [`${s.costCenter}|${s.personName}`, s]),
  );
  for (const sample of samples.filter(s => s.originPeriod === origin)) {
    const other = truncatedSamples.get(`${sample.costCenter}|${sample.personName}`);
    if (!other || other.x.some((v, i) => Math.abs(v - sample.x[i]) > 1e-9)) leaks++;
  }
}
check('features never depend on periods after the origin', leaks === 0, `${leaks} leaking rows`);

// --- Model -----------------------------------------------------------------
const trained = trainForecaster(parsed);
const bestBaseline = trained.baselines.reduce((a, b) => (b.metrics.mae < a.metrics.mae ? b : a));
check(
  'model beats the best naive baseline out of sample',
  trained.metrics.mae < bestBaseline.metrics.mae,
  `ridge ${trained.metrics.mae.toFixed(3)} vs ${bestBaseline.name} ${bestBaseline.metrics.mae.toFixed(3)}`,
);
check('every fold is scored', trained.byFold.length === 5 && trained.byFold.every(f => f.n === 60));
check(
  '80% interval covers roughly 80% of validation rows',
  trained.coverage80 > 0.7 && trained.coverage80 < 0.9,
  `${(trained.coverage80 * 100).toFixed(1)}%`,
);

const forecasts = forecastNextPeriod(trained, parsed);
check('one forecast per person', forecasts.length === new Set(parsed.map(r => `${r.costCenter}|${r.personName}`)).size);
check('forecasts target the period after the data', forecasts.every(f => f.targetPeriod === 13));
check(
  'forecasts and intervals stay in range',
  forecasts.every(f => f.forecastUtil >= 0 && f.forecastUtil <= 100 && f.low80 <= f.forecastUtil && f.forecastUtil <= f.high80),
);
check('rollup covers every cost center', rollupByCostCenter(forecasts).length === new Set(parsed.map(r => r.costCenter)).size);

console.log('');
if (failures > 0) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log('all checks passed');
