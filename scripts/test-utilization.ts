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
import {
  calibrateIntervals,
  crossValidateNested,
  forecastAll,
  forecastNextPeriod,
  rollupByCostCenter,
  selectLambdaBefore,
  trainForecaster,
} from '../src/lib/utilization/forecast.ts';
import { generatePlans, parsePlanCsv, plansToCsv, verifyPlan } from '../src/lib/utilization/plan.ts';
import { buildPlanSamples, PLAN_FEATURE_NAMES, runHeadToHead } from '../src/lib/utilization/headtohead.ts';

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

// --- Penalty selection and intervals ---------------------------------------
// The penalty for a fold must be a function of that fold's past and nothing
// else, so restricting the input to that past cannot change the answer.
let penaltyLeaks = 0;
for (const period of [8, 10, 12]) {
  const fromAll = selectLambdaBefore(samples, period);
  const fromPastOnly = selectLambdaBefore(
    samples.filter(s => s.targetPeriod < period),
    period,
  );
  if (fromAll !== fromPastOnly) penaltyLeaks++;
}
check(
  'penalty selection never sees the fold it is scored on',
  penaltyLeaks === 0,
  `${penaltyLeaks} folds differ`,
);
check(
  'every fold records the penalty it chose',
  trained.lambdaByFold.length === trained.byFold.length &&
    trained.lambdaByFold.every(l => Number.isFinite(l.lambda)),
);

const nested = crossValidateNested(samples);
const calibration = calibrateIntervals(nested.predictions);
check(
  'interval coverage is measured walk-forward, not on its own calibration set',
  calibration.nWalkForward > 0 && calibration.nWalkForward < nested.predictions.length,
  `${calibration.nWalkForward} of ${nested.predictions.length} rows`,
);
check(
  'the shipped interval is one of the measured methods',
  calibration.methods.some(m => m.name === calibration.shipped),
);
check(
  'every interval method brackets its prediction',
  calibration.methods.every(m => m.low < 0 && m.high > 0),
);

// --- Roster coverage -------------------------------------------------------
const full = forecastAll(trained, parsed);
const headcountLastPeriod = new Set(
  parsed.filter(r => r.periodIndex === 12).map(r => `${r.costCenter}|${r.personName}`),
).size;
check(
  'every person in the last period gets a forecast',
  full.forecasts.length === headcountLastPeriod,
  `${full.forecasts.length} vs ${headcountLastPeriod}`,
);
check(
  'coverage counts add up to the roster',
  full.coverage.reduce((a, c) => a + c.people, 0) ===
    full.forecasts.length + full.excluded.length,
);
check(
  'a complete panel needs no fallbacks',
  full.forecasts.every(f => f.method === 'model') && full.excluded.length === 0,
);

// Two periods of history is below what the model needs. Those people must still
// come back, flagged - not silently vanish from the forecast.
const shortPanel = parsed.filter(r => r.periodIndex >= 11);
const shortResult = forecastAll(trained, shortPanel);
check(
  'people with too little history are still forecast, and flagged',
  shortResult.forecasts.length === headcountLastPeriod &&
    shortResult.forecasts.every(f => f.method === 'short_history' && f.periodsOfHistory === 2),
  `${shortResult.forecasts.length} rows`,
);
// Fallback rows are given the naive baseline's own error spread rather than the
// model's. On this data the two are the same to four decimal places, because
// the model only beats "same as last period" by about 1% - so this asserts the
// width is derived from the right quantity, not that it comes out larger.
const unclamped = shortResult.forecasts.find(f => f.low80 > 0 && f.high80 < 100);
check(
  'fallback rows use the naive baseline width, not the model width',
  unclamped !== undefined &&
    Math.abs(unclamped.high80 - unclamped.low80 - 2 * 1.2816 * trained.fallbackSigma) < 1e-6,
);
check(
  'the fallback width is measured, not assumed',
  trained.fallbackSigma > 0 && Number.isFinite(trained.fallbackSigma),
);

// Someone who left mid-year has no period to forecast from.
const leaver = `${parsed[0].costCenter}|${parsed[0].personName}`;
const withLeaver = parsed.filter(
  r => !(r.periodIndex === 12 && `${r.costCenter}|${r.personName}` === leaver),
);
const leaverResult = forecastAll(trained, withLeaver);
check(
  'people absent from the last period are excluded, not silently dropped',
  leaverResult.excluded.length === 1 &&
    leaverResult.excluded[0].reason === 'absent_from_last_period' &&
    leaverResult.forecasts.length === headcountLastPeriod - 1,
  `${leaverResult.excluded.length} excluded`,
);

// --- PM allocations --------------------------------------------------------
const plans = generatePlans(parsed);
const parsedPlans = parsePlanCsv(plansToCsv(plans));

check('plan generator is deterministic', plansToCsv(generatePlans(parsed)) === plansToCsv(plans));
check(
  'a different plan seed gives different plans',
  plansToCsv(generatePlans(parsed, { seed: 7 })) !== plansToCsv(plans),
);
check(
  'plan CSV round-trips without loss',
  parsedPlans.length === plans.length &&
    parsedPlans.every((p, i) =>
      p.plannedUtilPct === plans[i].plannedUtilPct &&
      p.plannedAvailHours === plans[i].plannedAvailHours &&
      p.plannedDirectHours === plans[i].plannedDirectHours &&
      p.planPeriod === plans[i].planPeriod &&
      p.planAgePeriods === plans[i].planAgePeriods &&
      p.personName === plans[i].personName,
    ),
);

const planProblems = parsedPlans.flatMap(p => verifyPlan(p));
check('plan identities hold on every row', planProblems.length === 0, planProblems[0]);
check(
  'every plan is snapshotted exactly one period before the period it is about',
  parsedPlans.every(p => p.planSnapshotPeriod === p.planPeriod - 1),
);
check(
  'no plan exists for a period with no actual',
  Math.max(...parsedPlans.map(p => p.planPeriod)) === 12,
);
check(
  'one plan per person per period from P2 on',
  parsedPlans.length === new Set(parsed.map(r => `${r.costCenter}|${r.personName}`)).size * 11,
  `${parsedPlans.length} plan rows`,
);
check(
  'staleness actually happens, and is capped',
  parsedPlans.some(p => p.planAgePeriods > 0) &&
    Math.max(...parsedPlans.map(p => p.planAgePeriods)) <= 3,
);

// The plan is *allowed* to see the period it is about - that foresight is the
// whole point of it. What it must never see is anything after that period.
let planLeaks = 0;
for (const horizon of [5, 9]) {
  const truncated = generatePlans(parsed.filter(r => r.periodIndex <= horizon));
  const byKey = new Map(
    truncated.map(p => [`${p.costCenter}|${p.personName}|${p.planPeriod}`, p] as const),
  );
  for (const plan of plans.filter(p => p.planPeriod <= horizon)) {
    const other = byKey.get(`${plan.costCenter}|${plan.personName}|${plan.planPeriod}`);
    if (
      !other ||
      other.plannedUtilPct !== plan.plannedUtilPct ||
      other.plannedAvailHours !== plan.plannedAvailHours ||
      other.planAgePeriods !== plan.planAgePeriods
    ) {
      planLeaks++;
    }
  }
}
check(
  'a plan never depends on a period after the one it is about',
  planLeaks === 0,
  `${planLeaks} leaking plan rows`,
);

// --- Head-to-head ----------------------------------------------------------
const planSamples = buildPlanSamples(parsed, parsedPlans);
check(
  'the head-to-head scores the same rows as the history-only backtest',
  planSamples.length === samples.length,
  `${planSamples.length} vs ${samples.length}`,
);
check(
  'plan feature rows match the declared feature names',
  planSamples.every(s => s.planX.length === PLAN_FEATURE_NAMES.length),
);
check(
  'every sample is joined to the plan for its own target period',
  planSamples.every(
    s =>
      s.plan.planPeriod === s.base.targetPeriod &&
      s.plan.planSnapshotPeriod === s.base.originPeriod &&
      s.plan.personName === s.base.personName,
  ),
);

let planFeatureLeaks = 0;
for (const origin of [5, 8]) {
  // Everything knowable at the origin: actuals through the target period, and
  // plans for periods already frozen. Same feature row, or a feature saw more.
  const visibleRecords = parsed.filter(r => r.periodIndex <= origin + 1);
  const visiblePlans = parsedPlans.filter(p => p.planPeriod <= origin + 1);
  const rebuilt = new Map(
    buildPlanSamples(visibleRecords, visiblePlans)
      .filter(s => s.base.originPeriod === origin)
      .map(s => [`${s.base.costCenter}|${s.base.personName}`, s] as const),
  );
  for (const sample of planSamples.filter(s => s.base.originPeriod === origin)) {
    const other = rebuilt.get(`${sample.base.costCenter}|${sample.base.personName}`);
    if (!other || other.planX.some((v, i) => Math.abs(v - sample.planX[i]) > 1e-9)) {
      planFeatureLeaks++;
    }
  }
}
check(
  'plan features never depend on periods after the origin',
  planFeatureLeaks === 0,
  `${planFeatureLeaks} leaking rows`,
);

const head = runHeadToHead(parsed, parsedPlans);
check(
  'the head-to-head reproduces the history model it is compared against',
  Math.abs(head.metrics.history_ridge.mae - trained.metrics.mae) < 1e-9,
  `${head.metrics.history_ridge.mae.toFixed(4)} vs ${trained.metrics.mae.toFixed(4)}`,
);
check(
  'every forecaster is scored on identical rows',
  head.validationRows === 300 &&
    Object.values(head.metrics).every(m => m.n === head.validationRows),
);
check(
  'the blend weight is a genuine convex weight, never fitted on its own fold',
  head.byFold.every(f => f.blendWeight >= 0 && f.blendWeight <= 1) && head.byFold[0].blendWeight === 0.5,
);
check(
  'adding the plan to the model helps',
  head.metrics.plan_ridge.mae < head.metrics.history_ridge.mae,
  `${head.metrics.plan_ridge.mae.toFixed(3)} vs ${head.metrics.history_ridge.mae.toFixed(3)}`,
);

console.log('');
if (failures > 0) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log('all checks passed');
