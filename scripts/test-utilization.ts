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
import {
  forecasterFromArtifact,
  loadModel,
  ModelLoadError,
  MODEL_FORMAT_VERSION,
  serializeModel,
} from '../src/lib/utilization/model-io.ts';
import { validateDataset } from '../src/lib/utilization/validate.ts';
import {
  accuracyDrift,
  DEFAULT_THRESHOLDS,
  featureDrift,
  verdict,
} from '../src/lib/utilization/monitor.ts';
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

/** Asserts that `fn` refuses its input rather than returning something wrong. */
const throws = (label: string, fn: () => unknown) => {
  let threw = false;
  try {
    fn();
  } catch {
    threw = true;
  }
  check(label, threw);
};

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

// --- Multi-year generation -------------------------------------------------
// A second year exists to test seasonality, which one year cannot. The shipped
// dataset is one year, so the first requirement is that adding the option did
// not disturb it.
check(
  'the one-year dataset is unchanged by the years option',
  toCsv(generateDataset({ seed: 20260901, years: 1 })) === toCsv(records),
);
const twoYears = parseCsv(toCsv(generateDataset({ seed: 20260901, years: 2 })));
check(
  'two years gives 24 periods and twice the rows',
  new Set(twoYears.map(r => r.periodIndex)).size === 24 && twoYears.length === records.length * 2,
  `${twoYears.length} rows`,
);
check(
  'the fiscal year advances with the periods',
  twoYears.filter(r => r.periodIndex <= 12).every(r => r.sourceName.startsWith('FY26_')) &&
    twoYears.filter(r => r.periodIndex >= 13).every(r => r.sourceName.startsWith('FY27_')),
);
check(
  'periods are still recoverable from the second year of file names',
  twoYears.every(r => r.periodIndex >= 1 && r.periodIndex <= 24 && r.sourceName.includes(r.periodMonth)),
);
check('two-year rows satisfy the identities', twoYears.flatMap(r => verifyRecord(r)).length === 0);
// Year-to-date figures are fiscal-year-to-date, so P13 must restart rather than
// carry the first year's cumulative hours forward.
const firstOfYearTwo = twoYears.filter(r => r.periodIndex === 13);
check(
  'year-to-date resets at the start of the second fiscal year',
  firstOfYearTwo.length > 0 &&
    firstOfYearTwo.every(r => Math.abs(r.utilPctYtd - r.utilPct) < 0.011),
);
throws('a non-positive number of years is rejected', () => generateDataset({ years: 0 }));

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

// A joiner is on the roster but has no timesheet history. Without a roster the
// timesheet cannot know they exist, so `cold_start` must not occur; with one it
// must, and must be labelled rather than passed off as a forecast.
check(
  'cold_start never occurs without a roster',
  forecastAll(trained, parsed).forecasts.every(f => f.method !== 'cold_start'),
);
const joinerResult = forecastAll(trained, parsed, [
  { personName: 'New Joiner', costCenter: 'CC-1010', jobLevel: 'L2 Engineer', utilPctTarget: 85 },
  // Someone already in the panel must not be duplicated by the roster.
  { personName: parsed[0].personName, costCenter: parsed[0].costCenter },
]);
const joiner = joinerResult.forecasts.find(f => f.personName === 'New Joiner');
check(
  'a joiner on the roster is forecast, flagged cold_start',
  joiner !== undefined &&
    joiner.method === 'cold_start' &&
    joiner.periodsOfHistory === 0 &&
    joiner.forecastUtil > 0 &&
    joiner.forecastUtil <= 100,
  joiner ? `${joiner.forecastUtil.toFixed(1)}%` : 'missing',
);
check(
  'a roster entry who is already in the panel is not duplicated',
  joinerResult.forecasts.length === headcountLastPeriod + 1 &&
    joinerResult.forecasts.filter(f => f.personName === parsed[0].personName).length === 1,
  `${joinerResult.forecasts.length} rows`,
);
check(
  'a joiner gets the fallback interval, not the model interval',
  joiner !== undefined &&
    Math.abs(joiner.high80 - joiner.low80 - 2 * 1.2816 * trained.fallbackSigma) < 1e-6,
);
// CC-1010 holds exactly one L2 Engineer, so the narrowest peer group is one
// person's recent luck. It must widen rather than report that as a cohort.
check(
  'a thin peer group is widened until it describes enough people',
  joiner !== undefined &&
    /median of (\d+) peers/.test(joiner.basis ?? '') &&
    Number(/median of (\d+) peers/.exec(joiner.basis ?? '')?.[1]) >= 3,
  joiner?.basis,
);
// Peer utilization and peer targets are different quantities; deriving the
// target from the former would report a peer group's shortfall as this person's
// goal.
const untargeted = forecastAll(trained, parsed, [
  { personName: 'No Target', costCenter: 'CC-1010', jobLevel: 'L2 Engineer' },
]).forecasts.find(f => f.personName === 'No Target');
check(
  'a joiner target comes from peer targets, not peer utilization',
  untargeted !== undefined &&
    untargeted.utilTarget !== untargeted.forecastUtil &&
    untargeted.utilTarget > untargeted.forecastUtil,
  untargeted ? `target ${untargeted.utilTarget} vs forecast ${untargeted.forecastUtil.toFixed(1)}` : 'missing',
);
check(
  'a cold-start row has no last observation, and says so with NaN rather than a number',
  joiner !== undefined && Number.isNaN(joiner.lastUtil),
);

// Degenerate inputs must fail loudly. Both of these used to return quietly:
// a forecast for period -Infinity, and an interval with NaN bounds.
throws('forecasting with no history is refused', () => forecastAll(trained, []));
throws('forecasting a roster with no history is refused', () =>
  forecastAll(trained, [], [{ personName: 'X', costCenter: 'CC-1010' }]),
);
throws('calibrating an interval with no predictions is refused', () => calibrateIntervals([]));

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

// --- Model artifact --------------------------------------------------------
// A model artifact outlives the code that wrote it, and a ridge model will
// cheerfully multiply the wrong coefficient by the wrong column and return a
// plausible number. These check that it refuses instead.
const artifact = serializeModel(trained, 'data/utilization.csv');
const reloaded = loadModel(JSON.parse(JSON.stringify(artifact)));
check(
  'a saved model round-trips',
  reloaded.model.featureNames.length === FEATURE_NAMES.length &&
    reloaded.model.coefficients.every((c, i) => c === trained.model.coefficients[i]) &&
    reloaded.model.intercept === trained.model.intercept,
);

const scoredFromDisk = forecastAll(
  forecasterFromArtifact(reloaded.artifact, reloaded.model),
  parsed,
).forecasts;
check(
  'forecasting from a saved model matches forecasting from the trained one',
  scoredFromDisk.length === full.forecasts.length &&
    scoredFromDisk.every((f, i) => Math.abs(f.forecastUtil - full.forecasts[i].forecastUtil) < 1e-9),
);

const refuses = (label: string, mutate: (a: Record<string, unknown>) => void) => {
  const copy = JSON.parse(JSON.stringify(artifact)) as Record<string, unknown>;
  mutate(copy);
  let threw = false;
  try {
    loadModel(copy);
  } catch (error) {
    threw = error instanceof ModelLoadError;
  }
  check(label, threw);
};

refuses('a model of the wrong kind is refused', a => {
  a.kind = 'something-else';
});
refuses('a model with no format version is refused', a => {
  delete a.formatVersion;
});
refuses('a model from a future format is refused', a => {
  a.formatVersion = MODEL_FORMAT_VERSION + 1;
});
refuses('a model trained on different features is refused', a => {
  (a.model as { featureNames: string[] }).featureNames = FEATURE_NAMES.map((n, i) =>
    i === 0 ? 'some_other_feature' : n,
  );
});
refuses('a model with the wrong number of coefficients is refused', a => {
  (a.model as { coefficients: number[] }).coefficients = [1, 2, 3];
});
refuses('a model with a zero scale is refused', a => {
  (a.model as { sds: number[] }).sds[0] = 0;
});
refuses('a model with no interval calibration is refused', a => {
  delete a.interval;
});

// --- Dataset validation ----------------------------------------------------
const cleanReport = validateDataset(parsed);
check(
  'clean data validates with no issues',
  cleanReport.ok && cleanReport.issues.length === 0,
  cleanReport.issues.map(i => i.code).join(', '),
);

const codesFor = (rows: typeof parsed) => validateDataset(rows).issues.map(i => i.code);
check(
  'a duplicated person-period is an error',
  codesFor([...parsed, parsed[0]]).includes('duplicate_person_period'),
);
check(
  'a broken accounting identity is an error',
  codesFor([{ ...parsed[0], directTotal: parsed[0].directTotal + 25 }, ...parsed.slice(1)]).includes(
    'identity_violation',
  ),
);
check(
  'a missing period is an error',
  codesFor(parsed.filter(r => r.periodIndex !== 6)).includes('missing_periods'),
);
check(
  'too few periods to train is an error',
  codesFor(parsed.filter(r => r.periodIndex <= 4)).includes('insufficient_periods'),
);
check(
  'a leaver is a warning, not an error',
  (() => {
    const report = validateDataset(withLeaver);
    return (
      report.ok &&
      report.warnings.some(w => w.code === 'absent_from_last_period') &&
      report.errors.length === 0
    );
  })(),
);
check(
  'validation separates blocking errors from warnings',
  validateDataset([...parsed, parsed[0]]).ok === false,
);

// --- Horizons --------------------------------------------------------------
const h2 = buildTrainingSamples(parsed, 2);
check(
  'horizon-2 samples target exactly two periods after their origin',
  h2.length > 0 && h2.every(s => s.targetPeriod === s.originPeriod + 2),
);
check(
  'a longer horizon yields fewer samples',
  h2.length < samples.length && buildTrainingSamples(parsed, 3).length < h2.length,
  `${samples.length} / ${h2.length}`,
);
check(
  'horizon samples still carry the last-period anchor',
  h2.every(s => Number.isFinite(s.anchor) && s.anchor === s.baselines.last),
);
check(
  'a non-positive horizon is rejected',
  (() => {
    for (const bad of [0, -1, 1.5]) {
      try {
        buildTrainingSamples(parsed, bad);
        return false;
      } catch {
        /* expected */
      }
    }
    return true;
  })(),
);
// Features are computed at the origin, so a horizon-2 sample and a horizon-1
// sample from the same origin must have identical feature rows - only the
// question changes, not the information.
const byOrigin = new Map(samples.map(s => [`${s.costCenter}|${s.personName}|${s.originPeriod}`, s]));
let horizonFeatureDrift = 0;
for (const s of h2) {
  const one = byOrigin.get(`${s.costCenter}|${s.personName}|${s.originPeriod}`);
  if (one && one.x.some((v, i) => Math.abs(v - s.x[i]) > 1e-12)) horizonFeatureDrift++;
}
check(
  'the same origin gives the same features at any horizon',
  horizonFeatureDrift === 0,
  `${horizonFeatureDrift} rows differ`,
);

// --- Monitoring ------------------------------------------------------------
// Train on the first nine periods, then check the model against the three it
// never saw. This is also the honest demonstration of how much a backtest can
// flatter a model on a genuinely novel period.
const earlyRecords = parsed.filter(r => r.periodIndex <= 9);
const earlyTrained = trainForecaster(earlyRecords);
const earlyArtifact = serializeModel(earlyTrained, 'p1-p9');
const earlyLoaded = loadModel(JSON.parse(JSON.stringify(earlyArtifact)));

const unseen = accuracyDrift(earlyLoaded.artifact, earlyLoaded.model, parsed, 10);
check(
  'monitoring scores a model on periods it never saw',
  unseen !== undefined && unseen.window.samples === 180 && !unseen.overlapsTraining,
  `${unseen?.window.samples} samples`,
);
check(
  'monitoring detects real degradation on an unseen window',
  unseen !== undefined && unseen.maeRatio > 1.25 &&
    verdict(unseen, [], DEFAULT_THRESHOLDS).accuracyBreached,
  unseen ? `${unseen.observed.mae.toFixed(2)} vs claimed ${unseen.claimed.mae.toFixed(2)}` : 'none',
);
check(
  'an in-sample window never raises drift, however bad it looks',
  (() => {
    const inSample = accuracyDrift(reloaded.artifact, reloaded.model, parsed, 8);
    return (
      inSample !== undefined &&
      inSample.overlapsTraining &&
      !verdict(inSample, [], DEFAULT_THRESHOLDS).accuracyBreached
    );
  })(),
);
check(
  'monitoring returns nothing when no period has closed since training',
  accuracyDrift(reloaded.artifact, reloaded.model, parsed, 13) === undefined,
);

// A monitor that cannot measure must say so. Returning NaN would let `verdict`
// report "nothing exceeded the threshold" when nothing was checked at all.
throws('feature drift with too little history is refused, not silently NaN', () =>
  featureDrift(reloaded.model, parsed.filter(r => r.periodIndex <= 2)),
);
throws('feature drift with no records at all is refused', () => featureDrift(reloaded.model, []));

const drift = featureDrift(reloaded.model, parsed);
check(
  'feature drift covers every feature and is sorted by size',
  drift.length === FEATURE_NAMES.length &&
    drift.every((d, i) => i === 0 || Math.abs(drift[i - 1].drift) >= Math.abs(d.drift)),
);
check(
  'feature drift needs no outcomes',
  featureDrift(reloaded.model, parsed.filter(r => r.periodIndex <= 5)).length === FEATURE_NAMES.length,
);
check(
  'every drift figure is a real number',
  drift.every(d => Number.isFinite(d.now) && Number.isFinite(d.drift)),
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
