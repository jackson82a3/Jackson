/**
 * Checks a deployed model against the periods that have closed since it was fitted.
 *
 *   npm run util:monitor [-- --since 10 --model data/utilization-model.json]
 *
 * A backtest says how a model did on the past it was built from. This says how
 * it is doing now, which is the only question that matters once it is in use.
 * The logic lives in `src/lib/utilization/monitor.ts`; this is presentation and
 * an exit code, so it can run as a scheduled check rather than something someone
 * remembers to look at.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseCsv } from '../src/lib/utilization/csv.ts';
import { loadModel, ModelLoadError } from '../src/lib/utilization/model-io.ts';
import {
  accuracyDrift,
  DEFAULT_THRESHOLDS,
  featureDrift,
  verdict,
} from '../src/lib/utilization/monitor.ts';
import { formatValidationReport, validateDataset } from '../src/lib/utilization/validate.ts';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const dataPath = path.resolve(process.cwd(), arg('data', 'data/utilization.csv'));
const modelPath = path.resolve(process.cwd(), arg('model', 'data/utilization-model.json'));
const outPath = path.resolve(process.cwd(), arg('out', 'data/utilization-monitor.json'));
const thresholds = {
  maeRatio: Number(arg('mae-ratio', String(DEFAULT_THRESHOLDS.maeRatio))),
  featureSds: Number(arg('drift-sds', String(DEFAULT_THRESHOLDS.featureSds))),
};

const records = parseCsv(fs.readFileSync(dataPath, 'utf8'));
const validation = validateDataset(records);
if (validation.errors.some(e => e.code !== 'insufficient_periods')) {
  console.log(formatValidationReport(validation));
  console.error('Refusing to monitor on data with errors.');
  process.exit(1);
}

let loaded;
try {
  loaded = loadModel(JSON.parse(fs.readFileSync(modelPath, 'utf8')));
} catch (error) {
  if (error instanceof ModelLoadError) {
    console.error(`Cannot use ${path.relative(process.cwd(), modelPath)}:\n  ${error.message}`);
    process.exit(1);
  }
  throw error;
}
const { artifact, model } = loaded;

const periods = [...new Set(records.map(r => r.periodIndex))].sort((a, b) => a - b);
const trainedThrough = artifact.dataset.periodRange?.last;
const since = Number(
  arg(
    'since',
    String(trainedThrough !== undefined ? trainedThrough + 1 : periods[Math.max(0, periods.length - 3)]),
  ),
);

const accuracy = accuracyDrift(artifact, model, records, since);
const drift = featureDrift(model, records);
const result = verdict(accuracy, drift, thresholds);

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const signed = (x: number, d = 2) => `${x >= 0 ? '+' : '-'}${Math.abs(x).toFixed(d)}`;

console.log('Model monitoring');
console.log('='.repeat(78));
console.log(`Model     trained ${artifact.trainedAt}, claims MAE ${artifact.metrics.mae.toFixed(2)}pp`);
console.log(
  `          fitted on ${
    artifact.dataset.periodRange
      ? `P${artifact.dataset.periodRange.first}-P${artifact.dataset.periodRange.last}`
      : 'an unrecorded period range'
  }`,
);
console.log(`Window    periods >= P${since}${accuracy ? ` (${accuracy.window.samples} person-periods)` : ''}`);
console.log('');

if (!accuracy) {
  console.log('No closed period after the training range yet, so accuracy drift cannot be');
  console.log('measured. Feature drift below needs no outcomes and is still meaningful.');
  console.log('');
} else {
  if (accuracy.overlapsTraining) {
    console.log(`  WARNING: the window overlaps the training range (fitted through P${trainedThrough}).`);
    console.log('           In-sample accuracy flatters the model, so no drift is raised from it.');
    console.log('');
  }
  console.log('Accuracy now vs the accuracy the artifact claims');
  console.log('metric'.padEnd(20) + 'claimed'.padStart(10) + 'observed'.padStart(11) + 'change'.padStart(11));
  console.log('-'.repeat(78));
  const line = (label: string, was: number, now: number, d = 2) =>
    console.log(label.padEnd(20) + was.toFixed(d).padStart(10) + now.toFixed(d).padStart(11) + signed(now - was, d).padStart(11));
  line('MAE', accuracy.claimed.mae, accuracy.observed.mae);
  line('RMSE', accuracy.claimed.rmse, accuracy.observed.rmse);
  line('bias', accuracy.claimed.bias, accuracy.observed.bias);
  line('within 5pp', accuracy.claimed.within5, accuracy.observed.within5, 3);
  line('interval coverage', accuracy.claimedCoverage, accuracy.observedCoverage, 3);
  console.log('');
  console.log(
    result.accuracyBreached
      ? `  DRIFT: MAE is ${accuracy.maeRatio.toFixed(2)}x the claimed figure, past the ` +
          `${thresholds.maeRatio}x threshold. Retrain and re-review before trusting the next forecast.`
      : `  Accuracy is within ${thresholds.maeRatio}x the claimed MAE (${accuracy.maeRatio.toFixed(2)}x).`,
  );
  console.log('');
}

console.log('Feature drift (current mean vs the training mean, in training sds)');
console.log('feature'.padEnd(28) + 'trained'.padStart(10) + 'now'.padStart(10) + 'drift'.padStart(10));
console.log('-'.repeat(78));
for (const r of drift.slice(0, 8)) {
  console.log(
    r.feature.padEnd(28) + r.trained.toFixed(2).padStart(10) + r.now.toFixed(2).padStart(10) +
      signed(r.drift).padStart(10),
  );
}
console.log('');
console.log(
  result.driftedFeatures.length > 0
    ? `  DRIFT: ${result.driftedFeatures.length} feature(s) beyond ${thresholds.featureSds} sd: ` +
        `${result.driftedFeatures.map(d => d.feature).join(', ')}.`
    : `  No feature has moved more than ${thresholds.featureSds} sd from its training distribution.`,
);

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(
  outPath,
  JSON.stringify({ checkedAt: new Date().toISOString(), since, accuracy, drift, verdict: result }, null, 2) + '\n',
);
console.log(`\nWrote ${path.relative(process.cwd(), outPath)}`);

if (result.breached) {
  console.error('\nMonitoring thresholds breached.');
  process.exit(1);
}
console.log(`Coverage of the roster is checked by \`npm run util:predict\`.`);
