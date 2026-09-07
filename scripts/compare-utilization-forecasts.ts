/**
 * Scores PM plan vs model vs blend on identical rows, and writes
 * data/utilization-headtohead.json.
 *
 *   npm run util:compare
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseCsv } from '../src/lib/utilization/csv.ts';
import { generatePlans, parsePlanCsv } from '../src/lib/utilization/plan.ts';
import { FORECASTERS, runHeadToHead } from '../src/lib/utilization/headtohead.ts';
import type { ForecasterName } from '../src/lib/utilization/headtohead.ts';

/** Signs a number explicitly so a column of biases lines up. */
function signed(x: number, digits = 2): string {
  return `${x >= 0 ? '+' : '-'}${Math.abs(x).toFixed(digits)}`;
}

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const dataPath = path.resolve(process.cwd(), arg('data', 'data/utilization.csv'));
const planPath = path.resolve(process.cwd(), arg('plans', 'data/utilization-plan.csv'));
const outPath = path.resolve(process.cwd(), arg('out', 'data/utilization-headtohead.json'));

if (!fs.existsSync(planPath)) {
  console.error(`No plan file at ${path.relative(process.cwd(), planPath)} - run: npm run util:plans`);
  process.exit(1);
}

const records = parseCsv(fs.readFileSync(dataPath, 'utf8'));
const plans = parsePlanCsv(fs.readFileSync(planPath, 'utf8'));
const result = runHeadToHead(records, plans);

const LABELS: Record<ForecasterName, string> = {
  plan: 'PM plan (as-is)',
  history_ridge: 'model (history only)',
  plan_ridge: 'model on plan + history',
  blend: 'blend (plan + model)',
  blend_by_age: 'blend, weight by staleness',
};

const rule = '='.repeat(78);
const thin = '-'.repeat(78);

console.log('PM plan vs model vs blend');
console.log(rule);
console.log(
  `Rows      ${result.rows} plan-joined samples | ${result.validationRows} scored out-of-sample`,
);
console.log(
  `Penalty   history lambda=${result.lambdaHistory}, plan-augmented lambda=${result.lambdaPlan}`,
);
console.log('');

console.log('Out-of-sample accuracy (Util %, percentage points)');
console.log('forecaster                        MAE     RMSE     bias       R2   within5');
console.log(thin);
const ranked = [...FORECASTERS].sort((a, b) => result.metrics[a].mae - result.metrics[b].mae);
for (const name of ranked) {
  const m = result.metrics[name];
  console.log(
    LABELS[name].padEnd(28) +
      m.mae.toFixed(2).padStart(8) +
      m.rmse.toFixed(2).padStart(9) +
      signed(m.bias).padStart(9) +
      m.r2.toFixed(3).padStart(9) +
      `${(m.within5 * 100).toFixed(1)}%`.padStart(10),
  );
}
console.log('');

console.log('Against the PM plan, on the identical rows');
console.log('forecaster                    MAE delta   closer than plan');
console.log(thin);
for (const row of result.versusPlan) {
  console.log(
    LABELS[row.forecaster].padEnd(28) +
      `${signed(row.maeDelta)}pp`.padStart(11) +
      `${(row.winRate * 100).toFixed(1)}%`.padStart(19),
  );
}
console.log('');

console.log('Per fold (MAE, pp)');
console.log(
  'period      n' +
    FORECASTERS.map(n => n.padStart(15)).join('') +
    '   blend w(plan)',
);
console.log(thin);
for (const fold of result.byFold) {
  console.log(
    `P${String(fold.targetPeriod).padStart(2, '0')}`.padEnd(8) +
      String(fold.n).padStart(5) +
      FORECASTERS.map(n => fold.mae[n].toFixed(2).padStart(15)).join('') +
      fold.blendWeight.toFixed(2).padStart(16),
  );
}
console.log('');

console.log('PM plan accuracy by allocation staleness');
console.log('bucket                   n      MAE     bias');
console.log(thin);
for (const row of result.planByAge) {
  console.log(
    row.age.padEnd(20) +
      String(row.n).padStart(5) +
      row.mae.toFixed(2).padStart(9) +
      signed(row.bias).padStart(9),
  );
}
console.log('');

console.log('Top drivers of the plan correction (standardized coefficients)');
const drivers = result.planModel.featureNames
  .map((feature, i) => ({ feature, coefficient: result.planModel.coefficients[i] }))
  .sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient))
  .slice(0, 10);
for (const d of drivers) {
  console.log(`  ${d.feature.padEnd(26)} ${signed(d.coefficient, 3)}`);
}
console.log('');

// --- Robustness across plan draws -------------------------------------------
// One plan file is one draw. The gap between the top two forecasters turned out
// to be smaller than the spread between draws, so reporting a single ranking
// would overstate what this shows; the seeds below say which conclusions
// survive re-drawing the allocations and which do not.
const seedCount = Number(arg('seeds', '12'));
const perSeed: { seed: number; mae: Record<ForecasterName, number>; winner: ForecasterName }[] = [];
for (let i = 0; i < seedCount; i++) {
  const seed = 20260902 + i * 101;
  const drawn = runHeadToHead(records, generatePlans(records, { seed }));
  const mae = {} as Record<ForecasterName, number>;
  for (const name of FORECASTERS) mae[name] = drawn.metrics[name].mae;
  const winner = [...FORECASTERS].sort((a, b) => mae[a] - mae[b])[0];
  perSeed.push({ seed, mae, winner });
}

console.log(`Across ${seedCount} independent plan draws (MAE, pp)`);
console.log('forecaster                       mean       sd      min      max   best of 4');
console.log(thin);
const summary = FORECASTERS.map(name => {
  const values = perSeed.map(s => s.mae[name]);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  const wins = perSeed.filter(s => s.winner === name).length;
  return { name, mean, sd, min: Math.min(...values), max: Math.max(...values), wins };
});
for (const row of [...summary].sort((a, b) => a.mean - b.mean)) {
  console.log(
    LABELS[row.name].padEnd(28) +
      row.mean.toFixed(2).padStart(9) +
      row.sd.toFixed(2).padStart(9) +
      row.min.toFixed(2).padStart(9) +
      row.max.toFixed(2).padStart(9) +
      `${row.wins}/${seedCount}`.padStart(12),
  );
}
const beatsHistory = perSeed.filter(
  s => Math.min(s.mae.plan_ridge, s.mae.blend) < s.mae.history_ridge,
).length;
const planBeatsHistory = perSeed.filter(s => s.mae.plan < s.mae.history_ridge).length;
console.log('');
console.log(
  `Using the plan beat the history-only model in ${beatsHistory}/${seedCount} draws; ` +
    `the raw plan beat it in ${planBeatsHistory}/${seedCount}.`,
);
console.log('');

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(
  outPath,
  JSON.stringify({ ...result, robustness: { seedCount, perSeed, summary } }, null, 2) + '\n',
);
console.log(`Wrote ${path.relative(process.cwd(), outPath)}`);
