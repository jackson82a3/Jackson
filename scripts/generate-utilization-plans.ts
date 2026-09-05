/**
 * Writes the simulated PM hour allocations to data/utilization-plan.csv.
 *
 *   npm run util:plans [-- --seed 20260902 --foresight 0.55 --staleness 0.28]
 *
 * Reads the actuals and derives plans from them, so the timesheet extract is
 * never modified and every number already measured on it still stands.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseCsv } from '../src/lib/utilization/csv.ts';
import { generatePlans, plansToCsv, verifyPlan } from '../src/lib/utilization/plan.ts';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const seed = Number(arg('seed', '20260902'));
const foresight = Number(arg('foresight', '0.55'));
const staleness = Number(arg('staleness', '0.28'));
const dataPath = path.resolve(process.cwd(), arg('data', 'data/utilization.csv'));
const outPath = path.resolve(process.cwd(), arg('out', 'data/utilization-plan.csv'));

const records = parseCsv(fs.readFileSync(dataPath, 'utf8'));
const plans = generatePlans(records, { seed, foresight, stalenessRate: staleness });

const problems: string[] = [];
for (const plan of plans) {
  for (const problem of verifyPlan(plan)) {
    problems.push(`P${plan.planPeriod} / ${plan.personName}: ${problem}`);
  }
}
if (problems.length > 0) {
  console.error(`Generated plans are inconsistent (${problems.length} issues):`);
  for (const problem of problems.slice(0, 10)) console.error(`  ${problem}`);
  process.exit(1);
}

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, plansToCsv(plans));

const actualByKey = new Map(
  records.map(r => [`${r.costCenter}|${r.personName}|${r.periodIndex}`, r.utilPct] as const),
);
const errors: number[] = [];
for (const plan of plans) {
  const actual = actualByKey.get(`${plan.costCenter}|${plan.personName}|${plan.planPeriod}`);
  if (actual !== undefined) errors.push(plan.plannedUtilPct - actual);
}
const mae = errors.reduce((a, b) => a + Math.abs(b), 0) / errors.length;
const bias = errors.reduce((a, b) => a + b, 0) / errors.length;
const stale = plans.filter(p => p.planAgePeriods > 0).length;
const periods = [...new Set(plans.map(p => p.planPeriod))].sort((a, b) => a - b);

console.log(`Wrote ${plans.length} plan rows to ${path.relative(process.cwd(), outPath)} (seed ${seed})`);
console.log(`  foresight ${foresight}, staleness ${staleness}`);
console.log(`  plan periods P${periods[0]}..P${periods[periods.length - 1]} (no plan past the last extract)`);
console.log(`  ${stale} of ${plans.length} rows (${((stale / plans.length) * 100).toFixed(1)}%) are carried-forward allocations`);
console.log(`  Plan vs actual over all rows: MAE ${mae.toFixed(2)}pp, bias ${bias >= 0 ? '+' : ''}${bias.toFixed(2)}pp`);
console.log('  All plan identities check out.');
