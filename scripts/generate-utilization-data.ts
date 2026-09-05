/**
 * Writes the twelve-period synthetic utilization extract to data/utilization.csv.
 *
 *   npm run util:generate [-- --seed 12345 --out data/utilization.csv]
 */
import fs from 'node:fs';
import path from 'node:path';
import { generateDataset, PERIODS } from '../src/lib/utilization/generate.ts';
import { toCsv } from '../src/lib/utilization/csv.ts';
import { verifyRecord } from '../src/lib/utilization/types.ts';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const seed = Number(arg('seed', '20260901'));
const outPath = path.resolve(process.cwd(), arg('out', 'data/utilization.csv'));

const records = generateDataset({ seed });

const problems: string[] = [];
for (const record of records) {
  for (const problem of verifyRecord(record)) {
    problems.push(`${record.sourceName} / ${record.personName}: ${problem}`);
  }
}
if (problems.length > 0) {
  console.error(`Generated data violates its own identities (${problems.length} issues):`);
  for (const problem of problems.slice(0, 10)) console.error(`  ${problem}`);
  process.exit(1);
}

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, toCsv(records));

const people = new Set(records.map(r => r.personName));
const utils = records.map(r => r.utilPct);
const mean = utils.reduce((a, b) => a + b, 0) / utils.length;
const sd = Math.sqrt(utils.reduce((a, b) => a + (b - mean) ** 2, 0) / utils.length);

console.log(`Wrote ${records.length} rows to ${path.relative(process.cwd(), outPath)} (seed ${seed})`);
console.log(`  ${people.size} people x ${PERIODS.length} periods (${PERIODS[0].month} .. ${PERIODS[PERIODS.length - 1].month})`);
console.log(`  Util %: mean ${mean.toFixed(2)}, sd ${sd.toFixed(2)}, min ${Math.min(...utils).toFixed(2)}, max ${Math.max(...utils).toFixed(2)}`);
console.log('  All accounting identities check out.');
