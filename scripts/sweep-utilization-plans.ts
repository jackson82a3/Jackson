/**
 * How good would PM plans have to be for the plan to beat the model?
 *
 *   npm run util:sweep
 *
 * The head-to-head at any single setting is a statement about assumed planner
 * quality, not a finding. This sweeps the two assumptions that drive it -
 * foresight (how much of the coming period the planner genuinely sees) and
 * optimism (how hard plans are pulled up toward target) - regenerating the
 * allocations from the same unchanged actuals at each cell and re-running the
 * identical rolling-origin protocol.
 *
 * `history_ridge` never sees a plan, so its MAE is constant down every column;
 * that constancy is a useful check that the sweep is only moving what it means
 * to move.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseCsv } from '../src/lib/utilization/csv.ts';
import { generatePlans } from '../src/lib/utilization/plan.ts';
import { runHeadToHead } from '../src/lib/utilization/headtohead.ts';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const dataPath = path.resolve(process.cwd(), arg('data', 'data/utilization.csv'));
const outPath = path.resolve(process.cwd(), arg('out', 'data/utilization-sweep.json'));
const seed = Number(arg('seed', '20260902'));
const staleness = Number(arg('staleness', '0.28'));

const FORESIGHT = [0.2, 0.35, 0.5, 0.55, 0.65, 0.8];
const OPTIMISM = [0, 0.1, 0.2, 0.35, 0.5];

const records = parseCsv(fs.readFileSync(dataPath, 'utf8'));

interface Cell {
  foresight: number;
  optimismUp: number;
  planMae: number;
  planBias: number;
  historyMae: number;
  planRidgeMae: number;
  blendMae: number;
  /** True where the raw PM plan is at least as accurate as the history model. */
  planBeatsModel: boolean;
}

const cells: Cell[] = [];
for (const foresight of FORESIGHT) {
  for (const optimismUp of OPTIMISM) {
    const plans = generatePlans(records, {
      seed,
      foresight,
      stalenessRate: staleness,
      optimismUp,
      // The downward pull is held at a third of the upward one so the asymmetry
      // that defines optimism is preserved as the knob moves.
      optimismDown: optimismUp / 3,
    });
    const result = runHeadToHead(records, plans);
    cells.push({
      foresight,
      optimismUp,
      planMae: result.metrics.plan.mae,
      planBias: result.metrics.plan.bias,
      historyMae: result.metrics.history_ridge.mae,
      planRidgeMae: result.metrics.plan_ridge.mae,
      blendMae: result.metrics.blend.mae,
      planBeatsModel: result.metrics.plan.mae <= result.metrics.history_ridge.mae,
    });
  }
}

const historyMae = cells[0].historyMae;
const drift = Math.max(...cells.map(c => Math.abs(c.historyMae - historyMae)));

const thin = '-'.repeat(78);

function at(foresight: number, optimismUp: number): Cell {
  const found = cells.find(c => c.foresight === foresight && c.optimismUp === optimismUp);
  if (!found) throw new Error(`No sweep cell at foresight=${foresight}, optimism=${optimismUp}`);
  return found;
}

function grid(title: string, pick: (c: Cell) => number): void {
  console.log(title);
  console.log('foresight ' + OPTIMISM.map(o => `opt=${o.toFixed(2)}`.padStart(9)).join(''));
  console.log(thin);
  for (const foresight of FORESIGHT) {
    console.log(
      foresight.toFixed(2).padStart(9) +
        OPTIMISM.map(o => pick(at(foresight, o)).toFixed(2).padStart(9)).join(''),
    );
  }
  console.log('');
}

console.log('Sensitivity of the head-to-head to assumed planner quality');
console.log('='.repeat(78));
console.log(`Rows unchanged; history-only model MAE ${historyMae.toFixed(2)}pp in every cell`);
console.log(`(max drift across the sweep ${drift.toFixed(4)}pp - it never sees a plan)`);
console.log('');

grid('PM plan MAE (pp)', c => c.planMae);
grid('PM plan bias (pp)', c => c.planBias);
grid('model on plan + history, MAE (pp)', c => c.planRidgeMae);
grid('blend (plan + model), MAE (pp)', c => c.blendMae);

console.log('Where the raw PM plan beats the history-only model');
console.log('foresight ' + OPTIMISM.map(o => `opt=${o.toFixed(2)}`.padStart(9)).join(''));
console.log(thin);
for (const foresight of FORESIGHT) {
  console.log(
    foresight.toFixed(2).padStart(9) +
      OPTIMISM.map(o => (at(foresight, o).planBeatsModel ? 'plan' : '-').padStart(9)).join(''),
  );
}
console.log('');

const winners = cells.filter(c => c.planBeatsModel);
if (winners.length === 0) {
  console.log('The raw plan never beats the model anywhere on this grid.');
} else {
  const easiest = winners.reduce((a, b) => (b.optimismUp > a.optimismUp ? b : a));
  const minForesight = Math.min(...winners.map(c => c.foresight));
  console.log(
    `The raw plan wins in ${winners.length} of ${cells.length} cells; it needs foresight >= ${minForesight.toFixed(2)}`,
  );
  console.log(
    `and tolerates optimism up to ${easiest.optimismUp.toFixed(2)} (at foresight ${easiest.foresight.toFixed(2)}).`,
  );
}

const contenders = [
  ['PM plan', (c: Cell) => c.planMae],
  ['history-only model', (c: Cell) => c.historyMae],
  ['plan+history model', (c: Cell) => c.planRidgeMae],
  ['blend', (c: Cell) => c.blendMae],
] as const;

console.log('Best forecaster per cell, counted over the grid');
for (const [label, pick] of contenders) {
  const wins = cells.filter(c =>
    contenders.every(([, other]) => pick(c) <= other(c) + 1e-9),
  ).length;
  console.log(`  ${label.padEnd(22)} best in ${String(wins).padStart(2)} of ${cells.length} cells`);
}
console.log('');

// The interesting failure mode: anchoring on the plan is only safe when the
// plan is good. The blend, which learns how far to trust it, is not.
const planRidgeWorse = cells.filter(c => c.planRidgeMae > c.historyMae + 1e-9);
const blendWorse = cells.filter(c => c.blendMae > c.historyMae + 1e-9);
console.log(
  `The plan-anchored model is worse than ignoring plans entirely in ${planRidgeWorse.length} of ${cells.length} cells` +
    (planRidgeWorse.length > 0
      ? ` (all at foresight <= ${Math.max(...planRidgeWorse.map(c => c.foresight)).toFixed(2)}).`
      : '.'),
);
console.log(
  `The blend is worse than ignoring plans entirely in ${blendWorse.length} of ${cells.length} cells` +
    (blendWorse.length > 0
      ? `, by at most ${Math.max(...blendWorse.map(c => c.blendMae - c.historyMae)).toFixed(3)}pp.`
      : '.'),
);

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(
  outPath,
  JSON.stringify({ generatedAt: new Date().toISOString(), staleness, seed, cells }, null, 2) + '\n',
);
console.log(`\nWrote ${path.relative(process.cwd(), outPath)}`);
