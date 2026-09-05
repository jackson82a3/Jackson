/**
 * Is the model's edge over the naive baseline real, or is it this seed?
 *
 *   npm run util:seedstudy [-- --seeds 20]
 *
 * Every headline number in this project is measured on one simulated panel,
 * generated from one seed. That panel is a single draw from the simulator, so a
 * margin of ~1% over "same as last period" could be a property of the model or a
 * property of the draw. Nothing published so far distinguishes those.
 *
 * This regenerates the world from many seeds and scores the model the same way
 * each time. Two things are reported, because they answer different questions:
 *
 *   - **Across seeds**: how often the model beats the baseline at all. This is
 *     the question "would this have worked on a different year".
 *   - **Within a seed, paired and clustered by person**: whether the difference
 *     on that panel is larger than the noise on that panel. Rows are paired
 *     (both forecasters see identical rows) and clustered by person, because the
 *     same person appears in several folds and treating those as independent
 *     would overstate significance.
 *
 * The comparison is against the *strongest* naive baseline on each panel, not a
 * fixed one, so the model cannot win by the baseline happening to be weak.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseCsv, toCsv } from '../src/lib/utilization/csv.ts';
import { generateDataset } from '../src/lib/utilization/generate.ts';
import { generatePlans } from '../src/lib/utilization/plan.ts';
import { runHeadToHead } from '../src/lib/utilization/headtohead.ts';
import { buildTrainingSamples } from '../src/lib/utilization/features.ts';
import {
  BASELINES,
  baselinePrediction,
  crossValidateNested,
  evaluate,
} from '../src/lib/utilization/forecast.ts';
import type { BaselineName } from '../src/lib/utilization/forecast.ts';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

// 200 because 20 was not enough: the win rate moved from 7/20 to 101/200 as the
// study grew, so a small study of a small effect was itself too noisy to quote.
// 80 and 200 agree closely, so this has converged. It takes about 90 seconds.
const seedCount = Number(arg('seeds', '200'));
const shippedSeed = Number(arg('shipped-seed', '20260901'));
const outPath = path.resolve(process.cwd(), arg('out', 'data/utilization-seedstudy.json'));
// The PM-plan comparison costs a head-to-head run per seed, so it can be skipped.
const withPlans = !process.argv.includes('--no-plans');

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}
function sd(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / (values.length - 1));
}

interface SeedResult {
  seed: number;
  modelMae: number;
  baselineName: BaselineName;
  baselineMae: number;
  /** Positive means the model is better, in percent of the baseline's MAE. */
  marginPct: number;
  /** Mean of (model error - baseline error) per person; negative favours the model. */
  pairedMean: number;
  /** t over person clusters. |t| > ~2 is the usual "larger than noise" line. */
  clusteredT: number;
  people: number;
  /** The other claims, which were always said to be the stronger ones. */
  modelBias: number;
  baselineBias: number;
  modelWithin5: number;
  baselineWithin5: number;
  modelRollupMae: number;
  baselineRollupMae: number;
  /** The PM-plan forecasters, on the same panel. Undefined when --no-plans. */
  planMae?: number;
  planRidgeMae?: number;
  blendMae?: number;
}

const results: SeedResult[] = [];
const seeds = [shippedSeed, ...Array.from({ length: seedCount - 1 }, (_, i) => 1000 + i * 17)];

for (const seed of seeds) {
  const records = parseCsv(toCsv(generateDataset({ seed })));
  const cv = crossValidateNested(buildTrainingSamples(records));
  const actual = cv.predictions.map(p => p.sample.y);
  const modelMetrics = evaluate(actual, cv.predictions.map(p => p.predicted));

  const baselines = BASELINES.map(name => ({
    name,
    metrics: evaluate(actual, cv.predictions.map(p => baselinePrediction(p.sample, name))),
  }));
  const best = baselines.reduce((a, b) => (b.metrics.mae < a.metrics.mae ? b : a));

  // Paired per-row differences, then averaged within each person. The same
  // person appears in several folds, so person is the unit that is plausibly
  // independent - not the row.
  const byPerson = new Map<string, number[]>();
  for (const p of cv.predictions) {
    const key = `${p.sample.costCenter}|${p.sample.personName}`;
    const modelError = Math.abs(p.predicted - p.sample.y);
    const baselineError = Math.abs(baselinePrediction(p.sample, best.name) - p.sample.y);
    const list = byPerson.get(key) ?? [];
    list.push(modelError - baselineError);
    byPerson.set(key, list);
  }
  const perPerson = [...byPerson.values()].map(mean);
  const spread = sd(perPerson);
  const pairedMean = mean(perPerson);

  // Hours-weighted cost-centre rollup, for the model and for the same baseline,
  // on identical groupings. This is the number a resourcing conversation runs on.
  const rollup = (pick: (p: (typeof cv.predictions)[number]) => number) => {
    const groups = new Map<string, { a: number; p: number; w: number }[]>();
    for (const prediction of cv.predictions) {
      const key = `${prediction.sample.costCenter}|${prediction.sample.targetPeriod}`;
      const list = groups.get(key) ?? [];
      list.push({ a: prediction.sample.y, p: pick(prediction), w: prediction.sample.weight });
      groups.set(key, list);
    }
    const actualRollup: number[] = [];
    const predictedRollup: number[] = [];
    for (const rows of groups.values()) {
      const w = rows.reduce((x, r) => x + r.w, 0);
      actualRollup.push(rows.reduce((x, r) => x + r.a * r.w, 0) / w);
      predictedRollup.push(rows.reduce((x, r) => x + r.p * r.w, 0) / w);
    }
    return evaluate(actualRollup, predictedRollup).mae;
  };

  results.push({
    seed,
    modelMae: modelMetrics.mae,
    baselineName: best.name,
    baselineMae: best.metrics.mae,
    marginPct: (1 - modelMetrics.mae / best.metrics.mae) * 100,
    pairedMean,
    clusteredT: spread > 0 ? pairedMean / (spread / Math.sqrt(perPerson.length)) : 0,
    people: perPerson.length,
    modelBias: modelMetrics.bias,
    baselineBias: best.metrics.bias,
    modelWithin5: modelMetrics.within5,
    baselineWithin5: best.metrics.within5,
    modelRollupMae: rollup(p => p.predicted),
    baselineRollupMae: rollup(p => baselinePrediction(p.sample, best.name)),
    ...(withPlans
      ? (() => {
          const head = runHeadToHead(records, generatePlans(records, { seed: seed + 1 }));
          return {
            planMae: head.metrics.plan.mae,
            planRidgeMae: head.metrics.plan_ridge.mae,
            blendMae: head.metrics.blend.mae,
          };
        })()
      : {}),
  });
}

const thin = '-'.repeat(78);
const shipped = results[0];
const others = results.slice(1);
const wins = results.filter(r => r.marginPct > 0);
const significantWins = results.filter(r => r.clusteredT < -2);
const significantLosses = results.filter(r => r.clusteredT > 2);

console.log('Is the edge over the naive baseline real, or is it the seed?');
console.log('='.repeat(78));
console.log(`${results.length} seeds | same protocol each time | compared against the best baseline on each panel`);
console.log('');

console.log('seed'.padEnd(14) + 'model'.padStart(8) + 'baseline'.padStart(20) + 'margin'.padStart(10) + 'paired'.padStart(10) + 'clustered t'.padStart(14));
console.log(thin);
for (const r of results) {
  const label = r.seed === shippedSeed ? `${r.seed} *` : String(r.seed);
  console.log(
    label.padEnd(14) +
      r.modelMae.toFixed(3).padStart(8) +
      `${r.baselineName} ${r.baselineMae.toFixed(3)}`.padStart(20) +
      `${r.marginPct >= 0 ? '+' : ''}${r.marginPct.toFixed(1)}%`.padStart(10) +
      `${r.pairedMean >= 0 ? '+' : ''}${r.pairedMean.toFixed(3)}`.padStart(10) +
      r.clusteredT.toFixed(2).padStart(14),
  );
}
console.log('* the seed every published number is measured on.');
console.log('');

console.log('Summary');
console.log(thin);
console.log(`  Model beats the best baseline on ${wins.length} of ${results.length} seeds.`);
console.log(
  `  Margin across seeds: mean ${mean(results.map(r => r.marginPct)).toFixed(2)}%, ` +
    `sd ${sd(results.map(r => r.marginPct)).toFixed(2)}%, ` +
    `range ${Math.min(...results.map(r => r.marginPct)).toFixed(1)}% to ` +
    `${Math.max(...results.map(r => r.marginPct)).toFixed(1)}%.`,
);
console.log(
  `  Differences larger than the noise on their own panel (|clustered t| > 2): ` +
    `${significantWins.length} win, ${significantLosses.length} lose.`,
);
console.log(
  `  The shipped seed gives ${shipped.marginPct >= 0 ? '+' : ''}${shipped.marginPct.toFixed(1)}% ` +
    `(t = ${shipped.clusteredT.toFixed(2)}); the other ${others.length} average ` +
    `${mean(others.map(r => r.marginPct)).toFixed(2)}%.`,
);
console.log('');

// Which claims survive re-drawing the world? Level accuracy is only one of
// them, and it was never the one the write-ups leaned on hardest.
console.log('Do the other claims survive re-drawing the world?');
console.log(thin);
const claim = (
  label: string,
  modelValues: number[],
  baselineValues: number[],
  betterIsLower: boolean,
  digits = 3,
) => {
  const modelMean = mean(modelValues);
  const baselineMean = mean(baselineValues);
  const modelWins = modelValues.filter((v, i) =>
    betterIsLower ? v < baselineValues[i] : v > baselineValues[i],
  ).length;
  console.log(
    `  ${label.padEnd(30)} model ${modelMean.toFixed(digits).padStart(7)}  ` +
      `baseline ${baselineMean.toFixed(digits).padStart(7)}  ` +
      `model better on ${modelWins}/${results.length} seeds`,
  );
};
claim('MAE (pp)', results.map(r => r.modelMae), results.map(r => r.baselineMae), true);
claim(
  'absolute bias (pp)',
  results.map(r => Math.abs(r.modelBias)),
  results.map(r => Math.abs(r.baselineBias)),
  true,
);
claim('within 5pp', results.map(r => r.modelWithin5), results.map(r => r.baselineWithin5), false);
claim(
  'cost-centre rollup MAE (pp)',
  results.map(r => r.modelRollupMae),
  results.map(r => r.baselineRollupMae),
  true,
);
if (withPlans && results.every(r => r.blendMae !== undefined)) {
  claim(
    'blend with PM plans, MAE (pp)',
    results.map(r => r.blendMae as number),
    results.map(r => r.baselineMae),
    true,
  );
  claim(
    'model + plans, MAE (pp)',
    results.map(r => r.planRidgeMae as number),
    results.map(r => r.baselineMae),
    true,
  );
}
console.log('');

const meanMargin = mean(results.map(r => r.marginPct));
const verdict =
  wins.length >= results.length * 0.75 && meanMargin > 0.5
    ? 'The edge holds across seeds. The published figure is representative.'
    : wins.length <= results.length * 0.25
      ? 'The edge does NOT hold. On most panels the model is no better than "same as last\n' +
        'period", and the published figure reflects the seed it was measured on rather than\n' +
        'the model. The model should not be recommended over the naive baseline on level\n' +
        'accuracy alone - its defensible claims are the bias correction, the calibrated\n' +
        'interval, and the cost-centre rollup.'
      : 'The edge is not reliable across seeds. Treat the published figure as one draw, and\n' +
        'do not present the model as a dependable accuracy improvement over the baseline.';
console.log(verdict);

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(
  outPath,
  JSON.stringify(
    {
      generatedAt: new Date().toISOString(),
      shippedSeed,
      results,
      summary: {
        seeds: results.length,
        wins: wins.length,
        meanMarginPct: meanMargin,
        sdMarginPct: sd(results.map(r => r.marginPct)),
        significantWins: significantWins.length,
        significantLosses: significantLosses.length,
      },
    },
    null,
    2,
  ) + '\n',
);
console.log(`\nWrote ${path.relative(process.cwd(), outPath)}`);
