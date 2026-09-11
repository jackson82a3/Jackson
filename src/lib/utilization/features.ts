import type { PeriodedRecord } from './types.ts';

/**
 * Feature engineering for the next-period utilization forecast.
 *
 * One sample = one person observed through period t (the "origin"), with the
 * response being that person's Util % in period t+1. Every feature is computed
 * strictly from rows at or before the origin, so a sample never sees its own
 * future; that is what makes the rolling-origin evaluation in `forecast.ts`
 * honest.
 */

/** Periods of history a person needs before they produce a training sample. */
export const MIN_HISTORY = 3;

export const FEATURE_NAMES = [
  // Mean reversion: how far this person sits from their own run rate.
  'util_vs_person_mean',
  'ma3_vs_person_mean',
  'util_momentum',
  'util_person_sd',
  'gap_to_target',
  'ytd_vs_util',
  // Where last period's non-billable time went, as a share of available hours.
  'bench_share',
  'bench_share_ma3',
  'training_share',
  'leadership_share',
  'opportunity_share',
  'fringe_share',
  'avail_ratio',
  'ot_share',
  // Cross-sectional position. Absolute cost-center and firm levels are left out
  // on purpose: with one period per calendar month they are close to a period
  // label, and a model that leans on them just refits the period mean.
  'util_vs_cost_center',
  'cost_center_vs_firm',
  'cost_center_momentum_rel',
  'firm_util_momentum',
];

export interface Sample {
  personName: string;
  costCenter: string;
  costCenterName: string;
  jobLevel: string;
  targetType: string;
  originPeriod: number;
  targetPeriod: number;
  utilTarget: number;
  x: number[];
  /** Util % in the target period; undefined for a forecast row. */
  y: number;
  /** Trailing available hours, used to weight people into a cost-center number. */
  weight: number;
  /**
   * What the model is corrected away from: the person's last observed Util %.
   * The ridge fit predicts `y - anchor`, so shrinking the coefficients walks the
   * forecast back to "same as last period" rather than to the firm-wide mean.
   */
  anchor: number;
  /** Handles for the naive baselines, so they see exactly the same rows. */
  baselines: { last: number; ma3: number; personMean: number; target: number; ccLast: number };
}

export interface PeriodAggregate {
  /** Available-hours-weighted mean Util % per cost center. */
  byCostCenter: Map<string, number>;
  firmUtil: number;
}

export function groupByPerson(records: PeriodedRecord[]): Map<string, PeriodedRecord[]> {
  const panel = new Map<string, PeriodedRecord[]>();
  for (const record of records) {
    const key = `${record.costCenter}|${record.personName}`;
    const rows = panel.get(key);
    if (rows) rows.push(record);
    else panel.set(key, [record]);
  }
  for (const rows of panel.values()) rows.sort((a, b) => a.periodIndex - b.periodIndex);
  return panel;
}

export function periodAggregates(records: PeriodedRecord[]): Map<number, PeriodAggregate> {
  const byPeriod = new Map<number, PeriodedRecord[]>();
  for (const record of records) {
    const rows = byPeriod.get(record.periodIndex);
    if (rows) rows.push(record);
    else byPeriod.set(record.periodIndex, [record]);
  }

  const aggregates = new Map<number, PeriodAggregate>();
  for (const [periodIndex, rows] of byPeriod) {
    const byCostCenter = new Map<string, number>();
    const sums = new Map<string, { direct: number; avail: number }>();
    let firmDirect = 0;
    let firmAvail = 0;
    for (const row of rows) {
      const entry = sums.get(row.costCenter) ?? { direct: 0, avail: 0 };
      entry.direct += row.directTotal;
      entry.avail += row.availTotal;
      sums.set(row.costCenter, entry);
      firmDirect += row.directTotal;
      firmAvail += row.availTotal;
    }
    for (const [costCenter, entry] of sums) {
      byCostCenter.set(costCenter, (entry.direct / entry.avail) * 100);
    }
    aggregates.set(periodIndex, { byCostCenter, firmUtil: (firmDirect / firmAvail) * 100 });
  }
  return aggregates;
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function sd(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(mean(values.map(v => (v - m) ** 2)));
}

function ccUtilAt(
  aggregates: Map<number, PeriodAggregate>,
  costCenter: string,
  periodIndex: number,
  fallback: number,
): number {
  return aggregates.get(periodIndex)?.byCostCenter.get(costCenter) ?? fallback;
}

function firmUtilAt(
  aggregates: Map<number, PeriodAggregate>,
  periodIndex: number,
  fallback: number,
): number {
  return aggregates.get(periodIndex)?.firmUtil ?? fallback;
}

/**
 * Builds the feature row for a person whose history is `rows` (ordered, ending
 * at the origin period) forecasting `targetPeriod`.
 */
export function buildSample(
  rows: PeriodedRecord[],
  aggregates: Map<number, PeriodAggregate>,
  targetPeriod: number,
  y: number,
): Sample {
  const last = rows[rows.length - 1];
  const utils = rows.map(r => r.utilPct);
  const recent = utils.slice(-3);
  const lag0 = utils[utils.length - 1];
  const lag1 = utils[utils.length - 2];
  const ma3 = mean(recent);
  const personMean = mean(utils);

  const benchShares = rows.slice(-3).map(r => (r.bench / r.availTotal) * 100);
  const ccLag0 = ccUtilAt(aggregates, last.costCenter, last.periodIndex, personMean);
  const ccLag1 = ccUtilAt(aggregates, last.costCenter, last.periodIndex - 1, ccLag0);
  const firmLag0 = firmUtilAt(aggregates, last.periodIndex, personMean);
  const firmLag1 = firmUtilAt(aggregates, last.periodIndex - 1, firmLag0);

  const x = [
    lag0 - personMean,
    ma3 - personMean,
    lag0 - lag1,
    sd(utils),
    last.utilPctTarget - lag0,
    last.utilPctYtd - lag0,

    (last.bench / last.availTotal) * 100,
    mean(benchShares),
    (last.training / last.availTotal) * 100,
    ((last.busDev + last.mngmt) / last.availTotal) * 100,
    (last.opportunity / last.availTotal) * 100,
    (last.fringeTotal / last.total) * 100,
    (last.availTotal / last.total) * 100,
    (last.otHours / last.availTotal) * 100,

    lag0 - ccLag0,
    ccLag0 - firmLag0,
    ccLag0 - ccLag1 - (firmLag0 - firmLag1),
    firmLag0 - firmLag1,
  ];

  if (x.length !== FEATURE_NAMES.length) {
    throw new Error(`Feature row has ${x.length} values, expected ${FEATURE_NAMES.length}`);
  }

  return {
    personName: last.personName,
    costCenter: last.costCenter,
    costCenterName: last.costCenterName,
    jobLevel: last.jobLevel,
    targetType: last.targetType,
    originPeriod: last.periodIndex,
    targetPeriod,
    utilTarget: last.utilPctTarget,
    x,
    y,
    weight: mean(rows.slice(-3).map(r => r.availTotal)),
    anchor: lag0,
    baselines: {
      last: lag0,
      ma3,
      personMean,
      target: last.utilPctTarget,
      ccLast: ccLag0,
    },
  };
}

/**
 * Every (person, origin -> origin+horizon) pair that has both history and an outcome.
 *
 * `horizon` is how many periods ahead the sample predicts. It defaults to 1,
 * which is what the shipped model uses; larger values exist so the decay beyond
 * one period can be *measured* rather than asserted. The features are unchanged
 * either way - they are all computed at the origin - so a horizon-3 sample is
 * the same information being asked a harder question.
 */
export function buildTrainingSamples(records: PeriodedRecord[], horizon = 1): Sample[] {
  if (!Number.isInteger(horizon) || horizon < 1) {
    throw new Error(`Horizon must be a positive integer, got ${horizon}`);
  }
  const panel = groupByPerson(records);
  const aggregates = periodAggregates(records);
  const samples: Sample[] = [];

  for (const rows of panel.values()) {
    for (let i = MIN_HISTORY - 1; i < rows.length - horizon; i++) {
      const history = rows.slice(0, i + 1);
      const next = rows[i + horizon];
      // No gap-spanning samples: the target must be exactly `horizon` periods
      // after the origin, not merely the next row that happens to exist.
      if (next.periodIndex !== rows[i].periodIndex + horizon) continue;
      samples.push(buildSample(history, aggregates, next.periodIndex, next.utilPct));
    }
  }
  return samples.sort((a, b) => a.targetPeriod - b.targetPeriod);
}

/** Feature rows for the period after the last one present in the data. */
export function buildForecastSamples(records: PeriodedRecord[]): Sample[] {
  const panel = groupByPerson(records);
  const aggregates = periodAggregates(records);
  const lastPeriod = Math.max(...records.map(r => r.periodIndex));
  const samples: Sample[] = [];

  for (const rows of panel.values()) {
    if (rows.length < MIN_HISTORY) continue;
    if (rows[rows.length - 1].periodIndex !== lastPeriod) continue; // left the roster
    samples.push(buildSample(rows, aggregates, lastPeriod + 1, Number.NaN));
  }
  return samples;
}
