/**
 * Workforce utilization dataset.
 *
 * The schema mirrors a monthly timesheet extract that has been combined from one
 * workbook per period, so the period itself is only carried in `Source.Name`
 * (the file the row came from). Everything else in `UtilizationRecord` maps 1:1
 * to a column in that extract.
 *
 * Accounting identities every row satisfies (see `verifyRecord`):
 *   Total        = Direct Total + Indirect Total + Fringe Total
 *   Avail Total  = Total - Fringe Total
 *   Indirect Total = Admin + Bench + Bus Dev + Mngmt + Opportunity + Training + Other
 *   Fringe Total = Stat & Disc + Vacation + Wellness
 *   Util %          = Direct Total / Avail Total * 100
 *   Billability %   = Direct Total / Total * 100
 *   Util % Variance        = Util % - Util % Target
 *   Billability % Variance = Billability % - Billability % Target
 *
 * `OT Hours` are paid overtime already counted inside Direct Total (and Total).
 * `Unpaid OT` and `Unpaid Regular` are memo-only hours: unpaid, so excluded from
 * Total. YTD columns are fiscal-year-to-date ratios of the cumulative hours.
 */

export type TargetType = 'Billable Staff' | 'Project Leadership' | 'Overhead';

export type JobLevel =
  | 'L1 Analyst'
  | 'L2 Engineer'
  | 'L3 Senior Engineer'
  | 'L4 Manager'
  | 'L5 Principal';

export interface UtilizationRecord {
  // File and organizational data
  sourceName: string;
  costCenter: string;
  costCenterName: string;
  personName: string;
  targetType: TargetType;
  jobLevel: JobLevel;

  // Utilization metrics (percentages, e.g. 84.25)
  utilPctYtd: number;
  utilPct: number;
  utilPctTarget: number;
  utilPctVariance: number;

  // Billability metrics (percentages)
  billabilityPctYtd: number;
  billabilityPct: number;
  billabilityPctTarget: number;
  billabilityPctVariance: number;

  // Core hour data
  total: number;
  availTotal: number;
  directTotal: number;
  indirectTotal: number;
  otHours: number;

  // Indirect and nonproductive hours
  admin: number;
  bench: number;
  busDev: number;
  mngmt: number;
  opportunity: number;
  training: number;
  other: number;
  unpaidOt: number;

  // Leave and fringe hours
  fringeTotal: number;
  statAndDisc: number;
  vacation: number;
  wellness: number;
  unpaidRegular: number;
}

/** A record plus the period decoded from `Source.Name`. */
export interface PeriodedRecord extends UtilizationRecord {
  /** 1-based fiscal period, 1..12. */
  periodIndex: number;
  /** Calendar month of the period, `YYYY-MM`. */
  periodMonth: string;
}

/** CSV header -> record key, in the exact column order of the extract. */
export const COLUMNS: ReadonlyArray<readonly [string, keyof UtilizationRecord]> = [
  ['Source.Name', 'sourceName'],
  ['Cost Center', 'costCenter'],
  ['Cost Center Name', 'costCenterName'],
  ['Person Name', 'personName'],
  ['Target Type', 'targetType'],
  ['Job Level', 'jobLevel'],
  ['Util % YTD', 'utilPctYtd'],
  ['Util %', 'utilPct'],
  ['Util % Target', 'utilPctTarget'],
  ['Util % Variance', 'utilPctVariance'],
  ['Billability % YTD', 'billabilityPctYtd'],
  ['Billability %', 'billabilityPct'],
  ['Billability % Target', 'billabilityPctTarget'],
  ['Billability % Variance', 'billabilityPctVariance'],
  ['Total', 'total'],
  ['Avail Total', 'availTotal'],
  ['Direct Total', 'directTotal'],
  ['Indirect Total', 'indirectTotal'],
  ['OT Hours', 'otHours'],
  ['Admin', 'admin'],
  ['Bench', 'bench'],
  ['Bus Dev', 'busDev'],
  ['Mngmt', 'mngmt'],
  ['Opportunity', 'opportunity'],
  ['Training', 'training'],
  ['Other', 'other'],
  ['Unpaid OT', 'unpaidOt'],
  ['Fringe Total', 'fringeTotal'],
  ['Stat & Disc', 'statAndDisc'],
  ['Vacation', 'vacation'],
  ['Wellness', 'wellness'],
  ['Unpaid Regular', 'unpaidRegular'],
];

const TEXT_KEYS = new Set<keyof UtilizationRecord>([
  'sourceName',
  'costCenter',
  'costCenterName',
  'personName',
  'targetType',
  'jobLevel',
]);

export function isTextColumn(key: keyof UtilizationRecord): boolean {
  return TEXT_KEYS.has(key);
}

/** Returns the identity violations in a row, empty when the row is consistent. */
export function verifyRecord(r: UtilizationRecord, tol = 0.051): string[] {
  const problems: string[] = [];
  const check = (label: string, actual: number, expected: number) => {
    if (Math.abs(actual - expected) > tol) {
      problems.push(`${label}: ${actual.toFixed(2)} != ${expected.toFixed(2)}`);
    }
  };
  const categories =
    r.admin + r.bench + r.busDev + r.mngmt + r.opportunity + r.training + r.other;
  check('Indirect Total', r.indirectTotal, categories);
  check('Fringe Total', r.fringeTotal, r.statAndDisc + r.vacation + r.wellness);
  check('Avail Total', r.availTotal, r.total - r.fringeTotal);
  check('Total', r.total, r.directTotal + r.indirectTotal + r.fringeTotal);
  check('Util %', r.utilPct, (r.directTotal / r.availTotal) * 100);
  check('Billability %', r.billabilityPct, (r.directTotal / r.total) * 100);
  check('Util % Variance', r.utilPctVariance, r.utilPct - r.utilPctTarget);
  check(
    'Billability % Variance',
    r.billabilityPctVariance,
    r.billabilityPct - r.billabilityPctTarget,
  );
  if (r.otHours > r.directTotal + tol) problems.push('OT Hours exceed Direct Total');
  return problems;
}
