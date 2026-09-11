import { periodFromSourceName, sourceNameFor } from './csv.ts';
import { makeNormal, makeRng } from './generate.ts';
import { groupByPerson } from './features.ts';
import type { PeriodedRecord } from './types.ts';

/**
 * PM hour allocations: what the planners said each person would do, recorded
 * before the period happened.
 *
 * This is a *separate* extract from the timesheet data on purpose. In a real
 * firm the allocations live in the project system, not the timesheet system,
 * and here it also means `data/utilization.csv` is untouched by this file -
 * the actuals, and every number measured on them, stay exactly as they were.
 *
 * The one property the whole head-to-head rests on is that a plan row is
 * **snapshotted at forecast time**: the plan for period t+1 is frozen at the
 * end of period t and never revised afterwards. Scoring a later-revised
 * allocation would invent accuracy that never existed, so `planSnapshotPeriod`
 * is always exactly one period before `planPeriod`, and `verifyPlan` enforces
 * it.
 *
 * What the simulated planner knows, and does not:
 *   - it has genuine **foresight** the history-only model structurally cannot
 *     have (project end dates, new awards, approved leave), modelled as a noisy
 *     read on the coming period;
 *   - it is **optimistic**, and asymmetrically so: plans get pulled up toward
 *     target much harder than they get pulled down, because people are booked
 *     to fill their available hours;
 *   - it goes **stale**: a share of allocations are not refreshed after a slip,
 *     and simply carry the previous snapshot forward, foresight and all.
 *
 * Actuals are generated first and are never a function of the plan, so nothing
 * here can feed back into the dataset. That also means this simulation contains
 * no self-fulfilling booking effect - see the caveat in the write-up.
 */

export interface PlanRecord {
  /** `Source.Name` of the extract the plan was frozen against (period t). */
  snapshotSourceName: string;
  costCenter: string;
  costCenterName: string;
  personName: string;
  /** 1-based fiscal period the plan is *about* (period t+1). */
  planPeriod: number;
  /** Calendar month of `planPeriod`, `YYYY-MM`. */
  planMonth: string;
  plannedAvailHours: number;
  plannedDirectHours: number;
  plannedUtilPct: number;
  /**
   * Periods since the underlying allocation was last refreshed. 0 means it was
   * rebuilt at this snapshot; n > 0 means it is a carry-forward of a plan
   * originally written n periods ago, and is about a period it never targeted.
   */
  planAgePeriods: number;
}

/** A plan row plus the period its snapshot was taken in. */
export interface SnapshottedPlan extends PlanRecord {
  /** 1-based period the plan was frozen in; always `planPeriod - 1`. */
  planSnapshotPeriod: number;
}

/** CSV header -> record key, in column order. */
export const PLAN_COLUMNS: ReadonlyArray<readonly [string, keyof PlanRecord]> = [
  ['Snapshot Source.Name', 'snapshotSourceName'],
  ['Cost Center', 'costCenter'],
  ['Cost Center Name', 'costCenterName'],
  ['Person Name', 'personName'],
  ['Plan Period', 'planPeriod'],
  ['Plan Month', 'planMonth'],
  ['Planned Avail Hours', 'plannedAvailHours'],
  ['Planned Direct Hours', 'plannedDirectHours'],
  ['Planned Util %', 'plannedUtilPct'],
  ['Plan Age Periods', 'planAgePeriods'],
];

const PLAN_TEXT_KEYS = new Set<keyof PlanRecord>([
  'snapshotSourceName',
  'costCenter',
  'costCenterName',
  'personName',
  'planMonth',
]);

/** Returns the problems in a plan row, empty when the row is consistent. */
export function verifyPlan(plan: SnapshottedPlan, tol = 0.051): string[] {
  const problems: string[] = [];
  if (plan.planSnapshotPeriod !== plan.planPeriod - 1) {
    problems.push(
      `snapshot period ${plan.planSnapshotPeriod} is not one before plan period ${plan.planPeriod}`,
    );
  }
  if (plan.plannedAvailHours <= 0) problems.push('planned available hours must be positive');
  // Planned direct hours are the rounded quantity here (Util % is the input the
  // planner works in), so the identity is checked in hours - checking it in
  // percentage points would flag a tenth of an hour of rounding as a violation.
  const impliedDirect = (plan.plannedUtilPct / 100) * plan.plannedAvailHours;
  if (Math.abs(plan.plannedDirectHours - impliedDirect) > tol) {
    problems.push(
      `Planned Direct Hours: ${plan.plannedDirectHours.toFixed(1)} != ${impliedDirect.toFixed(1)}`,
    );
  }
  if (plan.plannedDirectHours < -tol) problems.push('planned direct hours are negative');
  if (plan.plannedDirectHours > plan.plannedAvailHours + tol) {
    problems.push('planned direct hours exceed planned available hours');
  }
  if (plan.planAgePeriods < 0) problems.push('plan age is negative');
  return problems;
}

function round1(x: number): number {
  return Math.round(x * 10) / 10;
}

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** FNV-1a over the person key, mixed with the run seed, to seed that person's stream. */
function personSeed(seed: number, key: string): number {
  let hash = 0x811c9dc5 ^ (seed >>> 0);
  for (let i = 0; i < key.length; i++) {
    hash ^= key.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

export interface PlanGenerateOptions {
  seed?: number;
  /**
   * How much of the coming period the planner genuinely sees, 0..1, before
   * person-level variation. This is the whole reason a plan can beat a
   * history-only model, so it is the number to move when asking "how good would
   * PM plans have to be for this conclusion to flip?".
   */
  foresight?: number;
  /** Probability an allocation is carried forward instead of refreshed. */
  stalenessRate?: number;
  /**
   * How hard a plan sitting *below* target is pulled up toward it. This is the
   * optimism knob, and the results are sensitive to it, so it is exposed rather
   * than buried: the head-to-head sweeps it.
   */
  optimismUp?: number;
  /** How hard a plan sitting *above* target is pulled back down toward it. */
  optimismDown?: number;
}

/** How the planner's view is built before foresight and optimism are applied. */
const PRIOR_WEIGHTS = { last: 0.5, recentMean: 0.2, target: 0.3 };
/**
 * Default pull toward target when the plan sits below it, and when it sits
 * above it. The asymmetry is the point: people get booked to fill their
 * available hours, so plans are pulled up much harder than they are pulled
 * down. At these defaults the simulated plans run about +2.5pp optimistic,
 * which is the order of bias professional-services planning tends to carry.
 */
const OPTIMISM_UP = 0.35;
const OPTIMISM_DOWN = 0.1;
/** Planner noise in percentage points, and noise on planned available hours. */
const PLAN_NOISE_PP = 3;
const AVAIL_NOISE_HOURS = 8;
/** A carried-forward allocation gets refreshed by force after this many periods. */
const MAX_PLAN_AGE = 3;

/**
 * Builds one plan row per person per period from period 2 onward.
 *
 * There is deliberately **no plan for the period after the data ends**: the
 * planner's foresight is modelled as a noisy read on the actual, and past the
 * last extract there is no actual to read. Inventing a foresight-free plan for
 * that period would put a row in front of the model that looks nothing like the
 * rows it was fitted on.
 */
export function generatePlans(
  records: PeriodedRecord[],
  options: PlanGenerateOptions = {},
): SnapshottedPlan[] {
  const seed = options.seed ?? 20260902;
  const baseForesight = options.foresight ?? 0.55;
  const stalenessRate = options.stalenessRate ?? 0.28;
  const optimismUp = options.optimismUp ?? OPTIMISM_UP;
  const optimismDown = options.optimismDown ?? OPTIMISM_DOWN;

  const panel = groupByPerson(records);
  const plans: SnapshottedPlan[] = [];

  const keys = [...panel.keys()].sort();

  for (const key of keys) {
    const rows = panel.get(key) as PeriodedRecord[];
    // One PRNG stream per person, so a person's plans depend only on that
    // person's own history. Two things follow, and both are worth having: the
    // roster can change without perturbing everybody else's allocations, and
    // truncating the data to period p reproduces the first p-1 plan rows
    // exactly - which is what makes the "a plan never depends on a period after
    // the one it is about" invariant testable rather than merely asserted.
    const rng = makeRng(personSeed(seed, key));
    const normal = makeNormal(rng);
    const personForesight = clamp(normal(baseForesight, 0.12), 0.2, 0.85);
    const personStaleness = clamp(normal(stalenessRate, 0.08), 0, 0.6);

    let previous: { util: number; avail: number; age: number } | undefined;

    for (let i = 1; i < rows.length; i++) {
      const snapshot = rows[i - 1];
      const actual = rows[i];
      // A gap in someone's history breaks the snapshot-is-one-period-before
      // invariant, so no plan is written across it.
      if (actual.periodIndex !== snapshot.periodIndex + 1) {
        previous = undefined;
        continue;
      }

      const history = rows.slice(0, i).map(r => r.utilPct);
      const lastActual = history[history.length - 1];
      const recentMean = mean(history.slice(-3));
      const target = snapshot.utilPctTarget;

      const carriedForward =
        previous !== undefined && previous.age < MAX_PLAN_AGE && rng() < personStaleness;

      let plannedUtil: number;
      let plannedAvail: number;
      let planAge: number;

      if (carriedForward && previous !== undefined) {
        // Nobody rebuilt the allocation: last snapshot's numbers ride along,
        // still describing the period they were written for.
        plannedUtil = previous.util;
        plannedAvail = previous.avail;
        planAge = previous.age + 1;
      } else {
        const view =
          PRIOR_WEIGHTS.last * lastActual +
          PRIOR_WEIGHTS.recentMean * recentMean +
          PRIOR_WEIGHTS.target * target;
        // Foresight: the part of the coming period the planner actually knows.
        const informed = view + personForesight * (actual.utilPct - view);
        const gap = target - informed;
        const pull = gap > 0 ? optimismUp : optimismDown;
        plannedUtil = clamp(informed + pull * gap + normal(0, PLAN_NOISE_PP), 5, 100);
        // Approved leave is known in advance, so planned capacity is close to
        // the capacity that materialises - but not exact.
        plannedAvail = Math.max(8, actual.availTotal + normal(0, AVAIL_NOISE_HOURS));
        planAge = 0;
      }

      const util = round2(plannedUtil);
      const avail = round1(plannedAvail);
      previous = { util, avail, age: planAge };

      plans.push({
        snapshotSourceName: sourceNameFor(snapshot.periodIndex, snapshot.periodMonth),
        costCenter: actual.costCenter,
        costCenterName: actual.costCenterName,
        personName: actual.personName,
        planPeriod: actual.periodIndex,
        planMonth: actual.periodMonth,
        planSnapshotPeriod: snapshot.periodIndex,
        plannedAvailHours: avail,
        plannedDirectHours: round1((util / 100) * avail),
        plannedUtilPct: util,
        planAgePeriods: planAge,
      });
    }
  }

  return plans.sort(
    (a, b) =>
      a.planPeriod - b.planPeriod ||
      a.costCenter.localeCompare(b.costCenter) ||
      a.personName.localeCompare(b.personName),
  );
}

function escapeCell(value: string): string {
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

function formatPlanCell(plan: PlanRecord, key: keyof PlanRecord): string {
  const value = plan[key];
  if (typeof value === 'string') return escapeCell(value);
  if (key === 'planPeriod' || key === 'planAgePeriods') return String(value);
  return value.toFixed(key === 'plannedUtilPct' ? 2 : 1);
}

export function plansToCsv(plans: PlanRecord[]): string {
  const header = PLAN_COLUMNS.map(([name]) => escapeCell(name)).join(',');
  const lines = plans.map(p => PLAN_COLUMNS.map(([, key]) => formatPlanCell(p, key)).join(','));
  return [header, ...lines].join('\n') + '\n';
}

function parseCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"' && line[i + 1] === '"') {
        current += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      cells.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  cells.push(current);
  return cells;
}

export function parsePlanCsv(text: string): SnapshottedPlan[] {
  const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length === 0) return [];
  const header = parseCsvLine(lines[0]).map(h => h.trim());
  const expected = PLAN_COLUMNS.map(([name]) => name);
  const missing = expected.filter(name => !header.includes(name));
  if (missing.length > 0) throw new Error(`Missing plan columns: ${missing.join(', ')}`);
  const indexOf = new Map(header.map((name, i) => [name, i] as const));

  return lines.slice(1).map((line, row) => {
    const cells = parseCsvLine(line);
    const plan = {} as PlanRecord;
    for (const [name, key] of PLAN_COLUMNS) {
      const raw = (cells[indexOf.get(name) as number] ?? '').trim();
      if (PLAN_TEXT_KEYS.has(key)) {
        (plan[key] as string) = raw;
      } else {
        const num = Number(raw);
        if (!Number.isFinite(num)) {
          throw new Error(`Plan row ${row + 2}: "${name}" is not numeric ("${raw}")`);
        }
        (plan[key] as number) = num;
      }
    }
    return {
      ...plan,
      planSnapshotPeriod: periodFromSourceName(plan.snapshotSourceName).periodIndex,
    };
  });
}

/** Index plans by person and the period they are about, for joining onto samples. */
export function indexPlans(plans: SnapshottedPlan[]): Map<string, SnapshottedPlan> {
  const index = new Map<string, SnapshottedPlan>();
  for (const plan of plans) {
    index.set(planKey(plan.costCenter, plan.personName, plan.planPeriod), plan);
  }
  return index;
}

export function planKey(costCenter: string, personName: string, planPeriod: number): string {
  return `${costCenter}|${personName}|${planPeriod}`;
}
