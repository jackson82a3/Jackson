import { sourceNameFor } from './csv.ts';
import type { JobLevel, TargetType, UtilizationRecord } from './types.ts';

/**
 * Synthetic generator for twelve monthly utilization extracts (FY26, Oct-Sep).
 *
 * The point of the generator is that the series has structure a forecaster can
 * actually learn: a person effect, a job-level and cost-center effect, calendar
 * seasonality (vacation-heavy summers, the December holiday period, the fiscal
 * year-end push), an AR(1) shock that persists across periods, a firm-wide
 * demand factor shared by everybody, and occasional bench events that drop one
 * person for a period or two. Everything is driven by a seeded PRNG, so the same
 * seed always produces the same twelve files.
 */

export interface GenerateOptions {
  seed?: number;
  /**
   * Fiscal years to generate, 1 by default.
   *
   * The shipped dataset is one year, and every published number is a number
   * about it. A second year exists to answer the question one year cannot: with
   * a single period per calendar month, every validation month is a month the
   * model has never seen, so month effects can be neither learned nor
   * validated. Two years is the smallest panel where that stops being true.
   */
  years?: number;
}

interface Period {
  index: number;
  month: string;
  workdays: number;
  statHours: number;
}

const CALENDAR: Period[] = [
  { index: 1, month: '2025-10', workdays: 23, statHours: 8 },
  { index: 2, month: '2025-11', workdays: 20, statHours: 8 },
  { index: 3, month: '2025-12', workdays: 23, statHours: 24 },
  { index: 4, month: '2026-01', workdays: 22, statHours: 8 },
  { index: 5, month: '2026-02', workdays: 20, statHours: 8 },
  { index: 6, month: '2026-03', workdays: 22, statHours: 0 },
  { index: 7, month: '2026-04', workdays: 22, statHours: 8 },
  { index: 8, month: '2026-05', workdays: 21, statHours: 8 },
  { index: 9, month: '2026-06', workdays: 22, statHours: 0 },
  { index: 10, month: '2026-07', workdays: 23, statHours: 8 },
  { index: 11, month: '2026-08', workdays: 21, statHours: 8 },
  { index: 12, month: '2026-09', workdays: 22, statHours: 8 },
];

/** Demand seasonality in percentage points, indexed by period. */
const SEASONALITY = [0.5, 1.0, -2.5, -1.5, 0.5, 1.5, 1.0, 0.5, 0.0, -1.5, -1.0, 2.0];
/** Share of the annual vacation budget typically taken in each period. */
const VACATION_WEIGHT = [0.04, 0.05, 0.15, 0.05, 0.04, 0.08, 0.06, 0.05, 0.07, 0.24, 0.12, 0.05];

interface CostCenter {
  code: string;
  name: string;
  headcount: number;
  utilOffset: number;
  benchRisk: number;
  otRate: number;
}

const COST_CENTERS: CostCenter[] = [
  { code: 'CC-1010', name: 'Structural Engineering', headcount: 11, utilOffset: 2, benchRisk: 0.04, otRate: 0.18 },
  { code: 'CC-1020', name: 'Mechanical Engineering', headcount: 10, utilOffset: 1, benchRisk: 0.05, otRate: 0.15 },
  { code: 'CC-1030', name: 'Electrical Engineering', headcount: 9, utilOffset: 0, benchRisk: 0.05, otRate: 0.16 },
  { code: 'CC-2010', name: 'Project Management', headcount: 10, utilOffset: -1, benchRisk: 0.04, otRate: 0.12 },
  { code: 'CC-3010', name: 'Field Services', headcount: 10, utilOffset: 3, benchRisk: 0.03, otRate: 0.35 },
  { code: 'CC-4010', name: 'Digital Solutions', headcount: 10, utilOffset: -3, benchRisk: 0.09, otRate: 0.2 },
];

interface LevelProfile {
  level: JobLevel;
  weight: number;
  baseUtil: number;
  utilTarget: number;
  mngmtShare: number;
  busDevShare: number;
}

const LEVELS: LevelProfile[] = [
  { level: 'L1 Analyst', weight: 0.2, baseUtil: 81, utilTarget: 80, mngmtShare: 0.005, busDevShare: 0.005 },
  { level: 'L2 Engineer', weight: 0.28, baseUtil: 88, utilTarget: 85, mngmtShare: 0.01, busDevShare: 0.01 },
  { level: 'L3 Senior Engineer', weight: 0.26, baseUtil: 89, utilTarget: 85, mngmtShare: 0.03, busDevShare: 0.02 },
  { level: 'L4 Manager', weight: 0.16, baseUtil: 64, utilTarget: 60, mngmtShare: 0.17, busDevShare: 0.05 },
  { level: 'L5 Principal', weight: 0.1, baseUtil: 57, utilTarget: 55, mngmtShare: 0.11, busDevShare: 0.1 },
];

const FIRST_NAMES = [
  'Amara', 'Bennett', 'Camille', 'Devon', 'Elena', 'Farid', 'Grace', 'Hassan',
  'Imani', 'Jonas', 'Katya', 'Liam', 'Maren', 'Nikhil', 'Odessa', 'Priya',
  'Quentin', 'Rosalind', 'Soren', 'Tamsin', 'Ulises', 'Vera', 'Wren', 'Xiomara',
  'Yusuf', 'Zara', 'Adaeze', 'Bruno', 'Clara', 'Dmitri',
];

const LAST_NAMES = [
  'Okafor', 'Lindqvist', 'Moreau', 'Ashby', 'Petrova', 'Haddad', 'Whitfield', 'Nakamura',
  'Beaumont', 'Iverson', 'Salazar', 'Duarte', 'Kowalski', 'Rathore', 'Fontaine', 'Mbeki',
  'Vasquez', 'Thorne', 'Halvorsen', 'Bergstrom', 'Carrasco', 'Novak', 'Adeyemi', 'Larkin',
  'Sandoval', 'Ferreira', 'Delacroix', 'Yilmaz', 'Ashworth', 'Mensah',
];

interface Person {
  name: string;
  costCenter: CostCenter;
  profile: LevelProfile;
  targetType: TargetType;
  utilTarget: number;
  billabilityTarget: number;
  /** Persistent individual effect on utilization, in percentage points. */
  effect: number;
  volatility: number;
  benchRisk: number;
  otRate: number;
  vacationBudget: number;
  /** Number of opening periods a new hire spends ramping up; 0 for tenured staff. */
  rampPeriods: number;
}

/** Deterministic 32-bit PRNG (mulberry32) so a seed reproduces the dataset. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeNormal(rng: () => number): (mean: number, sd: number) => number {
  return (mean, sd) => {
    // Box-Muller; the second variate is discarded to keep the draw order simple.
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    return mean + sd * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
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

function pickLevel(rng: () => number): LevelProfile {
  const draw = rng();
  let cumulative = 0;
  for (const profile of LEVELS) {
    cumulative += profile.weight;
    if (draw <= cumulative) return profile;
  }
  return LEVELS[LEVELS.length - 1];
}

function buildRoster(rng: () => number, normal: (m: number, sd: number) => number): Person[] {
  const people: Person[] = [];
  const used = new Set<string>();
  let nameSeq = 0;

  for (const costCenter of COST_CENTERS) {
    for (let i = 0; i < costCenter.headcount; i++) {
      let name = '';
      do {
        // The `lap` term walks the surname list on every pass through the given
        // names, so the pairs stay unique well past the first thirty people.
        const first = FIRST_NAMES[nameSeq % FIRST_NAMES.length];
        const lap = Math.floor(nameSeq / FIRST_NAMES.length);
        const last = LAST_NAMES[(nameSeq * 7 + 3 + lap) % LAST_NAMES.length];
        name = `${first} ${last}`;
        nameSeq++;
      } while (used.has(name) && nameSeq < FIRST_NAMES.length * LAST_NAMES.length);
      used.add(name);

      const profile = pickLevel(rng);
      // A small share of staff sit on overhead targets regardless of level.
      const isOverhead = rng() < 0.08 && profile.level !== 'L5 Principal';
      const targetType: TargetType = isOverhead
        ? 'Overhead'
        : profile.level === 'L4 Manager' || profile.level === 'L5 Principal'
          ? 'Project Leadership'
          : 'Billable Staff';
      const utilTarget = isOverhead ? 25 : profile.utilTarget;

      people.push({
        name,
        costCenter,
        profile,
        targetType,
        utilTarget,
        billabilityTarget: Math.round(utilTarget * 0.92 * 2) / 2,
        effect: normal(0, 3.6) + (isOverhead ? -profile.baseUtil + 26 : 0),
        volatility: clamp(normal(3.4, 0.9), 1.6, 6.5),
        benchRisk:
          costCenter.benchRisk +
          (profile.level === 'L1 Analyst' || profile.level === 'L2 Engineer' ? 0.03 : 0),
        otRate: costCenter.otRate,
        vacationBudget: Math.round(clamp(normal(160, 25), 100, 220)),
        rampPeriods: rng() < 0.08 ? 2 + Math.floor(rng() * 2) : 0,
      });
    }
  }
  return people;
}

/** Repeats the twelve-month calendar, advancing the period index and the year. */
function buildCalendar(years: number): Period[] {
  const periods: Period[] = [];
  for (let year = 0; year < years; year++) {
    for (const base of CALENDAR) {
      const [y, m] = base.month.split('-').map(Number);
      periods.push({
        ...base,
        index: base.index + year * CALENDAR.length,
        month: `${y + year}-${String(m).padStart(2, '0')}`,
      });
    }
  }
  return periods;
}

export function generateDataset(options: GenerateOptions = {}): UtilizationRecord[] {
  const rng = makeRng(options.seed ?? 20260901);
  const normal = makeNormal(rng);
  const years = options.years ?? 1;
  if (!Number.isInteger(years) || years < 1) {
    throw new Error(`years must be a positive integer, got ${years}`);
  }
  const calendar = buildCalendar(years);
  const people = buildRoster(rng, normal);

  // Firm-wide demand factor: one AR(1) path shared by everybody, which is what
  // makes cost-center averages informative about an individual's next period.
  const marketFactor: number[] = [];
  let market = normal(0, 1.5);
  for (let t = 0; t < calendar.length; t++) {
    market = 0.7 * market + normal(0, 1.6);
    marketFactor.push(market);
  }

  const records: UtilizationRecord[] = [];

  for (const person of people) {
    let shock = normal(0, person.volatility);
    let benchHangover = 0;
    let cumulativeDirect = 0;
    let cumulativeAvail = 0;
    let cumulativeTotal = 0;

    for (const period of calendar) {
      const t = period.index - 1;
      // Seasonality and leave patterns repeat each fiscal year; the market
      // factor and the personal shock do not.
      const monthOfYear = t % CALENDAR.length;
      const standardHours = period.workdays * 8;

      // Year-to-date figures are fiscal-year-to-date, so they reset each year.
      if (monthOfYear === 0) {
        cumulativeDirect = 0;
        cumulativeAvail = 0;
        cumulativeTotal = 0;
      }

      // --- Leave and fringe -------------------------------------------------
      // New hires ramp through their opening periods: little leave, heavy training.
      const ramping = period.index <= person.rampPeriods;
      const vacationDraw =
        person.vacationBudget * VACATION_WEIGHT[monthOfYear] * clamp(normal(1, 0.55), 0, 2.6) *
        (ramping ? 0.25 : 1);
      const vacation = round1(clamp(Math.round(vacationDraw / 4) * 4, 0, standardHours * 0.55));
      const statAndDisc = round1(period.statHours + (rng() < 0.06 ? 8 : 0));
      const wellness = rng() < 0.28 ? (rng() < 0.5 ? 4 : 8) : 0;
      const unpaidRegular = rng() < 0.03 ? Math.round(clamp(normal(16, 8), 4, 40) / 4) * 4 : 0;
      const fringeTotal = round1(statAndDisc + vacation + wellness);

      const otHoursDraw =
        rng() < person.otRate ? Math.round(clamp(normal(14, 8), 2, 44) / 2) * 2 : 0;
      const total = round1(standardHours - unpaidRegular + otHoursDraw);
      const availTotal = round1(Math.max(8, total - fringeTotal));

      // --- Latent utilization ----------------------------------------------
      shock = 0.55 * shock + normal(0, person.volatility);
      if (benchHangover > 0) benchHangover *= 0.45;
      if (rng() < person.benchRisk) {
        benchHangover -= clamp(normal(13, 5), 5, 26);
      }
      const rampPenalty = ramping ? -30 + 9 * (period.index - 1) : 0;
      const latent =
        person.profile.baseUtil +
        person.effect +
        person.costCenter.utilOffset +
        SEASONALITY[monthOfYear] +
        1.1 * marketFactor[t] +
        shock +
        benchHangover +
        rampPenalty;
      const utilFraction = clamp(latent, 12, 98.5) / 100;

      // --- Hour split -------------------------------------------------------
      const indirectBudget = Math.max(0, availTotal * (1 - utilFraction));
      const wanted = {
        admin: availTotal * (0.025 + rng() * 0.025),
        busDev:
          availTotal *
          (person.profile.busDevShare + (person.targetType === 'Overhead' ? 0.03 : 0)) *
          clamp(normal(1, 0.35), 0.2, 2),
        mngmt: availTotal * person.profile.mngmtShare * clamp(normal(1, 0.25), 0.3, 1.8),
        opportunity:
          availTotal * (rng() < 0.45 ? rng() * 0.05 * (monthOfYear >= 10 ? 1.6 : 1) : 0),
        training:
          (ramping ? clamp(normal(46, 12), 16, 80) : clamp(normal(4, 3), 0, 14)) +
          (monthOfYear === 0 || monthOfYear === 6 ? 4 + rng() * 4 : 0),
        other: availTotal * rng() * 0.015,
      };

      let admin = wanted.admin;
      let busDev = wanted.busDev;
      let mngmt = wanted.mngmt;
      let opportunity = wanted.opportunity;
      let training = wanted.training;
      let other = wanted.other;
      const wantedSum = admin + busDev + mngmt + opportunity + training + other;
      let bench = 0;
      if (wantedSum > indirectBudget) {
        // Not enough indirect time to go around: everything is squeezed pro rata.
        const scale = indirectBudget / Math.max(wantedSum, 1e-9);
        admin *= scale;
        busDev *= scale;
        mngmt *= scale;
        opportunity *= scale;
        training *= scale;
        other *= scale;
      } else {
        bench = indirectBudget - wantedSum;
      }

      admin = round1(admin);
      busDev = round1(busDev);
      mngmt = round1(mngmt);
      opportunity = round1(opportunity);
      training = round1(training);
      other = round1(other);
      bench = round1(Math.max(0, bench));
      let indirectTotal = round1(admin + bench + busDev + mngmt + opportunity + training + other);
      if (indirectTotal > availTotal) {
        // Rounding can push the split a tenth past the available hours.
        bench = round1(Math.max(0, bench - (indirectTotal - availTotal)));
        other = round1(Math.max(0, other - Math.max(0, indirectTotal - availTotal - bench)));
        indirectTotal = round1(admin + bench + busDev + mngmt + opportunity + training + other);
      }

      const directTotal = round1(availTotal - indirectTotal);
      const otHours = round1(Math.min(otHoursDraw, directTotal));
      const unpaidOt =
        person.profile.level === 'L4 Manager' || person.profile.level === 'L5 Principal'
          ? rng() < 0.35
            ? Math.round(clamp(normal(8, 5), 2, 24) / 2) * 2
            : 0
          : rng() < 0.12
            ? Math.round(clamp(normal(5, 3), 2, 14) / 2) * 2
            : 0;

      cumulativeDirect += directTotal;
      cumulativeAvail += availTotal;
      cumulativeTotal += total;

      const utilPct = round2((directTotal / availTotal) * 100);
      const billabilityPct = round2((directTotal / total) * 100);

      records.push({
        sourceName: sourceNameFor(period.index, period.month),
        costCenter: person.costCenter.code,
        costCenterName: person.costCenter.name,
        personName: person.name,
        targetType: person.targetType,
        jobLevel: person.profile.level,

        utilPctYtd: round2((cumulativeDirect / cumulativeAvail) * 100),
        utilPct,
        utilPctTarget: round2(person.utilTarget),
        utilPctVariance: round2(utilPct - person.utilTarget),

        billabilityPctYtd: round2((cumulativeDirect / cumulativeTotal) * 100),
        billabilityPct,
        billabilityPctTarget: round2(person.billabilityTarget),
        billabilityPctVariance: round2(billabilityPct - person.billabilityTarget),

        total,
        availTotal,
        directTotal,
        indirectTotal,
        otHours,

        admin,
        bench,
        busDev,
        mngmt,
        opportunity,
        training,
        other,
        unpaidOt: round1(unpaidOt),

        fringeTotal,
        statAndDisc,
        vacation,
        wellness: round1(wellness),
        unpaidRegular: round1(unpaidRegular),
      });
    }
  }

  // Files land period by period, so sort the combined extract the same way.
  records.sort(
    (a, b) =>
      a.sourceName.localeCompare(b.sourceName) ||
      a.costCenter.localeCompare(b.costCenter) ||
      a.personName.localeCompare(b.personName),
  );
  return records;
}

export const PERIODS = CALENDAR;
