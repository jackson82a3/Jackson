import { MIN_HISTORY, groupByPerson } from './features.ts';
import { FIRST_VALIDATION_PERIOD } from './forecast.ts';
import { verifyRecord } from './types.ts';
import type { PeriodedRecord } from './types.ts';

/**
 * Checks a parsed extract before anything is fitted to it.
 *
 * The synthetic generator produces data that is correct by construction. A real
 * extract does not: periods arrive late, a person appears twice because they
 * transferred mid-month, a rounding change breaks an identity, someone's
 * available hours come through as zero. Every one of those produces either a
 * crash deep inside the fit or - worse - a number that looks fine and is wrong.
 *
 * So this runs first and says plainly what is wrong with the data, separating
 * problems that make training unsound (`error`) from ones that are normal in
 * real feeds but change how the output should be read (`warning`).
 */

export type Severity = 'error' | 'warning';

export interface ValidationIssue {
  severity: Severity;
  /** Stable identifier, so callers can suppress a known-acceptable issue. */
  code: string;
  message: string;
  /** How many rows or people are affected, when that is meaningful. */
  count?: number;
  /** A few concrete instances, to make the problem findable in the source file. */
  examples?: string[];
}

export interface ValidationReport {
  issues: ValidationIssue[];
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  /** True when nothing blocks training. Warnings do not block. */
  ok: boolean;
  summary: { rows: number; people: number; periods: number };
}

function issue(
  severity: Severity,
  code: string,
  message: string,
  count?: number,
  examples?: string[],
): ValidationIssue {
  return { severity, code, message, count, examples: examples?.slice(0, 5) };
}

export function validateDataset(records: PeriodedRecord[]): ValidationReport {
  const issues: ValidationIssue[] = [];

  if (records.length === 0) {
    const empty = issue('error', 'empty_dataset', 'The extract has no rows.');
    return {
      issues: [empty],
      errors: [empty],
      warnings: [],
      ok: false,
      summary: { rows: 0, people: 0, periods: 0 },
    };
  }

  const periods = [...new Set(records.map(r => r.periodIndex))].sort((a, b) => a - b);
  const panel = groupByPerson(records);

  // --- Duplicates ----------------------------------------------------------
  const seen = new Map<string, number>();
  for (const record of records) {
    const key = `${record.costCenter}|${record.personName}|${record.periodIndex}`;
    seen.set(key, (seen.get(key) ?? 0) + 1);
  }
  const duplicates = [...seen.entries()].filter(([, n]) => n > 1);
  if (duplicates.length > 0) {
    issues.push(
      issue(
        'error',
        'duplicate_person_period',
        'The same person appears more than once in the same period. Utilization would be ' +
          'double counted, and the panel would silently disagree with headcount.',
        duplicates.length,
        duplicates.map(([key, n]) => `${key} x${n}`),
      ),
    );
  }

  // --- Accounting identities ----------------------------------------------
  const broken: string[] = [];
  for (const record of records) {
    const problems = verifyRecord(record);
    if (problems.length > 0) broken.push(`${record.sourceName} / ${record.personName}: ${problems[0]}`);
  }
  if (broken.length > 0) {
    issues.push(
      issue(
        'error',
        'identity_violation',
        'Rows do not satisfy the accounting identities (direct + indirect + fringe = total, ' +
          'Util % = direct / available, and so on). Fix the extract rather than the tolerance.',
        broken.length,
        broken,
      ),
    );
  }

  // --- Impossible values ---------------------------------------------------
  const nonPositiveAvail = records.filter(r => r.availTotal <= 0);
  if (nonPositiveAvail.length > 0) {
    issues.push(
      issue(
        'error',
        'non_positive_available_hours',
        'Available hours are zero or negative, so Util % is undefined for these rows. ' +
          'A person on leave for a whole period usually belongs out of the denominator, ' +
          'not in it with a zero.',
        nonPositiveAvail.length,
        nonPositiveAvail.map(r => `${r.personName} in ${r.sourceName}`),
      ),
    );
  }
  const impossibleUtil = records.filter(r => r.utilPct < -0.01 || r.utilPct > 150);
  if (impossibleUtil.length > 0) {
    issues.push(
      issue(
        'error',
        'util_out_of_range',
        'Util % is negative or above 150, which no correct extract produces.',
        impossibleUtil.length,
        impossibleUtil.map(r => `${r.personName} in ${r.sourceName}: ${r.utilPct}`),
      ),
    );
  }

  // --- Enough history to train --------------------------------------------
  if (periods.length < FIRST_VALIDATION_PERIOD + 1) {
    issues.push(
      issue(
        'error',
        'insufficient_periods',
        `Training needs at least ${FIRST_VALIDATION_PERIOD + 1} periods so the rolling origin ` +
          `has folds to score; this extract has ${periods.length}. Forecasting from an existing ` +
          'model does not have this requirement.',
        periods.length,
      ),
    );
  }

  // --- Contiguity ----------------------------------------------------------
  const gaps: string[] = [];
  for (let i = 1; i < periods.length; i++) {
    if (periods[i] !== periods[i - 1] + 1) gaps.push(`P${periods[i - 1]} -> P${periods[i]}`);
  }
  if (gaps.length > 0) {
    issues.push(
      issue(
        'error',
        'missing_periods',
        'Periods are not contiguous. Lagged features would silently compare across the gap ' +
          'as though the periods were adjacent.',
        gaps.length,
        gaps,
      ),
    );
  }

  // --- Roster churn --------------------------------------------------------
  const lastPeriod = periods[periods.length - 1];
  const leavers: string[] = [];
  const shortHistory: string[] = [];
  const personGaps: string[] = [];
  for (const [key, rows] of panel) {
    const personPeriods = rows.map(r => r.periodIndex);
    if (personPeriods[personPeriods.length - 1] !== lastPeriod) {
      leavers.push(`${key} (last seen P${personPeriods[personPeriods.length - 1]})`);
    } else if (rows.length < MIN_HISTORY) {
      shortHistory.push(`${key} (${rows.length} periods)`);
    }
    for (let i = 1; i < personPeriods.length; i++) {
      if (personPeriods[i] !== personPeriods[i - 1] + 1) {
        personGaps.push(`${key} (P${personPeriods[i - 1]} -> P${personPeriods[i]})`);
        break;
      }
    }
  }
  if (leavers.length > 0) {
    issues.push(
      issue(
        'warning',
        'absent_from_last_period',
        'People are missing from the final period. They are reported as exclusions rather ' +
          'than forecast; check they are genuine leavers and not a late-arriving extract.',
        leavers.length,
        leavers,
      ),
    );
  }
  if (shortHistory.length > 0) {
    issues.push(
      issue(
        'warning',
        'short_history',
        `People have fewer than ${MIN_HISTORY} periods of history, so the model cannot build a ` +
          'feature row for them. They fall back to their last observed Util %, flagged as ' +
          '`short_history` in the output.',
        shortHistory.length,
        shortHistory,
      ),
    );
  }
  if (personGaps.length > 0) {
    issues.push(
      issue(
        'warning',
        'person_period_gap',
        'People have gaps in their own history (a transfer, unpaid leave, or a missing row). ' +
          'Samples are not built across a gap, so these people contribute fewer training rows.',
        personGaps.length,
        personGaps,
      ),
    );
  }

  // --- Targets -------------------------------------------------------------
  const movingTargets: string[] = [];
  for (const [key, rows] of panel) {
    const targets = new Set(rows.map(r => r.utilPctTarget));
    if (targets.size > 1) movingTargets.push(`${key} (${[...targets].join(', ')})`);
  }
  if (movingTargets.length > 0) {
    issues.push(
      issue(
        'warning',
        'target_changed_mid_year',
        'Util % targets change during the year for some people. That is legitimate after a ' +
          'promotion, but `gap_to_target` is a feature, so a retrospective target change ' +
          'rewrites history the model learned from.',
        movingTargets.length,
        movingTargets,
      ),
    );
  }

  const errors = issues.filter(i => i.severity === 'error');
  const warnings = issues.filter(i => i.severity === 'warning');
  return {
    issues,
    errors,
    warnings,
    ok: errors.length === 0,
    summary: { rows: records.length, people: panel.size, periods: periods.length },
  };
}

/** Renders a report for a terminal, one block per issue. */
export function formatValidationReport(report: ValidationReport): string {
  const lines: string[] = [];
  const { rows, people, periods } = report.summary;
  lines.push(`Validation: ${rows} rows | ${people} people | ${periods} periods`);
  if (report.issues.length === 0) {
    lines.push('  No issues found.');
    return lines.join('\n');
  }
  for (const i of report.issues) {
    const label = i.severity === 'error' ? 'ERROR' : 'warn ';
    lines.push(`  ${label} [${i.code}]${i.count !== undefined ? ` x${i.count}` : ''}`);
    lines.push(`        ${i.message}`);
    for (const example of i.examples ?? []) lines.push(`          - ${example}`);
  }
  lines.push(
    report.ok
      ? `  ${report.warnings.length} warning(s), nothing blocking.`
      : `  ${report.errors.length} error(s) block training.`,
  );
  return lines.join('\n');
}
