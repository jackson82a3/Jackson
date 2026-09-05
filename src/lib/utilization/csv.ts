import { COLUMNS, isTextColumn } from './types.ts';
import type { PeriodedRecord, UtilizationRecord } from './types.ts';

/**
 * `Source.Name` looks like `FY26_P03_Utilization_2025-12.csv`; the period index
 * and calendar month are read back out of it.
 */
export function sourceNameFor(periodIndex: number, month: string): string {
  const p = String(periodIndex).padStart(2, '0');
  return `FY26_P${p}_Utilization_${month}.csv`;
}

export function periodFromSourceName(sourceName: string): {
  periodIndex: number;
  periodMonth: string;
} {
  const match = /_P(\d{2})_Utilization_(\d{4}-\d{2})/.exec(sourceName);
  if (!match) throw new Error(`Cannot read a period out of Source.Name "${sourceName}"`);
  return { periodIndex: Number(match[1]), periodMonth: match[2] };
}

function escapeCell(value: string): string {
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

function formatCell(record: UtilizationRecord, key: keyof UtilizationRecord): string {
  const value = record[key];
  if (typeof value === 'string') return escapeCell(value);
  // Percentages carry two decimals, hours one - matching how the extract prints.
  const decimals = key.includes('Pct') ? 2 : 1;
  return value.toFixed(decimals);
}

export function toCsv(records: UtilizationRecord[]): string {
  const header = COLUMNS.map(([name]) => escapeCell(name)).join(',');
  const lines = records.map(r => COLUMNS.map(([, key]) => formatCell(r, key)).join(','));
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

export function parseCsv(text: string): PeriodedRecord[] {
  const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length === 0) return [];
  const header = parseCsvLine(lines[0]).map(h => h.trim());
  const expected = COLUMNS.map(([name]) => name);
  const missing = expected.filter(name => !header.includes(name));
  if (missing.length > 0) throw new Error(`Missing columns: ${missing.join(', ')}`);
  const indexOf = new Map(header.map((name, i) => [name, i] as const));

  return lines.slice(1).map((line, row) => {
    const cells = parseCsvLine(line);
    const record = {} as UtilizationRecord;
    for (const [name, key] of COLUMNS) {
      const raw = (cells[indexOf.get(name) as number] ?? '').trim();
      if (isTextColumn(key)) {
        (record[key] as string) = raw;
      } else {
        const num = Number(raw);
        if (!Number.isFinite(num)) {
          throw new Error(`Row ${row + 2}: "${name}" is not numeric ("${raw}")`);
        }
        (record[key] as number) = num;
      }
    }
    return { ...record, ...periodFromSourceName(record.sourceName) };
  });
}
