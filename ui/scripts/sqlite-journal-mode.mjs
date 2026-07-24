export const DEFAULT_SQLITE_JOURNAL_MODE = 'WAL';
export const VALID_SQLITE_JOURNAL_MODES = new Set([
  'DELETE',
  'TRUNCATE',
  'PERSIST',
  'MEMORY',
  'WAL',
  'OFF',
]);

export function resolveSqliteJournalMode(value, warn = console.warn) {
  const requested = typeof value === 'string' && value.trim() ? value.trim().toUpperCase() : DEFAULT_SQLITE_JOURNAL_MODE;
  if (VALID_SQLITE_JOURNAL_MODES.has(requested)) return requested;

  warn(
    `Invalid AI_TOOLKIT_DB_JOURNAL_MODE "${value}". Expected one of ${[...VALID_SQLITE_JOURNAL_MODES].join(', ')}; using ${DEFAULT_SQLITE_JOURNAL_MODE}.`,
  );
  return DEFAULT_SQLITE_JOURNAL_MODE;
}
