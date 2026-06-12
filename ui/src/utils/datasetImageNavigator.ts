import { parseIdeogramCaption } from './ideogramCaption';

export type DatasetNavigatorFilter = 'all' | 'needs-caption' | 'has-boxes';
export type DatasetNavigatorStatus = 'unknown' | 'missing' | 'has-boxes' | 'json' | 'plain';

export type DatasetNavigatorEntry = {
  index: number;
  name: string;
  status: DatasetNavigatorStatus;
};

export function parseNavigatorJump(value: string, itemCount: number) {
  if (itemCount <= 0) return null;
  const match = value.trim().match(/\d+/);
  if (!match) return null;
  const parsed = Number.parseInt(match[0], 10);
  if (!Number.isFinite(parsed) || parsed < 1) return null;
  return Math.max(0, Math.min(itemCount - 1, parsed - 1));
}

export function navigatorStatusForCaption(caption: string, loaded: boolean): DatasetNavigatorStatus {
  if (!loaded) return 'unknown';
  if (!caption.trim()) return 'missing';
  const parsed = parseIdeogramCaption(caption);
  if (parsed.kind === 'ideogram') return parsed.boxes.length > 0 ? 'has-boxes' : 'json';
  if (parsed.kind === 'json') return 'json';
  return 'plain';
}

export function matchesNavigatorSearch(entry: Pick<DatasetNavigatorEntry, 'index' | 'name'>, query: string) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return true;
  const oneBasedIndex = String(entry.index + 1);
  return oneBasedIndex === normalized || oneBasedIndex.startsWith(normalized) || entry.name.toLowerCase().includes(normalized);
}

export function matchesNavigatorFilter(status: DatasetNavigatorStatus, filter: DatasetNavigatorFilter) {
  if (filter === 'all') return true;
  if (filter === 'needs-caption') return status === 'missing';
  if (filter === 'has-boxes') return status === 'has-boxes';
  return true;
}

export function filterNavigatorEntries(
  entries: DatasetNavigatorEntry[],
  query: string,
  filter: DatasetNavigatorFilter,
) {
  return entries.filter(entry => matchesNavigatorSearch(entry, query) && matchesNavigatorFilter(entry.status, filter));
}

export function groupNavigatorRows(indexes: number[], columns: number) {
  const safeColumns = Math.max(1, Math.floor(columns) || 1);
  const rows: number[][] = [];
  for (let index = 0; index < indexes.length; index += safeColumns) {
    rows.push(indexes.slice(index, index + safeColumns));
  }
  return rows;
}

export function navigatorColumnCount(width: number, tileWidth: number, gap = 8) {
  if (width <= 0 || tileWidth <= 0) return 1;
  return Math.max(1, Math.floor((width + gap) / (tileWidth + gap)));
}

export function navigatorStatusCounts(entries: DatasetNavigatorEntry[]) {
  return entries.reduce(
    (counts, entry) => {
      counts.total += 1;
      if (entry.status === 'missing') counts.missing += 1;
      if (entry.status === 'has-boxes') counts.hasBoxes += 1;
      if (entry.status === 'unknown') counts.unknown += 1;
      return counts;
    },
    { total: 0, missing: 0, hasBoxes: 0, unknown: 0 },
  );
}
