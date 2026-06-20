import { parseIdeogramCaption } from './ideogramCaption';
import { isFailedCaption } from './captionQuality';

export type DatasetNavigatorFilter = 'all' | 'needs-caption' | 'has-boxes';
export type DatasetNavigatorStatus = 'unknown' | 'missing' | 'has-boxes' | 'json' | 'plain';
export type DatasetNavigatorSortMode =
  | 'original'
  | 'name'
  | 'extension'
  | 'media-type'
  | 'size'
  | 'added'
  | 'captioned'
  | 'caption-status'
  | 'caption-length';
export type DatasetNavigatorSortDirection = 'asc' | 'desc';

export type DatasetNavigatorEntry = {
  index: number;
  name: string;
  status: DatasetNavigatorStatus;
  extension?: string | null;
  mediaType?: string | null;
  sizeBytes?: number | null;
  addedAt?: string | null;
  captionedAt?: string | null;
  captionLength?: number | null;
};

const naturalCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

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
  if (isFailedCaption(caption)) return 'missing';
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
  if (filter === 'needs-caption') return status === 'missing' || status === 'unknown';
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

function dateMs(value: string | null | undefined) {
  if (!value) return null;
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : null;
}

function cleanString(value: string | null | undefined) {
  const trimmed = typeof value === 'string' ? value.trim() : '';
  return trimmed || null;
}

function cleanNumber(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function compareNullable<T>(
  left: T | null,
  right: T | null,
  direction: DatasetNavigatorSortDirection,
  compare: (left: T, right: T) => number,
) {
  if (left == null && right == null) return 0;
  if (left == null) return 1;
  if (right == null) return -1;
  const result = compare(left, right);
  return direction === 'asc' ? result : -result;
}

function statusRank(status: DatasetNavigatorStatus) {
  if (status === 'unknown') return null;
  if (status === 'missing') return 0;
  if (status === 'plain') return 1;
  if (status === 'json') return 2;
  return 3;
}

export function sortNavigatorEntries(
  entries: DatasetNavigatorEntry[],
  mode: DatasetNavigatorSortMode,
  direction: DatasetNavigatorSortDirection,
) {
  if (mode === 'original') return entries;

  return [...entries].sort((left, right) => {
    let result = 0;

    if (mode === 'name') {
      result = compareNullable(cleanString(left.name), cleanString(right.name), direction, (a, b) =>
        naturalCollator.compare(a, b),
      );
    } else if (mode === 'extension') {
      result = compareNullable(cleanString(left.extension), cleanString(right.extension), direction, (a, b) =>
        naturalCollator.compare(a, b),
      );
    } else if (mode === 'media-type') {
      result = compareNullable(cleanString(left.mediaType), cleanString(right.mediaType), direction, (a, b) =>
        naturalCollator.compare(a, b),
      );
    } else if (mode === 'size') {
      result = compareNullable(cleanNumber(left.sizeBytes), cleanNumber(right.sizeBytes), direction, (a, b) => a - b);
    } else if (mode === 'added' || mode === 'captioned') {
      const field = mode === 'added' ? 'addedAt' : 'captionedAt';
      result = compareNullable(dateMs(left[field]), dateMs(right[field]), direction, (a, b) => a - b);
    } else if (mode === 'caption-status') {
      result = compareNullable(statusRank(left.status), statusRank(right.status), direction, (a, b) => a - b);
    } else if (mode === 'caption-length') {
      result = compareNullable(
        cleanNumber(left.captionLength),
        cleanNumber(right.captionLength),
        direction,
        (a, b) => a - b,
      );
    }

    return result || left.index - right.index;
  });
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
      if (entry.status === 'missing' || entry.status === 'unknown') counts.missing += 1;
      if (entry.status === 'has-boxes') counts.hasBoxes += 1;
      if (entry.status === 'unknown') counts.unknown += 1;
      return counts;
    },
    { total: 0, missing: 0, hasBoxes: 0, unknown: 0 },
  );
}
