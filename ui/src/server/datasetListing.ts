import fs from 'fs';
import path from 'path';
import { DATASET_MEDIA_EXTENSIONS } from './datasetImages';
import { DATASET_CAPTION_SIDECAR_EXTENSIONS, DATASET_TEXT_CAPTION_EXTENSIONS, captionSidecarPath } from './captionFiles';
import { isDatasetRootCaptionEntry } from './datasetRootCaption';
import { pendingLayeredImagePathsAsync } from './layeredPublication';
import { listLayeredImages } from './layeredImages';

const MAX_DIRECTORIES = 256;
const MAX_CACHED_ENTRIES = 100_000;
const RECHECK_MS = 30_000;
export const DATASET_LISTING_CONCURRENCY = 16;
type Item = { filename: string; caption: string | null };
type DirectoryIndex = {
  signature: string;
  checkedAt: number;
  items: Item[];
  directories: string[];
  entryCount: number;
};
const directories = new Map<string, DirectoryIndex>();
const pendingIndexes = new Map<string, Promise<DirectoryIndex>>();
let cachedEntries = 0;
const snapshots = new Map<string, { indexes: DirectoryIndex[]; items: Item[] }>();
const mediaExtensions = new Set(DATASET_MEDIA_EXTENSIONS);
const textExtensions = new Set(DATASET_TEXT_CAPTION_EXTENSIONS);
const fileKey = (value: string) => process.platform === 'win32' ? value.toLowerCase() : value;

export type DatasetListingEntry = {
  img_path: string;
  added_at: string | null;
  captioned_at: string | null;
  size_bytes: number;
  layer_count?: number;
};

export async function mapDatasetItemsBounded<T, R>(items: readonly T[], mapper: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(DATASET_LISTING_CONCURRENCY, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await mapper(items[index]);
    }
  }));
  return results;
}

export function datasetListingCacheStats() {
  const admitted = new Set(directories.values());
  return {
    directories: directories.size,
    entries: cachedEntries,
    snapshots: snapshots.size,
    unadmittedSnapshotIndexes: [...snapshots.values()].reduce((count, snapshot) =>
      count + snapshot.indexes.filter(index => !admitted.has(index)).length, 0),
  };
}

export function clearDatasetListingCache() {
  directories.clear();
  snapshots.clear();
  cachedEntries = 0;
}

async function readDirectoryIndex(directory: string, root: string): Promise<DirectoryIndex> {
  // Never reuse an index through a replaced directory/symlink. File metadata is
  // deliberately not cached: external edits need not change directory mtime.
  const stat = await fs.promises.lstat(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('Dataset directory cannot be a symbolic link');
  const signature = `${stat.dev}:${stat.ino}:${stat.mtimeMs}:${stat.ctimeMs}:${stat.size}`;
  const key = `${root}\0${directory}`;
  const previous = directories.get(key);
  if (previous && previous.signature === signature && Date.now() - previous.checkedAt < RECHECK_MS) {
    directories.delete(key);
    directories.set(key, previous);
    return previous;
  }
  const entries = await fs.promises.readdir(directory, { withFileTypes: true });
  const files = new Map(entries.filter(entry => entry.isFile()).map(entry => [fileKey(entry.name), entry.name]));
  const mediaStems = new Set<string>();
  const items: Item[] = [];
  const texts: string[] = [];
  const children: string[] = [];
  for (const entry of entries) {
    if (entry.name.startsWith('.') || entry.name === '_controls' && entry.isDirectory()) continue;
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) children.push(filename);
    if (!entry.isFile() || isDatasetRootCaptionEntry(root, directory, entry.name)) continue;
    const extension = path.extname(entry.name).toLowerCase();
    if (mediaExtensions.has(extension)) {
      mediaStems.add(filename.slice(0, -extension.length).toLowerCase());
      let caption: string | null = null;
      for (const extension of DATASET_CAPTION_SIDECAR_EXTENSIONS) {
        const candidate = files.get(fileKey(path.basename(captionSidecarPath(filename, extension))));
        if (candidate) { caption = path.join(directory, candidate); break; }
      }
      items.push({ filename, caption });
    } else if (textExtensions.has(extension)) texts.push(filename);
  }
  for (const filename of texts) {
    if (!mediaStems.has(filename.slice(0, -path.extname(filename).length).toLowerCase())) items.push({ filename, caption: filename });
  }
  const current = { signature, checkedAt: Date.now(), items, directories: children, entryCount: entries.length };
  snapshots.delete(root);
  const replaced = directories.get(key);
  if (replaced) { cachedEntries -= replaced.entryCount; directories.delete(key); }
  if (entries.length <= MAX_CACHED_ENTRIES) {
    directories.set(key, current);
    cachedEntries += entries.length;
  }
  while (directories.size > MAX_DIRECTORIES || cachedEntries > MAX_CACHED_ENTRIES) {
    const oldest = directories.keys().next().value;
    if (oldest === undefined) break;
    cachedEntries -= directories.get(oldest)!.entryCount;
    directories.delete(oldest);
    // Snapshots retain indexes; evict them along with the directory budget.
    snapshots.clear();
  }
  return current;
}

async function directoryIndex(directory: string, root: string): Promise<DirectoryIndex> {
  const key = `${root}\0${directory}`;
  const pending = pendingIndexes.get(key);
  if (pending) return pending;
  const promise = readDirectoryIndex(directory, root);
  pendingIndexes.set(key, promise);
  try { return await promise; }
  finally { if (pendingIndexes.get(key) === promise) pendingIndexes.delete(key); }
}

async function indexedItems(root: string): Promise<Item[]> {
  const indexes: DirectoryIndex[] = [];
  const pending = [root];
  // Directory traversal is bounded as well; no recursive Promise.all fan-out.
  while (pending.length) {
    const index = await directoryIndex(pending.pop()!, root);
    indexes.push(index);
    for (const directory of index.directories) pending.push(directory);
  }
  const previous = snapshots.get(root);
  if (previous && previous.indexes.length === indexes.length && indexes.every((index, i) => index === previous.indexes[i])) return previous.items;
  const items = indexes.flatMap(index => index.items).sort((a, b) => a.filename.localeCompare(b.filename));
  snapshots.delete(root);
  const admitted = new Set(directories.values());
  if (items.length <= MAX_CACHED_ENTRIES && indexes.every(index => admitted.has(index))) {
    // At most one dataset snapshot per directory budget; old datasets cannot pin
    // unlimited sorted arrays independently of directory eviction.
    snapshots.set(root, { indexes, items });
    while (snapshots.size > 16) snapshots.delete(snapshots.keys().next().value!);
  }
  return items;
}

function iso(date: Date) {
  return Number.isFinite(date.getTime()) && date.getTime() > 0 ? date.toISOString() : null;
}

async function regularFile(filename: string) {
  try {
    const stat = await fs.promises.lstat(filename);
    return stat.isFile() && !stat.isSymbolicLink() ? stat : null;
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && (error.code === 'ENOENT' || error.code === 'ENOTDIR')) return null;
    throw error;
  }
}

/** Root must already be canonically resolved within the configured dataset scope. */
export async function listDatasetImages(root: string, options: {
  compact?: boolean;
  offset?: number;
  limit?: number;
  // Validated configured spelling for operational URLs (e.g. Windows 8.3 paths).
  // This is never used for filesystem reads or index/security decisions.
  responseRoot?: string;
} = {}) {
  const responseRoot = options.responseRoot ?? root;
  const indexed = await indexedItems(root);
  // Never cache publication/security decisions or layered manifest validation.
  const pending = await pendingLayeredImagePathsAsync(root);
  const items = indexed.filter(item => !pending.has(path.resolve(item.filename)));
  const layered = new Map((await listLayeredImages(root)).map(group => [path.resolve(root, group.composite), group.layers.length]));
  const offset = options.offset ?? 0;
  const selected = options.limit === undefined ? items : items.slice(offset, offset + options.limit);
  const entries = await mapDatasetItemsBounded(selected, async item => {
    const stat = await regularFile(item.filename);
    if (!stat) return null; // Concurrent removals are harmless, never a stale cached entry.
    const caption = item.caption === item.filename ? stat : item.caption ? await regularFile(item.caption).catch(() => null) : null;
    const count = layered.get(path.resolve(item.filename));
    return {
      img_path: options.compact ? path.relative(root, item.filename) : path.join(responseRoot, path.relative(root, item.filename)),
      added_at: iso(stat.birthtime) || iso(stat.ctime) || iso(stat.mtime),
      captioned_at: caption ? iso(caption.mtime) : null,
      size_bytes: stat.size,
      ...(count !== undefined ? { layer_count: count } : {}),
    } satisfies DatasetListingEntry;
  });
  const images = entries.filter((entry): entry is NonNullable<typeof entry> => entry !== null);
  return {
    ...(options.compact ? { root: responseRoot + path.sep } : {}), images,
    ...(options.limit === undefined ? {} : { total: items.length, next_offset: offset + selected.length < items.length ? offset + selected.length : null }),
  };
}
