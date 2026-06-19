import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { randomUUID } from 'crypto';
import { db } from './db';
import { resolveDatasetScope } from './datasetScope';
import { DATASET_MEDIA_EXTENSIONS } from './datasetImages';
import { DATASET_TEXT_CAPTION_EXTENSIONS, captionSidecarPath } from './captionFiles';
import { isEncryptedDatasetFolder, resolveDatasetFolder } from './encryptedDatasets';
import { readDatasetRootCaption } from './datasetRootCaption';
import { getOpenRouterApiKey } from './settings';
import { generateSingleImageRecaption, type RecaptionProvider, type RecaptionOutputFormat } from './datasetSingleRecaption';
import { TOOLKIT_ROOT } from '../paths';

export const DATASET_WATCHERS_SETTING_KEY = 'DATASET_WATCHERS_V1';
export const DATASET_WATCHER_STATUS_SETTING_KEY = 'DATASET_WATCHER_STATUS_V1';
export const DATASET_WATCHER_IMPORT_MANIFEST = '.aitk_dataset_watch_imports.json';
export const DATASET_WATCHER_POLL_INTERVAL_MS = 10_000;
export const DATASET_WATCHER_STABLE_MS = 2_000;

const DATASET_WATCHER_MANIFEST_VERSION = 1;
const DATASET_WATCHER_STATUS_VERSION = 1;
const DATASET_WATCHER_LOCK_STALE_MS = 30 * 60 * 1000;
const DATASET_WATCHER_LOCK_HEARTBEAT_MS = 60_000;
const DATASET_WATCHER_SETTINGS_LOCK_STALE_MS = 60_000;
const DATASET_WATCHER_SETTINGS_LOCK_WAIT_MS = 5_000;
const MAX_WATCHERS = 100;
const MAX_WATCHER_PATH_LENGTH = 2048;
const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.jxl', '.gif', '.bmp']);
const MEDIA_EXTENSIONS = new Set(DATASET_MEDIA_EXTENSIONS);
const CAPTION_EXTENSIONS = new Set(DATASET_TEXT_CAPTION_EXTENSIONS);
const WINDOWS_DRIVE_PATH = /^([a-zA-Z]):[\\/](.*)$/;
const WSL_MOUNT_PATH = /^\/mnt\/([a-zA-Z])(?:\/(.*))?$/;
const WSL_UNC_PATH = /^\\\\(?:wsl\$|wsl\.localhost)\\([^\\]+)\\(.*)$/i;

export type DatasetWatcherAutoCaptionConfig = {
  enabled: boolean;
  provider: RecaptionProvider;
  model: string;
  prompt?: string;
  systemPrompt?: string;
  outputFormat?: RecaptionOutputFormat;
  maxNewTokens?: number | null;
  remoteWorkerId?: string;
};

export type DatasetWatcherConfig = {
  id: string;
  datasetName: string;
  projectID: string | null;
  enabled: boolean;
  sourcePath: string;
  includeSubfolders: boolean;
  preserveRelativePaths: boolean;
  autoCaption: DatasetWatcherAutoCaptionConfig | null;
  createdAt: string;
  updatedAt: string;
};

export type DatasetWatcherStatus = {
  state: 'idle' | 'disabled' | 'scanning' | 'importing' | 'captioning' | 'error';
  lastScanAt: string | null;
  lastImportedAt: string | null;
  lastImportedCount: number;
  lastCaptionedCount: number;
  lastError: string | null;
  warnings: string[];
};

export type DatasetWatcherRunResult = DatasetWatcherStatus & {
  watcherID: string;
  importedPaths: string[];
  captionedPaths: string[];
};

type DatasetWatchManifestEntry = {
  sourceKey: string;
  sourcePath: string;
  destinationRelativePath: string;
  importedAt: string;
  size: number;
  mtimeMs: number;
};

type DatasetWatchManifest = {
  version: 1;
  imports: Record<string, DatasetWatchManifestEntry>;
};

type PendingSourceObservation = {
  size: number;
  mtimeMs: number;
  firstSeenAt: number;
  lastSeenAt: number;
};

type DatasetWatcherRunOptions = {
  now?: number;
  stableMs?: number;
};

type SourceCandidate = {
  absolutePath: string;
  relativePath: string;
  size: number;
  mtimeMs: number;
};

type WatcherStatusStore = {
  version: 1;
  statuses: Record<string, DatasetWatcherStatus>;
};

declare global {
  // eslint-disable-next-line no-var
  var __aitkDatasetWatcherPending: Map<string, Map<string, PendingSourceObservation>> | undefined;
  // eslint-disable-next-line no-var
  var __aitkDatasetWatcherLastPollAt: number | undefined;
}

function pendingByWatcher() {
  if (!globalThis.__aitkDatasetWatcherPending) {
    globalThis.__aitkDatasetWatcherPending = new Map();
  }
  return globalThis.__aitkDatasetWatcherPending;
}

function nowIso(now = Date.now()) {
  return new Date(now).toISOString();
}

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function defaultWatcherStatus(state: DatasetWatcherStatus['state'] = 'idle'): DatasetWatcherStatus {
  return {
    state,
    lastScanAt: null,
    lastImportedAt: null,
    lastImportedCount: 0,
    lastCaptionedCount: 0,
    lastError: null,
    warnings: [],
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asString(value: unknown) {
  return typeof value === 'string' ? value.trim() : '';
}

function asBoolean(value: unknown, fallback: boolean) {
  return typeof value === 'boolean' ? value : fallback;
}

function normalizeProjectID(value: unknown) {
  const projectID = asString(value);
  return projectID || null;
}

function normalizeRecaptionProvider(value: unknown): RecaptionProvider {
  return value === 'ollama' || value === 'remote_ollama' ? value : 'openrouter';
}

function normalizeRecaptionOutputFormat(value: unknown): RecaptionOutputFormat {
  return value === 'ideogram_json' || value === 'json' ? 'ideogram_json' : 'text';
}

function normalizeMaxNewTokens(value: unknown) {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return Math.round(parsed);
}

function normalizeAutoCaption(value: unknown): DatasetWatcherAutoCaptionConfig | null {
  if (!isRecord(value) || value.enabled !== true) return null;
  const provider = normalizeRecaptionProvider(value.provider);
  return {
    enabled: true,
    provider,
    model: asString(value.model),
    prompt: asString(value.prompt),
    systemPrompt: asString(value.systemPrompt),
    outputFormat: normalizeRecaptionOutputFormat(value.outputFormat),
    maxNewTokens: normalizeMaxNewTokens(value.maxNewTokens),
    remoteWorkerId: asString(value.remoteWorkerId),
  };
}

function normalizeWatcher(raw: unknown): DatasetWatcherConfig | null {
  if (!isRecord(raw)) return null;
  const id = asString(raw.id);
  const datasetName = asString(raw.datasetName);
  const sourcePath = asString(raw.sourcePath);
  const createdAt = asString(raw.createdAt) || nowIso();
  if (!id || !datasetName || !sourcePath) return null;

  return {
    id,
    datasetName,
    projectID: normalizeProjectID(raw.projectID),
    enabled: raw.enabled !== false,
    sourcePath,
    includeSubfolders: asBoolean(raw.includeSubfolders, true),
    preserveRelativePaths: asBoolean(raw.preserveRelativePaths, true),
    autoCaption: normalizeAutoCaption(raw.autoCaption),
    createdAt,
    updatedAt: asString(raw.updatedAt) || createdAt,
  };
}

function normalizeIncomingWatcher(raw: unknown, existing?: DatasetWatcherConfig | null): DatasetWatcherConfig {
  if (!isRecord(raw)) throw new Error('Watcher config is required');
  const now = nowIso();
  const id = asString(raw.id) || existing?.id || randomUUID();
  const datasetName = asString(raw.datasetName ?? existing?.datasetName);
  const sourcePath = asString(raw.sourcePath ?? existing?.sourcePath);
  if (!datasetName) throw new Error('Dataset name is required');
  if (!sourcePath) throw new Error('Watch folder path is required');
  if (datasetName.includes('/') || datasetName.includes('\\') || datasetName.startsWith('.')) {
    throw new Error('Invalid dataset name');
  }
  if (sourcePath.length > MAX_WATCHER_PATH_LENGTH) {
    throw new Error('Watch folder path is too long');
  }

  return {
    id,
    datasetName,
    projectID: normalizeProjectID(raw.projectID ?? existing?.projectID),
    enabled: asBoolean(raw.enabled, existing?.enabled ?? true),
    sourcePath,
    includeSubfolders: asBoolean(raw.includeSubfolders, existing?.includeSubfolders ?? true),
    preserveRelativePaths: asBoolean(raw.preserveRelativePaths, existing?.preserveRelativePaths ?? true),
    autoCaption: normalizeAutoCaption(raw.autoCaption ?? existing?.autoCaption),
    createdAt: existing?.createdAt || now,
    updatedAt: now,
  };
}

function normalizeStatus(raw: unknown): DatasetWatcherStatus | null {
  if (!isRecord(raw)) return null;
  const state = ['idle', 'disabled', 'scanning', 'importing', 'captioning', 'error'].includes(String(raw.state))
    ? (raw.state as DatasetWatcherStatus['state'])
    : 'idle';
  return {
    state,
    lastScanAt: typeof raw.lastScanAt === 'string' ? raw.lastScanAt : null,
    lastImportedAt: typeof raw.lastImportedAt === 'string' ? raw.lastImportedAt : null,
    lastImportedCount: Number.isFinite(Number(raw.lastImportedCount)) ? Number(raw.lastImportedCount) : 0,
    lastCaptionedCount: Number.isFinite(Number(raw.lastCaptionedCount)) ? Number(raw.lastCaptionedCount) : 0,
    lastError: typeof raw.lastError === 'string' && raw.lastError.trim() ? raw.lastError : null,
    warnings: Array.isArray(raw.warnings) ? raw.warnings.filter((item): item is string => typeof item === 'string') : [],
  };
}

async function readWatcherRows() {
  const row = await db.settings.get(DATASET_WATCHERS_SETTING_KEY);
  if (!row?.value) return [];
  try {
    const parsed = JSON.parse(row.value);
    const values = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.watchers) ? parsed.watchers : [];
    return values.map(normalizeWatcher).filter(Boolean) as DatasetWatcherConfig[];
  } catch {
    return [];
  }
}

async function writeWatcherRows(watchers: DatasetWatcherConfig[]) {
  const normalized = watchers.map(normalizeWatcher).filter(Boolean) as DatasetWatcherConfig[];
  await db.settings.upsert(DATASET_WATCHERS_SETTING_KEY, JSON.stringify({ version: 1, watchers: normalized }));
  return normalized;
}

async function readStatusStore(): Promise<WatcherStatusStore> {
  const row = await db.settings.get(DATASET_WATCHER_STATUS_SETTING_KEY);
  if (!row?.value) return { version: DATASET_WATCHER_STATUS_VERSION, statuses: {} };
  try {
    const parsed = JSON.parse(row.value);
    const rawStatuses = isRecord(parsed?.statuses) ? parsed.statuses : {};
    const statuses: Record<string, DatasetWatcherStatus> = {};
    for (const [id, rawStatus] of Object.entries(rawStatuses)) {
      const status = normalizeStatus(rawStatus);
      if (status) statuses[id] = status;
    }
    return { version: DATASET_WATCHER_STATUS_VERSION, statuses };
  } catch {
    return { version: DATASET_WATCHER_STATUS_VERSION, statuses: {} };
  }
}

async function writeStatus(id: string, status: DatasetWatcherStatus) {
  return withSettingsLock('status', async () => {
    const store = await readStatusStore();
    store.statuses[id] = status;
    await db.settings.upsert(DATASET_WATCHER_STATUS_SETTING_KEY, JSON.stringify(store));
    return status;
  });
}

export function isDatasetWatchersSettingKey(key: string) {
  return key === DATASET_WATCHERS_SETTING_KEY || key === DATASET_WATCHER_STATUS_SETTING_KEY;
}

export async function listDatasetWatchers(filter: { datasetName?: string; projectID?: string | null } = {}) {
  const watchers = await readWatcherRows();
  return watchers.filter(watcher => {
    if (filter.datasetName && watcher.datasetName !== filter.datasetName) return false;
    if ('projectID' in filter && (watcher.projectID || null) !== (filter.projectID || null)) return false;
    return true;
  });
}

export async function getDatasetWatcherStatuses(ids?: string[]) {
  const store = await readStatusStore();
  if (!ids) return store.statuses;
  return Object.fromEntries(ids.map(id => [id, store.statuses[id] || defaultWatcherStatus()]));
}

function assertWatcherLimit(watchers: DatasetWatcherConfig[], id: string) {
  if (watchers.some(watcher => watcher.id === id)) return;
  if (watchers.length >= MAX_WATCHERS) throw new Error(`At most ${MAX_WATCHERS} dataset watchers can be configured`);
}

export async function saveDatasetWatcher(rawWatcher: unknown) {
  const watcher = await withSettingsLock('registry', async () => {
    const watchers = await readWatcherRows();
    const requestedId = isRecord(rawWatcher) ? asString(rawWatcher.id) : '';
    const existing = requestedId ? watchers.find(item => item.id === requestedId) || null : null;
    const nextWatcher = normalizeIncomingWatcher(rawWatcher, existing);
    assertWatcherLimit(watchers, nextWatcher.id);
    await validateDatasetWatcher(nextWatcher);

    const nextWatchers = existing
      ? watchers.map(item => (item.id === nextWatcher.id ? nextWatcher : item))
      : [...watchers, nextWatcher];
    await writeWatcherRows(nextWatchers);
    return nextWatcher;
  });
  await writeStatus(watcher.id, watcher.enabled ? defaultWatcherStatus('idle') : defaultWatcherStatus('disabled'));
  return watcher;
}

export async function deleteDatasetWatcher(id: string) {
  const watcherID = asString(id);
  if (!watcherID) throw new Error('Watcher id is required');
  const existing = await withSettingsLock('registry', async () => {
    const watchers = await readWatcherRows();
    const watcher = watchers.find(item => item.id === watcherID) || null;
    if (!watcher) return null;
    await writeWatcherRows(watchers.filter(item => item.id !== watcherID));
    return watcher;
  });
  if (!existing) return null;
  await withSettingsLock('status', async () => {
    const store = await readStatusStore();
    delete store.statuses[watcherID];
    await db.settings.upsert(DATASET_WATCHER_STATUS_SETTING_KEY, JSON.stringify(store));
  });
  pendingByWatcher().delete(watcherID);
  return existing;
}

export function isWindowsDrivePath(value: string) {
  return WINDOWS_DRIVE_PATH.test(value.trim());
}

export function isWslMountPath(value: string) {
  return WSL_MOUNT_PATH.test(value.trim().replace(/\\/g, '/'));
}

export function isWslUncPath(value: string) {
  return WSL_UNC_PATH.test(value.trim());
}

function isLikelyWsl(platform: NodeJS.Platform, env: NodeJS.ProcessEnv) {
  return platform === 'linux' && Boolean(env.WSL_DISTRO_NAME || env.WSL_INTEROP || env.WSLENV);
}

function dedupe(values: string[]) {
  const seen = new Set<string>();
  const results: string[] = [];
  for (const value of values) {
    if (!value || seen.has(value)) continue;
    seen.add(value);
    results.push(value);
  }
  return results;
}

function settingsLockPath(name: string) {
  const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '_') || 'settings';
  return path.join(TOOLKIT_ROOT, `.aitk_dataset_watchers_${safeName}.lock`);
}

async function acquireShortFileLock(lockPath: string, label: string) {
  const startedAt = Date.now();
  const lockID = randomUUID();
  const payload = JSON.stringify({ lockID, label, pid: process.pid, lockedAt: nowIso(startedAt) }, null, 2);

  while (true) {
    try {
      const handle = await fsp.open(lockPath, 'wx');
      try {
        await handle.writeFile(payload, 'utf-8');
      } finally {
        await handle.close();
      }
      return async () => {
        const current = await fsp.readFile(lockPath, 'utf-8').catch(() => '');
        try {
          if (JSON.parse(current)?.lockID !== lockID) return;
        } catch {
          return;
        }
        await fsp.rm(lockPath, { force: true }).catch(() => undefined);
      };
    } catch (error: any) {
      if (error?.code !== 'EEXIST') throw error;
      const now = Date.now();
      const stat = await fsp.stat(lockPath).catch(() => null);
      if (stat && now - stat.mtimeMs > DATASET_WATCHER_SETTINGS_LOCK_STALE_MS) {
        await fsp.rm(lockPath, { force: true }).catch(() => undefined);
        continue;
      }
      if (now - startedAt > DATASET_WATCHER_SETTINGS_LOCK_WAIT_MS) {
        throw new Error(`Timed out waiting for dataset watcher ${label} lock`);
      }
      await sleep(50);
    }
  }
}

async function withSettingsLock<T>(name: string, fn: () => Promise<T>): Promise<T> {
  const release = await acquireShortFileLock(settingsLockPath(name), name);
  try {
    return await fn();
  } finally {
    await release();
  }
}

export function datasetWatcherPathCandidates(
  rawPath: string,
  options: { platform?: NodeJS.Platform; env?: NodeJS.ProcessEnv } = {},
) {
  const platform = options.platform || process.platform;
  const env = options.env || process.env;
  const value = rawPath.trim().replace(/^["']|["']$/g, '');
  const candidates = [value];
  const driveMatch = value.match(WINDOWS_DRIVE_PATH);
  const slashNormalized = value.replace(/\\/g, '/');
  const wslMountMatch = slashNormalized.match(WSL_MOUNT_PATH);
  const wslUncMatch = value.match(WSL_UNC_PATH);

  if (driveMatch && isLikelyWsl(platform, env)) {
    const drive = driveMatch[1].toLowerCase();
    const rest = driveMatch[2].replace(/\\/g, '/');
    candidates.push(`/mnt/${drive}/${rest}`);
  }

  if (wslMountMatch && platform === 'win32') {
    const drive = wslMountMatch[1].toUpperCase();
    const rest = (wslMountMatch[2] || '').replace(/\//g, '\\');
    candidates.push(`${drive}:\\${rest}`);
  }

  if (wslUncMatch) {
    const distro = wslUncMatch[1];
    const rest = wslUncMatch[2].replace(/\\/g, '/');
    if (platform === 'linux' && (!env.WSL_DISTRO_NAME || distro.toLowerCase() === env.WSL_DISTRO_NAME.toLowerCase())) {
      candidates.push(`/${rest}`);
    }
  }

  if (platform === 'win32' && value.startsWith('/') && env.WSL_DISTRO_NAME && !wslMountMatch) {
    candidates.push(`\\\\wsl$\\${env.WSL_DISTRO_NAME}${value.replace(/\//g, '\\')}`);
  }

  return dedupe(candidates);
}

function realpathOrResolve(value: string) {
  return fsp.realpath(value).catch(() => path.resolve(value));
}

function pathKey(value: string, platform: NodeJS.Platform = process.platform) {
  const normalized = path.resolve(value).replace(/\\/g, '/');
  return platform === 'win32' ? normalized.toLowerCase() : normalized;
}

export function isPathInsideOrSame(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

async function resolveAccessibleDirectory(rawPath: string) {
  const errors: string[] = [];
  for (const candidate of datasetWatcherPathCandidates(rawPath)) {
    try {
      const stat = await fsp.stat(candidate);
      if (!stat.isDirectory()) {
        errors.push(`${candidate} is not a folder`);
        continue;
      }
      return await fsp.realpath(candidate);
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }
  throw new Error(`Watch folder was not found or is not accessible: ${rawPath}${errors[0] ? ` (${errors[0]})` : ''}`);
}

async function assertNotRootDirectory(directory: string) {
  const resolved = path.resolve(directory);
  if (resolved === path.parse(resolved).root) {
    throw new Error('Watch folder cannot be a filesystem root');
  }
}

async function resolveWatcherDataset(watcher: Pick<DatasetWatcherConfig, 'datasetName' | 'projectID'>) {
  const scope = await resolveDatasetScope(watcher.projectID);
  const datasetFolder = resolveDatasetFolder(scope.datasetsRoot, watcher.datasetName);
  const stat = await fsp.stat(datasetFolder).catch(() => null);
  if (!stat?.isDirectory()) throw new Error('Dataset folder was not found');
  if (isEncryptedDatasetFolder(datasetFolder)) {
    throw new Error('Dataset watch folders are not supported for encrypted datasets');
  }
  return { scope, datasetFolder };
}

export async function validateDatasetWatcher(watcher: DatasetWatcherConfig) {
  const { datasetFolder } = await resolveWatcherDataset(watcher);
  const [sourceRealPath, datasetRealPath] = await Promise.all([
    resolveAccessibleDirectory(watcher.sourcePath),
    realpathOrResolve(datasetFolder),
  ]);
  await assertNotRootDirectory(sourceRealPath);
  if (isPathInsideOrSame(sourceRealPath, datasetRealPath) || isPathInsideOrSame(datasetRealPath, sourceRealPath)) {
    throw new Error('Watch folder cannot overlap the destination dataset folder');
  }
  return { sourceRealPath, datasetFolder, datasetRealPath };
}

function manifestPath(datasetFolder: string) {
  return path.join(datasetFolder, DATASET_WATCHER_IMPORT_MANIFEST);
}

async function readImportManifest(datasetFolder: string): Promise<DatasetWatchManifest> {
  try {
    const parsed = JSON.parse(await fsp.readFile(manifestPath(datasetFolder), 'utf-8'));
    if (parsed?.version !== DATASET_WATCHER_MANIFEST_VERSION || !isRecord(parsed.imports)) {
      return { version: DATASET_WATCHER_MANIFEST_VERSION, imports: {} };
    }
    return parsed;
  } catch {
    return { version: DATASET_WATCHER_MANIFEST_VERSION, imports: {} };
  }
}

async function writeImportManifest(datasetFolder: string, manifest: DatasetWatchManifest) {
  const target = manifestPath(datasetFolder);
  const tmp = `${target}.${process.pid}.${Date.now()}.tmp`;
  await fsp.writeFile(tmp, JSON.stringify(manifest, null, 2), 'utf-8');
  await fsp.rename(tmp, target);
}

function isHiddenPathSegment(name: string) {
  return name.startsWith('.');
}

function isMediaFile(filePath: string) {
  return MEDIA_EXTENSIONS.has(path.extname(filePath).toLowerCase());
}

function isCaptionSidecar(filePath: string) {
  return CAPTION_EXTENSIONS.has(path.extname(filePath).toLowerCase());
}

function isAutoCaptionableImage(filePath: string) {
  return IMAGE_EXTENSIONS.has(path.extname(filePath).toLowerCase());
}

function cleanPathSegment(segment: string, fallback: string) {
  const cleaned = segment
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_')
    .replace(/\s+/g, '_')
    .replace(/^\.+/, '')
    .slice(0, 120);
  return cleaned || fallback;
}

function cleanUploadFileName(fileName: string) {
  const base = path.basename(fileName || 'file');
  const ext = path.extname(base).replace(/[^a-zA-Z0-9._-]/g, '_');
  const stem = ext ? base.slice(0, -ext.length) : base;
  return `${cleanPathSegment(stem, 'file')}${ext}`;
}

function cleanRelativeImportPath(relativePath: string, fallbackName: string) {
  const normalized = relativePath.replace(/\\/g, '/');
  const rawParts = normalized.split('/').filter(Boolean);
  if (rawParts.length === 0 || normalized.startsWith('/') || rawParts.some(part => part === '..')) {
    return cleanUploadFileName(fallbackName);
  }
  const parts = rawParts.map((part, index) =>
    index === rawParts.length - 1 ? cleanUploadFileName(part) : cleanPathSegment(part, `folder_${index + 1}`),
  );
  return path.join(...parts);
}

function destinationForSuffix(datasetFolder: string, relativePath: string, suffix: number) {
  const parsed = path.parse(relativePath);
  const fileName = suffix > 1 ? `${parsed.name}_${suffix}${parsed.ext}` : `${parsed.name}${parsed.ext}`;
  const candidate = path.resolve(datasetFolder, parsed.dir, fileName);
  const relative = path.relative(path.resolve(datasetFolder), candidate);
  if (relative === '' || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error('Invalid watcher destination path');
  }
  return candidate;
}

function sameObservedFile(candidate: SourceCandidate, stat: fs.Stats) {
  return candidate.size === stat.size && candidate.mtimeMs === stat.mtimeMs;
}

function watcherLockPath(datasetFolder: string) {
  return path.join(datasetFolder, '.aitk_dataset_watch.lock');
}

async function acquireWatcherLock(datasetFolder: string, watcherID: string, now: number) {
  const lockPath = watcherLockPath(datasetFolder);
  const lockID = randomUUID();
  const payload = JSON.stringify({ lockID, watcherID, pid: process.pid, lockedAt: nowIso(now) }, null, 2);

  const openLock = async () => {
    const handle = await fsp.open(lockPath, 'wx');
    try {
      await handle.writeFile(payload, 'utf-8');
    } finally {
      await handle.close();
    }
  };

  try {
    await openLock();
  } catch (error: any) {
    if (error?.code !== 'EEXIST') throw error;
    const stat = await fsp.stat(lockPath).catch(() => null);
    if (!stat || now - stat.mtimeMs <= DATASET_WATCHER_LOCK_STALE_MS) {
      return null;
    }
    await fsp.rm(lockPath, { force: true }).catch(() => undefined);
    try {
      await openLock();
    } catch (retryError: any) {
      if (retryError?.code === 'EEXIST') return null;
      throw retryError;
    }
  }

  const heartbeat = setInterval(() => {
    const date = new Date();
    void fsp.utimes(lockPath, date, date).catch(() => undefined);
  }, DATASET_WATCHER_LOCK_HEARTBEAT_MS);
  heartbeat.unref?.();

  return async () => {
    clearInterval(heartbeat);
    const current = await fsp.readFile(lockPath, 'utf-8').catch(() => '');
    try {
      if (JSON.parse(current)?.lockID !== lockID) return;
    } catch {
      return;
    }
    await fsp.rm(lockPath, { force: true }).catch(() => undefined);
  };
}

async function findSourceMedia(sourceRoot: string, includeSubfolders: boolean) {
  const results: SourceCandidate[] = [];
  const stack = [{ absolutePath: sourceRoot, relativePath: '' }];

  while (stack.length > 0) {
    const current = stack.pop() as { absolutePath: string; relativePath: string };
    const entries = await fsp.readdir(current.absolutePath, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (isHiddenPathSegment(entry.name)) continue;
      const absolutePath = path.join(current.absolutePath, entry.name);
      const relativePath = current.relativePath ? path.join(current.relativePath, entry.name) : entry.name;
      if (entry.isDirectory()) {
        if (includeSubfolders) stack.push({ absolutePath, relativePath });
        continue;
      }
      if (!entry.isFile() || isCaptionSidecar(entry.name) || !isMediaFile(entry.name)) continue;
      const stat = await fsp.stat(absolutePath).catch(() => null);
      if (!stat?.isFile()) continue;
      results.push({ absolutePath, relativePath, size: stat.size, mtimeMs: stat.mtimeMs });
    }
  }

  return results.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
}

function sourceKey(filePath: string) {
  return pathKey(filePath);
}

function isStableCandidate(
  watcherID: string,
  key: string,
  candidate: SourceCandidate,
  now: number,
  stableMs: number,
) {
  const allPending = pendingByWatcher();
  const watcherPending = allPending.get(watcherID) || new Map<string, PendingSourceObservation>();
  allPending.set(watcherID, watcherPending);
  const previous = watcherPending.get(key);
  watcherPending.set(key, {
    size: candidate.size,
    mtimeMs: candidate.mtimeMs,
    firstSeenAt:
      previous && previous.size === candidate.size && previous.mtimeMs === candidate.mtimeMs ? previous.firstSeenAt : now,
    lastSeenAt: now,
  });
  return (
    !!previous &&
    previous.size === candidate.size &&
    previous.mtimeMs === candidate.mtimeMs &&
    now - previous.firstSeenAt >= stableMs
  );
}

function prunePending(watcherID: string, activeKeys: Set<string>) {
  const pending = pendingByWatcher().get(watcherID);
  if (!pending) return;
  for (const key of pending.keys()) {
    if (!activeKeys.has(key)) pending.delete(key);
  }
}

function mimeTypeForImagePath(filePath: string) {
  const ext = path.extname(filePath).toLowerCase();
  const map: Record<string, string> = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.jxl': 'image/jxl',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
  };
  return map[ext] || 'application/octet-stream';
}

async function imageDataUrlFromPath(filePath: string) {
  const bytes = await fsp.readFile(filePath);
  return `data:${mimeTypeForImagePath(filePath)};base64,${bytes.toString('base64')}`;
}

async function autoCaptionImportedFile(options: {
  watcher: DatasetWatcherConfig;
  datasetFolder: string;
  importedPath: string;
}) {
  const autoCaption = options.watcher.autoCaption;
  if (!autoCaption?.enabled) return false;
  if (!isAutoCaptionableImage(options.importedPath)) return false;
  if (autoCaption.provider === 'remote_ollama' && !autoCaption.remoteWorkerId) {
    throw new Error('Remote Ollama worker is required for auto-caption');
  }

  const rootCaption = await readDatasetRootCaption(options.datasetFolder).catch(() => ({ found: false, systemPrompt: '' }));
  const systemPrompt = autoCaption.systemPrompt || (rootCaption.found ? rootCaption.systemPrompt : '');
  const outputFormat = autoCaption.outputFormat || 'text';
  const result = await generateSingleImageRecaption({
    provider: autoCaption.provider,
    model: autoCaption.model,
    prompt: autoCaption.prompt,
    systemPrompt,
    outputFormat,
    existingCaption: '',
    remoteWorkerId: autoCaption.remoteWorkerId,
    maxNewTokens: autoCaption.maxNewTokens,
    imageDataUrl: await imageDataUrlFromPath(options.importedPath),
    openRouterApiKey: await getOpenRouterApiKey(),
  });
  const extension = result.outputFormat === 'ideogram_json' ? '.json' : '.txt';
  await fsp.writeFile(captionSidecarPath(options.importedPath, extension), result.caption, 'utf-8');
  return true;
}

async function copyCandidateIntoDataset(options: {
  watcher: DatasetWatcherConfig;
  datasetFolder: string;
  candidate: SourceCandidate;
}) {
  const relativePath = options.watcher.preserveRelativePaths
    ? cleanRelativeImportPath(options.candidate.relativePath, path.basename(options.candidate.absolutePath))
    : cleanUploadFileName(path.basename(options.candidate.absolutePath));

  for (let suffix = 1; suffix < 10_000; suffix += 1) {
    const destination = destinationForSuffix(options.datasetFolder, relativePath, suffix);
    await fsp.mkdir(path.dirname(destination), { recursive: true });
    try {
      await fsp.copyFile(options.candidate.absolutePath, destination, fs.constants.COPYFILE_EXCL);
    } catch (error: any) {
      if (error?.code === 'EEXIST') continue;
      throw error;
    }

    const sourceStat = await fsp.stat(options.candidate.absolutePath).catch(() => null);
    if (!sourceStat?.isFile() || !sameObservedFile(options.candidate, sourceStat)) {
      await fsp.rm(destination, { force: true }).catch(() => undefined);
      return null;
    }

    return destination;
  }

  throw new Error('Could not find an available destination file name');
}

export async function runDatasetWatcherOnce(
  watcher: DatasetWatcherConfig,
  options: DatasetWatcherRunOptions = {},
): Promise<DatasetWatcherRunResult> {
  const now = options.now ?? Date.now();
  const stableMs = options.stableMs ?? DATASET_WATCHER_STABLE_MS;
  const importedPaths: string[] = [];
  const captionedPaths: string[] = [];
  const warnings: string[] = [];
  let lastError: string | null = null;
  let state: DatasetWatcherStatus['state'] = watcher.enabled ? 'scanning' : 'disabled';

  if (!watcher.enabled) {
    const disabled = await writeStatus(watcher.id, {
      ...defaultWatcherStatus('disabled'),
      lastScanAt: nowIso(now),
    });
    return { watcherID: watcher.id, ...disabled, importedPaths, captionedPaths };
  }

  let releaseLock: (() => Promise<void>) | null = null;

  try {
    const { datasetFolder } = await resolveWatcherDataset(watcher);
    releaseLock = await acquireWatcherLock(datasetFolder, watcher.id, now);
    if (!releaseLock) {
      const current = (await readStatusStore()).statuses[watcher.id] || defaultWatcherStatus('scanning');
      const warning = 'Watcher is already running.';
      return {
        watcherID: watcher.id,
        ...current,
        warnings: current.warnings.includes(warning) ? current.warnings : [...current.warnings, warning],
        importedPaths,
        captionedPaths,
      };
    }

    await writeStatus(watcher.id, {
      ...defaultWatcherStatus('scanning'),
      lastScanAt: nowIso(now),
    });

    const [sourceRoot, datasetRealPath] = await Promise.all([
      resolveAccessibleDirectory(watcher.sourcePath),
      realpathOrResolve(datasetFolder),
    ]);
    await assertNotRootDirectory(sourceRoot);
    if (isPathInsideOrSame(sourceRoot, datasetRealPath) || isPathInsideOrSame(datasetRealPath, sourceRoot)) {
      throw new Error('Watch folder cannot overlap the destination dataset folder');
    }

    const manifest = await readImportManifest(datasetFolder);
    const candidates = await findSourceMedia(sourceRoot, watcher.includeSubfolders);
    const activeKeys = new Set(candidates.map(candidate => sourceKey(candidate.absolutePath)));
    prunePending(watcher.id, activeKeys);

    for (const candidate of candidates) {
      const key = sourceKey(candidate.absolutePath);
      if (manifest.imports[key]) continue;
      if (!isStableCandidate(watcher.id, key, candidate, now, stableMs)) continue;

      state = 'importing';
      await writeStatus(watcher.id, {
        state,
        lastScanAt: nowIso(now),
        lastImportedAt: importedPaths.length > 0 ? nowIso(now) : null,
        lastImportedCount: importedPaths.length,
        lastCaptionedCount: captionedPaths.length,
        lastError,
        warnings,
      });

      const destination = await copyCandidateIntoDataset({ watcher, datasetFolder, candidate });
      if (!destination) {
        warnings.push(`${candidate.relativePath}: Source file changed during import; it will be retried on the next scan.`);
        pendingByWatcher().get(watcher.id)?.delete(key);
        continue;
      }
      const destinationRelativePath = path.relative(datasetFolder, destination);
      manifest.imports[key] = {
        sourceKey: key,
        sourcePath: candidate.absolutePath,
        destinationRelativePath,
        importedAt: nowIso(now),
        size: candidate.size,
        mtimeMs: candidate.mtimeMs,
      };
      importedPaths.push(destination);
      pendingByWatcher().get(watcher.id)?.delete(key);
      await writeImportManifest(datasetFolder, manifest);

      if (watcher.autoCaption?.enabled && isAutoCaptionableImage(destination)) {
        state = 'captioning';
        await writeStatus(watcher.id, {
          state,
          lastScanAt: nowIso(now),
          lastImportedAt: nowIso(now),
          lastImportedCount: importedPaths.length,
          lastCaptionedCount: captionedPaths.length,
          lastError,
          warnings,
        });
        try {
          if (await autoCaptionImportedFile({ watcher, datasetFolder, importedPath: destination })) {
            captionedPaths.push(destination);
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Auto-caption failed';
          lastError = message;
          warnings.push(`${path.basename(destination)}: ${message}`);
        }
      }
    }
  } catch (error) {
    lastError = error instanceof Error ? error.message : 'Dataset watcher failed';
  } finally {
    await releaseLock?.();
  }

  const finalStatus: DatasetWatcherStatus = {
    state: lastError ? 'error' : 'idle',
    lastScanAt: nowIso(now),
    lastImportedAt: importedPaths.length > 0 ? nowIso(now) : null,
    lastImportedCount: importedPaths.length,
    lastCaptionedCount: captionedPaths.length,
    lastError,
    warnings,
  };
  await writeStatus(watcher.id, finalStatus);
  return { watcherID: watcher.id, ...finalStatus, importedPaths, captionedPaths };
}

export async function runEnabledDatasetWatchers(options: DatasetWatcherRunOptions = {}) {
  const watchers = await readWatcherRows();
  const results: DatasetWatcherRunResult[] = [];
  for (const watcher of watchers) {
    if (!watcher.enabled) {
      await writeStatus(watcher.id, { ...defaultWatcherStatus('disabled'), lastScanAt: nowIso(options.now ?? Date.now()) });
      continue;
    }
    results.push(await runDatasetWatcherOnce(watcher, options));
  }
  return results;
}

export async function runDatasetWatchersIfDue(now = Date.now()) {
  const lastPollAt = globalThis.__aitkDatasetWatcherLastPollAt || 0;
  if (now - lastPollAt < DATASET_WATCHER_POLL_INTERVAL_MS) return [];
  globalThis.__aitkDatasetWatcherLastPollAt = now;
  return runEnabledDatasetWatchers({ now });
}
