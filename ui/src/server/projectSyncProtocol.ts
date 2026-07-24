import { createHash } from 'crypto';
import { createReadStream, createWriteStream } from 'fs';
import fs from 'fs/promises';
import path from 'path';
import { pipeline } from 'stream/promises';

export const PROJECT_SYNC_PROTOCOL = 'project-sync-v1' as const;
export const PROJECT_SYNC_CHUNK_BYTES = 8 * 1024 * 1024;
export const PROJECT_SYNC_MAX_FILE_BYTES = 1024 * 1024 * 1024 * 1024;
export const PROJECT_SYNC_MAX_OPERATION_BYTES = 2 * 1024 * 1024 * 1024 * 1024;
export const PROJECT_SYNC_MAX_FILES = 100_000;

export const PROJECT_SYNC_PROFILES = ['full', 'launch', 'results'] as const;
export type ProjectSyncProfileName = (typeof PROJECT_SYNC_PROFILES)[number];
export const PROJECT_SYNC_TRANSFER_BLOCKING_JOB_STATUSES = ['starting', 'running', 'stopping'] as const;
export const PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES = [
  'queued',
  ...PROJECT_SYNC_TRANSFER_BLOCKING_JOB_STATUSES,
] as const;

const PROFILE_ZONES = {
  full: ['datasets', 'configs', 'runs', 'outputs', 'models', 'assets', 'notes'],
  // Launch replicas only need held-out validation images from the assets
  // zone. Other project assets may be private or unrelated to execution.
  launch: ['datasets', 'configs', 'runs', 'models', 'assets/validation'],
  results: ['runs', 'outputs', 'models'],
} as const satisfies Record<ProjectSyncProfileName, readonly string[]>;

const LAUNCH_VALIDATION_IMAGE_EXTENSIONS = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.gif',
  '.webp',
  '.jxl',
  '.bmp',
]);

const BLOCKED_SEGMENTS = new Set([
  '.git',
  '.aitk-sync',
  '.tmp',
  '__pycache__',
  'cache',
  'node_modules',
  'locks',
  'tmp',
  'temp',
]);

const BLOCKED_FILE_NAMES = new Set([
  '.env',
  '.env.local',
  '.env.production',
  '.env.development',
  '.ds_store',
  'thumbs.db',
  'credentials.json',
  'secrets.json',
  'tokens.json',
]);

const BLOCKED_FILE_SUFFIXES = ['.lock', '.part', '.tmp', '.temp', '.swp', '.pid', '.key', '.pem', '.p12', '.pfx'];
const BLOCKED_FILE_FRAGMENTS = ['credential', 'api-token', 'api_token', 'access-token', 'access_token', 'private-key'];
const CREDENTIAL_CONFIG_KEYS = new Set([
  'ai_toolkit_auth',
  'api_key',
  'api_token',
  'access_token',
  'auth_token',
  'bearer_token',
  'durable_encryption_key',
  'encryption_key',
  'hf_token',
  'keyb64',
  'openrouter_api_key',
  'password',
  'private_key',
  'refresh_token',
]);
const CREDENTIAL_TEXT_PATTERN =
  /(?:^|["'\s{,])(?:ai_toolkit_auth|api[_-]?(?:key|token)|access[_-]?token|auth[_-]?token|bearer[_-]?token|durable[_-]?encryption[_-]?key|encryption[_-]?key|hf[_-]?token|keyb64|openrouter[_-]?api[_-]?key|password|private[_-]?key|refresh[_-]?token)["']?\s*[:=]\s*["']?[^\s"',}\]]+/im;
const CREDENTIAL_SCAN_EXTENSIONS = new Set(['.json', '.json5', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf']);

export type ProjectManifestEntry = {
  path: string;
  size: number;
  sha256: string;
  modified_at: string;
};

export type ProjectSyncManifest = {
  protocol: typeof PROJECT_SYNC_PROTOCOL;
  project_id: string;
  profile: ProjectSyncProfileName;
  generated_at: string;
  hash: string;
  files: ProjectManifestEntry[];
};

export type ProjectSyncConflictResolution = 'keep-home' | 'keep-worker' | 'keep-both';

export type ProjectSyncConflict = {
  path: string;
  kind: 'both-modified' | 'home-deleted-worker-modified' | 'worker-deleted-home-modified';
  base: ProjectManifestEntry | null;
  home: ProjectManifestEntry | null;
  worker: ProjectManifestEntry | null;
  resolution?: ProjectSyncConflictResolution;
};

export type ProjectManifestDiff = {
  add_or_update: ProjectManifestEntry[];
  delete: string[];
  unchanged: string[];
};

export type ChunkReceipt = {
  sha256: string;
  total: number;
  received: number;
  complete: boolean;
};

export type ProjectCommitPlan = {
  project_id: string;
  profile: ProjectSyncProfileName;
  operation_id: string;
  files: ProjectManifestEntry[];
  delete_paths?: string[];
  preserve_paths?: Array<{ path: string; preserve_as: string }>;
};

export type PortableProjectJob = {
  id: string;
  name: string;
  source_worker_id: string;
  remote_job_id: string | null;
  remote_sync_at: string | null;
  remote_error: string | null;
  gpu_ids: string;
  job_config: unknown;
  status: string;
  stop: boolean;
  return_to_queue: boolean;
  step: number;
  info: string;
  speed_string: string;
  queue_position: number;
  job_type: string;
  job_ref: string | null;
  save_now: boolean;
  sample_now: boolean;
  created_at: string;
  updated_at: string;
};

export type PortableProjectJobSnapshot = {
  protocol: typeof PROJECT_SYNC_PROTOCOL;
  project_id: string;
  home_instance_id: string;
  generated_at: string;
  hash: string;
  jobs: PortableProjectJob[];
};

export type ProjectReplicaExecutionAuthorization = {
  protocol: typeof PROJECT_SYNC_PROTOCOL;
  projectID: string;
  homeInstanceID: string;
};

export class ProjectSyncProtocolError extends Error {
  status: number;
  code: string;
  details?: unknown;

  constructor(message: string, options: { status?: number; code?: string; details?: unknown } = {}) {
    super(message);
    this.name = 'ProjectSyncProtocolError';
    this.status = options.status ?? 400;
    this.code = options.code ?? 'PROJECT_SYNC_INVALID_REQUEST';
    this.details = options.details;
  }
}

export function parseReplicaExecutionAuthorization(
  headers: Pick<Headers, 'get'>,
  expectedBearerToken: string | null | undefined,
): ProjectReplicaExecutionAuthorization | null {
  if (headers.get('x-aitk-project-sync') !== PROJECT_SYNC_PROTOCOL) return null;
  const expected = expectedBearerToken?.trim() || '';
  const supplied = headers.get('authorization')?.match(/^Bearer\s+(.+)$/i)?.[1] || '';
  if (!expected || supplied !== expected) {
    throw new ProjectSyncProtocolError('Authenticated project-sync worker access is required', {
      status: 401,
      code: 'PROJECT_SYNC_EXECUTION_UNAUTHORIZED',
    });
  }
  const projectID = headers.get('x-aitk-project-id')?.trim() || '';
  const homeInstanceID = headers.get('x-aitk-home-instance')?.trim() || '';
  if (!projectID || !homeInstanceID) {
    throw new ProjectSyncProtocolError('Project replica execution headers are incomplete', {
      status: 403,
      code: 'PROJECT_SYNC_EXECUTION_UNAUTHORIZED',
    });
  }
  return { protocol: PROJECT_SYNC_PROTOCOL, projectID, homeInstanceID };
}

function posixPath(value: string) {
  return value.replace(/\\/g, '/');
}

export function normalizePortableRelativePath(value: string) {
  if (typeof value !== 'string' || value.includes('\0')) {
    throw new ProjectSyncProtocolError('Portable project paths must be strings without NUL bytes');
  }
  const normalized = path.posix.normalize(posixPath(value).replace(/^\/+/, ''));
  if (!normalized || normalized === '.' || normalized === '..' || normalized.startsWith('../') || path.posix.isAbsolute(normalized)) {
    throw new ProjectSyncProtocolError('Portable project path escapes the project root', {
      code: 'PROJECT_SYNC_PATH_INVALID',
      details: { path: value },
    });
  }
  return normalized;
}

export function makePortableProjectRef(projectID: string, relativePath: string) {
  const safeProjectID = encodeURIComponent(projectID.trim());
  if (!safeProjectID) throw new ProjectSyncProtocolError('Project ID is required');
  const encodedPath = normalizePortableRelativePath(relativePath).split('/').map(encodeURIComponent).join('/');
  return `aitk-project://${safeProjectID}/${encodedPath}`;
}

export function parsePortableProjectRef(value: string) {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new ProjectSyncProtocolError('Invalid portable project reference', { code: 'PROJECT_SYNC_REF_INVALID' });
  }
  if (url.protocol !== 'aitk-project:' || !url.hostname) {
    throw new ProjectSyncProtocolError('Invalid portable project reference', { code: 'PROJECT_SYNC_REF_INVALID' });
  }
  return {
    project_id: decodeURIComponent(url.hostname),
    relative_path: normalizePortableRelativePath(decodeURIComponent(url.pathname.replace(/^\/+/, ''))),
  };
}

function mapProjectConfigValue(
  value: unknown,
  transform: (value: string) => string,
  seen: WeakSet<object>,
): unknown {
  if (typeof value === 'string') return transform(value);
  if (!value || typeof value !== 'object') return value;
  if (seen.has(value)) {
    throw new ProjectSyncProtocolError('Project config contains a circular value', {
      code: 'PROJECT_SYNC_CONFIG_INVALID',
    });
  }
  seen.add(value);
  if (Array.isArray(value)) {
    const mapped = value.map(item => mapProjectConfigValue(item, transform, seen));
    seen.delete(value);
    return mapped;
  }
  const mapped = Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, mapProjectConfigValue(item, transform, seen)]),
  );
  seen.delete(value);
  return mapped;
}

export function portableizeProjectConfig(value: unknown, projectRoot: string, projectID: string) {
  const root = path.resolve(projectRoot);
  return mapProjectConfigValue(
    value,
    candidate => {
      if (!path.isAbsolute(candidate)) return candidate;
      const absolute = path.resolve(candidate);
      const relative = path.relative(root, absolute);
      if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return candidate;
      return makePortableProjectRef(projectID, posixPath(relative));
    },
    new WeakSet(),
  );
}

export function resolvePortableProjectConfig(value: unknown, projectRoot: string, projectID: string) {
  return mapProjectConfigValue(
    value,
    candidate => {
      if (!candidate.startsWith('aitk-project://')) return candidate;
      const reference = parsePortableProjectRef(candidate);
      if (reference.project_id !== projectID) {
        throw new ProjectSyncProtocolError('Project config contains a cross-project portable reference', {
          status: 409,
          code: 'PROJECT_SYNC_CROSS_PROJECT_REF',
        });
      }
      return resolveProjectSyncPath(projectRoot, reference.relative_path, 'full');
    },
    new WeakSet(),
  );
}

export function collectNonPortableAbsolutePaths(value: unknown) {
  const paths = new Set<string>();
  const visit = (item: unknown, seen: WeakSet<object>) => {
    if (typeof item === 'string') {
      if (path.isAbsolute(item)) paths.add(item);
      return;
    }
    if (!item || typeof item !== 'object' || seen.has(item)) return;
    seen.add(item);
    if (Array.isArray(item)) item.forEach(child => visit(child, seen));
    else Object.values(item as Record<string, unknown>).forEach(child => visit(child, seen));
  };
  visit(value, new WeakSet());
  return [...paths].sort();
}

export function collectCredentialConfigKeys(value: unknown) {
  const matches = new Set<string>();
  const visit = (item: unknown, seen: WeakSet<object>) => {
    if (!item || typeof item !== 'object' || seen.has(item)) return;
    seen.add(item);
    if (Array.isArray(item)) {
      item.forEach(child => visit(child, seen));
      return;
    }
    for (const [key, child] of Object.entries(item as Record<string, unknown>)) {
      const normalized = key.trim().toLowerCase().replace(/-/g, '_');
      if (CREDENTIAL_CONFIG_KEYS.has(normalized) && child != null && String(child).trim()) matches.add(key);
      visit(child, seen);
    }
  };
  visit(value, new WeakSet());
  return [...matches].sort();
}

async function assertFileHasNoCredentialMaterial(filePath: string, relativePath: string, size: number) {
  const extension = path.extname(filePath).toLowerCase();
  if (!CREDENTIAL_SCAN_EXTENSIONS.has(extension) || size > 16 * 1024 * 1024) return;
  const contents = await fs.readFile(filePath, 'utf8');
  if (CREDENTIAL_TEXT_PATTERN.test(contents)) {
    throw new ProjectSyncProtocolError('Project sync blocked a file containing credential material', {
      status: 409,
      code: 'PROJECT_SYNC_SECRET_BLOCKED',
      details: { path: relativePath },
    });
  }
}

export function profileIncludesPath(profile: ProjectSyncProfileName, relativePath: string) {
  const normalized = normalizePortableRelativePath(relativePath);
  return (PROFILE_ZONES[profile] as readonly string[]).some(
    includedPath =>
      normalized === includedPath ||
      normalized.startsWith(`${includedPath}/`),
  );
}

export function isProjectSyncPathExcluded(
  relativePath: string,
  profile: ProjectSyncProfileName,
  options: { directory?: boolean } = {},
) {
  let normalized: string;
  try {
    normalized = normalizePortableRelativePath(relativePath);
  } catch {
    return true;
  }
  if (!profileIncludesPath(profile, normalized)) return true;
  const segments = normalized.toLowerCase().split('/');
  if (segments.some(segment => BLOCKED_SEGMENTS.has(segment))) return true;
  const fileName = segments.at(-1) || '';
  if (
    !options.directory &&
    profile === 'launch' &&
    normalized.toLowerCase().startsWith('assets/validation/') &&
    !LAUNCH_VALIDATION_IMAGE_EXTENSIONS.has(path.posix.extname(fileName))
  ) {
    return true;
  }
  if (BLOCKED_FILE_NAMES.has(fileName)) return true;
  if (BLOCKED_FILE_SUFFIXES.some(suffix => fileName.endsWith(suffix))) return true;
  if (BLOCKED_FILE_FRAGMENTS.some(fragment => fileName.includes(fragment))) return true;
  return false;
}

export function resolveProjectSyncPath(projectRoot: string, relativePath: string, profile: ProjectSyncProfileName) {
  const normalized = normalizePortableRelativePath(relativePath);
  if (isProjectSyncPathExcluded(normalized, profile)) {
    throw new ProjectSyncProtocolError('Path is excluded from project sync', {
      status: 403,
      code: 'PROJECT_SYNC_PATH_EXCLUDED',
      details: { path: normalized, profile },
    });
  }
  const root = path.resolve(projectRoot);
  const target = path.resolve(root, ...normalized.split('/'));
  const relative = path.relative(root, target);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new ProjectSyncProtocolError('Path escapes the project root', {
      status: 403,
      code: 'PROJECT_SYNC_PATH_INVALID',
    });
  }
  return target;
}

export async function assertProjectSyncPathContained(projectRoot: string, targetPath: string) {
  const root = path.resolve(projectRoot);
  const target = path.resolve(targetPath);
  const lexicalRelative = path.relative(root, target);
  if (!lexicalRelative || lexicalRelative.startsWith('..') || path.isAbsolute(lexicalRelative)) {
    throw new ProjectSyncProtocolError('Path escapes the project root', {
      status: 403,
      code: 'PROJECT_SYNC_PATH_INVALID',
    });
  }
  const realRoot = await fs.realpath(root);
  let existingAncestor = target;
  while (true) {
    try {
      await fs.lstat(existingAncestor);
      break;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      const parent = path.dirname(existingAncestor);
      if (parent === existingAncestor) throw error;
      existingAncestor = parent;
    }
  }
  const realAncestor = await fs.realpath(existingAncestor);
  const realRelative = path.relative(realRoot, realAncestor);
  if (realRelative.startsWith('..') || path.isAbsolute(realRelative)) {
    throw new ProjectSyncProtocolError('Project sync path crosses a symlink outside the project root', {
      status: 403,
      code: 'PROJECT_SYNC_PATH_INVALID',
    });
  }
  return target;
}

export async function sha256File(filePath: string) {
  const digest = createHash('sha256');
  await pipeline(createReadStream(filePath), digest);
  return digest.digest('hex');
}

export function hashManifestEntries(entries: readonly ProjectManifestEntry[]) {
  const digest = createHash('sha256');
  for (const entry of [...entries].sort((left, right) => left.path.localeCompare(right.path))) {
    digest.update(entry.path);
    digest.update('\0');
    digest.update(String(entry.size));
    digest.update('\0');
    digest.update(entry.sha256);
    digest.update('\n');
  }
  return digest.digest('hex');
}

function asBoundedString(value: unknown, label: string, maxLength: number, allowEmpty = true) {
  if (typeof value !== 'string' || value.length > maxLength || (!allowEmpty && !value.trim())) {
    throw new ProjectSyncProtocolError(`Invalid ${label}`, { code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID' });
  }
  return value;
}

function asNullableBoundedString(value: unknown, label: string, maxLength: number) {
  if (value == null) return null;
  return asBoundedString(value, label, maxLength);
}

function asSafeNumber(value: unknown, label: string, minimum = Number.MIN_SAFE_INTEGER) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum) {
    throw new ProjectSyncProtocolError(`Invalid ${label}`, { code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID' });
  }
  return value;
}

function asBoolean(value: unknown, label: string) {
  if (typeof value !== 'boolean') {
    throw new ProjectSyncProtocolError(`Invalid ${label}`, { code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID' });
  }
  return value;
}

function asIsoDate(value: unknown, label: string) {
  const text = asBoundedString(value, label, 64, false);
  const timestamp = Date.parse(text);
  if (Number.isNaN(timestamp)) {
    throw new ProjectSyncProtocolError(`Invalid ${label}`, { code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID' });
  }
  return new Date(timestamp).toISOString();
}

function assertJsonValue(value: unknown, seen = new WeakSet<object>()): void {
  if (value == null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number') {
    if (Number.isFinite(value)) return;
    throw new ProjectSyncProtocolError('Job config contains a non-finite number', {
      code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID',
    });
  }
  if (typeof value !== 'object' || seen.has(value)) {
    throw new ProjectSyncProtocolError('Job config is not portable JSON', {
      code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID',
    });
  }
  seen.add(value);
  if (Array.isArray(value)) value.forEach(item => assertJsonValue(item, seen));
  else Object.values(value as Record<string, unknown>).forEach(item => assertJsonValue(item, seen));
  seen.delete(value);
}

export function parsePortableProjectJob(value: unknown): PortableProjectJob {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ProjectSyncProtocolError('Invalid project job snapshot entry', {
      code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID',
    });
  }
  const job = value as Record<string, unknown>;
  const id = asBoundedString(job.id, 'job id', 160, false);
  if (!/^[a-zA-Z0-9._-]+$/.test(id)) {
    throw new ProjectSyncProtocolError('Invalid project job id', { code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID' });
  }
  assertJsonValue(job.job_config);
  return {
    id,
    name: asBoundedString(job.name, 'job name', 240, false),
    source_worker_id: asBoundedString(job.source_worker_id, 'source worker id', 160, false),
    remote_job_id: asNullableBoundedString(job.remote_job_id, 'remote job id', 160),
    remote_sync_at: job.remote_sync_at == null ? null : asIsoDate(job.remote_sync_at, 'remote sync date'),
    remote_error: asNullableBoundedString(job.remote_error, 'remote error', 10_000),
    gpu_ids: asBoundedString(job.gpu_ids, 'GPU ids', 512, false),
    job_config: job.job_config,
    status: asBoundedString(job.status, 'job status', 64, false),
    stop: asBoolean(job.stop, 'job stop flag'),
    return_to_queue: asBoolean(job.return_to_queue, 'job return-to-queue flag'),
    step: asSafeNumber(job.step, 'job step', 0),
    info: asBoundedString(job.info, 'job info', 100_000),
    speed_string: asBoundedString(job.speed_string, 'job speed', 1_000),
    queue_position: asSafeNumber(job.queue_position, 'queue position'),
    job_type: asBoundedString(job.job_type, 'job type', 64, false),
    job_ref: asNullableBoundedString(job.job_ref, 'job ref', 4_096),
    save_now: asBoolean(job.save_now, 'job save-now flag'),
    sample_now:
      job.sample_now === undefined
        ? false
        : asBoolean(job.sample_now, 'job sample-now flag'),
    created_at: asIsoDate(job.created_at, 'created date'),
    updated_at: asIsoDate(job.updated_at, 'updated date'),
  };
}

export function hashPortableProjectJobs(jobs: readonly PortableProjectJob[]) {
  const ordered = [...jobs].sort((left, right) => left.id.localeCompare(right.id));
  return createHash('sha256').update(JSON.stringify(ordered)).digest('hex');
}

export function parsePortableProjectJobSnapshot(
  value: unknown,
  expectedProjectID: string,
  expectedHomeInstanceID: string,
): PortableProjectJobSnapshot {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ProjectSyncProtocolError('Invalid project job snapshot', { code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID' });
  }
  const snapshot = value as Record<string, unknown>;
  if (
    snapshot.protocol !== PROJECT_SYNC_PROTOCOL ||
    snapshot.project_id !== expectedProjectID ||
    snapshot.home_instance_id !== expectedHomeInstanceID ||
    !Array.isArray(snapshot.jobs) ||
    snapshot.jobs.length > PROJECT_SYNC_MAX_FILES ||
    typeof snapshot.hash !== 'string' ||
    !/^[a-f0-9]{64}$/.test(snapshot.hash)
  ) {
    throw new ProjectSyncProtocolError('Invalid project job snapshot', { code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID' });
  }
  const jobs = snapshot.jobs.map(parsePortableProjectJob).sort((left, right) => left.id.localeCompare(right.id));
  if (new Set(jobs.map(job => job.id)).size !== jobs.length || hashPortableProjectJobs(jobs) !== snapshot.hash) {
    throw new ProjectSyncProtocolError('Project job snapshot hash or identities are invalid', {
      status: 422,
      code: 'PROJECT_SYNC_HASH_MISMATCH',
    });
  }
  return {
    protocol: PROJECT_SYNC_PROTOCOL,
    project_id: expectedProjectID,
    home_instance_id: expectedHomeInstanceID,
    generated_at: asIsoDate(snapshot.generated_at, 'snapshot date'),
    hash: snapshot.hash,
    jobs,
  };
}

export function parseProjectManifestEntry(value: unknown, profile: ProjectSyncProfileName): ProjectManifestEntry {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ProjectSyncProtocolError('Invalid project manifest entry', { code: 'PROJECT_SYNC_MANIFEST_INVALID' });
  }
  const entry = value as Record<string, unknown>;
  if (
    typeof entry.path !== 'string' ||
    typeof entry.sha256 !== 'string' ||
    !/^[a-f0-9]{64}$/.test(entry.sha256) ||
    typeof entry.modified_at !== 'string' ||
    Number.isNaN(Date.parse(entry.modified_at)) ||
    !Number.isSafeInteger(entry.size) ||
    Number(entry.size) < 0 ||
    Number(entry.size) > PROJECT_SYNC_MAX_FILE_BYTES
  ) {
    throw new ProjectSyncProtocolError('Invalid project manifest entry', { code: 'PROJECT_SYNC_MANIFEST_INVALID' });
  }
  const portablePath = normalizePortableRelativePath(entry.path);
  if (isProjectSyncPathExcluded(portablePath, profile)) {
    throw new ProjectSyncProtocolError('Project manifest includes an excluded path', {
      status: 403,
      code: 'PROJECT_SYNC_PATH_EXCLUDED',
      details: { path: portablePath },
    });
  }
  return {
    path: portablePath,
    sha256: entry.sha256,
    size: Number(entry.size),
    modified_at: entry.modified_at,
  };
}

export function assertProjectSyncQuota(entries: readonly ProjectManifestEntry[]) {
  if (entries.length > PROJECT_SYNC_MAX_FILES) {
    throw new ProjectSyncProtocolError('Project sync manifest contains too many files', {
      status: 413,
      code: 'PROJECT_SYNC_QUOTA_EXCEEDED',
    });
  }
  let total = 0;
  const paths = new Set<string>();
  for (const entry of entries) {
    if (paths.has(entry.path)) {
      throw new ProjectSyncProtocolError('Project sync manifest contains duplicate paths', {
        code: 'PROJECT_SYNC_MANIFEST_INVALID',
        details: { path: entry.path },
      });
    }
    paths.add(entry.path);
    total += entry.size;
    if (!Number.isSafeInteger(total) || total > PROJECT_SYNC_MAX_OPERATION_BYTES) {
      throw new ProjectSyncProtocolError('Project sync manifest exceeds the operation quota', {
        status: 413,
        code: 'PROJECT_SYNC_QUOTA_EXCEEDED',
      });
    }
  }
  return total;
}

export function parseProjectSyncManifest(
  value: unknown,
  expectedProjectID: string,
  expectedProfile: ProjectSyncProfileName,
): ProjectSyncManifest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ProjectSyncProtocolError('Invalid project sync manifest', { code: 'PROJECT_SYNC_MANIFEST_INVALID' });
  }
  const manifest = value as Record<string, unknown>;
  if (
    manifest.protocol !== PROJECT_SYNC_PROTOCOL ||
    manifest.project_id !== expectedProjectID ||
    manifest.profile !== expectedProfile ||
    typeof manifest.generated_at !== 'string' ||
    Number.isNaN(Date.parse(manifest.generated_at)) ||
    typeof manifest.hash !== 'string' ||
    !/^[a-f0-9]{64}$/.test(manifest.hash) ||
    !Array.isArray(manifest.files)
  ) {
    throw new ProjectSyncProtocolError('Invalid project sync manifest', { code: 'PROJECT_SYNC_MANIFEST_INVALID' });
  }
  const files = manifest.files.map(entry => parseProjectManifestEntry(entry, expectedProfile));
  assertProjectSyncQuota(files);
  const computedHash = hashManifestEntries(files);
  if (computedHash !== manifest.hash) {
    throw new ProjectSyncProtocolError('Project sync manifest hash is invalid', {
      status: 422,
      code: 'PROJECT_SYNC_HASH_MISMATCH',
    });
  }
  return {
    protocol: PROJECT_SYNC_PROTOCOL,
    project_id: expectedProjectID,
    profile: expectedProfile,
    generated_at: manifest.generated_at,
    hash: manifest.hash,
    files: files.sort((left, right) => left.path.localeCompare(right.path)),
  };
}

async function collectManifestFiles(
  projectRoot: string,
  absoluteDirectory: string,
  profile: ProjectSyncProfileName,
  files: ProjectManifestEntry[],
) {
  const children = await fs.readdir(absoluteDirectory, { withFileTypes: true }).catch(error => {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw error;
  });
  children.sort((left, right) => left.name.localeCompare(right.name));
  for (const child of children) {
    const absolutePath = path.join(absoluteDirectory, child.name);
    const relativePath = posixPath(path.relative(projectRoot, absolutePath));
    const stat = await fs.lstat(absolutePath);
    if (stat.isSymbolicLink()) continue;
    if (isProjectSyncPathExcluded(relativePath, profile, { directory: stat.isDirectory() })) continue;
    if (stat.isDirectory()) {
      await collectManifestFiles(projectRoot, absolutePath, profile, files);
      continue;
    }
    if (!stat.isFile()) continue;
    if (!Number.isSafeInteger(stat.size) || stat.size < 0 || stat.size > PROJECT_SYNC_MAX_FILE_BYTES) {
      throw new ProjectSyncProtocolError('Project file exceeds the sync size limit', {
        status: 413,
        code: 'PROJECT_SYNC_FILE_TOO_LARGE',
        details: { path: relativePath, size: stat.size },
      });
    }
    if (relativePath.startsWith('configs/') || relativePath.startsWith('runs/')) {
      await assertFileHasNoCredentialMaterial(absolutePath, relativePath, stat.size);
    }
    files.push({
      path: normalizePortableRelativePath(relativePath),
      size: stat.size,
      sha256: await sha256File(absolutePath),
      modified_at: stat.mtime.toISOString(),
    });
  }
}

export async function buildProjectSyncManifest(
  projectRoot: string,
  projectID: string,
  profile: ProjectSyncProfileName,
): Promise<ProjectSyncManifest> {
  if (!(PROJECT_SYNC_PROFILES as readonly string[]).includes(profile)) {
    throw new ProjectSyncProtocolError('Unsupported project sync profile');
  }
  const root = path.resolve(projectRoot);
  const files: ProjectManifestEntry[] = [];
  for (const includedPath of PROFILE_ZONES[profile]) {
    await collectManifestFiles(
      root,
      path.join(root, ...includedPath.split('/')),
      profile,
      files,
    );
  }
  files.sort((left, right) => left.path.localeCompare(right.path));
  assertProjectSyncQuota(files);
  return {
    protocol: PROJECT_SYNC_PROTOCOL,
    project_id: projectID,
    profile,
    generated_at: new Date().toISOString(),
    hash: hashManifestEntries(files),
    files,
  };
}

function entriesByPath(manifest: Pick<ProjectSyncManifest, 'files'> | null | undefined) {
  return new Map((manifest?.files || []).map(entry => [entry.path, entry]));
}

function sameEntry(left: ProjectManifestEntry | null | undefined, right: ProjectManifestEntry | null | undefined) {
  if (!left || !right) return !left && !right;
  return left.sha256 === right.sha256 && left.size === right.size;
}

export function diffProjectSyncManifests(source: ProjectSyncManifest, target: ProjectSyncManifest): ProjectManifestDiff {
  const sourceByPath = entriesByPath(source);
  const targetByPath = entriesByPath(target);
  const addOrUpdate: ProjectManifestEntry[] = [];
  const unchanged: string[] = [];
  for (const entry of source.files) {
    if (sameEntry(entry, targetByPath.get(entry.path))) unchanged.push(entry.path);
    else addOrUpdate.push(entry);
  }
  const deleted = [...targetByPath.keys()].filter(filePath => !sourceByPath.has(filePath)).sort();
  return { add_or_update: addOrUpdate, delete: deleted, unchanged };
}

export function detectProjectSyncConflicts(
  base: Pick<ProjectSyncManifest, 'files'> | null,
  home: Pick<ProjectSyncManifest, 'files'>,
  worker: Pick<ProjectSyncManifest, 'files'>,
) {
  const baseByPath = entriesByPath(base);
  const homeByPath = entriesByPath(home);
  const workerByPath = entriesByPath(worker);
  const allPaths = new Set([...baseByPath.keys(), ...homeByPath.keys(), ...workerByPath.keys()]);
  const conflicts: ProjectSyncConflict[] = [];
  for (const filePath of [...allPaths].sort()) {
    const baseEntry = baseByPath.get(filePath) || null;
    const homeEntry = homeByPath.get(filePath) || null;
    const workerEntry = workerByPath.get(filePath) || null;
    const homeChanged = !sameEntry(homeEntry, baseEntry);
    const workerChanged = !sameEntry(workerEntry, baseEntry);
    if (!homeChanged || !workerChanged || sameEntry(homeEntry, workerEntry)) continue;
    conflicts.push({
      path: filePath,
      kind: !homeEntry
        ? 'home-deleted-worker-modified'
        : !workerEntry
          ? 'worker-deleted-home-modified'
          : 'both-modified',
      base: baseEntry,
      home: homeEntry,
      worker: workerEntry,
    });
  }
  return conflicts;
}

export function deterministicKeepBothPath(relativePath: string, workerInstanceID: string) {
  const normalized = normalizePortableRelativePath(relativePath);
  const extension = path.posix.extname(normalized);
  const stem = extension ? normalized.slice(0, -extension.length) : normalized;
  const suffix = workerInstanceID.replace(/[^a-zA-Z0-9]/g, '').slice(0, 8) || 'worker';
  return `${stem}.worker-${suffix}${extension}`;
}

function validateHash(value: string) {
  if (!/^[a-f0-9]{64}$/.test(value)) {
    throw new ProjectSyncProtocolError('Invalid SHA-256 digest', { code: 'PROJECT_SYNC_HASH_INVALID' });
  }
  return value;
}

function syncMetadataRoot(projectRoot: string) {
  return path.join(projectRoot, '.aitk-sync');
}

function chunkPaths(projectRoot: string, sha256: string) {
  const hash = validateHash(sha256);
  const directory = path.join(syncMetadataRoot(projectRoot), 'chunks');
  return {
    directory,
    partial: path.join(directory, `${hash}.part`),
    complete: path.join(directory, `${hash}.blob`),
  };
}

export async function getProjectSyncChunkReceipt(projectRoot: string, sha256: string, total: number): Promise<ChunkReceipt> {
  validateHash(sha256);
  if (!Number.isSafeInteger(total) || total < 0 || total > PROJECT_SYNC_MAX_FILE_BYTES) {
    throw new ProjectSyncProtocolError('Invalid chunk total', { code: 'PROJECT_SYNC_SIZE_INVALID' });
  }
  const chunk = chunkPaths(projectRoot, sha256);
  await assertProjectSyncPathContained(projectRoot, chunk.complete);
  await assertProjectSyncPathContained(projectRoot, chunk.partial);
  const completeStat = await fs.stat(chunk.complete).catch(() => null);
  if (completeStat) {
    if (completeStat.size !== total) {
      throw new ProjectSyncProtocolError('A staged blob has an unexpected size', {
        status: 409,
        code: 'PROJECT_SYNC_CHUNK_MISMATCH',
      });
    }
    return { sha256, total, received: total, complete: true };
  }
  const partialStat = await fs.stat(chunk.partial).catch(() => null);
  const received = partialStat?.size ?? 0;
  if (received > total) {
    throw new ProjectSyncProtocolError('A staged blob exceeds its declared total', {
      status: 409,
      code: 'PROJECT_SYNC_CHUNK_MISMATCH',
    });
  }
  return { sha256, total, received, complete: false };
}

export async function appendProjectSyncChunk(
  projectRoot: string,
  sha256: string,
  total: number,
  offset: number,
  bytes: Uint8Array,
): Promise<ChunkReceipt> {
  if (bytes.byteLength > PROJECT_SYNC_CHUNK_BYTES) {
    throw new ProjectSyncProtocolError('Chunk exceeds the 8 MiB protocol limit', {
      status: 413,
      code: 'PROJECT_SYNC_CHUNK_TOO_LARGE',
    });
  }
  if (!Number.isSafeInteger(offset) || offset < 0 || offset + bytes.byteLength > total) {
    throw new ProjectSyncProtocolError('Invalid chunk byte range', { code: 'PROJECT_SYNC_RANGE_INVALID' });
  }
  const current = await getProjectSyncChunkReceipt(projectRoot, sha256, total);
  if (current.complete) return current;
  if (current.received !== offset) {
    throw new ProjectSyncProtocolError('Chunk offset does not match the resumable upload position', {
      status: 409,
      code: 'PROJECT_SYNC_OFFSET_MISMATCH',
      details: { expected_offset: current.received },
    });
  }
  const chunk = chunkPaths(projectRoot, sha256);
  await fs.mkdir(chunk.directory, { recursive: true });
  await fs.appendFile(chunk.partial, bytes);
  const received = offset + bytes.byteLength;
  if (received !== total) return { sha256, total, received, complete: false };
  const actualHash = await sha256File(chunk.partial);
  if (actualHash !== sha256) {
    await fs.rm(chunk.partial, { force: true });
    throw new ProjectSyncProtocolError('Completed upload failed SHA-256 verification', {
      status: 422,
      code: 'PROJECT_SYNC_HASH_MISMATCH',
    });
  }
  await fs.rename(chunk.partial, chunk.complete).catch(async error => {
    if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
    await fs.rm(chunk.partial, { force: true });
  });
  return { sha256, total, received, complete: true };
}

async function copyFileVerified(source: string, destination: string, expectedHash: string, expectedSize: number) {
  await fs.mkdir(path.dirname(destination), { recursive: true });
  await pipeline(createReadStream(source), createWriteStream(destination, { flags: 'wx' }));
  const stat = await fs.stat(destination);
  if (stat.size !== expectedSize || (await sha256File(destination)) !== expectedHash) {
    await fs.rm(destination, { force: true });
    throw new ProjectSyncProtocolError('Staged file failed verification', {
      status: 422,
      code: 'PROJECT_SYNC_HASH_MISMATCH',
    });
  }
}

type ProjectSyncCommitMutation = {
  relative_path: string;
  kind: 'write' | 'delete';
  staged_path: string | null;
};

type ProjectSyncCommitJournalRecord = ProjectSyncCommitMutation & {
  backup_path: string;
  had_existing: boolean;
  backed_up: boolean;
  installed: boolean;
};

type ProjectSyncCommitJournal = {
  version: 1;
  sequence: number;
  operation_id: string;
  profile: ProjectSyncProfileName;
  phase: 'prepared' | 'backing-up' | 'backed-up' | 'installing' | 'completed';
  records: ProjectSyncCommitJournalRecord[];
};

export type ProjectSyncCommitTestOptions = {
  failAfterInstalledFiles?: number;
};

function projectCommitRoot(projectRoot: string, operationID: string) {
  return path.join(syncMetadataRoot(projectRoot), 'commits', operationID);
}

async function writeCommitJournal(commitRoot: string, journal: ProjectSyncCommitJournal) {
  journal.sequence += 1;
  const journalPath = path.join(commitRoot, `journal-${String(journal.sequence).padStart(8, '0')}.json`);
  const handle = await fs.open(journalPath, 'wx');
  try {
    await handle.writeFile(JSON.stringify(journal), 'utf8');
    await handle.sync();
  } finally {
    await handle.close();
  }
}

async function readLatestCommitJournal(commitRoot: string): Promise<ProjectSyncCommitJournal | null> {
  const names = await fs.readdir(commitRoot).catch(() => []);
  const journals = names.filter(name => /^journal-\d{8}\.json$/.test(name)).sort();
  if (journals.length === 0) return null;
  try {
    return JSON.parse(await fs.readFile(path.join(commitRoot, journals[journals.length - 1]), 'utf8')) as ProjectSyncCommitJournal;
  } catch {
    throw new ProjectSyncProtocolError('Project sync commit journal is unreadable', {
      status: 409,
      code: 'PROJECT_SYNC_COMMIT_RECOVERY_REQUIRED',
    });
  }
}

export async function recoverProjectSyncCommit(projectRoot: string, operationID: string) {
  if (!operationID || !/^[a-zA-Z0-9._-]+$/.test(operationID)) {
    throw new ProjectSyncProtocolError('Invalid sync operation ID');
  }
  const commitRoot = projectCommitRoot(projectRoot, operationID);
  await assertProjectSyncPathContained(projectRoot, commitRoot);
  const journal = await readLatestCommitJournal(commitRoot);
  if (!journal) {
    await fs.rm(commitRoot, { recursive: true, force: true });
    return { recovered: false };
  }
  if (journal.operation_id !== operationID || journal.version !== 1) {
    throw new ProjectSyncProtocolError('Project sync commit journal does not match the operation', {
      status: 409,
      code: 'PROJECT_SYNC_COMMIT_RECOVERY_REQUIRED',
    });
  }
  if (journal.phase !== 'completed') {
    for (const record of [...journal.records].reverse()) {
      const target = resolveProjectSyncPath(projectRoot, record.relative_path, journal.profile);
      await assertProjectSyncPathContained(projectRoot, target);
      const backup = path.join(commitRoot, record.backup_path);
      await assertProjectSyncPathContained(projectRoot, backup);
      const backupStat = await fs.lstat(backup).catch(() => null);
      if (backupStat?.isFile() && !backupStat.isSymbolicLink()) {
        await fs.rm(target, { force: true });
        await fs.mkdir(path.dirname(target), { recursive: true });
        await fs.rename(backup, target);
      } else if (!record.had_existing && (record.installed || journal.phase === 'installing')) {
        await fs.rm(target, { force: true });
      }
    }
  }
  await fs.rm(commitRoot, { recursive: true, force: true });
  return { recovered: journal.phase !== 'completed' };
}

export async function commitProjectSyncPlan(
  projectRoot: string,
  plan: ProjectCommitPlan,
  testOptions: ProjectSyncCommitTestOptions = {},
) {
  if (!plan.operation_id || !/^[a-zA-Z0-9._-]+$/.test(plan.operation_id)) {
    throw new ProjectSyncProtocolError('Invalid sync operation ID');
  }
  assertProjectSyncQuota(plan.files);
  const commitRoot = projectCommitRoot(projectRoot, plan.operation_id);
  await assertProjectSyncPathContained(projectRoot, commitRoot);
  await recoverProjectSyncCommit(projectRoot, plan.operation_id);
  const stagedRoot = path.join(commitRoot, 'staged');
  const backupRoot = path.join(commitRoot, 'backups');
  await fs.mkdir(stagedRoot, { recursive: true });
  await fs.mkdir(backupRoot, { recursive: true });

  const mutations = new Map<string, ProjectSyncCommitMutation>();
  try {
    for (const [index, entry] of plan.files.entries()) {
      const target = resolveProjectSyncPath(projectRoot, entry.path, plan.profile);
      await assertProjectSyncPathContained(projectRoot, target);
      const chunk = chunkPaths(projectRoot, entry.sha256);
      const receipt = await getProjectSyncChunkReceipt(projectRoot, entry.sha256, entry.size);
      if (!receipt.complete) {
        throw new ProjectSyncProtocolError('Commit references an incomplete blob', {
          status: 409,
          code: 'PROJECT_SYNC_BLOB_INCOMPLETE',
          details: { path: entry.path, received: receipt.received },
        });
      }
      const stagedPath = path.join(stagedRoot, `${index}.new`);
      await copyFileVerified(chunk.complete, stagedPath, entry.sha256, entry.size);
      mutations.set(entry.path, { relative_path: entry.path, kind: 'write', staged_path: path.relative(commitRoot, stagedPath) });
    }

    for (const [index, preserve] of (plan.preserve_paths || []).entries()) {
      const source = resolveProjectSyncPath(projectRoot, preserve.path, plan.profile);
      const destination = resolveProjectSyncPath(projectRoot, preserve.preserve_as, plan.profile);
      await assertProjectSyncPathContained(projectRoot, source);
      await assertProjectSyncPathContained(projectRoot, destination);
      const stat = await fs.lstat(source).catch(() => null);
      if (!stat) continue;
      if (!stat.isFile() || stat.isSymbolicLink()) {
        throw new ProjectSyncProtocolError('Keep-both source is not a regular file', {
          status: 409,
          code: 'PROJECT_SYNC_PATH_INVALID',
        });
      }
      if (mutations.has(preserve.preserve_as)) {
        throw new ProjectSyncProtocolError('Project sync commit has conflicting destination paths', {
          status: 409,
          code: 'PROJECT_SYNC_COMMIT_CONFLICT',
          details: { path: preserve.preserve_as },
        });
      }
      const stagedPath = path.join(stagedRoot, `preserve-${index}.new`);
      const sourceHash = await sha256File(source);
      await copyFileVerified(source, stagedPath, sourceHash, stat.size);
      mutations.set(preserve.preserve_as, {
        relative_path: preserve.preserve_as,
        kind: 'write',
        staged_path: path.relative(commitRoot, stagedPath),
      });
    }

    for (const relativePath of plan.delete_paths || []) {
      const target = resolveProjectSyncPath(projectRoot, relativePath, plan.profile);
      await assertProjectSyncPathContained(projectRoot, target);
      if (!mutations.has(relativePath)) {
        mutations.set(relativePath, { relative_path: relativePath, kind: 'delete', staged_path: null });
      }
    }

    const records: ProjectSyncCommitJournalRecord[] = [];
    for (const [index, mutation] of [...mutations.values()].sort((left, right) => left.relative_path.localeCompare(right.relative_path)).entries()) {
      const target = resolveProjectSyncPath(projectRoot, mutation.relative_path, plan.profile);
      const stat = await fs.lstat(target).catch(() => null);
      if (stat && (!stat.isFile() || stat.isSymbolicLink())) {
        throw new ProjectSyncProtocolError('Project sync target is not a regular file', {
          status: 409,
          code: 'PROJECT_SYNC_PATH_INVALID',
          details: { path: mutation.relative_path },
        });
      }
      records.push({
        ...mutation,
        backup_path: path.relative(commitRoot, path.join(backupRoot, `${index}.bak`)),
        had_existing: !!stat,
        backed_up: false,
        installed: false,
      });
    }

    const journal: ProjectSyncCommitJournal = {
      version: 1,
      sequence: 0,
      operation_id: plan.operation_id,
      profile: plan.profile,
      phase: 'prepared',
      records,
    };
    await writeCommitJournal(commitRoot, journal);
    journal.phase = 'backing-up';
    await writeCommitJournal(commitRoot, journal);
    for (const record of journal.records) {
      const target = resolveProjectSyncPath(projectRoot, record.relative_path, plan.profile);
      if (record.had_existing) {
        const backup = path.join(commitRoot, record.backup_path);
        await fs.mkdir(path.dirname(backup), { recursive: true });
        await fs.rename(target, backup);
      }
      record.backed_up = true;
      await writeCommitJournal(commitRoot, journal);
    }
    journal.phase = 'backed-up';
    await writeCommitJournal(commitRoot, journal);
    journal.phase = 'installing';
    await writeCommitJournal(commitRoot, journal);
    let installedFiles = 0;
    for (const record of journal.records) {
      const target = resolveProjectSyncPath(projectRoot, record.relative_path, plan.profile);
      if (record.kind === 'write' && record.staged_path) {
        const staged = path.join(commitRoot, record.staged_path);
        await fs.mkdir(path.dirname(target), { recursive: true });
        await fs.rename(staged, target);
      }
      record.installed = true;
      installedFiles += 1;
      await writeCommitJournal(commitRoot, journal);
      if (
        testOptions.failAfterInstalledFiles != null &&
        installedFiles >= testOptions.failAfterInstalledFiles
      ) {
        throw new ProjectSyncProtocolError('Injected project sync commit failure', {
          status: 500,
          code: 'PROJECT_SYNC_TEST_FAILURE',
        });
      }
    }
    journal.phase = 'completed';
    await writeCommitJournal(commitRoot, journal);
    await fs.rm(commitRoot, { recursive: true, force: true });
  } catch (error) {
    await recoverProjectSyncCommit(projectRoot, plan.operation_id).catch(recoveryError => {
      console.error('Project sync commit rollback failed:', recoveryError);
    });
    throw error;
  }
}

export function parseHttpByteRange(rangeHeader: string | null, size: number) {
  if (!rangeHeader) return { start: 0, end: Math.max(0, size - 1), partial: false };
  const match = /^bytes=(\d*)-(\d*)$/i.exec(rangeHeader.trim());
  if (!match || size < 0) throw new ProjectSyncProtocolError('Invalid byte range', { status: 416, code: 'PROJECT_SYNC_RANGE_INVALID' });
  let start: number;
  let end: number;
  if (!match[1]) {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) throw new ProjectSyncProtocolError('Invalid byte range', { status: 416 });
    start = Math.max(0, size - suffix);
    end = Math.max(0, size - 1);
  } else {
    start = Number(match[1]);
    end = match[2] ? Number(match[2]) : Math.max(0, size - 1);
  }
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || start >= size) {
    throw new ProjectSyncProtocolError('Requested range is not satisfiable', { status: 416, code: 'PROJECT_SYNC_RANGE_INVALID' });
  }
  return { start, end: Math.min(end, size - 1), partial: true };
}

export async function writeProjectSyncBaseManifest(
  projectRoot: string,
  workerID: string,
  profile: ProjectSyncProfileName,
  manifest: ProjectSyncManifest,
) {
  const safeWorker = workerID.replace(/[^a-zA-Z0-9._-]/g, '_');
  const directory = path.join(syncMetadataRoot(projectRoot), 'bases');
  const target = path.join(directory, `${safeWorker}.${profile}.json`);
  await assertProjectSyncPathContained(projectRoot, target);
  await fs.mkdir(directory, { recursive: true });
  const temporary = `${target}.${process.pid}.tmp`;
  await fs.writeFile(temporary, JSON.stringify(manifest), 'utf8');
  await fs.rename(temporary, target).catch(async error => {
    if (process.platform !== 'win32') throw error;
    await fs.rm(target, { force: true });
    await fs.rename(temporary, target);
  });
}

export async function readProjectSyncBaseManifest(
  projectRoot: string,
  workerID: string,
  profile: ProjectSyncProfileName,
): Promise<ProjectSyncManifest | null> {
  const safeWorker = workerID.replace(/[^a-zA-Z0-9._-]/g, '_');
  const target = path.join(syncMetadataRoot(projectRoot), 'bases', `${safeWorker}.${profile}.json`);
  try {
    await assertProjectSyncPathContained(projectRoot, target);
    const value: unknown = JSON.parse(await fs.readFile(target, 'utf8'));
    if (!value || typeof value !== 'object') return null;
    const manifest = value as Partial<ProjectSyncManifest>;
    if (
      manifest.protocol !== PROJECT_SYNC_PROTOCOL ||
      manifest.profile !== profile ||
      !Array.isArray(manifest.files) ||
      typeof manifest.hash !== 'string'
    ) {
      return null;
    }
    return manifest as ProjectSyncManifest;
  } catch {
    return null;
  }
}
