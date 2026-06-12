import { execFile } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { promisify } from 'util';
import { TOOLKIT_ROOT } from '../paths';

export type RepoUpdateState =
  | 'pending'
  | 'checking'
  | 'up_to_date'
  | 'update_available'
  | 'unknown_current'
  | 'updating'
  | 'restarting'
  | 'updated'
  | 'update_failed'
  | 'update_blocked'
  | 'update_conflict'
  | 'error'
  | 'unsupported'
  | 'disabled'
  | 'stopped';

export interface RepoUpdateStatus {
  schemaVersion: 1;
  state: RepoUpdateState;
  message: string;
  updatedAt: string;
  checkedAt?: string | null;
  startedAt?: string | null;
  lastSuccessfulCheckAt?: string | null;
  nextCheckAt?: string | null;
  trigger?: 'startup' | 'schedule' | 'manual' | string;
  updaterPid?: number | null;
  updaterGeneration?: number | null;
  intervalMinutes?: number | null;
  branch?: string | null;
  upstream?: string | null;
  installKind?: 'git' | 'archive' | string | null;
  repoFullName?: string | null;
  repoWebUrl?: string | null;
  downloadUrl?: string | null;
  latestVersion?: string | null;
  latestReleaseUrl?: string | null;
  latestReleasePublishedAt?: string | null;
  remote?: string | null;
  remoteWebUrl?: string | null;
  remoteCommitDate?: string | null;
  sourceRemote?: string | null;
  sourceRemoteWebUrl?: string | null;
  sourceRemoteMatchesCanonical?: boolean | null;
  compareUrl?: string | null;
  compareMode?: 'commit' | 'version' | 'none' | string | null;
  localVersion?: string | null;
  localCommit?: string | null;
  localShortCommit?: string | null;
  remoteCommit?: string | null;
  remoteShortCommit?: string | null;
  recentCommits?: RepoUpdateCommit[];
  ahead?: number | null;
  behind?: number | null;
  canApplyUpdate?: boolean | null;
  applyUpdateUnavailableReason?: string | null;
  updateStartedAt?: string | null;
  updateCompletedAt?: string | null;
  updateStep?: string | null;
  updateError?: string | null;
  restartStartedAt?: string | null;
  restartStep?: string | null;
  restartPid?: number | null;
  restartChildPid?: number | null;
  restartError?: string | null;
  previousLocalCommit?: string | null;
  stashCreated?: boolean | null;
  stashRef?: string | null;
  localChangesRestored?: boolean | null;
  needsRestart?: boolean | null;
  error?: string | null;
  stale?: boolean;
}

export interface RepoUpdateCommit {
  sha: string;
  shortSha: string;
  message: string;
  body?: string | null;
  authorName?: string | null;
  authorDate?: string | null;
  committerName?: string | null;
  committerDate?: string | null;
  url?: string | null;
}

const TMP_ROOT = path.join(TOOLKIT_ROOT, '.tmp');
const STATUS_PATH = path.join(TMP_ROOT, 'repo-update-status.json');
const REQUEST_PATH = path.join(TMP_ROOT, 'repo-update-request.json');
const STALE_CHECKING_MS = 10 * 60 * 1000;
const GIT_REMOTE_TIMEOUT_MS = 5000;
const DEFAULT_REPO_OWNER = 'rmcc3';
const DEFAULT_REPO_NAME = 'ai-toolkit-revamped';
const execFileAsync = promisify(execFile);

function nowIso() {
  return new Date().toISOString();
}

function redactRemoteCredentials(remoteUrl?: string | null) {
  const raw = (remoteUrl || '').trim();
  if (!raw) return null;

  return raw.replace(/^(https?:\/\/)(?:[^/?#]*@)/i, '$1');
}

function normalizeRemoteWebUrl(remoteUrl?: string | null) {
  const raw = redactRemoteCredentials(remoteUrl);
  if (!raw) return null;

  const sshMatch = raw.match(/^git@([^:]+):(.+?)(?:\.git)?$/);
  if (sshMatch) {
    return `https://${sshMatch[1]}/${sshMatch[2]}`;
  }

  const httpMatch = raw.match(/^(https?:\/\/.+?)(?:\.git)?$/);
  if (httpMatch) {
    return httpMatch[1];
  }

  return null;
}

function normalizeRepoUrlForCompare(url?: string | null) {
  return (url || '').replace(/\.git$/, '').replace(/\/$/, '').toLowerCase();
}

function getDefaultRepoWebUrl() {
  const owner = (process.env.AITK_UPDATE_REPO_OWNER || DEFAULT_REPO_OWNER).trim();
  const name = (process.env.AITK_UPDATE_REPO_NAME || DEFAULT_REPO_NAME).trim();
  return `https://github.com/${owner}/${name}`;
}

async function readLiveSourceRemote() {
  try {
    const result = await execFileAsync('git', ['remote', 'get-url', 'origin'], {
      cwd: TOOLKIT_ROOT,
      env: {
        ...process.env,
        GIT_TERMINAL_PROMPT: '0',
      },
      timeout: GIT_REMOTE_TIMEOUT_MS,
      windowsHide: true,
    });
    return result.stdout.trim() || null;
  } catch {
    return null;
  }
}

async function withLiveSourceRemote(status: RepoUpdateStatus) {
  const liveSourceRemote = await readLiveSourceRemote();
  const sourceRemote = redactRemoteCredentials(liveSourceRemote);
  if (!sourceRemote) {
    return status;
  }

  const sourceRemoteWebUrl = normalizeRemoteWebUrl(sourceRemote);
  const repoWebUrl = status.repoWebUrl || status.remoteWebUrl || getDefaultRepoWebUrl();

  return {
    ...status,
    sourceRemote,
    sourceRemoteWebUrl,
    sourceRemoteMatchesCanonical: sourceRemoteWebUrl
      ? normalizeRepoUrlForCompare(sourceRemoteWebUrl) === normalizeRepoUrlForCompare(repoWebUrl)
      : null,
  };
}

function defaultStatus(): RepoUpdateStatus {
  return {
    schemaVersion: 1,
    state: 'pending',
    message: 'Waiting for update checker',
    updatedAt: nowIso(),
    checkedAt: null,
    nextCheckAt: null,
    updaterPid: null,
  };
}

function normalizeStatus(raw: unknown): RepoUpdateStatus {
  if (!raw || typeof raw !== 'object') {
    return defaultStatus();
  }

  const status = raw as Partial<RepoUpdateStatus>;
  const normalized: RepoUpdateStatus = {
    ...defaultStatus(),
    ...status,
    schemaVersion: 1,
    state: status.state || 'pending',
    message: status.message || 'Waiting for update checker',
    updatedAt: status.updatedAt || nowIso(),
  };

  normalized.sourceRemote = redactRemoteCredentials(normalized.sourceRemote);
  normalized.sourceRemoteWebUrl = normalizeRemoteWebUrl(normalized.sourceRemoteWebUrl);

  if (normalized.state === 'checking') {
    const updatedAt = new Date(normalized.updatedAt).getTime();
    if (Number.isFinite(updatedAt) && Date.now() - updatedAt > STALE_CHECKING_MS) {
      normalized.state = 'error';
      normalized.message = 'Update checker stopped while checking';
      normalized.stale = true;
    }
  }

  return normalized;
}

async function readJson(filePath: string) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch {
    return null;
  }
}

async function writeJsonAtomic(filePath: string, value: unknown) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tmpPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  await fs.writeFile(tmpPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  await fs.rename(tmpPath, filePath);
}

export async function getRepoUpdateStatus() {
  return withLiveSourceRemote(normalizeStatus(await readJson(STATUS_PATH)));
}

export type RepoUpdateRequestAction = 'check' | 'apply' | 'restart';

export async function requestRepoUpdateCheck(action: RepoUpdateRequestAction = 'check') {
  const request = {
    action,
    requestedAt: nowIso(),
    requestedBy: 'ui',
    nonce: `${process.pid}-${Date.now()}`,
  };
  await writeJsonAtomic(REQUEST_PATH, request);
  return {
    success: true,
    request,
    status: await getRepoUpdateStatus(),
  };
}
