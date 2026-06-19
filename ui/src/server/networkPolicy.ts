import dns from 'dns/promises';
import net from 'net';
import { db } from './db';

export const OFFLINE_MODE_SETTING_KEY = 'OFFLINE_MODE';
export const OFFLINE_ALLOWED_HOSTS_ENV = 'AITK_OFFLINE_ALLOWED_HOSTS';

const OFFLINE_MODE_ENV_KEYS = ['AITK_OFFLINE_MODE', 'AI_TOOLKIT_OFFLINE_MODE'] as const;
const REMOTE_OLLAMA_WORKERS_SETTING_KEY = 'REMOTE_OLLAMA_WORKERS';
const TRUTHY_VALUES = new Set(['1', 'true', 'yes', 'on']);
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);
const MAX_OFFLINE_REDIRECTS = 20;

const localPrivateBlockList = new net.BlockList();
localPrivateBlockList.addSubnet('127.0.0.0', 8, 'ipv4');
localPrivateBlockList.addSubnet('10.0.0.0', 8, 'ipv4');
localPrivateBlockList.addSubnet('172.16.0.0', 12, 'ipv4');
localPrivateBlockList.addSubnet('192.168.0.0', 16, 'ipv4');
localPrivateBlockList.addSubnet('169.254.0.0', 16, 'ipv4');
localPrivateBlockList.addAddress('::1', 'ipv6');
localPrivateBlockList.addSubnet('fc00::', 7, 'ipv6');
localPrivateBlockList.addSubnet('fe80::', 10, 'ipv6');

export class OfflineModeError extends Error {
  status = 403;
  code = 'OFFLINE_MODE_BLOCKED';

  constructor(message: string) {
    super(message);
    this.name = 'OfflineModeError';
  }
}

function boolValue(value: unknown, defaultValue = false) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['false', '0', 'off', 'disabled', 'no'].includes(normalized)) return false;
    if (['true', '1', 'on', 'enabled', 'yes'].includes(normalized)) return true;
  }
  return defaultValue;
}

export function isOfflineModeLockedByEnv(env: NodeJS.ProcessEnv = process.env) {
  return OFFLINE_MODE_ENV_KEYS.some(key =>
    TRUTHY_VALUES.has(
      String(env[key] || '')
        .trim()
        .toLowerCase(),
    ),
  );
}

export async function isOfflineModeEnabled() {
  if (isOfflineModeLockedByEnv()) return true;
  const row = await db.settings.get(OFFLINE_MODE_SETTING_KEY);
  return boolValue(row?.value, false);
}

export async function getOfflineModeState() {
  const locked = isOfflineModeLockedByEnv();
  const row = await db.settings.get(OFFLINE_MODE_SETTING_KEY);
  return {
    enabled: locked || boolValue(row?.value, false),
    lockedByEnv: locked,
  };
}

export function normalizeHostname(value: unknown) {
  const hostname = String(value || '')
    .trim()
    .toLowerCase()
    .replace(/^\[|\]$/g, '');
  return hostname.endsWith('.') ? hostname.slice(0, -1) : hostname;
}

function hostFromUrl(value: unknown) {
  try {
    return normalizeHostname(new URL(String(value)).hostname);
  } catch {
    return '';
  }
}

function isIpAddress(value: string) {
  return net.isIP(value) !== 0;
}

function ipv4MappedAddress(value: string) {
  const match = value.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/i);
  return match?.[1] || null;
}

export function isLocalPrivateIp(value: string) {
  const normalized = normalizeHostname(value);
  if (!normalized) return false;

  const mapped = ipv4MappedAddress(normalized);
  const address = mapped || normalized;
  const family = net.isIP(address);
  if (family === 0) return false;
  if (address === '0.0.0.0' || address === '::') return false;
  return localPrivateBlockList.check(address, family === 4 ? 'ipv4' : 'ipv6');
}

async function resolveHostAddresses(hostname: string) {
  const host = normalizeHostname(hostname);
  if (!host) return [];
  if (isIpAddress(host)) return [host];
  const addresses = await dns.lookup(host, { all: true, verbatim: true });
  return Array.from(new Set(addresses.map(item => item.address)));
}

function envAllowedHostnames() {
  return new Set(
    String(process.env[OFFLINE_ALLOWED_HOSTS_ENV] || '')
      .split(',')
      .map(normalizeHostname)
      .filter(Boolean),
  );
}

type RawRemoteOllamaWorker = {
  base_url?: string;
  offline_bypass_enabled?: boolean;
};

export async function getOfflineBypassHostnames() {
  const hosts = envAllowedHostnames();

  const workerNodes = await db.workerNodes.list().catch(() => []);
  for (const worker of workerNodes) {
    if ((worker as any).offline_bypass_enabled) {
      const host = hostFromUrl(worker.base_url);
      if (host) hosts.add(host);
    }
  }

  const remoteOllamaRow = await db.settings.get(REMOTE_OLLAMA_WORKERS_SETTING_KEY).catch(() => null);
  if (remoteOllamaRow?.value) {
    try {
      const workers = JSON.parse(remoteOllamaRow.value) as RawRemoteOllamaWorker[];
      if (Array.isArray(workers)) {
        for (const worker of workers) {
          if (worker?.offline_bypass_enabled) {
            const host = hostFromUrl(worker.base_url);
            if (host) hosts.add(host);
          }
        }
      }
    } catch {
      // Ignore malformed legacy settings; they should not grant bypasses.
    }
  }

  return hosts;
}

function describeUrl(url: URL) {
  return `${url.protocol}//${url.host}`;
}

export async function assertUrlAllowedByOfflineMode(input: string | URL, feature = 'request') {
  if (!(await isOfflineModeEnabled())) return;

  const url = input instanceof URL ? input : new URL(String(input));
  if (!['http:', 'https:', 'ws:', 'wss:'].includes(url.protocol)) {
    throw new OfflineModeError(`Offline mode blocked ${feature}: unsupported URL protocol ${url.protocol}`);
  }

  const hostname = normalizeHostname(url.hostname);
  const bypassHosts = await getOfflineBypassHostnames();
  if (bypassHosts.has(hostname)) return;

  let addresses: string[];
  try {
    addresses = await resolveHostAddresses(hostname);
  } catch (error) {
    throw new OfflineModeError(
      `Offline mode blocked ${feature} to ${describeUrl(url)}: DNS lookup failed${
        error instanceof Error && error.message ? `: ${error.message}` : ''
      }`,
    );
  }

  if (!addresses.length) {
    throw new OfflineModeError(
      `Offline mode blocked ${feature} to ${describeUrl(url)}: DNS lookup returned no addresses`,
    );
  }

  const blocked = addresses.filter(address => !isLocalPrivateIp(address));
  if (blocked.length > 0) {
    throw new OfflineModeError(
      `Offline mode blocked ${feature} to ${describeUrl(url)}: resolved outside local/private IP space (${blocked.join(', ')})`,
    );
  }
}

function requestUrl(input: RequestInfo | URL) {
  if (typeof input === 'string' || input instanceof URL) return String(input);
  if (typeof Request !== 'undefined' && input instanceof Request) return input.url;
  return String(input);
}

function requestMethod(input: RequestInfo | URL, init?: RequestInit) {
  const method =
    init?.method || (typeof Request !== 'undefined' && input instanceof Request ? input.method : undefined) || 'GET';
  return method.toUpperCase();
}

function cloneHeadersWithoutBodyHeaders(headers: Headers) {
  const next = new Headers(headers);
  next.delete('content-length');
  next.delete('content-type');
  return next;
}

function redirectTarget(response: Response, currentUrl: string) {
  if (!REDIRECT_STATUSES.has(response.status)) return null;
  const location = response.headers.get('location');
  if (!location) return null;
  return new URL(location, currentUrl).toString();
}

export async function guardedFetch(input: RequestInfo | URL, init?: RequestInit, feature = 'request') {
  if (!(await isOfflineModeEnabled())) {
    return fetch(input, init);
  }

  const url =
    typeof input === 'string' || input instanceof URL
      ? input
      : typeof Request !== 'undefined' && input instanceof Request
        ? input.url
        : String(input);
  await assertUrlAllowedByOfflineMode(url, feature);

  if (init?.redirect === 'manual' || init?.redirect === 'error') {
    return fetch(input, init);
  }

  let currentInput = input;
  let currentInit = init;
  let currentUrl = requestUrl(currentInput);
  let currentMethod = requestMethod(currentInput, currentInit);

  for (let redirectCount = 0; redirectCount <= MAX_OFFLINE_REDIRECTS; redirectCount += 1) {
    const response = await fetch(currentInput, { ...currentInit, redirect: 'manual' });
    const nextUrl = redirectTarget(response, currentUrl);
    if (!nextUrl) return response;

    await assertUrlAllowedByOfflineMode(nextUrl, `${feature} redirect`);

    if (redirectCount === MAX_OFFLINE_REDIRECTS) {
      throw new OfflineModeError(`Offline mode blocked ${feature}: too many redirects`);
    }

    const nextHeaders = new Headers(currentInit?.headers);
    const shouldSwitchToGet =
      response.status === 303 || ((response.status === 301 || response.status === 302) && currentMethod === 'POST');
    if (shouldSwitchToGet) {
      currentInit = {
        ...currentInit,
        method: 'GET',
        body: undefined,
        headers: cloneHeadersWithoutBodyHeaders(nextHeaders),
      };
      currentMethod = 'GET';
    } else {
      currentInit = {
        ...currentInit,
        headers: nextHeaders,
      };
    }
    currentInput = nextUrl;
    currentUrl = nextUrl;
  }

  throw new OfflineModeError(`Offline mode blocked ${feature}: too many redirects`);
}

export function offlineChildProcessEnv(enabled: boolean, allowedHosts: Iterable<string> = []) {
  if (!enabled) return {};
  const hosts = Array.from(new Set(Array.from(allowedHosts).map(normalizeHostname).filter(Boolean)));
  return {
    AITK_OFFLINE_MODE: '1',
    HF_HUB_OFFLINE: '1',
    TRANSFORMERS_OFFLINE: '1',
    HF_DATASETS_OFFLINE: '1',
    ...(hosts.length > 0 ? { [OFFLINE_ALLOWED_HOSTS_ENV]: hosts.join(',') } : {}),
  };
}

export function hostnamesFromUrls(values: Iterable<string | null | undefined>) {
  const hosts = new Set<string>();
  for (const value of values) {
    const host = value ? hostFromUrl(value) : '';
    if (host) hosts.add(host);
  }
  return hosts;
}
