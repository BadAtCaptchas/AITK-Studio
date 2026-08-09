import * as nodeModule from 'module';
import os from 'os';
import path from 'path';

type UnknownRecord = Record<string, unknown>;

export interface MacstatsModule {
  getCpuDataSync(): unknown;
  getFanDataSync(): unknown;
  getGpuDataSync(): unknown;
  getPowerDataSync(): unknown;
  getRAMUsageSync(): unknown;
}

let cachedModule: MacstatsModule | null | undefined;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function numberFromRecord(value: unknown, key: string, fallback = 0): number {
  if (!isRecord(value)) return fallback;
  const candidate = value[key];
  return typeof candidate === 'number' && Number.isFinite(candidate) ? candidate : fallback;
}

export function firstFanRpm(value: unknown): number {
  if (!isRecord(value)) return 0;
  for (const fan of Object.values(value)) {
    const rpm = numberFromRecord(fan, 'rpm', Number.NaN);
    if (Number.isFinite(rpm)) return rpm;
  }
  return 0;
}

export function parseMacstatsModule(value: unknown): MacstatsModule | null {
  if (!isRecord(value)) return null;
  const methodNames = [
    'getCpuDataSync',
    'getFanDataSync',
    'getGpuDataSync',
    'getPowerDataSync',
    'getRAMUsageSync',
  ] as const;
  if (!methodNames.every(methodName => typeof value[methodName] === 'function')) return null;
  return value as unknown as MacstatsModule;
}

/**
 * Load the native, macOS-only optional dependency without allowing Next's
 * bundler to turn it into a hard dependency on other platforms.
 */
export function loadMacstats(): MacstatsModule | null {
  if (cachedModule !== undefined) return cachedModule;
  if (os.platform() !== 'darwin') {
    cachedModule = null;
    return cachedModule;
  }

  try {
    const moduleNamespace: unknown = nodeModule;
    if (!isRecord(moduleNamespace)) throw new Error('Node module namespace is unavailable');
    const createRequireValue = moduleNamespace['create' + 'Require'];
    if (typeof createRequireValue !== 'function') throw new Error('createRequire is unavailable');
    const createRequire = createRequireValue as typeof nodeModule.createRequire;
    const nativeRequire = createRequire(path.join(process.cwd(), 'package.json'));
    cachedModule = parseMacstatsModule(nativeRequire('macstats'));
    if (!cachedModule) throw new Error('macstats has an unexpected API');
  } catch (error) {
    console.warn('macstats not available:', error);
    cachedModule = null;
  }

  return cachedModule ?? null;
}
