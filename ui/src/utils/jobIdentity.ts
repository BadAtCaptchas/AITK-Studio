import type { Job } from '../types';

export function jobStorageKey(job: Pick<Job, 'name' | 'storage_key'>): string {
  return job.storage_key || job.name;
}

export function validJobName(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value.length > 0 &&
    value.length <= 180 &&
    value === value.trim() &&
    !/[<>:"/\\|?*\u0000-\u001f\u007f]/.test(value) &&
    !/[. ]$/.test(value) &&
    !value.includes('..') &&
    !/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(value)
  );
}

export function deviceIds(value: unknown): string[] {
  if (value === 'mps' || value === 'cpu') return [value];
  if (typeof value !== 'string' || !/^\d+(,\d+)*$/.test(value)) throw new Error('Invalid device selection');
  const ids = value.split(',').map(id => String(Number(id)));
  if (ids.some(id => !Number.isSafeInteger(Number(id)))) throw new Error('Invalid device index');
  return [...new Set(ids)];
}
export function devicesOverlap(left: string, right: string): boolean {
  const used = new Set(deviceIds(left));
  return deviceIds(right).some(id => used.has(id));
}
