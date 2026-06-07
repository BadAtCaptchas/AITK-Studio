const DATASET_DEFAULT_PREFIX = 'config.process[0].datasets[x].';

function cloneDefaultValue<T>(value: T): T {
  if (value == null || typeof value !== 'object') return value;
  return JSON.parse(JSON.stringify(value));
}

function selectedDefaultValue(value: any) {
  return Array.isArray(value) ? value[0] : value;
}

function parsePath(path: string) {
  const parts: Array<string | number> = [];
  const re = /([^[.\]]+)|\[(\d+)\]/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(path)) !== null) {
    if (match[1] !== undefined) parts.push(match[1]);
    if (match[2] !== undefined) parts.push(Number(match[2]));
  }
  return parts;
}

function setPathValue(target: Record<string, any>, path: string, value: unknown) {
  const parts = parsePath(path);
  if (parts.length === 0) return;

  let cursor: any = target;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    if (cursor[part] == null) {
      cursor[part] = typeof parts[i + 1] === 'number' ? [] : {};
    }
    cursor = cursor[part];
  }

  cursor[parts[parts.length - 1] as any] = cloneDefaultValue(value);
}

export function applySelectedDatasetDefaults<T extends Record<string, any>>(
  dataset: T,
  defaults?: Record<string, any>,
): T {
  const next = { ...dataset };
  for (const [key, value] of Object.entries(defaults || {})) {
    if (!key.startsWith(DATASET_DEFAULT_PREFIX)) continue;
    setPathValue(next, key.slice(DATASET_DEFAULT_PREFIX.length), selectedDefaultValue(value));
  }
  return next;
}
