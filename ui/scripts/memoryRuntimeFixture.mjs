import { db } from '../dist/src/server/db.js';

// Tests must never authenticate against or mutate the user's runtime database.
export function installMemoryRuntime() {
  const records = new Map();
  const original = { ...db.runtime };
  db.runtime.get = async key => structuredClone(records.get(key) ?? null);
  db.runtime.compareAndSwap = async (key, version, value) => {
    const old = records.get(key);
    if ((old?.version ?? null) !== version) return false;
    records.set(key, {
      key,
      value: structuredClone(value),
      version: (version ?? 0) + 1,
      updated_at: new Date().toISOString(),
    });
    return true;
  };
  db.runtime.delete = async (key, version) => {
    if (version !== undefined && records.get(key)?.version !== version) return false;
    return records.delete(key);
  };
  db.runtime.list = async (prefix, limit = 1000, after = '') =>
    [...records.values()]
      .filter(row => row.key.startsWith(prefix) && row.key > after)
      .sort((a, b) => a.key.localeCompare(b.key))
      .slice(0, limit)
      .map(row => structuredClone(row));
  return { records, restore: () => Object.assign(db.runtime, original) };
}
