import fsp from 'fs/promises';
import path from 'path';

type NormalizePathSettingOptions = {
  allowExternal?: boolean;
};

export function isPathWithinRoot(root: string, target: string) {
  const relativePath = path.relative(path.resolve(root), path.resolve(target));
  return (
    relativePath === '' ||
    (relativePath !== '..' && !relativePath.startsWith(`..${path.sep}`) && !path.isAbsolute(relativePath))
  );
}

async function resolvePathWithExistingAncestors(value: string) {
  const resolved = path.resolve(value);
  const unresolvedSegments: string[] = [];
  let current = resolved;

  while (true) {
    try {
      const realCurrent = await fsp.realpath(current);
      return path.resolve(realCurrent, ...unresolvedSegments.reverse());
    } catch {
      const parent = path.dirname(current);
      if (parent === current) return resolved;
      unresolvedSegments.push(path.basename(current));
      current = parent;
    }
  }
}

export async function normalizeStoragePathSetting(
  value: unknown,
  fallbackRoot: string,
  options: NormalizePathSettingOptions = {},
) {
  const fallbackPath = path.resolve(fallbackRoot);
  const rawPath = typeof value === 'string' && value.trim() !== '' ? value.trim() : fallbackPath;
  const normalizedPath = path.resolve(rawPath);
  const canonicalPath = await resolvePathWithExistingAncestors(normalizedPath);

  if (canonicalPath === path.parse(canonicalPath).root) {
    return null;
  }

  if (options.allowExternal) {
    return normalizedPath;
  }

  const canonicalFallbackRoot = await resolvePathWithExistingAncestors(fallbackPath);
  if (!isPathWithinRoot(canonicalFallbackRoot, canonicalPath)) {
    return null;
  }

  return normalizedPath;
}
