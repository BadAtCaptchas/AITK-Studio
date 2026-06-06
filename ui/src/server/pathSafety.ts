import path from 'path';

export function resolvePathWithinRoot(root: string, target: unknown) {
  if (typeof target !== 'string' || target.trim().length === 0) {
    return null;
  }

  const resolvedRoot = path.resolve(root);
  const resolvedPath = path.resolve(resolvedRoot, target);
  const relativePath = path.relative(resolvedRoot, resolvedPath);

  if (relativePath === '' || relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    return null;
  }

  return resolvedPath;
}

export function isSafePathSegment(value: unknown) {
  if (typeof value !== 'string' || value.trim().length === 0) {
    return false;
  }

  if (value === '.' || value.includes('..')) {
    return false;
  }

  return value === path.basename(value) && value === path.win32.basename(value);
}
