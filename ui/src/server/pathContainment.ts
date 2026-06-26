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

export function normalizeStoragePathSetting(
  value: unknown,
  fallbackRoot: string,
  options: NormalizePathSettingOptions = {},
) {
  if (typeof value !== 'string' || value.trim() === '') {
    return fallbackRoot;
  }

  const normalizedPath = path.resolve(value.trim());
  if (normalizedPath === path.parse(normalizedPath).root) {
    return null;
  }

  if (options.allowExternal) {
    return normalizedPath;
  }

  const normalizedFallbackRoot = path.resolve(fallbackRoot);
  if (!isPathWithinRoot(normalizedFallbackRoot, normalizedPath)) {
    return null;
  }

  return normalizedPath;
}
