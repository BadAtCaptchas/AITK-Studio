import path from 'path';
import { normalizeStoragePathSetting } from './pathContainment';

export type ModelsPathState = {
  path: string;
  lockedByEnv: boolean;
};

export function modelsPathFromEnv(value: unknown = process.env.MODELS_PATH): string | null {
  if (typeof value !== 'string' || value.trim() === '') return null;
  const resolved = path.resolve(value.trim());
  if (resolved === path.parse(resolved).root) {
    throw new Error('MODELS_PATH cannot be the filesystem root');
  }
  return resolved;
}

export async function resolveModelsPathState(options: {
  defaultRoot: string;
  settingValue?: unknown;
  allowExternal?: boolean;
  envValue?: unknown;
}): Promise<ModelsPathState> {
  const environmentPath = modelsPathFromEnv(options.envValue);
  if (environmentPath) {
    return { path: environmentPath, lockedByEnv: true };
  }

  const normalizedSetting = await normalizeStoragePathSetting(
    options.settingValue,
    options.defaultRoot,
    { allowExternal: options.allowExternal },
  );
  return {
    path: normalizedSetting || path.resolve(options.defaultRoot),
    lockedByEnv: false,
  };
}
