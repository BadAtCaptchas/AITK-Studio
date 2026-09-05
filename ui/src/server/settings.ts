import path from 'path';
import { defaultDatasetsFolder, defaultDataRoot, defaultModelsFolder, defaultTrainFolder } from '../paths';
import NodeCache from 'node-cache';
import { db } from './db';
import { normalizeStoragePathSetting } from './pathContainment';
import { modelsPathFromEnv, resolveModelsPathState } from './modelsPath';

const myCache = new NodeCache();
export function normalizeBooleanSetting(value: unknown, defaultValue: boolean) {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['false', '0', 'off', 'disabled'].includes(normalized)) return 'false';
    if (['true', '1', 'on', 'enabled'].includes(normalized)) return 'true';
  }
  return defaultValue ? 'true' : 'false';
}

export const flushCache = () => {
  myCache.flushAll();
};

export const getDatasetsRoot = async () => {
  const key = 'DATASETS_FOLDER';
  let datasetsPath = myCache.get(key) as string;
  if (datasetsPath) {
    return datasetsPath;
  }
  const row = await db.settings.get('DATASETS_FOLDER');
  datasetsPath = defaultDatasetsFolder;
  const normalizedDatasetsPath = await normalizeStoragePathSetting(row?.value, defaultDatasetsFolder, {
    allowExternal: Boolean(process.env.AI_TOOLKIT_AUTH),
  });
  if (normalizedDatasetsPath) {
    datasetsPath = normalizedDatasetsPath;
  }
  // Strip trailing slashes; the routes' `root + path.sep` prefix checks 403
  // on every file if the stored path ends with a separator.
  datasetsPath = path.resolve(datasetsPath);
  myCache.set(key, datasetsPath);
  return datasetsPath as string;
};

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let trainingRoot = myCache.get(key) as string;
  if (trainingRoot) {
    return trainingRoot;
  }
  const row = await db.settings.get(key);
  trainingRoot = defaultTrainFolder;
  const normalizedTrainingRoot = await normalizeStoragePathSetting(row?.value, defaultTrainFolder, {
    allowExternal: Boolean(process.env.AI_TOOLKIT_AUTH),
  });
  if (normalizedTrainingRoot) {
    trainingRoot = normalizedTrainingRoot;
  }
  trainingRoot = path.resolve(trainingRoot);
  myCache.set(key, trainingRoot);
  return trainingRoot as string;
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let token = myCache.get(key) as string;
  if (token) {
    return token;
  }
  const row = await db.settings.get(key);
  token = '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  myCache.set(key, token);
  return token;
};

export const getOpenRouterApiKey = async () => {
  const key = 'OPENROUTER_API_KEY';
  let token = myCache.get(key) as string;
  if (token) {
    return token;
  }
  const row = await db.settings.get(key);
  token = process.env.OPENROUTER_API_KEY?.trim() || process.env.AITK_OPENROUTER_API_KEY?.trim() || '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  myCache.set(key, token);
  return token;
};

export const getDataRoot = async () => {
  const key = 'DATA_ROOT';
  let dataRoot = myCache.get(key) as string;
  if (dataRoot) {
    return dataRoot;
  }
  const row = await db.settings.get(key);
  dataRoot = defaultDataRoot;
  const normalizedDataRoot = await normalizeStoragePathSetting(row?.value, defaultDataRoot, {
    allowExternal: Boolean(process.env.AI_TOOLKIT_AUTH),
  });
  if (normalizedDataRoot) {
    dataRoot = normalizedDataRoot;
  }
  dataRoot = path.resolve(dataRoot);
  myCache.set(key, dataRoot);
  return dataRoot;
};

export const getModelsRoot = async () => {
  const environmentPath = modelsPathFromEnv();
  if (environmentPath) return environmentPath;

  const key = 'MODELS_PATH';
  const cached = myCache.get(key);
  if (typeof cached === 'string' && cached) return cached;

  const row = await db.settings.get(key);
  const state = await resolveModelsPathState({
    defaultRoot: defaultModelsFolder,
    settingValue: row?.value,
    allowExternal: Boolean(process.env.AI_TOOLKIT_AUTH),
    envValue: null,
  });
  myCache.set(key, state.path);
  return state.path;
};
