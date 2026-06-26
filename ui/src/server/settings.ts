import { defaultDatasetsFolder, defaultDataRoot, defaultProjectsFolder, defaultTrainFolder } from '../paths';
import NodeCache from 'node-cache';
import { db } from './db';
import { normalizeStoragePathSetting } from './pathContainment';

const myCache = new NodeCache();
export const PROJECTS_ENABLED_KEY = 'PROJECTS_ENABLED';
export const PROJECT_SPACES_DISABLED_MESSAGE = 'Project spaces are disabled';

export class ProjectSpacesDisabledError extends Error {
  status = 403;

  constructor() {
    super(PROJECT_SPACES_DISABLED_MESSAGE);
    this.name = 'ProjectSpacesDisabledError';
  }
}

export function normalizeBooleanSetting(value: unknown, defaultValue: boolean) {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['false', '0', 'off', 'disabled'].includes(normalized)) return 'false';
    if (['true', '1', 'on', 'enabled'].includes(normalized)) return 'true';
  }
  return defaultValue ? 'true' : 'false';
}

export function isProjectSpacesDisabledError(error: unknown) {
  return (
    error instanceof ProjectSpacesDisabledError ||
    ((error as any)?.name === 'ProjectSpacesDisabledError' && (error as any)?.status === 403)
  );
}

export const flushCache = () => {
  myCache.flushAll();
};

export const areProjectsEnabled = async () => {
  const cached = myCache.get(PROJECTS_ENABLED_KEY) as string | undefined;
  if (typeof cached === 'string') return cached === 'true';

  const row = await db.settings.get(PROJECTS_ENABLED_KEY);
  const normalized = normalizeBooleanSetting(row?.value, false);
  myCache.set(PROJECTS_ENABLED_KEY, normalized);
  return normalized === 'true';
};

export const assertProjectsEnabled = async () => {
  if (!(await areProjectsEnabled())) {
    throw new ProjectSpacesDisabledError();
  }
};

export const getDatasetsRoot = async () => {
  const key = 'DATASETS_FOLDER';
  let datasetsPath = myCache.get(key) as string;
  if (datasetsPath) {
    return datasetsPath;
  }
  let row = await db.settings.get('DATASETS_FOLDER');
  datasetsPath = defaultDatasetsFolder;
  const normalizedDatasetsPath = await normalizeStoragePathSetting(row?.value, defaultDatasetsFolder, {
    allowExternal: Boolean(process.env.AI_TOOLKIT_AUTH),
  });
  if (normalizedDatasetsPath) {
    datasetsPath = normalizedDatasetsPath;
  }
  myCache.set(key, datasetsPath);
  return datasetsPath as string;
};

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let trainingRoot = myCache.get(key) as string;
  if (trainingRoot) {
    return trainingRoot;
  }
  let row = await db.settings.get(key);
  trainingRoot = defaultTrainFolder;
  const normalizedTrainingRoot = await normalizeStoragePathSetting(row?.value, defaultTrainFolder, {
    allowExternal: Boolean(process.env.AI_TOOLKIT_AUTH),
  });
  if (normalizedTrainingRoot) {
    trainingRoot = normalizedTrainingRoot;
  }
  myCache.set(key, trainingRoot);
  return trainingRoot as string;
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let token = myCache.get(key) as string;
  if (token) {
    return token;
  }
  let row = await db.settings.get(key);
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
  let row = await db.settings.get(key);
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
  let row = await db.settings.get(key);
  dataRoot = defaultDataRoot;
  const normalizedDataRoot = await normalizeStoragePathSetting(row?.value, defaultDataRoot, {
    allowExternal: Boolean(process.env.AI_TOOLKIT_AUTH),
  });
  if (normalizedDataRoot) {
    dataRoot = normalizedDataRoot;
  }
  myCache.set(key, dataRoot);
  return dataRoot;
};

export const getProjectsRoot = async () => {
  const key = 'PROJECTS_FOLDER';
  let projectsRoot = myCache.get(key) as string;
  if (projectsRoot) {
    return projectsRoot;
  }
  let row = await db.settings.get(key);
  projectsRoot = defaultProjectsFolder;
  const normalizedProjectsRoot = await normalizeStoragePathSetting(row?.value, defaultProjectsFolder, {
    allowExternal: Boolean(process.env.AI_TOOLKIT_AUTH),
  });
  if (normalizedProjectsRoot) {
    projectsRoot = normalizedProjectsRoot;
  }
  myCache.set(key, projectsRoot);
  return projectsRoot;
};
