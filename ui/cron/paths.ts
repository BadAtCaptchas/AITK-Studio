import path from 'path';
import { db } from '../src/server/db';
import { resolveModelsPathState } from '../src/server/modelsPath';

export const TOOLKIT_ROOT = path.resolve('@', '..', '..');
export const defaultTrainFolder = path.join(TOOLKIT_ROOT, 'output');
export const defaultDatasetsFolder = path.join(TOOLKIT_ROOT, 'datasets');
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');
export const defaultModelsFolder = path.join(TOOLKIT_ROOT, 'models');

console.log('TOOLKIT_ROOT:', TOOLKIT_ROOT);

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let row = await db.settings.get(key);
  let trainingRoot = defaultTrainFolder;
  if (row?.value && row.value !== '') {
    trainingRoot = row.value;
  }
  return trainingRoot as string;
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let row = await db.settings.get(key);
  let token = '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  return token;
};

export const getOpenRouterApiKey = async () => {
  const key = 'OPENROUTER_API_KEY';
  let row = await db.settings.get(key);
  let token = process.env.OPENROUTER_API_KEY?.trim() || process.env.AITK_OPENROUTER_API_KEY?.trim() || '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  return token;
};

export const getModelsRoot = async () => {
  const row = await db.settings.get('MODELS_PATH');
  return (
    await resolveModelsPathState({
      defaultRoot: defaultModelsFolder,
      settingValue: row?.value,
      allowExternal: Boolean(process.env.AI_TOOLKIT_AUTH),
    })
  ).path;
};
