import fsp from 'fs/promises';
import { db } from './db';
import { resolveDatasetFolder } from './encryptedDatasets';
import { getDatasetsRoot } from './settings';

export const SECURE_CAPTION_SYSTEM_PROMPT_SETTING_PREFIX = 'SECURE_REMOTE_CAPTION_SYSTEM_PROMPT:';
export const SECURE_CAPTION_SYSTEM_PROMPT_MAX_LENGTH = 4000;

function normalizeDatasetName(datasetName: string) {
  const normalized = datasetName.trim();
  if (!normalized) throw new Error('Dataset is required');
  return normalized;
}

export function secureCaptionSystemPromptSettingKey(datasetName: string) {
  return `${SECURE_CAPTION_SYSTEM_PROMPT_SETTING_PREFIX}${Buffer.from(
    normalizeDatasetName(datasetName),
    'utf8',
  ).toString('base64url')}`;
}

export function isSecureCaptionSystemPromptSettingKey(key: string) {
  return key.startsWith(SECURE_CAPTION_SYSTEM_PROMPT_SETTING_PREFIX);
}

export function normalizeSecureCaptionSystemPrompt(value: unknown) {
  if (value == null) return '';
  if (typeof value !== 'string') throw new Error('System prompt must be text');
  const prompt = value.trim();
  if (prompt.length > SECURE_CAPTION_SYSTEM_PROMPT_MAX_LENGTH) {
    throw new Error(`System prompt must be ${SECURE_CAPTION_SYSTEM_PROMPT_MAX_LENGTH} characters or less`);
  }
  return prompt;
}

export async function validateSecureCaptionDataset(datasetName: string) {
  const normalizedDatasetName = normalizeDatasetName(datasetName);
  const datasetsRoot = await getDatasetsRoot();
  const datasetFolder = resolveDatasetFolder(datasetsRoot, normalizedDatasetName);
  const stat = await fsp.stat(datasetFolder).catch(() => null);
  if (!stat?.isDirectory()) {
    throw new Error('Dataset not found');
  }
  return { datasetName: normalizedDatasetName, datasetFolder };
}

export async function getSecureCaptionSystemPrompt(datasetName: string) {
  const key = secureCaptionSystemPromptSettingKey(datasetName);
  const row = await db.settings.get(key);
  return normalizeSecureCaptionSystemPrompt(row?.value || '');
}

export async function setSecureCaptionSystemPrompt(datasetName: string, systemPrompt: string) {
  const key = secureCaptionSystemPromptSettingKey(datasetName);
  const prompt = normalizeSecureCaptionSystemPrompt(systemPrompt);
  if (!prompt) {
    await db.settings.delete(key);
    return '';
  }
  await db.settings.upsert(key, prompt);
  return prompt;
}

export async function renameSecureCaptionSystemPrompt(oldDatasetName: string, newDatasetName: string) {
  const oldKey = secureCaptionSystemPromptSettingKey(oldDatasetName);
  const newKey = secureCaptionSystemPromptSettingKey(newDatasetName);
  if (oldKey === newKey) return;

  const existing = await db.settings.get(oldKey);
  if (!existing) return;

  await db.settings.upsert(newKey, existing.value);
  await db.settings.delete(oldKey);
}
