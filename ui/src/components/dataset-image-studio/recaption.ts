import { defaultIdeogramJsonCaptionPrompt, defaultImageCaptionPrompt } from '@/helpers/captionOptions';
import {
  AUTO_BOX_PROVIDERS,
  DEFAULT_OLLAMA_VISION_MODEL,
  DEFAULT_OPENROUTER_BOX_MODEL,
  OLLAMA_VISION_MODELS,
  OPENROUTER_BOX_MODELS,
} from './constants';
import type { DatasetStudioItem } from './types';

export type StudioBoxProvider = (typeof AUTO_BOX_PROVIDERS)[number]['value'];
export type RecaptionProvider = StudioBoxProvider;
export type RecaptionOutputFormat = 'text' | 'ideogram_json';
export type RecaptionModelOption = { value: string; label: string };
export type RecaptionLoadStatus = 'idle' | 'loading' | 'success' | 'error';
export type RecaptionSettingsPreset = {
  provider: RecaptionProvider;
  model: string;
  outputFormat: RecaptionOutputFormat;
  prompt: string;
  systemPrompt: string;
  remoteWorkerId: string;
  maxNewTokens: number;
};
export type RecaptionQueueEntry = {
  id: string;
  item: DatasetStudioItem;
  key: string;
  name: string;
  existingCaption: string;
  settings: RecaptionSettingsPreset;
};
export type PersistedRecaptionQueueEntry = Omit<RecaptionQueueEntry, 'item'> & {
  status: 'queued' | 'running';
  updatedAt: string;
};
export type PersistedRecaptionQueue = {
  version: 1;
  active: PersistedRecaptionQueueEntry | null;
  queue: PersistedRecaptionQueueEntry[];
  updatedAt: string;
};

type RecaptionStorageScope = {
  datasetName: string;
  projectID?: string | null;
  datasetPath?: string | null;
  workerID: string;
};

const RECAPTION_SETTINGS_STORAGE_PREFIX = 'aitk.datasetEditor.recaptionSettings.v1';
const RECAPTION_QUEUE_STORAGE_PREFIX = 'aitk.datasetEditor.recaptionQueue.v1';

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function stringValue(value: unknown) {
  return typeof value === 'string' ? value : '';
}

function recaptionScopedStorageKey(
  prefix: string,
  { datasetName, projectID, datasetPath, workerID }: RecaptionStorageScope,
) {
  const dataset = datasetName.trim();
  if (!dataset) return '';
  const scope = (projectID || datasetPath || workerID || 'global').trim();
  return `${prefix}:${encodeURIComponent(scope)}:${encodeURIComponent(dataset)}`;
}

function normalizeRecaptionSettingsPreset(value: unknown): RecaptionSettingsPreset {
  const parsed = isRecord(value) ? value : {};
  const provider = normalizeRecaptionProvider(parsed.provider);
  const fallbackModel = provider === 'openrouter' ? DEFAULT_OPENROUTER_BOX_MODEL : DEFAULT_OLLAMA_VISION_MODEL;
  const maxNewTokens = Number(parsed.maxNewTokens);

  return {
    provider,
    model: stringValue(parsed.model).trim() || fallbackModel,
    outputFormat: normalizeRecaptionOutputFormat(parsed.outputFormat),
    prompt: stringValue(parsed.prompt).trim() ? stringValue(parsed.prompt) : defaultImageCaptionPrompt,
    systemPrompt: stringValue(parsed.systemPrompt),
    remoteWorkerId: stringValue(parsed.remoteWorkerId),
    maxNewTokens: Number.isFinite(maxNewTokens) && maxNewTokens > 0 ? Math.floor(maxNewTokens) : 256,
  };
}

function normalizePersistedRecaptionQueueEntry(value: unknown): PersistedRecaptionQueueEntry | null {
  if (!isRecord(value)) return null;
  const key = stringValue(value.key);
  if (!key) return null;
  const status = value.status === 'running' ? 'running' : 'queued';

  return {
    id: stringValue(value.id),
    key,
    name: stringValue(value.name),
    existingCaption: stringValue(value.existingCaption),
    settings: normalizeRecaptionSettingsPreset(value.settings),
    status,
    updatedAt: stringValue(value.updatedAt) || new Date().toISOString(),
  };
}

export function providerLabel(providerValue: string) {
  return AUTO_BOX_PROVIDERS.find(provider => provider.value === providerValue)?.label || providerValue || 'OpenRouter';
}

export function ollamaModelOptions(models: unknown): RecaptionModelOption[] {
  if (!Array.isArray(models)) return [];

  return models.flatMap(model => {
    if (!isRecord(model)) return [];
    const value = stringValue(model.model).trim() || stringValue(model.name).trim();
    if (!value) return [];

    const details = isRecord(model.details) ? model.details : {};
    const detail = [details.parameter_size, details.quantization_level]
      .filter((part): part is string => typeof part === 'string' && part.trim().length > 0)
      .join(' ');

    return [{ value, label: detail ? `${value} (${detail})` : value }];
  });
}

export function recaptionSettingsStorageKey(scope: RecaptionStorageScope) {
  return recaptionScopedStorageKey(RECAPTION_SETTINGS_STORAGE_PREFIX, scope);
}

export function recaptionQueueStorageKey(scope: RecaptionStorageScope) {
  return recaptionScopedStorageKey(RECAPTION_QUEUE_STORAGE_PREFIX, scope);
}

export function normalizeStudioBoxProvider(value: unknown): StudioBoxProvider {
  return AUTO_BOX_PROVIDERS.some(provider => provider.value === value) ? (value as StudioBoxProvider) : 'openrouter';
}

export function normalizeRecaptionProvider(value: unknown): RecaptionProvider {
  return normalizeStudioBoxProvider(value);
}

export function normalizeRecaptionOutputFormat(value: unknown): RecaptionOutputFormat {
  return value === 'ideogram_json' ? 'ideogram_json' : 'text';
}

export function readRecaptionSettingsPreset(storageKey: string): RecaptionSettingsPreset | null {
  if (!storageKey || typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (!isRecord(parsed)) return null;
    return normalizeRecaptionSettingsPreset(parsed);
  } catch (error) {
    console.warn('Could not load saved recaption settings:', error);
    return null;
  }
}

export function persistedRecaptionQueueEntry(
  entry: RecaptionQueueEntry,
  status: PersistedRecaptionQueueEntry['status'],
): PersistedRecaptionQueueEntry {
  return {
    id: entry.id,
    key: entry.key,
    name: entry.name,
    existingCaption: entry.existingCaption,
    settings: entry.settings,
    status,
    updatedAt: new Date().toISOString(),
  };
}

export function readPersistedRecaptionQueue(storageKey: string): PersistedRecaptionQueue | null {
  if (!storageKey || typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (!isRecord(parsed) || parsed.version !== 1) return null;

    return {
      version: 1,
      active: normalizePersistedRecaptionQueueEntry(parsed.active),
      queue: Array.isArray(parsed.queue)
        ? parsed.queue.flatMap(entry => {
            const normalized = normalizePersistedRecaptionQueueEntry(entry);
            return normalized ? [normalized] : [];
          })
        : [],
      updatedAt: stringValue(parsed.updatedAt) || new Date().toISOString(),
    };
  } catch (error) {
    console.warn('Could not load saved recaption queue:', error);
    return null;
  }
}

export function modelOptionsForRecaptionProvider(
  provider: RecaptionProvider,
  remoteModelOptions: RecaptionModelOption[],
) {
  if (provider === 'openrouter') return OPENROUTER_BOX_MODELS;
  if (provider === 'remote_ollama' && remoteModelOptions.length > 0) return remoteModelOptions;
  return OLLAMA_VISION_MODELS;
}

export function nextAutoBoxModelForProvider(provider: StudioBoxProvider, currentModel: string) {
  const trimmed = currentModel.trim();
  if (provider === 'openrouter') {
    return !trimmed || trimmed.startsWith('qwen3.5:') ? DEFAULT_OPENROUTER_BOX_MODEL : trimmed;
  }
  return !trimmed || trimmed === DEFAULT_OPENROUTER_BOX_MODEL ? DEFAULT_OLLAMA_VISION_MODEL : trimmed;
}

export function nextRecaptionModelForProvider(
  provider: RecaptionProvider,
  currentModel: string,
  remoteModelOptions: RecaptionModelOption[],
) {
  const trimmed = currentModel.trim();
  if (provider === 'openrouter') {
    return !trimmed || trimmed.startsWith('qwen3.5:') || trimmed.startsWith('gemma4:')
      ? DEFAULT_OPENROUTER_BOX_MODEL
      : trimmed;
  }
  if (provider === 'remote_ollama') {
    return remoteModelOptions.some(option => option.value === trimmed) ? trimmed : remoteModelOptions[0]?.value || '';
  }
  return !trimmed || trimmed === DEFAULT_OPENROUTER_BOX_MODEL ? DEFAULT_OLLAMA_VISION_MODEL : trimmed;
}

export function promptForRecaptionOutputFormat(format: RecaptionOutputFormat, currentPrompt: string) {
  const trimmed = currentPrompt.trim();
  if (format === 'ideogram_json') {
    return !trimmed || trimmed === defaultImageCaptionPrompt ? defaultIdeogramJsonCaptionPrompt : currentPrompt;
  }
  return !trimmed || trimmed === defaultIdeogramJsonCaptionPrompt ? defaultImageCaptionPrompt : currentPrompt;
}

export function maxNewTokensForRecaptionOutputFormat(format: RecaptionOutputFormat, currentMaxNewTokens: number) {
  if (format === 'ideogram_json') return currentMaxNewTokens < 2048 ? 2048 : currentMaxNewTokens;
  return currentMaxNewTokens >= 2048 ? 256 : currentMaxNewTokens;
}

export function normalizedRecaptionMaxNewTokens(value: number) {
  return Math.max(1, Math.floor(Number(value) || 1));
}

export function appendRecaptionFields(
  formData: FormData,
  settings: RecaptionSettingsPreset,
  existingCaption: string,
  datasetName: string,
) {
  formData.append('provider', settings.provider);
  formData.append('model', settings.model);
  formData.append('outputFormat', settings.outputFormat);
  formData.append('prompt', settings.prompt);
  formData.append('systemPrompt', settings.systemPrompt);
  formData.append('existingCaption', existingCaption);
  formData.append('datasetName', datasetName);
  formData.append('remoteWorkerId', settings.remoteWorkerId);
  formData.append('maxNewTokens', String(settings.maxNewTokens));
}
