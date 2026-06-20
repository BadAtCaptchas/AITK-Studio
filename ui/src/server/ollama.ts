import sharp from 'sharp';
import { guardedFetch } from './networkPolicy';
import { sanitizeCaptionText } from '../utils/captionQuality';

export type OllamaModel = {
  name?: string;
  model?: string;
  modified_at?: string;
  size?: number;
  digest?: string;
  details?: Record<string, unknown>;
};

export type OllamaStatus = {
  ok: boolean;
  baseUrl: string;
  modelCount: number;
  error: string | null;
};

export type OllamaGenerateOptions = {
  model: string;
  systemPrompt?: string;
  prompt: string;
  imageBase64: string;
  maxNewTokens?: number;
};

export type OllamaEndpointConfig = {
  baseUrl?: string | null;
  authToken?: string | null;
};

export type OllamaEndpoint = string | OllamaEndpointConfig;

type ResolvedOllamaEndpoint = {
  baseUrl: string;
  authToken: string;
};

type OllamaGenerationAttempt = {
  endpoint: 'generate' | 'chat';
  attempt: number;
  numPredict: number;
  caption: string;
  doneReason: string | null;
  hadThinking: boolean;
};

const OLLAMA_CAPTION_MAX_ATTEMPTS = 3;
const OLLAMA_CAPTION_EMPTY_RETRY_DELAY_MS = 2000;
const OLLAMA_CAPTION_MIN_NUM_PREDICT = 2048;
const OLLAMA_CAPTION_MAX_NUM_PREDICT = 4096;
const JPEG_CONVERSION_QUALITY = 95;

export type OllamaModelPullStatus = {
  status: 'ready' | 'pulling' | 'error';
  error: string | null;
  startedAt: string;
  updatedAt: string;
  phase?: 'checking' | 'pulling' | 'warming' | 'ready';
};

declare global {
  // eslint-disable-next-line no-var
  var __aitkOllamaModelPulls: Map<string, OllamaModelPullStatus> | undefined;
}

function trimTrailingSlash(value: string) {
  return value.replace(/\/+$/, '');
}

export function getOllamaBaseUrl() {
  return trimTrailingSlash(process.env.AITK_OLLAMA_BASE_URL?.trim() || 'http://127.0.0.1:11434');
}

export function resolveOllamaEndpoint(endpoint?: OllamaEndpoint): ResolvedOllamaEndpoint {
  if (typeof endpoint === 'string') {
    return { baseUrl: trimTrailingSlash(endpoint.trim()), authToken: '' };
  }

  return {
    baseUrl: trimTrailingSlash(endpoint?.baseUrl?.trim() || getOllamaBaseUrl()),
    authToken: endpoint?.authToken?.trim() || '',
  };
}

function normalizeModelName(value: string) {
  const trimmed = value.trim();
  return trimmed.includes(':') ? trimmed : `${trimmed}:latest`;
}

export function hasOllamaModel(models: OllamaModel[], requestedModel: string) {
  const normalizedRequested = normalizeModelName(requestedModel);
  return models.some(model => {
    const candidates = [model.model, model.name].filter((value): value is string => typeof value === 'string');
    return candidates.some(
      candidate => candidate === requestedModel || normalizeModelName(candidate) === normalizedRequested,
    );
  });
}

async function readOllamaError(response: Response) {
  const body = await response.text().catch(() => '');
  if (!body) return `Ollama returned ${response.status}`;
  try {
    const parsed = JSON.parse(body);
    if (typeof parsed?.error === 'string') return parsed.error;
  } catch {
    // Keep the trimmed text below.
  }
  return body.slice(0, 500);
}

function ollamaFetchErrorMessage(error: unknown) {
  return error instanceof Error && error.message ? error.message : 'fetch failed';
}

async function fetchOllama(
  path: string,
  init: RequestInit,
  endpoint: OllamaEndpoint | undefined,
  operation: string,
): Promise<{ response: Response; url: string }> {
  const { baseUrl, authToken } = resolveOllamaEndpoint(endpoint);
  const url = `${baseUrl}${path}`;
  const headers = new Headers(init.headers);
  if (authToken && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${authToken}`);
  }
  try {
    return { response: await guardedFetch(url, { ...init, headers }, `Ollama ${operation}`), url };
  } catch (error) {
    throw new Error(
      `Ollama ${operation} failed at ${url}: ${ollamaFetchErrorMessage(error)}. Confirm Ollama is running and AITK_OLLAMA_BASE_URL points to the remote server's local Ollama.`,
    );
  }
}

async function throwOllamaResponseError(response: Response, url: string, operation: string): Promise<never> {
  throw new Error(`Ollama ${operation} failed at ${url}: ${await readOllamaError(response)}`);
}

function extractOllamaCaptionText(data: unknown) {
  if (typeof data !== 'object' || data === null) return '';
  const record = data as Record<string, unknown>;
  if (typeof record.response === 'string') return sanitizeCaptionText(record.response).trim();

  const message = record.message;
  if (typeof message === 'object' && message !== null) {
    const content = (message as Record<string, unknown>).content;
    if (typeof content === 'string') return sanitizeCaptionText(content).trim();
  }

  const text = record.text;
  if (typeof text === 'string') return sanitizeCaptionText(text).trim();

  return '';
}

function extractOllamaDoneReason(data: unknown) {
  if (typeof data !== 'object' || data === null) return null;
  const doneReason = (data as Record<string, unknown>).done_reason;
  return typeof doneReason === 'string' ? doneReason : null;
}

function hasOllamaThinking(data: unknown) {
  if (typeof data !== 'object' || data === null) return false;
  const record = data as Record<string, unknown>;
  if (typeof record.thinking === 'string' && record.thinking.trim()) return true;

  const message = record.message;
  if (typeof message === 'object' && message !== null) {
    const thinking = (message as Record<string, unknown>).thinking;
    return typeof thinking === 'string' && thinking.trim().length > 0;
  }

  return false;
}

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isWebpBuffer(buffer: Buffer) {
  return (
    buffer.length >= 12 &&
    buffer.subarray(0, 4).toString('ascii') === 'RIFF' &&
    buffer.subarray(8, 12).toString('ascii') === 'WEBP'
  );
}

async function prepareOllamaImageBase64(imageBase64: string) {
  const imageBuffer = Buffer.from(imageBase64, 'base64');
  if (!isWebpBuffer(imageBuffer)) return imageBase64;

  try {
    const jpegBuffer = await sharp(imageBuffer, { pages: 1 })
      .flatten({ background: '#ffffff' })
      .jpeg({ quality: JPEG_CONVERSION_QUALITY })
      .toBuffer();
    return jpegBuffer.toString('base64');
  } catch (error) {
    const message = error instanceof Error ? error.message : 'unknown conversion error';
    throw new Error(`Failed to convert WebP image to JPEG for Ollama captioning: ${message}`);
  }
}

function captionNumPredict(maxNewTokens: number | undefined, attempt: number) {
  const requested = Number.isFinite(maxNewTokens) && (maxNewTokens || 0) > 0 ? Math.round(maxNewTokens as number) : 0;
  const baseBudget = Math.max(OLLAMA_CAPTION_MIN_NUM_PREDICT, requested * 4);
  const retryBudget = baseBudget * 2 ** Math.max(0, attempt - 1);
  return Math.min(OLLAMA_CAPTION_MAX_NUM_PREDICT, retryBudget);
}

function captionBodyForAttempt(body: Record<string, unknown>, maxNewTokens: number | undefined, attempt: number) {
  return {
    ...body,
    options: { num_predict: captionNumPredict(maxNewTokens, attempt) },
  };
}

export async function listOllamaModels(endpoint?: OllamaEndpoint) {
  const { response, url } = await fetchOllama('/api/tags', { cache: 'no-store' }, endpoint, 'model list');
  if (!response.ok) {
    await throwOllamaResponseError(response, url, 'model list');
  }
  const data = (await response.json()) as { models?: OllamaModel[] };
  return Array.isArray(data.models) ? data.models : [];
}

export async function getOllamaStatus(endpoint?: OllamaEndpoint): Promise<OllamaStatus> {
  const { baseUrl } = resolveOllamaEndpoint(endpoint);
  try {
    const models = await listOllamaModels(endpoint);
    return { ok: true, baseUrl, modelCount: models.length, error: null };
  } catch (error) {
    return {
      ok: false,
      baseUrl,
      modelCount: 0,
      error: error instanceof Error ? error.message : 'Ollama is unavailable',
    };
  }
}

export async function ensureOllamaModel(model: string, endpoint?: OllamaEndpoint) {
  const trimmedModel = model.trim();
  if (!trimmedModel) {
    throw new Error('Ollama model is required');
  }

  const models = await listOllamaModels(endpoint);
  if (hasOllamaModel(models, trimmedModel)) {
    return { pulled: false };
  }

  const { response, url } = await fetchOllama(
    '/api/pull',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: trimmedModel, stream: false }),
    },
    endpoint,
    `model pull for ${trimmedModel}`,
  );
  if (!response.ok) {
    await throwOllamaResponseError(response, url, `model pull for ${trimmedModel}`);
  }
  return { pulled: true };
}

function pullStateMap() {
  if (!globalThis.__aitkOllamaModelPulls) {
    globalThis.__aitkOllamaModelPulls = new Map();
  }
  return globalThis.__aitkOllamaModelPulls;
}

function modelPullKey(model: string, baseUrl: string) {
  return `${trimTrailingSlash(baseUrl)}::${normalizeModelName(model)}`;
}

function copyPullState(state: OllamaModelPullStatus): OllamaModelPullStatus {
  return { ...state };
}

export async function startOllamaModelPull(model: string, endpoint?: OllamaEndpoint): Promise<OllamaModelPullStatus> {
  const trimmedModel = model.trim();
  if (!trimmedModel) throw new Error('Ollama model is required');

  const resolvedEndpoint = resolveOllamaEndpoint(endpoint);
  const key = modelPullKey(trimmedModel, resolvedEndpoint.baseUrl);
  const pulls = pullStateMap();
  const existing = pulls.get(key);
  if (existing?.status === 'pulling' || existing?.status === 'ready') {
    return copyPullState(existing);
  }

  if (existing?.status === 'error') {
    return copyPullState(existing);
  }

  const now = new Date().toISOString();
  const state: OllamaModelPullStatus = {
    status: 'pulling',
    phase: 'checking',
    error: null,
    startedAt: now,
    updatedAt: now,
  };
  pulls.set(key, state);

  void (async () => {
    try {
      const models = await listOllamaModels(resolvedEndpoint);
      if (!hasOllamaModel(models, trimmedModel)) {
        state.phase = 'pulling';
        state.updatedAt = new Date().toISOString();
        const { response, url } = await fetchOllama(
          '/api/pull',
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: trimmedModel, stream: false }),
          },
          resolvedEndpoint,
          `model pull for ${trimmedModel}`,
        );
        if (!response.ok) {
          await throwOllamaResponseError(response, url, `model pull for ${trimmedModel}`);
        }
      }
      state.phase = 'warming';
      state.updatedAt = new Date().toISOString();
      await warmOllamaModel(trimmedModel, resolvedEndpoint);
      state.status = 'ready';
      state.phase = 'ready';
      state.error = null;
      state.updatedAt = new Date().toISOString();
    } catch (error) {
      state.status = 'error';
      state.error = error instanceof Error ? error.message : 'Ollama model pull failed';
      state.updatedAt = new Date().toISOString();
    }
  })();

  return copyPullState(state);
}

async function warmOllamaModel(model: string, endpoint?: OllamaEndpoint) {
  const { response, url } = await fetchOllama(
    '/api/generate',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt: '',
        stream: false,
        keep_alive: '10m',
      }),
    },
    endpoint,
    `model warm-up for ${model}`,
  );
  if (!response.ok) {
    await throwOllamaResponseError(response, url, `model warm-up for ${model}`);
  }
}

export async function generateOllamaImageCaption(options: OllamaGenerateOptions, endpoint?: OllamaEndpoint) {
  const model = options.model.trim();
  const prompt = options.prompt.trim();
  if (!model) throw new Error('Ollama model is required');
  if (!prompt) throw new Error('Caption prompt is required');
  if (!options.imageBase64) throw new Error('Image payload is required');

  const imageBase64 = await prepareOllamaImageBase64(options.imageBase64);

  await ensureOllamaModel(model, endpoint);

  const generateBody: Record<string, unknown> = {
    model,
    prompt,
    images: [imageBase64],
    stream: false,
    keep_alive: '10m',
  };
  if (options.systemPrompt?.trim()) {
    generateBody.system = options.systemPrompt.trim();
  }

  const messages: Array<Record<string, unknown>> = [];
  if (options.systemPrompt?.trim()) {
    messages.push({ role: 'system', content: options.systemPrompt.trim() });
  }
  messages.push({
    role: 'user',
    content: prompt,
    images: [imageBase64],
  });
  const chatBody: Record<string, unknown> = {
    model,
    messages,
    stream: false,
    keep_alive: '10m',
  };

  const attempts: OllamaGenerationAttempt[] = [];
  for (let attempt = 1; attempt <= OLLAMA_CAPTION_MAX_ATTEMPTS; attempt += 1) {
    const generateAttempt = await runOllamaGenerationAttempt(
      'generate',
      captionBodyForAttempt(generateBody, options.maxNewTokens, attempt),
      endpoint,
      attempt,
    );
    attempts.push(generateAttempt);
    if (generateAttempt.caption) return generateAttempt.caption;

    const chatAttempt = await runOllamaGenerationAttempt(
      'chat',
      captionBodyForAttempt(chatBody, options.maxNewTokens, attempt),
      endpoint,
      attempt,
    );
    attempts.push(chatAttempt);
    if (chatAttempt.caption) return chatAttempt.caption;

    if (attempt < OLLAMA_CAPTION_MAX_ATTEMPTS) {
      await sleep(OLLAMA_CAPTION_EMPTY_RETRY_DELAY_MS);
    }
  }

  const reasons = attempts
    .filter(attempt => attempt.doneReason)
    .map(
      attempt =>
        `${attempt.endpoint} attempt ${attempt.attempt}: ${attempt.doneReason} at num_predict ${attempt.numPredict}${
          attempt.hadThinking ? ' with thinking' : ''
        }`,
    )
    .join(', ');
  throw new Error(
    `Ollama returned an empty caption${reasons ? ` (${reasons})` : ''}. Confirm the selected model supports image inputs and try a stronger caption prompt.`,
  );
}

async function runOllamaGenerationAttempt(
  endpoint: OllamaGenerationAttempt['endpoint'],
  body: Record<string, unknown>,
  ollamaEndpoint: OllamaEndpoint | undefined,
  attempt: number,
): Promise<OllamaGenerationAttempt> {
  const { response, url } = await fetchOllama(
    `/api/${endpoint}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    ollamaEndpoint,
    `${endpoint} caption attempt ${attempt}`,
  );
  if (!response.ok) {
    await throwOllamaResponseError(response, url, `${endpoint} caption attempt ${attempt}`);
  }

  const data = await response.json();
  const numPredict = ((body.options as Record<string, unknown> | undefined)?.num_predict as number | undefined) || 0;
  return {
    endpoint,
    attempt,
    numPredict,
    caption: extractOllamaCaptionText(data),
    doneReason: extractOllamaDoneReason(data),
    hadThinking: hasOllamaThinking(data),
  };
}

export async function unloadOllamaModel(model: string, endpoint?: OllamaEndpoint) {
  const trimmedModel = model.trim();
  if (!trimmedModel) throw new Error('Ollama model is required');
  const resolvedEndpoint = resolveOllamaEndpoint(endpoint);

  const { response, url } = await fetchOllama(
    '/api/generate',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: trimmedModel,
        prompt: '',
        stream: false,
        keep_alive: 0,
      }),
    },
    resolvedEndpoint,
    `model unload for ${trimmedModel}`,
  );
  if (!response.ok) {
    const message = await readOllamaError(response);
    if (response.status === 404 || message.toLowerCase().includes('not found')) {
      pullStateMap().delete(modelPullKey(trimmedModel, resolvedEndpoint.baseUrl));
      return { unloaded: false, reason: 'model_not_found' };
    }
    throw new Error(`Ollama model unload for ${trimmedModel} failed at ${url}: ${message}`);
  }

  pullStateMap().delete(modelPullKey(trimmedModel, resolvedEndpoint.baseUrl));

  return { unloaded: true };
}
