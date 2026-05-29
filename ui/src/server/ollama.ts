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

type OllamaGenerationAttempt = {
  endpoint: 'generate' | 'chat';
  attempt: number;
  caption: string;
  doneReason: string | null;
};

const OLLAMA_CAPTION_MAX_ATTEMPTS = 3;
const OLLAMA_CAPTION_EMPTY_RETRY_DELAY_MS = 2000;

export type OllamaModelPullStatus = {
  status: 'ready' | 'pulling' | 'error';
  error: string | null;
  startedAt: string;
  updatedAt: string;
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

function normalizeModelName(value: string) {
  const trimmed = value.trim();
  return trimmed.includes(':') ? trimmed : `${trimmed}:latest`;
}

export function hasOllamaModel(models: OllamaModel[], requestedModel: string) {
  const normalizedRequested = normalizeModelName(requestedModel);
  return models.some(model => {
    const candidates = [model.model, model.name].filter((value): value is string => typeof value === 'string');
    return candidates.some(candidate => candidate === requestedModel || normalizeModelName(candidate) === normalizedRequested);
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

function extractOllamaCaptionText(data: unknown) {
  if (typeof data !== 'object' || data === null) return '';
  const record = data as Record<string, unknown>;
  if (typeof record.response === 'string') return record.response.trim();

  const message = record.message;
  if (typeof message === 'object' && message !== null) {
    const content = (message as Record<string, unknown>).content;
    if (typeof content === 'string') return content.trim();
  }

  const text = record.text;
  if (typeof text === 'string') return text.trim();

  return '';
}

function extractOllamaDoneReason(data: unknown) {
  if (typeof data !== 'object' || data === null) return null;
  const doneReason = (data as Record<string, unknown>).done_reason;
  return typeof doneReason === 'string' ? doneReason : null;
}

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export async function listOllamaModels(baseUrl = getOllamaBaseUrl()) {
  const response = await fetch(`${trimTrailingSlash(baseUrl)}/api/tags`, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(await readOllamaError(response));
  }
  const data = (await response.json()) as { models?: OllamaModel[] };
  return Array.isArray(data.models) ? data.models : [];
}

export async function getOllamaStatus(baseUrl = getOllamaBaseUrl()): Promise<OllamaStatus> {
  try {
    const models = await listOllamaModels(baseUrl);
    return { ok: true, baseUrl: trimTrailingSlash(baseUrl), modelCount: models.length, error: null };
  } catch (error) {
    return {
      ok: false,
      baseUrl: trimTrailingSlash(baseUrl),
      modelCount: 0,
      error: error instanceof Error ? error.message : 'Ollama is unavailable',
    };
  }
}

export async function ensureOllamaModel(model: string, baseUrl = getOllamaBaseUrl()) {
  const trimmedModel = model.trim();
  if (!trimmedModel) {
    throw new Error('Ollama model is required');
  }

  const models = await listOllamaModels(baseUrl);
  if (hasOllamaModel(models, trimmedModel)) {
    return { pulled: false };
  }

  const response = await fetch(`${trimTrailingSlash(baseUrl)}/api/pull`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: trimmedModel, stream: false }),
  });
  if (!response.ok) {
    throw new Error(await readOllamaError(response));
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

export async function startOllamaModelPull(model: string, baseUrl = getOllamaBaseUrl()): Promise<OllamaModelPullStatus> {
  const trimmedModel = model.trim();
  if (!trimmedModel) throw new Error('Ollama model is required');

  const normalizedBaseUrl = trimTrailingSlash(baseUrl);
  const key = modelPullKey(trimmedModel, normalizedBaseUrl);
  const pulls = pullStateMap();
  const existing = pulls.get(key);
  if (existing?.status === 'pulling' || existing?.status === 'ready') {
    return copyPullState(existing);
  }

  const models = await listOllamaModels(normalizedBaseUrl);
  if (hasOllamaModel(models, trimmedModel)) {
    const now = new Date().toISOString();
    const ready: OllamaModelPullStatus = { status: 'ready', error: null, startedAt: now, updatedAt: now };
    pulls.set(key, ready);
    return copyPullState(ready);
  }

  if (existing?.status === 'error') {
    return copyPullState(existing);
  }

  const now = new Date().toISOString();
  const state: OllamaModelPullStatus = { status: 'pulling', error: null, startedAt: now, updatedAt: now };
  pulls.set(key, state);

  void (async () => {
    try {
      const response = await fetch(`${normalizedBaseUrl}/api/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: trimmedModel, stream: false }),
      });
      if (!response.ok) {
        throw new Error(await readOllamaError(response));
      }
      state.status = 'ready';
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

export async function generateOllamaImageCaption(options: OllamaGenerateOptions, baseUrl = getOllamaBaseUrl()) {
  const model = options.model.trim();
  const prompt = options.prompt.trim();
  if (!model) throw new Error('Ollama model is required');
  if (!prompt) throw new Error('Caption prompt is required');
  if (!options.imageBase64) throw new Error('Image payload is required');

  await ensureOllamaModel(model, baseUrl);

  const generateBody: Record<string, unknown> = {
    model,
    prompt,
    images: [options.imageBase64],
    stream: false,
    keep_alive: '10m',
  };
  if (options.systemPrompt?.trim()) {
    generateBody.system = options.systemPrompt.trim();
  }
  if (Number.isFinite(options.maxNewTokens) && (options.maxNewTokens || 0) > 0) {
    generateBody.options = { num_predict: Math.round(options.maxNewTokens as number) };
  }

  const messages: Array<Record<string, unknown>> = [];
  if (options.systemPrompt?.trim()) {
    messages.push({ role: 'system', content: options.systemPrompt.trim() });
  }
  messages.push({
    role: 'user',
    content: prompt,
    images: [options.imageBase64],
  });
  const chatBody: Record<string, unknown> = {
    model,
    messages,
    stream: false,
    keep_alive: '10m',
  };
  if (Number.isFinite(options.maxNewTokens) && (options.maxNewTokens || 0) > 0) {
    chatBody.options = { num_predict: Math.round(options.maxNewTokens as number) };
  }

  const attempts: OllamaGenerationAttempt[] = [];
  for (let attempt = 1; attempt <= OLLAMA_CAPTION_MAX_ATTEMPTS; attempt += 1) {
    const generateAttempt = await runOllamaGenerationAttempt('generate', generateBody, baseUrl, attempt);
    attempts.push(generateAttempt);
    if (generateAttempt.caption) return generateAttempt.caption;

    const chatAttempt = await runOllamaGenerationAttempt('chat', chatBody, baseUrl, attempt);
    attempts.push(chatAttempt);
    if (chatAttempt.caption) return chatAttempt.caption;

    if (attempt < OLLAMA_CAPTION_MAX_ATTEMPTS) {
      await sleep(OLLAMA_CAPTION_EMPTY_RETRY_DELAY_MS);
    }
  }

  const reasons = attempts
    .filter(attempt => attempt.doneReason)
    .map(attempt => `${attempt.endpoint} attempt ${attempt.attempt}: ${attempt.doneReason}`)
    .join(', ');
  throw new Error(
    `Ollama returned an empty caption${reasons ? ` (${reasons})` : ''}. Confirm the selected model supports image inputs and try a stronger caption prompt.`,
  );
}

async function runOllamaGenerationAttempt(
  endpoint: OllamaGenerationAttempt['endpoint'],
  body: Record<string, unknown>,
  baseUrl: string,
  attempt: number,
): Promise<OllamaGenerationAttempt> {
  const response = await fetch(`${trimTrailingSlash(baseUrl)}/api/${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await readOllamaError(response));
  }

  const data = await response.json();
  return {
    endpoint,
    attempt,
    caption: extractOllamaCaptionText(data),
    doneReason: extractOllamaDoneReason(data),
  };
}

export async function unloadOllamaModel(model: string, baseUrl = getOllamaBaseUrl()) {
  const trimmedModel = model.trim();
  if (!trimmedModel) throw new Error('Ollama model is required');

  const response = await fetch(`${trimTrailingSlash(baseUrl)}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: trimmedModel,
      prompt: '',
      stream: false,
      keep_alive: 0,
    }),
  });
  if (!response.ok) {
    throw new Error(await readOllamaError(response));
  }

  return { unloaded: true };
}
