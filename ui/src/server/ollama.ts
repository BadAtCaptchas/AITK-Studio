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

  const body: Record<string, unknown> = {
    model,
    prompt,
    images: [options.imageBase64],
    stream: false,
  };
  if (options.systemPrompt?.trim()) {
    body.system = options.systemPrompt.trim();
  }
  if (Number.isFinite(options.maxNewTokens) && (options.maxNewTokens || 0) > 0) {
    body.options = { num_predict: Math.round(options.maxNewTokens as number) };
  }

  const response = await fetch(`${trimTrailingSlash(baseUrl)}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await readOllamaError(response));
  }

  const data = await response.json();
  const caption = extractOllamaCaptionText(data);
  if (!caption) {
    const doneReason =
      typeof data === 'object' && data !== null && typeof (data as Record<string, unknown>).done_reason === 'string'
        ? ` Done reason: ${(data as Record<string, unknown>).done_reason}.`
        : '';
    throw new Error(
      `Ollama returned an empty caption.${doneReason} Confirm the selected model supports image inputs and try a stronger caption prompt.`,
    );
  }
  return caption;
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
