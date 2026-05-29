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

  const data = (await response.json()) as { response?: string };
  return (data.response || '').trim();
}
