import { guardedFetch, isOfflineModeEnabled } from './networkPolicy';

type OpenRouterModel = {
  id: string;
  name?: string;
  pricing?: {
    prompt?: string;
    completion?: string;
  };
};

export type OpenRouterCaptionPricing = {
  modelName: string;
  prompt: number;
  completion: number;
  source: 'openrouter' | 'fallback';
};

const FALLBACK_OPENROUTER_PRICING: Record<string, { prompt: number; completion: number; name: string }> = {
  'x-ai/grok-4.3': {
    prompt: 0.00000125,
    completion: 0.0000025,
    name: 'xAI: Grok 4.3',
  },
};

async function fetchOpenRouterModel(modelId: string): Promise<OpenRouterModel | null> {
  if (await isOfflineModeEnabled()) return null;

  try {
    const response = await guardedFetch('https://openrouter.ai/api/v1/models', { cache: 'no-store' }, 'OpenRouter pricing');
    if (!response.ok) return null;
    const data = (await response.json()) as { data?: OpenRouterModel[] };
    return data.data?.find(model => model.id === modelId) || null;
  } catch {
    return null;
  }
}

export async function getOpenRouterCaptionPricing(modelId: string): Promise<OpenRouterCaptionPricing | null> {
  const model = await fetchOpenRouterModel(modelId);
  const prompt = Number(model?.pricing?.prompt);
  const completion = Number(model?.pricing?.completion);
  if (Number.isFinite(prompt) && prompt >= 0 && Number.isFinite(completion) && completion >= 0) {
    return {
      modelName: model?.name || modelId,
      prompt,
      completion,
      source: 'openrouter',
    };
  }

  const fallback = FALLBACK_OPENROUTER_PRICING[modelId];
  if (fallback) {
    return {
      modelName: fallback.name,
      prompt: fallback.prompt,
      completion: fallback.completion,
      source: 'fallback',
    };
  }

  return null;
}
