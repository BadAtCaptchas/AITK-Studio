import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { isEncryptedDatasetFolder } from '@/server/encryptedDatasets';
import { REMOTE_CAPTION_MEDIA_EXTENSIONS, resolveDatasetDirectoryInsideRoot } from '@/server/remoteCaptionSecurity';
import { guardedFetch } from '@/server/networkPolicy';

type OpenRouterModel = {
  id: string;
  name?: string;
  pricing?: {
    prompt?: string;
    completion?: string;
  };
};

const FALLBACK_OPENROUTER_PRICING: Record<string, { prompt: number; completion: number; name: string }> = {
  'x-ai/grok-4.3': {
    prompt: 0.00000125,
    completion: 0.0000025,
    name: 'xAI: Grok 4.3',
  },
};

function normalizeExtension(value: unknown, fallback: string) {
  return (typeof value === 'string' ? value : fallback).trim().replace(/^\.+/, '').toLowerCase() || fallback;
}

function normalizeMediaExtensions(value: unknown) {
  const requested = Array.isArray(value) ? value : [];
  const normalized = requested
    .map(item => (typeof item === 'string' ? item.trim().replace(/^\.+/, '').toLowerCase() : ''))
    .filter(Boolean);
  return normalized.length > 0 ? normalized : ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'jxl'];
}

function countCaptionTargets(
  datasetPath: string,
  extensions: string[],
  captionExtension: string,
  recaption: boolean,
) {
  const mediaExts = new Set(extensions.map(ext => `.${ext}`));
  let count = 0;

  const visit = (dir: string) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith('.')) continue;
      if (entry.isDirectory()) {
        if (entry.name === '_controls') continue;
        visit(path.join(dir, entry.name));
        continue;
      }
      if (!entry.isFile()) continue;
      const ext = path.extname(entry.name).toLowerCase();
      if (!mediaExts.has(ext) && !REMOTE_CAPTION_MEDIA_EXTENSIONS.has(ext)) continue;
      if (!mediaExts.has(ext)) continue;
      if (!recaption) {
        const targetCaptionPath = path.join(dir, `${entry.name.slice(0, -ext.length)}.${captionExtension}`);
        if (fs.existsSync(targetCaptionPath)) continue;
      }
      count += 1;
    }
  };

  visit(datasetPath);
  return count;
}

async function fetchOpenRouterModel(modelId: string): Promise<OpenRouterModel | null> {
  try {
    const response = await guardedFetch('https://openrouter.ai/api/v1/models', { cache: 'no-store' }, 'OpenRouter pricing');
    if (!response.ok) return null;
    const data = (await response.json()) as { data?: OpenRouterModel[] };
    return data.data?.find(model => model.id === modelId) || null;
  } catch {
    return null;
  }
}

async function openRouterPricing(modelId: string) {
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

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const datasetPath = typeof body?.datasetPath === 'string' ? body.datasetPath : '';
    if (!datasetPath.trim()) {
      return NextResponse.json({ error: 'datasetPath is required' }, { status: 400 });
    }

    const datasetsRoot = await getDatasetsRoot();
    const resolvedDatasetPath = await resolveDatasetDirectoryInsideRoot(path.resolve(datasetPath), datasetsRoot);
    const encrypted = isEncryptedDatasetFolder(resolvedDatasetPath);
    if (encrypted) {
      return NextResponse.json({
        encrypted: true,
        mediaCount: null,
        estimatedCostUsd: null,
        error: 'Cost estimates are unavailable while the encrypted dataset catalog is locked.',
      });
    }

    const captionExtension = normalizeExtension(body?.captionExtension, 'txt');
    const extensions = normalizeMediaExtensions(body?.extensions);
    const recaption = body?.recaption === true;
    const mediaCount = countCaptionTargets(resolvedDatasetPath, extensions, captionExtension, recaption);

    const provider = typeof body?.provider === 'string' ? body.provider : '';
    if (provider !== 'openrouter') {
      return NextResponse.json({
        encrypted: false,
        mediaCount,
        estimatedCostUsd: 0,
        pricing: null,
      });
    }

    const model = typeof body?.model === 'string' ? body.model.trim() : '';
    const pricing = model ? await openRouterPricing(model) : null;
    if (!pricing) {
      return NextResponse.json({
        encrypted: false,
        mediaCount,
        estimatedCostUsd: null,
        pricing: null,
        error: 'OpenRouter pricing was not available for this model.',
      });
    }

    const outputFormat = typeof body?.outputFormat === 'string' ? body.outputFormat : 'text';
    const isJson = outputFormat === 'ideogram_json' || outputFormat === 'json';
    const maxNewTokens = Number(body?.maxNewTokens);
    const outputTokensPerFile = Number.isFinite(maxNewTokens) && maxNewTokens > 0 ? Math.round(maxNewTokens) : isJson ? 900 : 180;
    const promptTokensPerFile = isJson ? 650 : 220;
    const sourceCaptionTokensPerFile = isJson ? 150 : 0;
    const imageTokensPerFile = 900;
    const inputTokensPerFile = promptTokensPerFile + sourceCaptionTokensPerFile + imageTokensPerFile;
    const estimatedCostUsd =
      mediaCount * inputTokensPerFile * pricing.prompt + mediaCount * outputTokensPerFile * pricing.completion;

    return NextResponse.json({
      encrypted: false,
      mediaCount,
      estimatedCostUsd,
      pricing,
      assumptions: {
        inputTokensPerFile,
        outputTokensPerFile,
        promptTokensPerFile,
        sourceCaptionTokensPerFile,
        imageTokensPerFile,
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to estimate captioning cost' },
      { status: 400 },
    );
  }
}
