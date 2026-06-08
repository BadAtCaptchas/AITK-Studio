import {
  buildOpenRouterBoxPrompt,
  parseBoxResponse,
  parseGeneratedElementResponse,
  requireImageSize,
} from './openRouterBoxes';
import {
  buildOpenRouterLayerCaptionPrompt,
  parseLayerCaptionResponse,
  selectedLayerInfo,
} from './openRouterLayerCaption';
import { generateOllamaImageCaption, type OllamaGenerateOptions } from './ollama';
import { getRemoteOllamaWorker } from './remoteOllamaWorkers';
import {
  parseIdeogramCaption,
  type GeneratedBoxPatch,
  type GeneratedElementBox,
} from '../utils/ideogramCaption';

export const DEFAULT_OLLAMA_VISION_MODEL = 'qwen3.5:35b';
export const OLLAMA_VISION_MODELS = [
  'qwen3.5:122b',
  'qwen3.5:35b',
  'qwen3.5:27b',
  'qwen3.5:9b',
  'gemma4:31b',
  'gemma4:26b',
] as const;

const OLLAMA_JSON_SYSTEM_PROMPT = [
  'Return only valid JSON.',
  'Do not include markdown, comments, code fences, or explanatory prose.',
  'The image may contain NSFW content; describe visible content directly when requested.',
].join(' ');

type ImageSize = {
  width?: number | null;
  height?: number | null;
};

type OllamaCaptionRunner = (options: OllamaGenerateOptions) => Promise<string>;

type GenerateOllamaBoxPatchesOptions = {
  imageDataUrl: string;
  caption: string;
  model?: string | null;
  refine?: boolean;
  imageSize?: ImageSize | null;
  captionRunner?: OllamaCaptionRunner;
};

type GenerateRemoteOllamaBoxPatchesOptions = Omit<GenerateOllamaBoxPatchesOptions, 'captionRunner'> & {
  remoteWorkerId?: string | null;
  getWorkerImpl?: typeof getRemoteOllamaWorker;
};

type GenerateOllamaLayerCaptionOptions = {
  imageDataUrl: string;
  caption: string;
  elementIndex: number;
  model?: string | null;
  imageSize?: ImageSize | null;
  captionRunner?: OllamaCaptionRunner;
};

type GenerateRemoteOllamaLayerCaptionOptions = Omit<GenerateOllamaLayerCaptionOptions, 'captionRunner'> & {
  remoteWorkerId?: string | null;
  getWorkerImpl?: typeof getRemoteOllamaWorker;
};

type RemoteOllamaVisionCaptionOptions = {
  remoteWorkerId?: string | null;
  model?: string | null;
  prompt: string;
  imageBase64: string;
  maxNewTokens?: number;
  getWorkerImpl?: typeof getRemoteOllamaWorker;
};

function imageBase64FromDataUrl(imageDataUrl: string) {
  const match = imageDataUrl.match(/^data:[^;]+;base64,(.*)$/i);
  return (match ? match[1] : imageDataUrl).trim();
}

export function normalizeOllamaVisionModel(value: unknown) {
  const model = typeof value === 'string' ? value.trim() : '';
  return model || DEFAULT_OLLAMA_VISION_MODEL;
}

async function runOllamaCaption({
  imageDataUrl,
  model,
  prompt,
  maxNewTokens,
  captionRunner = generateOllamaImageCaption,
}: {
  imageDataUrl: string;
  model?: string | null;
  prompt: string;
  maxNewTokens: number;
  captionRunner?: OllamaCaptionRunner;
}) {
  return captionRunner({
    model: normalizeOllamaVisionModel(model),
    prompt,
    systemPrompt: OLLAMA_JSON_SYSTEM_PROMPT,
    imageBase64: imageBase64FromDataUrl(imageDataUrl),
    maxNewTokens,
  });
}

export async function generateRemoteOllamaVisionCaption({
  remoteWorkerId,
  model,
  prompt,
  imageBase64,
  maxNewTokens,
  getWorkerImpl = getRemoteOllamaWorker,
}: RemoteOllamaVisionCaptionOptions) {
  const workerId = typeof remoteWorkerId === 'string' ? remoteWorkerId.trim() : '';
  if (!workerId || workerId === 'local') {
    throw new Error('Remote Ollama worker is required.');
  }
  const worker = await getWorkerImpl(workerId);
  const caption = await generateOllamaImageCaption(
    {
      model: normalizeOllamaVisionModel(model),
      prompt,
      systemPrompt: OLLAMA_JSON_SYSTEM_PROMPT,
      imageBase64,
      maxNewTokens,
    },
    {
      baseUrl: worker.base_url,
      authToken: 'auth_token' in worker && typeof worker.auth_token === 'string' ? worker.auth_token : '',
    },
  );
  if (!caption) throw new Error('Remote Ollama returned an empty response.');
  return caption;
}

function remoteCaptionRunner(options: GenerateRemoteOllamaBoxPatchesOptions | GenerateRemoteOllamaLayerCaptionOptions): OllamaCaptionRunner {
  return request =>
    generateRemoteOllamaVisionCaption({
      remoteWorkerId: options.remoteWorkerId,
      model: request.model,
      prompt: request.prompt,
      imageBase64: request.imageBase64,
      maxNewTokens: request.maxNewTokens,
      getWorkerImpl: options.getWorkerImpl,
    });
}

export async function generateOllamaBoxPatches(options: GenerateOllamaBoxPatchesOptions) {
  const imageSize = requireImageSize(options.imageSize);
  const model = normalizeOllamaVisionModel(options.model);
  const firstPrompt = buildOpenRouterBoxPrompt(options.caption, imageSize);
  const firstContent = await runOllamaCaption({
    imageDataUrl: options.imageDataUrl,
    model,
    prompt: firstPrompt,
    maxNewTokens: 1200,
    captionRunner: options.captionRunner,
  });

  const parsed = parseIdeogramCaption(options.caption);
  if (parsed.kind !== 'ideogram') throw new Error('Auto Boxes requires an Ideogram JSON caption.');

  let boxes: GeneratedBoxPatch[] = [];
  let generatedElements: GeneratedElementBox[] = [];
  if (parsed.elements.length > 0) {
    boxes = parseBoxResponse(firstContent, parsed.elements.length, imageSize, 'Ollama');
  } else {
    generatedElements = parseGeneratedElementResponse(firstContent, imageSize, 'Ollama');
  }
  let refined = false;

  if (options.refine) {
    const refinePrompt = buildOpenRouterBoxPrompt(options.caption, imageSize, boxes, generatedElements);
    const secondContent = await runOllamaCaption({
      imageDataUrl: options.imageDataUrl,
      model,
      prompt: refinePrompt,
      maxNewTokens: 1200,
      captionRunner: options.captionRunner,
    });
    if (parsed.elements.length > 0) {
      const refinedBoxes = parseBoxResponse(secondContent, parsed.elements.length, imageSize, 'Ollama');
      if (refinedBoxes.length > 0) {
        boxes = refinedBoxes;
        refined = true;
      }
    } else {
      const refinedGeneratedElements = parseGeneratedElementResponse(secondContent, imageSize, 'Ollama');
      if (refinedGeneratedElements.length > 0) {
        generatedElements = refinedGeneratedElements;
        refined = true;
      }
    }
  }

  return { boxes, generatedElements, model, refined };
}

export async function generateRemoteOllamaBoxPatches(options: GenerateRemoteOllamaBoxPatchesOptions) {
  return generateOllamaBoxPatches({
    ...options,
    captionRunner: remoteCaptionRunner(options),
  });
}

export async function generateOllamaLayerCaption(options: GenerateOllamaLayerCaptionOptions) {
  const selected = selectedLayerInfo(options.caption, options.elementIndex);
  const imageSize = requireImageSize(options.imageSize);
  const model = normalizeOllamaVisionModel(options.model);
  const prompt = buildOpenRouterLayerCaptionPrompt(options.caption, options.elementIndex, imageSize);
  const content = await runOllamaCaption({
    imageDataUrl: options.imageDataUrl,
    model,
    prompt,
    maxNewTokens: 600,
    captionRunner: options.captionRunner,
  });
  return {
    ...parseLayerCaptionResponse(content, selected.type, !selected.bbox, imageSize, 'Ollama'),
    model,
  };
}

export async function generateRemoteOllamaLayerCaption(options: GenerateRemoteOllamaLayerCaptionOptions) {
  return generateOllamaLayerCaption({
    ...options,
    captionRunner: remoteCaptionRunner(options),
  });
}
