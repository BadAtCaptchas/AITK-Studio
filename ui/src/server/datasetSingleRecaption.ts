import { assertUsableCaption } from '../utils/captionQuality';
import { parseIdeogramCaption } from '../utils/ideogramCaption';
import { generateOllamaImageCaption } from './ollama';
import { endpointForRemoteOllamaWorker, getRemoteOllamaWorker } from './remoteOllamaWorkers';

export type RecaptionProvider = 'openrouter' | 'ollama' | 'remote_ollama';
export type RecaptionOutputFormat = 'text' | 'ideogram_json';

export type SingleRecaptionRequest = {
  provider?: string | null;
  model?: string | null;
  prompt?: string | null;
  systemPrompt?: string | null;
  outputFormat?: string | null;
  existingCaption?: string | null;
  imageDataUrl: string;
  openRouterApiKey?: string | null;
  remoteWorkerId?: string | null;
  maxNewTokens?: number | null;
  fetchImpl?: typeof fetch;
};

const DEFAULT_OPENROUTER_MODEL = 'x-ai/grok-4.3';
const DEFAULT_OLLAMA_MODEL = 'qwen3.5:35b';
const DEFAULT_TEXT_RECAPTION_PROMPT =
  'Caption this image as if you were going to generate it with an image model. Be direct, detailed, and do not include preamble.';
const DEFAULT_IDEOGRAM_RECAPTION_PROMPT = [
  'Create an Ideogram 4 training caption for this image as a JSON object.',
  'Return only valid JSON. Do not wrap it in markdown.',
  'Include high_level_description, style_description, and compositional_deconstruction.elements.',
  'For important visible elements, include type ("obj" or "text"), desc, optional color_palette, and bbox when available.',
  'Preserve and refine this existing caption when present:',
  '{existing_caption}',
].join('\n');

const IDEOGRAM_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    high_level_description: { type: 'string' },
    style_description: {
      type: 'object',
      additionalProperties: false,
      properties: {
        aesthetics: { type: 'string' },
        lighting: { type: 'string' },
        photo: { type: 'string' },
        medium: { type: 'string' },
        art_style: { type: 'string' },
        color_palette: { type: 'array', items: { type: 'string' } },
      },
      required: ['aesthetics', 'lighting', 'medium', 'color_palette'],
    },
    compositional_deconstruction: {
      type: 'object',
      additionalProperties: false,
      properties: {
        background: { type: 'string' },
        elements: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              type: { type: 'string', enum: ['obj', 'text'] },
              bbox: { type: 'array', items: { type: 'integer' } },
              text: { type: 'string' },
              desc: { type: 'string' },
              color_palette: { type: 'array', items: { type: 'string' } },
            },
            required: ['type', 'desc'],
          },
        },
      },
      required: ['background', 'elements'],
    },
  },
  required: ['high_level_description', 'style_description', 'compositional_deconstruction'],
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function normalizeRecaptionProvider(value: unknown): RecaptionProvider {
  return value === 'ollama' || value === 'remote_ollama' ? value : 'openrouter';
}

export function normalizeRecaptionOutputFormat(value: unknown): RecaptionOutputFormat {
  return value === 'ideogram_json' || value === 'json' ? 'ideogram_json' : 'text';
}

function normalizeMaxNewTokens(value: unknown, outputFormat: RecaptionOutputFormat) {
  const parsed = Number(value);
  if (Number.isFinite(parsed) && parsed > 0) return Math.round(parsed);
  return outputFormat === 'ideogram_json' ? 2048 : 256;
}

function imageBase64FromDataUrl(imageDataUrl: string) {
  const match = imageDataUrl.match(/^data:[^;]+;base64,(.*)$/i);
  return (match ? match[1] : imageDataUrl).trim();
}

function defaultPromptForOutput(outputFormat: RecaptionOutputFormat) {
  return outputFormat === 'ideogram_json' ? DEFAULT_IDEOGRAM_RECAPTION_PROMPT : DEFAULT_TEXT_RECAPTION_PROMPT;
}

export function buildSingleRecaptionPrompt(
  prompt: string | null | undefined,
  outputFormat: RecaptionOutputFormat,
  existingCaption?: string | null,
) {
  const basePrompt = (prompt || defaultPromptForOutput(outputFormat)).trim();
  const existing = (existingCaption || '').trim();
  if (!existing) return basePrompt;
  if (basePrompt.includes('{existing_caption}')) return basePrompt.split('{existing_caption}').join(existing);
  return `${basePrompt}\n\nExisting caption to replace or improve:\n${existing}`;
}

function extractOpenRouterContent(content: unknown): string {
  if (typeof content === 'string') return content.trim();
  if (isRecord(content)) {
    if (typeof content.text === 'string' && content.text.trim()) return content.text.trim();
    if (typeof content.value === 'string' && content.value.trim()) return content.value.trim();
    if (content.value != null) return JSON.stringify(content.value);
    if (content.content != null) return extractOpenRouterContent(content.content);
  }
  if (Array.isArray(content)) {
    return content.map(extractOpenRouterContent).filter(Boolean).join('\n').trim();
  }
  return '';
}

function extractOpenRouterMessageText(data: unknown) {
  const choices = isRecord(data) && Array.isArray(data.choices) ? data.choices : [];
  const first = choices[0];
  const message = isRecord(first) ? first.message : null;
  return isRecord(message) ? extractOpenRouterContent(message.content) : '';
}

function validateGeneratedCaption(caption: string, outputFormat: RecaptionOutputFormat) {
  const trimmed = caption.trim();
  assertUsableCaption(trimmed);
  if (outputFormat === 'ideogram_json') {
    const parsed = parseIdeogramCaption(trimmed);
    if (parsed.kind !== 'ideogram') {
      throw new Error(parsed.kind === 'json' ? parsed.error : 'Captioner did not return valid Ideogram JSON.');
    }
  }
  return trimmed;
}

async function generateOpenRouterRecaption({
  apiKey,
  imageDataUrl,
  model,
  prompt,
  systemPrompt,
  outputFormat,
  maxNewTokens,
  fetchImpl,
}: {
  apiKey: string;
  imageDataUrl: string;
  model: string;
  prompt: string;
  systemPrompt: string;
  outputFormat: RecaptionOutputFormat;
  maxNewTokens: number;
  fetchImpl: typeof fetch;
}) {
  if (!apiKey) throw new Error('OpenRouter API key is missing. Save it in settings or set OPENROUTER_API_KEY.');
  const messages: Array<Record<string, unknown>> = [];
  if (systemPrompt.trim()) messages.push({ role: 'system', content: systemPrompt.trim() });
  messages.push({
    role: 'user',
    content: [
      { type: 'text', text: prompt },
      { type: 'image_url', image_url: { url: imageDataUrl } },
    ],
  });
  const body: Record<string, unknown> = {
    model,
    stream: false,
    temperature: outputFormat === 'ideogram_json' ? 0 : 0.2,
    max_tokens: maxNewTokens,
    messages,
  };
  if (outputFormat === 'ideogram_json') {
    body.provider = { require_parameters: true };
    body.response_format = {
      type: 'json_schema',
      json_schema: {
        name: 'single_image_ideogram_caption',
        strict: true,
        schema: IDEOGRAM_JSON_SCHEMA,
      },
    };
  }

  const response = await fetchImpl('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'X-Title': 'AI Toolkit Dataset Studio',
    },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      (isRecord(data?.error) && typeof data.error.message === 'string' && data.error.message) ||
      (typeof data?.error === 'string' && data.error) ||
      `OpenRouter request failed with status ${response.status}`;
    throw new Error(message);
  }
  return extractOpenRouterMessageText(data);
}

export async function generateSingleImageRecaption(request: SingleRecaptionRequest) {
  const provider = normalizeRecaptionProvider(request.provider);
  const outputFormat = normalizeRecaptionOutputFormat(request.outputFormat);
  const model = (request.model || '').trim() || (provider === 'openrouter' ? DEFAULT_OPENROUTER_MODEL : DEFAULT_OLLAMA_MODEL);
  const prompt = buildSingleRecaptionPrompt(request.prompt, outputFormat, request.existingCaption);
  const maxNewTokens = normalizeMaxNewTokens(request.maxNewTokens, outputFormat);
  let rawCaption = '';

  if (provider === 'openrouter') {
    rawCaption = await generateOpenRouterRecaption({
      apiKey: request.openRouterApiKey?.trim() || '',
      imageDataUrl: request.imageDataUrl,
      model,
      prompt,
      systemPrompt: request.systemPrompt || '',
      outputFormat,
      maxNewTokens,
      fetchImpl: request.fetchImpl || fetch,
    });
  } else {
    const endpoint =
      provider === 'remote_ollama'
        ? endpointForRemoteOllamaWorker(await getRemoteOllamaWorker((request.remoteWorkerId || '').trim()))
        : undefined;
    rawCaption = await generateOllamaImageCaption(
      {
        model,
        prompt,
        systemPrompt: request.systemPrompt || (outputFormat === 'ideogram_json' ? 'Return only valid JSON.' : ''),
        imageBase64: imageBase64FromDataUrl(request.imageDataUrl),
        maxNewTokens,
      },
      endpoint,
    );
  }

  return {
    caption: validateGeneratedCaption(rawCaption, outputFormat),
    provider,
    model,
    outputFormat,
  };
}
