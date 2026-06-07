import {
  arrayToBox,
  boxToArray,
  parseIdeogramCaption,
} from '../utils/ideogramCaption';
import {
  cleanString,
  extractMessageText,
  imageSizeLine,
  isRecord,
  normalizeOpenRouterBoxModel,
  parseJsonObject,
  type OpenRouterUsage,
} from './openRouterBoxes';

type ImageSize = {
  width?: number | null;
  height?: number | null;
};

type GenerateOpenRouterLayerCaptionOptions = {
  apiKey: string;
  imageDataUrl: string;
  caption: string;
  elementIndex: number;
  model?: string | null;
  imageSize?: ImageSize | null;
  fetchImpl?: typeof fetch;
};

type LayerType = 'obj' | 'text';

type SelectedLayerInfo = {
  type: LayerType;
  description: string;
  visibleText: string;
  bbox: [number, number, number, number] | null;
  targetClue: string;
};

type OpenRouterLayerCaptionCallResult = {
  content: string;
  usage?: OpenRouterUsage;
};

const OPENROUTER_LAYER_CAPTION_RESPONSE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    desc: {
      type: 'string',
      description: 'Concise visual description of only the selected layer target.',
    },
    text: {
      type: 'string',
      description: 'Readable visible text for text layers, or an empty string for object layers.',
    },
  },
  required: ['desc', 'text'],
};

function selectedLayerInfo(caption: string, elementIndex: number): SelectedLayerInfo {
  const parsed = parseIdeogramCaption(caption);
  if (parsed.kind !== 'ideogram') throw new Error('Layer captioning requires an Ideogram JSON caption.');
  if (!Number.isInteger(elementIndex) || elementIndex < 0 || elementIndex >= parsed.elements.length) {
    throw new Error('Selected layer was not found in the caption.');
  }

  const element = parsed.elements[elementIndex];
  if (!isRecord(element)) throw new Error('Selected layer is not a valid caption element.');

  const type: LayerType = element.type === 'text' ? 'text' : 'obj';
  const description = cleanString(element.desc);
  const visibleText = cleanString(element.text);
  const bbox = arrayToBox(element.bbox);
  const targetClue = type === 'text' ? visibleText || description : description || visibleText;
  if (!bbox && !targetClue) {
    throw new Error('Add a layer label or draw a box first.');
  }

  return {
    type,
    description,
    visibleText,
    bbox: bbox ? boxToArray(bbox) : null,
    targetClue,
  };
}

export function buildOpenRouterLayerCaptionPrompt(caption: string, elementIndex: number, imageSize?: ImageSize | null) {
  const parsed = parseIdeogramCaption(caption);
  if (parsed.kind !== 'ideogram') throw new Error('Layer captioning requires an Ideogram JSON caption.');
  const selected = selectedLayerInfo(caption, elementIndex);
  const sceneContext = {
    highLevelDescription: cleanString(parsed.data.high_level_description),
    background: cleanString(parsed.data.compositional_deconstruction?.background),
  };
  const selectedElement = {
    elementIndex,
    type: selected.type,
    description: selected.description,
    ...(selected.type === 'text' ? { visibleText: selected.visibleText } : {}),
    ...(selected.bbox ? { bbox: selected.bbox } : { targetClue: selected.targetClue }),
  };
  const targetLine = selected.bbox
    ? `Target bbox: ${JSON.stringify(selected.bbox)} as [ymin, xmin, ymax, xmax] normalized to 0-1000. Describe only the content inside that region.`
    : `No bbox exists for the selected layer. Identify the visible subject that best matches this selected-layer clue: ${JSON.stringify(selected.targetClue)}.`;

  return [
    'Caption only one selected Ideogram JSON caption layer in this image.',
    imageSizeLine(imageSize),
    targetLine,
    '',
    'Rules:',
    '- Do not caption the whole image.',
    '- Do not alter, mention, or create bounding boxes.',
    '- Return desc as a concise visual description of only the selected layer target.',
    '- For text layers, return text as the readable glyph text if visible; use an empty string if uncertain.',
    '- For object layers, return text as an empty string.',
    '- Do not include markdown, quotes around the whole answer, or explanatory prose outside JSON.',
    '',
    'Scene context:',
    JSON.stringify(sceneContext, null, 2),
    '',
    'Selected layer:',
    JSON.stringify(selectedElement, null, 2),
    '',
    'Return only JSON that matches the schema.',
  ].join('\n');
}

async function callOpenRouterLayerCaption({
  apiKey,
  imageDataUrl,
  model,
  prompt,
  fetchImpl,
}: {
  apiKey: string;
  imageDataUrl: string;
  model: string;
  prompt: string;
  fetchImpl: typeof fetch;
}): Promise<OpenRouterLayerCaptionCallResult> {
  const response = await fetchImpl('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'X-Title': 'AI Toolkit Dataset Studio',
    },
    body: JSON.stringify({
      model,
      stream: false,
      temperature: 0,
      max_tokens: 500,
      provider: { require_parameters: true },
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'dataset_layer_caption',
          strict: true,
          schema: OPENROUTER_LAYER_CAPTION_RESPONSE_SCHEMA,
        },
      },
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            { type: 'image_url', image_url: { url: imageDataUrl } },
          ],
        },
      ],
    }),
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      (isRecord(data?.error) && cleanString(data.error.message)) ||
      cleanString(data?.error) ||
      `OpenRouter request failed with status ${response.status}`;
    throw new Error(message);
  }

  const content = extractMessageText(data);
  if (!content) throw new Error('OpenRouter returned an empty layer caption response.');
  return { content, usage: isRecord(data?.usage) ? data.usage : undefined };
}

function parseLayerCaptionResponse(content: string, type: LayerType) {
  const parsed = parseJsonObject(content);
  if (!isRecord(parsed)) throw new Error('OpenRouter did not return a JSON object.');
  const text = cleanString(parsed.text || parsed.visibleText).slice(0, 240);
  const desc = (cleanString(parsed.desc || parsed.description || parsed.caption) || (type === 'text' && text ? `Visible text: ${text}` : '')).slice(0, 600);
  if (!desc) throw new Error('OpenRouter did not return a usable layer caption.');
  return {
    desc,
    ...(type === 'text' && text ? { text } : {}),
  };
}

export async function generateOpenRouterLayerCaption(options: GenerateOpenRouterLayerCaptionOptions) {
  const apiKey = options.apiKey.trim();
  if (!apiKey) throw new Error('OpenRouter API key is missing. Add it in Settings.');

  const selected = selectedLayerInfo(options.caption, options.elementIndex);
  const model = normalizeOpenRouterBoxModel(options.model);
  const prompt = buildOpenRouterLayerCaptionPrompt(options.caption, options.elementIndex, options.imageSize);
  const fetchImpl = options.fetchImpl || fetch;
  const result = await callOpenRouterLayerCaption({
    apiKey,
    imageDataUrl: options.imageDataUrl,
    model,
    prompt,
    fetchImpl,
  });
  return {
    ...parseLayerCaptionResponse(result.content, selected.type),
    model,
    usage: result.usage,
  };
}
