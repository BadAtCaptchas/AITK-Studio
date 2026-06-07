import {
  normalizeGeneratedElementBoxes,
  normalizeGeneratedBoxPatches,
  parseIdeogramCaption,
  type GeneratedElementBox,
  type GeneratedBoxPatch,
} from '../utils/ideogramCaption';

export const DEFAULT_OPENROUTER_BOX_MODEL = 'x-ai/grok-4.3';
export const OPENROUTER_BOX_MODELS = ['x-ai/grok-4.3'] as const;

type ImageSize = {
  width?: number | null;
  height?: number | null;
};

export type OpenRouterUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  [key: string]: unknown;
};

type GenerateOpenRouterBoxPatchesOptions = {
  apiKey: string;
  imageDataUrl: string;
  caption: string;
  model?: string | null;
  refine?: boolean;
  imageSize?: ImageSize | null;
  fetchImpl?: typeof fetch;
};

type OpenRouterCallResult = {
  content: string;
  usage?: OpenRouterUsage;
};

const OPENROUTER_BOX_RESPONSE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    boxes: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          elementIndex: {
            type: 'integer',
            minimum: 0,
            description: 'Index of the existing caption element this box belongs to.',
          },
          bbox: {
            type: 'array',
            description: 'Bounding box as [ymin, xmin, ymax, xmax], normalized to 0-1000.',
            items: { type: 'integer', minimum: 0, maximum: 1000 },
            minItems: 4,
            maxItems: 4,
          },
        },
        required: ['elementIndex', 'bbox'],
      },
    },
    generatedElements: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          type: {
            type: 'string',
            enum: ['obj', 'text'],
            description: 'Ideogram element type. Use text only for visible glyphs, signs, labels, or readable text regions.',
          },
          bbox: {
            type: 'array',
            description: 'Bounding box as [ymin, xmin, ymax, xmax], normalized to 0-1000.',
            items: { type: 'integer', minimum: 0, maximum: 1000 },
            minItems: 4,
            maxItems: 4,
          },
          desc: {
            type: 'string',
            description: 'Short object or region description. For text, describe where the text appears.',
          },
          text: {
            type: 'string',
            description: 'Visible text content for text elements. Use an empty string for ordinary object elements.',
          },
        },
        required: ['type', 'bbox', 'desc', 'text'],
      },
    },
  },
  required: ['boxes', 'generatedElements'],
};

export function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function normalizeOpenRouterBoxModel(value: unknown) {
  const model = typeof value === 'string' ? value.trim() : '';
  return OPENROUTER_BOX_MODELS.includes(model as any) ? model : DEFAULT_OPENROUTER_BOX_MODEL;
}

export function cleanString(value: unknown) {
  return typeof value === 'string' ? value.trim() : '';
}

function summarizeElements(elements: any[]) {
  return elements.map((element, index) => {
    const type = element?.type === 'text' ? 'text' : 'obj';
    return {
      elementIndex: index,
      type,
      description: cleanString(element?.desc),
      ...(type === 'text' ? { visibleText: cleanString(element?.text) } : {}),
      ...(Array.isArray(element?.bbox) ? { currentBbox: element.bbox } : {}),
    };
  });
}

export function imageSizeLine(imageSize?: ImageSize | null) {
  const width = Number(imageSize?.width);
  const height = Number(imageSize?.height);
  if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
    return `Image pixel size: ${Math.round(width)} x ${Math.round(height)}.`;
  }
  return 'Image pixel size: unknown.';
}

function captionSceneContext(parsed: Extract<ReturnType<typeof parseIdeogramCaption>, { kind: 'ideogram' }>) {
  return {
    highLevelDescription: cleanString(parsed.data.high_level_description),
    background: cleanString(parsed.data.compositional_deconstruction?.background),
  };
}

export function buildOpenRouterBoxPrompt(
  caption: string,
  imageSize?: ImageSize | null,
  previousBoxes?: GeneratedBoxPatch[],
  previousGeneratedElements?: GeneratedElementBox[],
) {
  const parsed = parseIdeogramCaption(caption);
  if (parsed.kind !== 'ideogram') {
    throw new Error('Auto Boxes requires an Ideogram JSON caption.');
  }

  const elements = summarizeElements(parsed.elements);
  const previous = previousBoxes?.length
    ? `\nCurrent proposed boxes to correct:\n${JSON.stringify({ boxes: previousBoxes, generatedElements: [] }, null, 2)}\n`
    : previousGeneratedElements?.length
      ? `\nCurrent proposed generated elements to correct:\n${JSON.stringify({ boxes: [], generatedElements: previousGeneratedElements }, null, 2)}\n`
      : '';

  if (parsed.elements.length === 0) {
    return [
      'Create Ideogram caption elements with accurate bounding boxes for this image.',
      imageSizeLine(imageSize),
      '',
      'The caption currently has no compositional_deconstruction.elements, so generate new elements for the most important visible regions.',
      'Scene context:',
      JSON.stringify(captionSceneContext(parsed), null, 2),
      '',
      'Coordinate contract:',
      '- Return [ymin, xmin, ymax, xmax] integers normalized to 0-1000.',
      '- Fit the visible extent of the object or text region tightly.',
      '- Do not add padding for aesthetics.',
      '- If an object is occluded or cropped, box only the visible part.',
      '- Use type "text" only for readable text glyphs, signs, labels, or UI text regions; put the visible glyph text in text.',
      '- Use type "obj" for ordinary objects and leave text as an empty string.',
      '- Prefer 3-10 salient regions. Do not cover the entire image unless the whole image is a single object.',
      '- Return boxes: [] because there are no existing element indexes to patch.',
      previous,
      'Return only JSON with both keys: boxes and generatedElements.',
    ].join('\n');
  }

  return [
    'Create accurate bounding boxes for existing Ideogram caption elements in this image.',
    imageSizeLine(imageSize),
    '',
    'Coordinate contract:',
    '- Return [ymin, xmin, ymax, xmax] integers normalized to 0-1000.',
    '- Fit the visible extent of the named object or text region tightly.',
    '- Do not add padding for aesthetics.',
    '- If an object is occluded or cropped, box only the visible part.',
    '- For text elements, box the visible text glyphs or sign/label area, not the larger object holding it.',
    '- Keep every elementIndex exactly as provided. Do not create, remove, reorder, or rename elements.',
    '- Return generatedElements: [] because existing caption elements are being patched.',
    '- If a currentBbox exists, use it only as a rough hint; correct it when the image contradicts it.',
    previous,
    'Existing caption elements:',
    JSON.stringify(elements, null, 2),
    '',
    'Return only JSON that matches the schema.',
  ].join('\n');
}

export function extractMessageText(data: any) {
  const choice = Array.isArray(data?.choices) ? data.choices[0] : null;
  const message = isRecord(choice?.message) ? choice.message : null;
  const content = message?.content;
  if (typeof content === 'string') return content.trim();
  if (Array.isArray(content)) {
    return content
      .map(item => (isRecord(item) && typeof item.text === 'string' ? item.text : ''))
      .filter(Boolean)
      .join('\n')
      .trim();
  }
  return '';
}

export function parseJsonObject(text: string) {
  try {
    return JSON.parse(text);
  } catch {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start < 0 || end <= start) throw new Error('OpenRouter did not return a JSON object.');
    return JSON.parse(text.slice(start, end + 1));
  }
}

function mergeUsage(left?: OpenRouterUsage, right?: OpenRouterUsage): OpenRouterUsage | undefined {
  if (!left) return right;
  if (!right) return left;
  return {
    ...left,
    ...right,
    prompt_tokens: Number(left.prompt_tokens || 0) + Number(right.prompt_tokens || 0),
    completion_tokens: Number(left.completion_tokens || 0) + Number(right.completion_tokens || 0),
    total_tokens: Number(left.total_tokens || 0) + Number(right.total_tokens || 0),
  };
}

async function callOpenRouterBoxes({
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
}): Promise<OpenRouterCallResult> {
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
      max_tokens: 1200,
      provider: { require_parameters: true },
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'dataset_image_boxes',
          strict: true,
          schema: OPENROUTER_BOX_RESPONSE_SCHEMA,
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
  if (!content) throw new Error('OpenRouter returned an empty box response.');
  return { content, usage: isRecord(data?.usage) ? data.usage : undefined };
}

function parseBoxResponse(content: string, elementCount: number) {
  const parsed = parseJsonObject(content);
  const boxes = normalizeGeneratedBoxPatches(parsed, elementCount, 2);
  if (boxes.length === 0) {
    throw new Error('OpenRouter did not return any usable boxes.');
  }
  return boxes;
}

function parseGeneratedElementResponse(content: string) {
  const parsed = parseJsonObject(content);
  const generatedElements = normalizeGeneratedElementBoxes(parsed, 2, 20);
  if (generatedElements.length === 0) {
    throw new Error('OpenRouter did not return any usable generated elements.');
  }
  return generatedElements;
}

export async function generateOpenRouterBoxPatches(options: GenerateOpenRouterBoxPatchesOptions) {
  const apiKey = options.apiKey.trim();
  if (!apiKey) throw new Error('OpenRouter API key is missing. Add it in Settings.');

  const parsed = parseIdeogramCaption(options.caption);
  if (parsed.kind !== 'ideogram') throw new Error('Auto Boxes requires an Ideogram JSON caption.');

  const model = normalizeOpenRouterBoxModel(options.model);
  const fetchImpl = options.fetchImpl || fetch;
  const firstPrompt = buildOpenRouterBoxPrompt(options.caption, options.imageSize);
  const first = await callOpenRouterBoxes({ apiKey, imageDataUrl: options.imageDataUrl, model, prompt: firstPrompt, fetchImpl });
  let boxes: GeneratedBoxPatch[] = [];
  let generatedElements: GeneratedElementBox[] = [];
  if (parsed.elements.length > 0) {
    boxes = parseBoxResponse(first.content, parsed.elements.length);
  } else {
    generatedElements = parseGeneratedElementResponse(first.content);
  }
  let usage = first.usage;
  let refined = false;

  if (options.refine) {
    const refinePrompt = buildOpenRouterBoxPrompt(options.caption, options.imageSize, boxes, generatedElements);
    const second = await callOpenRouterBoxes({ apiKey, imageDataUrl: options.imageDataUrl, model, prompt: refinePrompt, fetchImpl });
    if (parsed.elements.length > 0) {
      const refinedBoxes = parseBoxResponse(second.content, parsed.elements.length);
      if (refinedBoxes.length > 0) {
        boxes = refinedBoxes;
        refined = true;
      }
    } else {
      const refinedGeneratedElements = parseGeneratedElementResponse(second.content);
      if (refinedGeneratedElements.length > 0) {
        generatedElements = refinedGeneratedElements;
        refined = true;
      }
    }
    if (refined) {
      refined = true;
    }
    usage = mergeUsage(usage, second.usage);
  }

  return { boxes, generatedElements, model, refined, usage };
}
