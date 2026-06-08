import {
  boxToArray,
  normalizeIdeogramColorPalette,
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

type RequiredImageSize = {
  width: number;
  height: number;
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
            description: 'Deprecated compatibility field. Use bbox_px.',
            items: { type: 'integer', minimum: 0 },
            minItems: 4,
            maxItems: 4,
          },
          bbox_px: {
            type: 'array',
            description: 'Bounding box as [ymin, xmin, ymax, xmax] in image pixels.',
            items: { type: 'integer', minimum: 0 },
            minItems: 4,
            maxItems: 4,
          },
          color_palette: {
            type: 'array',
            description: 'Dominant visible colors for this element as #RRGGBB hex strings.',
            items: { type: 'string' },
            maxItems: 5,
          },
        },
        required: ['elementIndex', 'bbox_px', 'color_palette'],
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
            description: 'Deprecated compatibility field. Use bbox_px.',
            items: { type: 'integer', minimum: 0 },
            minItems: 4,
            maxItems: 4,
          },
          bbox_px: {
            type: 'array',
            description: 'Bounding box as [ymin, xmin, ymax, xmax] in image pixels.',
            items: { type: 'integer', minimum: 0 },
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
          color_palette: {
            type: 'array',
            description: 'Dominant visible colors for this element as #RRGGBB hex strings.',
            items: { type: 'string' },
            maxItems: 5,
          },
        },
        required: ['type', 'bbox_px', 'desc', 'text', 'color_palette'],
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

function summarizeElements(elements: any[], imageSize: RequiredImageSize) {
  return elements.map((element, index) => {
    const type = element?.type === 'text' ? 'text' : 'obj';
    const currentBboxPx = Array.isArray(element?.bbox) ? normalizedBboxToPixelArray(element.bbox, imageSize) : null;
    const currentColorPalette = normalizeIdeogramColorPalette(element?.color_palette);
    return {
      elementIndex: index,
      type,
      description: cleanString(element?.desc),
      ...(type === 'text' ? { visibleText: cleanString(element?.text) } : {}),
      ...(currentBboxPx ? { currentBbox_px: currentBboxPx } : {}),
      ...(currentColorPalette.length > 0 ? { currentColorPalette } : {}),
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

export function requireImageSize(imageSize?: ImageSize | null): RequiredImageSize {
  const width = Number(imageSize?.width);
  const height = Number(imageSize?.height);
  if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
    throw new Error('Image width and height are required before generating boxes.');
  }
  return { width: Math.round(width), height: Math.round(height) };
}

function numberTuple4(value: unknown): [number, number, number, number] | null {
  if (!Array.isArray(value) || value.length !== 4) return null;
  const values = value.map(item => (typeof item === 'number' && Number.isFinite(item) ? item : null));
  if (values.some(item => item == null)) return null;
  return values as [number, number, number, number];
}

export function pixelBboxToNormalizedArray(
  value: unknown,
  imageSize: RequiredImageSize,
): [number, number, number, number] | null {
  const values = numberTuple4(value);
  if (!values) return null;
  const [y1, x1, y2, x2] = values;
  return boxToArray({
    y1: (y1 / imageSize.height) * 1000,
    x1: (x1 / imageSize.width) * 1000,
    y2: (y2 / imageSize.height) * 1000,
    x2: (x2 / imageSize.width) * 1000,
  });
}

export function normalizedBboxToPixelArray(
  value: unknown,
  imageSize: RequiredImageSize,
): [number, number, number, number] | null {
  const values = numberTuple4(value);
  if (!values) return null;
  const [y1, x1, y2, x2] = values;
  return [
    Math.round((y1 / 1000) * imageSize.height),
    Math.round((x1 / 1000) * imageSize.width),
    Math.round((y2 / 1000) * imageSize.height),
    Math.round((x2 / 1000) * imageSize.width),
  ];
}

function rawPixelBbox(raw: Record<string, any>) {
  return raw.bbox_px || raw.bboxPx || raw.bbox;
}

function rawColorPalette(raw: Record<string, any>) {
  return raw.color_palette || raw.colorPalette || raw.palette || raw.colors;
}

function pixelGeneratedBoxPatches(
  value: unknown,
  elementCount: number,
  imageSize: RequiredImageSize,
  minSpan = 1,
): GeneratedBoxPatch[] {
  const rawBoxes = isRecord(value) && Array.isArray(value.boxes) ? value.boxes : Array.isArray(value) ? value : [];
  const boxes = rawBoxes.flatMap(rawBox => {
    if (!isRecord(rawBox)) return [];
    const bbox = pixelBboxToNormalizedArray(rawPixelBbox(rawBox), imageSize);
    if (!bbox) return [];
    const colorPalette = normalizeIdeogramColorPalette(rawColorPalette(rawBox));
    return [
      {
        elementIndex: rawBox.elementIndex,
        bbox,
        ...(colorPalette.length > 0 ? { color_palette: colorPalette } : {}),
      },
    ];
  });
  return normalizeGeneratedBoxPatches({ boxes }, elementCount, minSpan);
}

function pixelGeneratedElementBoxes(
  value: unknown,
  imageSize: RequiredImageSize,
  minSpan = 1,
  maxElements = 20,
): GeneratedElementBox[] {
  const rawElements: unknown[] =
    isRecord(value) && Array.isArray(value.generatedElements)
      ? value.generatedElements
      : isRecord(value) && Array.isArray(value.elements)
        ? value.elements
        : Array.isArray(value)
          ? value
          : [];

  const generatedElements = rawElements.flatMap(rawElement => {
    if (!isRecord(rawElement)) return [];
    const bbox = pixelBboxToNormalizedArray(rawPixelBbox(rawElement), imageSize);
    if (!bbox) return [];
    return [
      {
        type: rawElement.type,
        bbox,
        desc: rawElement.desc || rawElement.description || rawElement.label,
        text: rawElement.text || rawElement.visibleText || '',
        color_palette: normalizeIdeogramColorPalette(rawColorPalette(rawElement)),
      },
    ];
  });
  return normalizeGeneratedElementBoxes({ generatedElements }, minSpan, maxElements);
}

function normalizedPatchesToPixelPrompt(patches: GeneratedBoxPatch[], imageSize: RequiredImageSize) {
  return patches.flatMap(patch => {
    const bbox_px = normalizedBboxToPixelArray(patch.bbox, imageSize);
    return bbox_px
      ? [
          {
            elementIndex: patch.elementIndex,
            bbox_px,
            ...(patch.color_palette?.length ? { color_palette: patch.color_palette } : {}),
          },
        ]
      : [];
  });
}

function normalizedElementsToPixelPrompt(elements: GeneratedElementBox[], imageSize: RequiredImageSize) {
  return elements.flatMap(element => {
    const bbox_px = normalizedBboxToPixelArray(element.bbox, imageSize);
    return bbox_px ? [{ ...element, bbox_px, bbox: undefined }] : [];
  });
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
  const requiredImageSize = requireImageSize(imageSize);
  const parsed = parseIdeogramCaption(caption);
  if (parsed.kind !== 'ideogram') {
    throw new Error('Auto Boxes requires an Ideogram JSON caption.');
  }

  const elements = summarizeElements(parsed.elements, requiredImageSize);
  const previous = previousBoxes?.length
    ? `\nCurrent proposed boxes to correct:\n${JSON.stringify({ boxes: normalizedPatchesToPixelPrompt(previousBoxes, requiredImageSize), generatedElements: [] }, null, 2)}\n`
    : previousGeneratedElements?.length
      ? `\nCurrent proposed generated elements to correct:\n${JSON.stringify({ boxes: [], generatedElements: normalizedElementsToPixelPrompt(previousGeneratedElements, requiredImageSize) }, null, 2)}\n`
      : '';

  if (parsed.elements.length === 0) {
    return [
      'Create Ideogram caption elements with accurate bounding boxes for this image.',
      imageSizeLine(requiredImageSize),
      '',
      'The caption currently has no compositional_deconstruction.elements, so generate new elements for the most important visible regions.',
      'Scene context:',
      JSON.stringify(captionSceneContext(parsed), null, 2),
      '',
      'Coordinate contract:',
      '- Return bbox_px as [ymin, xmin, ymax, xmax] integers in image pixels.',
      `- Pixel coordinates must be relative to the submitted ${requiredImageSize.width} x ${requiredImageSize.height} image.`,
      '- Fit the visible extent of the object or text region tightly.',
      '- Do not add padding for aesthetics.',
      '- If an object is occluded or cropped, box only the visible part.',
      '- Use type "text" only for readable text glyphs, signs, labels, or UI text regions; put the visible glyph text in text.',
      '- Use type "obj" for ordinary objects and leave text as an empty string.',
      '- Return color_palette as up to 5 dominant visible colors for each generated element, ordered by prominence, using #RRGGBB hex strings.',
      '- Use color_palette: [] only when no reliable color can be identified for that element.',
      '- Prefer 3-10 salient regions. Do not cover the entire image unless the whole image is a single object.',
      '- Return boxes: [] because there are no existing element indexes to patch.',
      previous,
      'Return only JSON with both keys: boxes and generatedElements.',
    ].join('\n');
  }

  return [
    'Create accurate bounding boxes for existing Ideogram caption elements in this image.',
    imageSizeLine(requiredImageSize),
    '',
    'Coordinate contract:',
    '- Return bbox_px as [ymin, xmin, ymax, xmax] integers in image pixels.',
    `- Pixel coordinates must be relative to the submitted ${requiredImageSize.width} x ${requiredImageSize.height} image.`,
    '- Fit the visible extent of the named object or text region tightly.',
    '- Do not add padding for aesthetics.',
    '- If an object is occluded or cropped, box only the visible part.',
    '- For text elements, box the visible text glyphs or sign/label area, not the larger object holding it.',
    '- Return color_palette as up to 5 dominant visible colors for each existing element, ordered by prominence, using #RRGGBB hex strings.',
    '- For text elements, include visible glyph, fill, stroke, shadow, or sign/background colors that belong to that selected text target.',
    '- Use color_palette: [] only when no reliable color can be identified for that element.',
    '- Keep every elementIndex exactly as provided. Do not create, remove, reorder, or rename elements.',
    '- Return generatedElements: [] because existing caption elements are being patched.',
    '- If a currentBbox_px exists, use it only as a rough hint; correct it when the image contradicts it.',
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

export function parseJsonObject(text: string, providerName = 'OpenRouter') {
  try {
    return JSON.parse(text);
  } catch {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start < 0 || end <= start) throw new Error(`${providerName} did not return a JSON object.`);
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

export function parseBoxResponse(
  content: string,
  elementCount: number,
  imageSize: RequiredImageSize,
  providerName = 'OpenRouter',
) {
  const parsed = parseJsonObject(content, providerName);
  const boxes = pixelGeneratedBoxPatches(parsed, elementCount, imageSize, 2);
  if (boxes.length === 0) {
    throw new Error(`${providerName} did not return any usable boxes.`);
  }
  return boxes;
}

export function parseGeneratedElementResponse(content: string, imageSize: RequiredImageSize, providerName = 'OpenRouter') {
  const parsed = parseJsonObject(content, providerName);
  const generatedElements = pixelGeneratedElementBoxes(parsed, imageSize, 2, 20);
  if (generatedElements.length === 0) {
    throw new Error(`${providerName} did not return any usable generated elements.`);
  }
  return generatedElements;
}

export async function generateOpenRouterBoxPatches(options: GenerateOpenRouterBoxPatchesOptions) {
  const apiKey = options.apiKey.trim();
  if (!apiKey) throw new Error('OpenRouter API key is missing. Add it in Settings.');

  const parsed = parseIdeogramCaption(options.caption);
  if (parsed.kind !== 'ideogram') throw new Error('Auto Boxes requires an Ideogram JSON caption.');

  const imageSize = requireImageSize(options.imageSize);
  const model = normalizeOpenRouterBoxModel(options.model);
  const fetchImpl = options.fetchImpl || fetch;
  const firstPrompt = buildOpenRouterBoxPrompt(options.caption, imageSize);
  const first = await callOpenRouterBoxes({ apiKey, imageDataUrl: options.imageDataUrl, model, prompt: firstPrompt, fetchImpl });
  let boxes: GeneratedBoxPatch[] = [];
  let generatedElements: GeneratedElementBox[] = [];
  if (parsed.elements.length > 0) {
    boxes = parseBoxResponse(first.content, parsed.elements.length, imageSize);
  } else {
    generatedElements = parseGeneratedElementResponse(first.content, imageSize);
  }
  let usage = first.usage;
  let refined = false;

  if (options.refine) {
    const refinePrompt = buildOpenRouterBoxPrompt(options.caption, imageSize, boxes, generatedElements);
    const second = await callOpenRouterBoxes({ apiKey, imageDataUrl: options.imageDataUrl, model, prompt: refinePrompt, fetchImpl });
    if (parsed.elements.length > 0) {
      const refinedBoxes = parseBoxResponse(second.content, parsed.elements.length, imageSize);
      if (refinedBoxes.length > 0) {
        boxes = refinedBoxes;
        refined = true;
      }
    } else {
      const refinedGeneratedElements = parseGeneratedElementResponse(second.content, imageSize);
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
