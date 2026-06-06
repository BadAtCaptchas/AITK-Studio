export type IdeogramElementType = 'obj' | 'text';

export type NormalizedBox = {
  y1: number;
  x1: number;
  y2: number;
  x2: number;
};

export type IdeogramBox = NormalizedBox & {
  elementIndex: number;
  type: IdeogramElementType;
  label: string;
  color: string;
};

export type GeneratedBoxPatch = {
  elementIndex: number;
  bbox: [number, number, number, number];
};

export type IdeogramCaptionParse =
  | {
      kind: 'ideogram';
      data: Record<string, any>;
      elements: any[];
      boxes: IdeogramBox[];
    }
  | {
      kind: 'json';
      data: unknown;
      error: string;
    }
  | {
      kind: 'plain';
      error?: string;
    };

const TOP_LEVEL_ORDER = ['high_level_description', 'style_description', 'compositional_deconstruction'];
const STYLE_PHOTO_ORDER = ['aesthetics', 'lighting', 'photo', 'medium', 'color_palette'];
const STYLE_ART_ORDER = ['aesthetics', 'lighting', 'medium', 'art_style', 'color_palette'];
const COMPOSITION_ORDER = ['background', 'elements'];
const ELEMENT_OBJ_ORDER = ['type', 'bbox', 'desc', 'color_palette'];
const ELEMENT_TEXT_ORDER = ['type', 'bbox', 'text', 'desc', 'color_palette'];
const DEFAULT_COLORS = ['#22D3EE', '#F59E0B', '#A3E635', '#FB7185', '#818CF8', '#34D399'];

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function orderRecord(source: Record<string, any>, preferredOrder: string[]) {
  const next: Record<string, any> = {};
  preferredOrder.forEach(key => {
    if (Object.prototype.hasOwnProperty.call(source, key)) next[key] = source[key];
  });
  Object.keys(source).forEach(key => {
    if (!Object.prototype.hasOwnProperty.call(next, key)) next[key] = source[key];
  });
  return next;
}

function normalizeStyleDescription(value: unknown) {
  if (!isRecord(value)) return value;
  const hasPhoto = Object.prototype.hasOwnProperty.call(value, 'photo');
  const order = hasPhoto ? STYLE_PHOTO_ORDER : STYLE_ART_ORDER;
  return orderRecord(value, order);
}

function normalizeElement(value: unknown) {
  if (!isRecord(value)) return value;
  if (Array.isArray(value.bbox)) value.bbox = boxToArray(arrayToBox(value.bbox) || { y1: 0, x1: 0, y2: 0, x2: 0 });
  const order = value.type === 'text' ? ELEMENT_TEXT_ORDER : ELEMENT_OBJ_ORDER;
  return orderRecord(value, order);
}

function normalizeComposition(value: unknown) {
  if (!isRecord(value)) return value;
  const next = { ...value };
  if (Array.isArray(next.elements)) {
    next.elements = next.elements.map(normalizeElement);
  }
  return orderRecord(next, COMPOSITION_ORDER);
}

export function normalizeIdeogramCaption(data: Record<string, any>) {
  const next = { ...data };
  if (Object.prototype.hasOwnProperty.call(next, 'style_description')) {
    next.style_description = normalizeStyleDescription(next.style_description);
  }
  if (Object.prototype.hasOwnProperty.call(next, 'compositional_deconstruction')) {
    next.compositional_deconstruction = normalizeComposition(next.compositional_deconstruction);
  }
  return orderRecord(next, TOP_LEVEL_ORDER);
}

export function clampNorm(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1000, Math.round(value)));
}

export function normalizeBox(box: NormalizedBox): NormalizedBox {
  return {
    y1: clampNorm(Math.min(box.y1, box.y2)),
    x1: clampNorm(Math.min(box.x1, box.x2)),
    y2: clampNorm(Math.max(box.y1, box.y2)),
    x2: clampNorm(Math.max(box.x1, box.x2)),
  };
}

export function arrayToBox(value: unknown): NormalizedBox | null {
  if (!Array.isArray(value) || value.length !== 4) return null;
  const values = value.map(item => (typeof item === 'number' && Number.isFinite(item) ? item : null));
  if (values.some(item => item == null)) return null;
  return normalizeBox({
    y1: values[0] as number,
    x1: values[1] as number,
    y2: values[2] as number,
    x2: values[3] as number,
  });
}

export function boxToArray(box: NormalizedBox): [number, number, number, number] {
  const normalized = normalizeBox(box);
  return [normalized.y1, normalized.x1, normalized.y2, normalized.x2];
}

function hasUsableSpan(box: NormalizedBox, minSpan: number) {
  return box.x2 - box.x1 >= minSpan && box.y2 - box.y1 >= minSpan;
}

export function normalizeGeneratedBoxPatches(
  value: unknown,
  elementCount: number,
  minSpan = 1,
): GeneratedBoxPatch[] {
  const rawBoxes = isRecord(value) && Array.isArray(value.boxes) ? value.boxes : Array.isArray(value) ? value : [];
  const byElementIndex = new Map<number, GeneratedBoxPatch>();

  rawBoxes.forEach(rawBox => {
    if (!isRecord(rawBox)) return;
    const elementIndex = Number(rawBox.elementIndex);
    if (!Number.isInteger(elementIndex) || elementIndex < 0 || elementIndex >= elementCount) return;

    const box = arrayToBox(rawBox.bbox);
    if (!box || !hasUsableSpan(box, minSpan)) return;
    byElementIndex.set(elementIndex, { elementIndex, bbox: boxToArray(box) });
  });

  return Array.from(byElementIndex.values()).sort((left, right) => left.elementIndex - right.elementIndex);
}

export function boxToRect(box: NormalizedBox) {
  const normalized = normalizeBox(box);
  return {
    x: normalized.x1,
    y: normalized.y1,
    w: Math.max(0, normalized.x2 - normalized.x1),
    h: Math.max(0, normalized.y2 - normalized.y1),
  };
}

export function rectToBox(rect: { x: number; y: number; w: number; h: number }) {
  return normalizeBox({
    y1: rect.y,
    x1: rect.x,
    y2: rect.y + Math.max(0, rect.h),
    x2: rect.x + Math.max(0, rect.w),
  });
}

export function parseIdeogramCaption(text: string): IdeogramCaptionParse {
  const trimmed = text.trim();
  if (!trimmed.startsWith('{')) return { kind: 'plain' };

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (error) {
    return { kind: 'plain', error: error instanceof Error ? error.message : 'Invalid JSON' };
  }

  if (!isRecord(parsed)) return { kind: 'json', data: parsed, error: 'Caption JSON must be an object.' };
  const composition = parsed.compositional_deconstruction;
  if (!isRecord(composition) || !Array.isArray(composition.elements)) {
    return {
      kind: 'json',
      data: parsed,
      error: 'Caption JSON must include compositional_deconstruction.elements.',
    };
  }

  return {
    kind: 'ideogram',
    data: parsed,
    elements: composition.elements,
    boxes: extractIdeogramBoxes(parsed),
  };
}

export function extractIdeogramBoxes(data: unknown): IdeogramBox[] {
  if (!isRecord(data)) return [];
  const composition = data.compositional_deconstruction;
  if (!isRecord(composition) || !Array.isArray(composition.elements)) return [];

  return composition.elements.flatMap((rawElement, elementIndex) => {
    if (!isRecord(rawElement)) return [];
    const box = arrayToBox(rawElement.bbox);
    if (!box) return [];
    const type: IdeogramElementType = rawElement.type === 'text' ? 'text' : 'obj';
    const labelSource = type === 'text' ? rawElement.text || rawElement.desc : rawElement.desc;
    const palette = Array.isArray(rawElement.color_palette) ? rawElement.color_palette : [];
    const color = typeof palette[0] === 'string' ? palette[0] : DEFAULT_COLORS[elementIndex % DEFAULT_COLORS.length];
    return [
      {
        ...box,
        elementIndex,
        type,
        label: labelSource == null ? '' : String(labelSource),
        color,
      },
    ];
  });
}

export function serializeIdeogramCaption(data: Record<string, any>) {
  return JSON.stringify(normalizeIdeogramCaption(data), null, 2);
}

export function cloneIdeogramData(data: Record<string, any>) {
  return JSON.parse(JSON.stringify(data)) as Record<string, any>;
}

export function getIdeogramElements(data: Record<string, any>): any[] {
  const elements = data.compositional_deconstruction?.elements;
  if (!Array.isArray(elements)) {
    data.compositional_deconstruction = {
      ...(isRecord(data.compositional_deconstruction) ? data.compositional_deconstruction : {}),
      elements: [],
    };
  }
  return data.compositional_deconstruction.elements;
}

export function addIdeogramElement(data: Record<string, any>, type: IdeogramElementType, box: NormalizedBox) {
  const elements = getIdeogramElements(data);
  const element =
    type === 'text'
      ? { type: 'text', bbox: boxToArray(box), text: '', desc: '' }
      : { type: 'obj', bbox: boxToArray(box), desc: '' };
  elements.push(element);
  return elements.length - 1;
}

export function deleteIdeogramElement(data: Record<string, any>, elementIndex: number) {
  const elements = getIdeogramElements(data);
  if (!elements[elementIndex]) return;
  elements.splice(elementIndex, 1);
}

export function updateIdeogramElementBox(data: Record<string, any>, elementIndex: number, box: NormalizedBox) {
  const element = getIdeogramElements(data)[elementIndex];
  if (!isRecord(element)) return;
  element.bbox = boxToArray(box);
}

export function applyGeneratedBoxPatches(data: Record<string, any>, patches: GeneratedBoxPatch[]) {
  const elements = getIdeogramElements(data);
  const normalizedPatches = normalizeGeneratedBoxPatches({ boxes: patches }, elements.length);
  normalizedPatches.forEach(patch => {
    const box = arrayToBox(patch.bbox);
    if (box) updateIdeogramElementBox(data, patch.elementIndex, box);
  });
  return normalizedPatches.length;
}

export function updateIdeogramElementField(
  data: Record<string, any>,
  elementIndex: number,
  field: 'desc' | 'text',
  value: string,
) {
  const element = getIdeogramElements(data)[elementIndex];
  if (!isRecord(element)) return;
  element[field] = value;
}

export function updateIdeogramElementType(data: Record<string, any>, elementIndex: number, type: IdeogramElementType) {
  const elements = getIdeogramElements(data);
  const current = elements[elementIndex];
  if (!isRecord(current)) return;
  const next =
    type === 'text'
      ? {
          type: 'text',
          ...(Array.isArray(current.bbox) ? { bbox: current.bbox } : {}),
          text: typeof current.text === 'string' ? current.text : '',
          desc: typeof current.desc === 'string' ? current.desc : '',
          ...(Array.isArray(current.color_palette) ? { color_palette: current.color_palette } : {}),
        }
      : {
          type: 'obj',
          ...(Array.isArray(current.bbox) ? { bbox: current.bbox } : {}),
          desc: typeof current.desc === 'string' ? current.desc : typeof current.text === 'string' ? current.text : '',
          ...(Array.isArray(current.color_palette) ? { color_palette: current.color_palette } : {}),
        };
  elements[elementIndex] = next;
}

export function updateIdeogramElementPalette(data: Record<string, any>, elementIndex: number, colors: string[]) {
  const element = getIdeogramElements(data)[elementIndex];
  if (!isRecord(element)) return;
  const cleanColors = colors
    .map(color => color.trim().toUpperCase())
    .filter(color => /^#[0-9A-F]{6}$/.test(color))
    .slice(0, 5);
  if (cleanColors.length === 0) {
    delete element.color_palette;
  } else {
    element.color_palette = cleanColors;
  }
}

export function updateIdeogramHighLevelDescription(data: Record<string, any>, value: string) {
  data.high_level_description = value;
}
