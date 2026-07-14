export type CaptionJsonObject = Record<string, unknown>;

export type CaptionJsonIssueCode =
  | 'invalid-json'
  | 'non-object-json'
  | 'missing-high-level-description'
  | 'missing-style-description'
  | 'missing-compositional-deconstruction'
  | 'missing-background'
  | 'missing-elements'
  | 'invalid-element'
  | 'missing-element-bbox';

export type CaptionJsonIssue = {
  code: CaptionJsonIssueCode;
  message: string;
  path: string;
  line?: number;
  column?: number;
  endLine?: number;
  endColumn?: number;
};

export type CaptionJsonValidation =
  | {
      validity: 'valid';
      data: CaptionJsonObject;
      errors: readonly [];
    }
  | {
      validity: 'invalid';
      data: null;
      errors: readonly [CaptionJsonIssue];
    };

export type CaptionJsonFormatResult =
  | {
      ok: true;
      value: string;
      data: CaptionJsonObject;
    }
  | {
      ok: false;
      value: string;
      error: CaptionJsonIssue;
    };

export type CaptionJsonSectionKind = 'high-level' | 'style' | 'background' | 'elements';

export type CaptionJsonNavigatorSection = {
  id: string;
  label: string;
  path: string;
  kind: CaptionJsonSectionKind;
  count?: number;
};

export type CaptionJsonSourceRange = {
  startOffset: number;
  endOffset: number;
  startLine: number;
  startColumn: number;
  endLine: number;
  endColumn: number;
};

export type CaptionJsonNavigatorElement = {
  index: number;
  label: string;
  type?: string;
  path: string;
  startLine?: number;
  endLine?: number;
  sourceRange?: CaptionJsonSourceRange;
};

export type CaptionJsonNavigator = {
  sections: readonly CaptionJsonNavigatorSection[];
  elements: readonly CaptionJsonNavigatorElement[];
};

export type CaptionJsonWorkspaceAnalysis =
  | {
      validity: 'valid';
      data: CaptionJsonObject;
      errors: readonly [];
      warnings: readonly CaptionJsonIssue[];
      sections: readonly CaptionJsonNavigatorSection[];
      elements: readonly CaptionJsonNavigatorElement[];
    }
  | {
      validity: 'invalid';
      data: null;
      errors: readonly [CaptionJsonIssue];
      warnings: readonly [];
      sections: readonly CaptionJsonNavigatorSection[];
      elements: readonly [];
    };

type ScannedJsonNode = {
  kind: 'object' | 'array' | 'scalar';
  start: number;
  end: number;
  properties?: ReadonlyMap<string, ScannedJsonNode>;
  items?: readonly ScannedJsonNode[];
};

type StringToken = {
  value: string;
  start: number;
  end: number;
};

const CANONICAL_ELEMENTS_PATH = 'compositional_deconstruction.elements';
const FLAT_ELEMENTS_PATH = 'elements';

function isRecord(value: unknown): value is CaptionJsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasOwn(value: CaptionJsonObject, key: string) {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

function skipWhitespace(source: string, offset: number) {
  let cursor = offset;
  while (cursor < source.length && /\s/.test(source[cursor])) cursor += 1;
  return cursor;
}

function scanStringToken(source: string, offset: number): StringToken | null {
  if (source[offset] !== '"') return null;
  let cursor = offset + 1;
  while (cursor < source.length) {
    const character = source[cursor];
    if (character === '\\') {
      cursor += 2;
      continue;
    }
    if (character === '"') {
      const end = cursor + 1;
      try {
        const parsed: unknown = JSON.parse(source.slice(offset, end));
        return typeof parsed === 'string' ? { value: parsed, start: offset, end } : null;
      } catch {
        return null;
      }
    }
    cursor += 1;
  }
  return null;
}

function scanJsonNode(source: string, offset: number): ScannedJsonNode | null {
  const start = skipWhitespace(source, offset);
  const first = source[start];

  if (first === '"') {
    const token = scanStringToken(source, start);
    return token ? { kind: 'scalar', start, end: token.end } : null;
  }

  if (first === '{') {
    const properties = new Map<string, ScannedJsonNode>();
    let cursor = skipWhitespace(source, start + 1);
    if (source[cursor] === '}') return { kind: 'object', start, end: cursor + 1, properties };

    while (cursor < source.length) {
      const key = scanStringToken(source, cursor);
      if (!key) return null;
      cursor = skipWhitespace(source, key.end);
      if (source[cursor] !== ':') return null;
      const child = scanJsonNode(source, cursor + 1);
      if (!child) return null;
      properties.set(key.value, child);
      cursor = skipWhitespace(source, child.end);
      if (source[cursor] === '}') return { kind: 'object', start, end: cursor + 1, properties };
      if (source[cursor] !== ',') return null;
      cursor = skipWhitespace(source, cursor + 1);
    }
    return null;
  }

  if (first === '[') {
    const items: ScannedJsonNode[] = [];
    let cursor = skipWhitespace(source, start + 1);
    if (source[cursor] === ']') return { kind: 'array', start, end: cursor + 1, items };

    while (cursor < source.length) {
      const child = scanJsonNode(source, cursor);
      if (!child) return null;
      items.push(child);
      cursor = skipWhitespace(source, child.end);
      if (source[cursor] === ']') return { kind: 'array', start, end: cursor + 1, items };
      if (source[cursor] !== ',') return null;
      cursor = skipWhitespace(source, cursor + 1);
    }
    return null;
  }

  let cursor = start;
  while (cursor < source.length && !/[\s,\]}]/.test(source[cursor])) cursor += 1;
  return cursor > start ? { kind: 'scalar', start, end: cursor } : null;
}

function getScannedProperty(node: ScannedJsonNode | null, key: string) {
  return node?.kind === 'object' ? (node.properties?.get(key) ?? null) : null;
}

function makeLineStarts(source: string) {
  const starts = [0];
  for (let index = 0; index < source.length; index += 1) {
    if (source.charCodeAt(index) === 10) starts.push(index + 1);
  }
  return starts;
}

function locationAtOffset(lineStarts: readonly number[], offset: number) {
  const safeOffset = Math.max(0, offset);
  let low = 0;
  let high = lineStarts.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (lineStarts[middle] <= safeOffset) low = middle + 1;
    else high = middle - 1;
  }
  const lineIndex = Math.max(0, high);
  return {
    line: lineIndex + 1,
    column: safeOffset - lineStarts[lineIndex] + 1,
  };
}

function sourceRangeForNode(source: string, node: ScannedJsonNode): CaptionJsonSourceRange {
  const lineStarts = makeLineStarts(source);
  const start = locationAtOffset(lineStarts, node.start);
  const endCharacterOffset = Math.max(node.start, node.end - 1);
  const end = locationAtOffset(lineStarts, endCharacterOffset);
  return {
    startOffset: node.start,
    endOffset: node.end,
    startLine: start.line,
    startColumn: start.column,
    endLine: end.line,
    endColumn: end.column + 1,
  };
}

function parseErrorLocation(source: string, message: string) {
  const positionMatch = message.match(/position\s+(\d+)/i);
  if (positionMatch) {
    const offset = Math.min(source.length, Math.max(0, Number(positionMatch[1])));
    return locationAtOffset(makeLineStarts(source), offset);
  }
  const lineColumnMatch = message.match(/line\s+(\d+)\s+column\s+(\d+)/i);
  if (lineColumnMatch) {
    return { line: Number(lineColumnMatch[1]), column: Number(lineColumnMatch[2]) };
  }
  return { line: 1, column: 1 };
}

function invalidJsonIssue(source: string, error: unknown): CaptionJsonIssue {
  const rawMessage = error instanceof Error ? error.message : 'Invalid JSON';
  const location = parseErrorLocation(source, rawMessage);
  return {
    code: 'invalid-json',
    message: rawMessage,
    path: '',
    line: location.line,
    column: location.column,
  };
}

function nonObjectIssue(source: string): CaptionJsonIssue {
  const rootOffset = skipWhitespace(source, 0);
  const location = locationAtOffset(makeLineStarts(source), rootOffset);
  return {
    code: 'non-object-json',
    message: 'Caption JSON must be an object.',
    path: '',
    line: location.line,
    column: location.column,
  };
}

function getComposition(data: CaptionJsonObject) {
  return isRecord(data.compositional_deconstruction) ? data.compositional_deconstruction : null;
}

function getElements(data: CaptionJsonObject) {
  const composition = getComposition(data);
  if (composition && Array.isArray(composition.elements)) {
    return { elements: composition.elements, path: CANONICAL_ELEMENTS_PATH } as const;
  }
  if (Array.isArray(data.elements)) return { elements: data.elements, path: FLAT_ELEMENTS_PATH } as const;
  return { elements: [] as unknown[], path: CANONICAL_ELEMENTS_PATH } as const;
}

function hasValidBbox(value: unknown) {
  return (
    Array.isArray(value) &&
    value.length === 4 &&
    value.every(coordinate => typeof coordinate === 'number' && Number.isFinite(coordinate))
  );
}

function firstLabel(values: readonly unknown[]) {
  for (const value of values) {
    if (nonEmptyString(value)) return value.trim();
  }
  return '';
}

function elementLabel(value: unknown, index: number) {
  if (!isRecord(value)) return `Element ${index + 1}`;
  const type = nonEmptyString(value.type) ? value.type.trim() : '';
  const preferred = type === 'text' ? [value.text, value.label, value.desc] : [value.label, value.desc, value.text];
  return firstLabel(preferred) || `Element ${index + 1}`;
}

function elementType(value: unknown) {
  if (!isRecord(value) || !nonEmptyString(value.type)) return undefined;
  return value.type.trim();
}

function getElementsArrayNode(source: string) {
  const root = scanJsonNode(source, 0);
  if (!root || root.kind !== 'object') return null;
  const composition = getScannedProperty(root, 'compositional_deconstruction');
  const nestedElements = getScannedProperty(composition, 'elements');
  if (nestedElements?.kind === 'array') return nestedElements;
  const flatElements = getScannedProperty(root, 'elements');
  return flatElements?.kind === 'array' ? flatElements : null;
}

export function validateCaptionJsonDraft(source: string): CaptionJsonValidation {
  let parsed: unknown;
  try {
    parsed = JSON.parse(source);
  } catch (error) {
    return { validity: 'invalid', data: null, errors: [invalidJsonIssue(source, error)] };
  }

  if (!isRecord(parsed)) {
    return { validity: 'invalid', data: null, errors: [nonObjectIssue(source)] };
  }
  return { validity: 'valid', data: parsed, errors: [] };
}

export function formatCaptionJsonDraft(source: string, indentation = 2): CaptionJsonFormatResult {
  const validation = validateCaptionJsonDraft(source);
  if (validation.validity === 'invalid') {
    return { ok: false, value: source, error: validation.errors[0] };
  }
  const safeIndentation = Number.isFinite(indentation) ? Math.min(8, Math.max(0, Math.floor(indentation))) : 2;
  return {
    ok: true,
    value: JSON.stringify(validation.data, null, safeIndentation),
    data: validation.data,
  };
}

export function getCaptionJsonElementLineRanges(source: string): readonly CaptionJsonSourceRange[] {
  return getCaptionJsonElementRangesByIndex(source).flatMap(range => (range ? [range] : []));
}

function getCaptionJsonElementRangesByIndex(source: string): readonly (CaptionJsonSourceRange | null)[] {
  const validation = validateCaptionJsonDraft(source);
  if (validation.validity === 'invalid') return [];
  const arrayNode = getElementsArrayNode(source);
  if (!arrayNode?.items) return [];
  return arrayNode.items.map(node => (node.kind === 'object' ? sourceRangeForNode(source, node) : null));
}

export function collectCaptionJsonSchemaWarnings(
  data: CaptionJsonObject,
  source = JSON.stringify(data, null, 2),
): readonly CaptionJsonIssue[] {
  const warnings: CaptionJsonIssue[] = [];
  const composition = getComposition(data);
  const { elements, path: elementsPath } = getElements(data);
  const ranges = getCaptionJsonElementRangesByIndex(source);

  if (!nonEmptyString(data.high_level_description)) {
    warnings.push({
      code: 'missing-high-level-description',
      message: 'Add a high-level description for this image.',
      path: 'high_level_description',
    });
  }
  if (!isRecord(data.style_description)) {
    warnings.push({
      code: 'missing-style-description',
      message: 'Add a style description object.',
      path: 'style_description',
    });
  }
  if (!composition) {
    warnings.push({
      code: 'missing-compositional-deconstruction',
      message: 'Add a compositional_deconstruction object.',
      path: 'compositional_deconstruction',
    });
  }
  if (!composition || !hasOwn(composition, 'background')) {
    warnings.push({
      code: 'missing-background',
      message: 'Add a background description.',
      path: 'compositional_deconstruction.background',
    });
  }
  if ((!composition || !Array.isArray(composition.elements)) && !Array.isArray(data.elements)) {
    warnings.push({
      code: 'missing-elements',
      message: 'Add an elements array.',
      path: CANONICAL_ELEMENTS_PATH,
    });
  }

  elements.forEach((value, index) => {
    const range = ranges[index];
    const location = range
      ? {
          line: range.startLine,
          column: range.startColumn,
          endLine: range.endLine,
          endColumn: range.endColumn,
        }
      : {};
    const path = `${elementsPath}[${index}]`;
    if (!isRecord(value)) {
      warnings.push({
        code: 'invalid-element',
        message: `Element ${index + 1} must be an object.`,
        path,
        ...location,
      });
      return;
    }
    if (!hasValidBbox(value.bbox)) {
      warnings.push({
        code: 'missing-element-bbox',
        message: `${elementLabel(value, index)} needs a four-number bbox.`,
        path: `${path}.bbox`,
        ...location,
      });
    }
  });

  return warnings;
}

export function buildCaptionJsonNavigator(data: CaptionJsonObject, source: string): CaptionJsonNavigator {
  const composition = getComposition(data);
  const { elements, path: elementsPath } = getElements(data);
  const ranges = getCaptionJsonElementRangesByIndex(source);
  const highLevelPath = hasOwn(data, 'high_level_description')
    ? 'high_level_description'
    : hasOwn(data, 'caption')
      ? 'caption'
      : 'high_level_description';
  const backgroundPath =
    composition && hasOwn(composition, 'background')
      ? 'compositional_deconstruction.background'
      : hasOwn(data, 'background')
        ? 'background'
        : 'compositional_deconstruction.background';

  const sections: CaptionJsonNavigatorSection[] = [
    { id: 'high-level', label: 'High-level', path: highLevelPath, kind: 'high-level' },
    { id: 'style', label: 'Style', path: 'style_description', kind: 'style' },
    { id: 'background', label: 'Background', path: backgroundPath, kind: 'background' },
    { id: 'elements', label: 'Elements', path: elementsPath, kind: 'elements', count: elements.length },
  ];

  const navigatorElements = elements.map<CaptionJsonNavigatorElement>((value, index) => {
    const range = ranges[index];
    return {
      index,
      label: elementLabel(value, index),
      type: elementType(value),
      path: `${elementsPath}[${index}]`,
      ...(range
        ? {
            startLine: range.startLine,
            endLine: range.endLine,
            sourceRange: range,
          }
        : {}),
    };
  });

  return { sections, elements: navigatorElements };
}

export function analyzeCaptionJsonDraft(source: string): CaptionJsonWorkspaceAnalysis {
  const validation = validateCaptionJsonDraft(source);
  if (validation.validity === 'invalid') {
    return {
      ...validation,
      warnings: [],
      sections: [
        { id: 'high-level', label: 'High-level', path: 'high_level_description', kind: 'high-level' },
        { id: 'style', label: 'Style', path: 'style_description', kind: 'style' },
        {
          id: 'background',
          label: 'Background',
          path: 'compositional_deconstruction.background',
          kind: 'background',
        },
        { id: 'elements', label: 'Elements', path: CANONICAL_ELEMENTS_PATH, kind: 'elements', count: 0 },
      ],
      elements: [],
    };
  }

  const navigator = buildCaptionJsonNavigator(validation.data, source);
  return {
    ...validation,
    warnings: collectCaptionJsonSchemaWarnings(validation.data, source),
    ...navigator,
  };
}
