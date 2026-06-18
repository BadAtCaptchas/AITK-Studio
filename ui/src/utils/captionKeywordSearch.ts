import { parseIdeogramCaption, serializeIdeogramCaption } from './ideogramCaption';
import { isFailedCaption } from './captionQuality';

export type CaptionKeywordMatchMode = 'whole-word' | 'partial';

const JSON_NON_TEXT_KEYS = new Set(['bbox', 'color_palette', 'colorPalette', 'palette', 'type']);
const WORD_CHAR_CLASS = '\\p{L}\\p{N}_';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function escapeRegex(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function normalizeSpaceAfterRemoval(value: string) {
  return value
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1')
    .replace(/,\s*,+/g, ', ')
    .replace(/^[\s,;:]+|[\s,;:]+$/g, '')
    .trim();
}

function uniqueTerms(terms: string[]) {
  const seen = new Set<string>();
  const result: string[] = [];
  terms.forEach(term => {
    const normalized = term.trim();
    const key = normalized.toLocaleLowerCase();
    if (!normalized || seen.has(key)) return;
    seen.add(key);
    result.push(normalized);
  });
  return result;
}

export function parseCaptionKeywordQuery(query: string) {
  return uniqueTerms(query.split(/[\s,]+/g));
}

function termRegex(term: string, mode: CaptionKeywordMatchMode) {
  const escaped = escapeRegex(term);
  if (mode === 'partial') return new RegExp(escaped, 'giu');
  return new RegExp(`(^|[^${WORD_CHAR_CLASS}])(${escaped})(?=$|[^${WORD_CHAR_CLASS}])`, 'giu');
}

function countMatchesInText(text: string, terms: string[], mode: CaptionKeywordMatchMode) {
  return terms.reduce((count, term) => {
    const matches = text.match(termRegex(term, mode));
    return count + (matches?.length || 0);
  }, 0);
}

function collectJsonTextValues(value: unknown, parentKey = ''): string[] {
  if (typeof value === 'string') {
    return JSON_NON_TEXT_KEYS.has(parentKey) ? [] : [value];
  }
  if (Array.isArray(value)) {
    if (JSON_NON_TEXT_KEYS.has(parentKey)) return [];
    return value.flatMap(item => collectJsonTextValues(item, parentKey));
  }
  if (!isRecord(value)) return [];
  return Object.entries(value).flatMap(([key, child]) => collectJsonTextValues(child, key));
}

function removeFromText(text: string, terms: string[], mode: CaptionKeywordMatchMode) {
  let removedCount = 0;
  let next = text;
  terms.forEach(term => {
    next = next.replace(termRegex(term, mode), (...parts) => {
      removedCount += 1;
      return mode === 'whole-word' ? String(parts[1] || '') : '';
    });
  });
  return {
    text: removedCount > 0 ? normalizeSpaceAfterRemoval(next) : text,
    removedCount,
  };
}

function removeFromJsonTextValues(value: unknown, terms: string[], mode: CaptionKeywordMatchMode, parentKey = ''): {
  value: unknown;
  removedCount: number;
} {
  if (typeof value === 'string') {
    if (JSON_NON_TEXT_KEYS.has(parentKey)) return { value, removedCount: 0 };
    const result = removeFromText(value, terms, mode);
    return { value: result.text, removedCount: result.removedCount };
  }
  if (Array.isArray(value)) {
    if (JSON_NON_TEXT_KEYS.has(parentKey)) return { value, removedCount: 0 };
    let removedCount = 0;
    const next = value.map(item => {
      const result = removeFromJsonTextValues(item, terms, mode, parentKey);
      removedCount += result.removedCount;
      return result.value;
    });
    return { value: next, removedCount };
  }
  if (!isRecord(value)) return { value, removedCount: 0 };
  let removedCount = 0;
  const next: Record<string, unknown> = {};
  Object.entries(value).forEach(([key, child]) => {
    const result = removeFromJsonTextValues(child, terms, mode, key);
    next[key] = result.value;
    removedCount += result.removedCount;
  });
  return { value: next, removedCount };
}

export function captionKeywordSearchText(caption: string) {
  if (isFailedCaption(caption)) return '';
  const parsed = parseIdeogramCaption(caption);
  if (parsed.kind === 'ideogram') return collectJsonTextValues(parsed.data).join('\n');
  if (parsed.kind === 'json') return collectJsonTextValues(parsed.data).join('\n');
  return caption;
}

export function countCaptionKeywordMatches(
  caption: string,
  terms: string[],
  mode: CaptionKeywordMatchMode = 'whole-word',
) {
  if (terms.length === 0) return 0;
  return countMatchesInText(captionKeywordSearchText(caption), terms, mode);
}

export function captionMatchesKeywords(
  caption: string,
  terms: string[],
  mode: CaptionKeywordMatchMode = 'whole-word',
) {
  return countCaptionKeywordMatches(caption, terms, mode) > 0;
}

export function removeCaptionKeywords(
  caption: string,
  terms: string[],
  mode: CaptionKeywordMatchMode = 'whole-word',
) {
  if (terms.length === 0) return { caption, removedCount: 0, changed: false };

  const parsed = parseIdeogramCaption(caption);
  if (parsed.kind === 'ideogram' || parsed.kind === 'json') {
    const result = removeFromJsonTextValues(parsed.data, terms, mode);
    if (result.removedCount === 0) return { caption, removedCount: 0, changed: false };
    const nextCaption =
      parsed.kind === 'ideogram' && isRecord(result.value)
        ? serializeIdeogramCaption(result.value)
        : JSON.stringify(result.value, null, 2);
    return {
      caption: nextCaption,
      removedCount: result.removedCount,
      changed: nextCaption !== caption,
    };
  }

  const result = removeFromText(caption, terms, mode);
  return {
    caption: result.text,
    removedCount: result.removedCount,
    changed: result.removedCount > 0 && result.text !== caption,
  };
}
