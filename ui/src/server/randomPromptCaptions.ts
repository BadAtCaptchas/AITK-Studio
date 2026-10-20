export const RANDOM_PROMPT_CAPTION_EXTENSION_PRIORITY = ['.json', '.txt', '.caption', '.sdxl', '.md'];
export const RANDOM_PROMPT_CAPTION_EXTENSIONS = new Set(RANDOM_PROMPT_CAPTION_EXTENSION_PRIORITY);

export type DatasetPromptRequest = {
  folderPath: string;
  captionExt?: string;
  defaultCaption?: string;
};

export type RandomPromptCandidate = {
  prompt: string;
  datasetPath: string;
  mediaPath: string;
  captionPath?: string;
  source: 'caption' | 'default_caption';
};

export type RandomPromptState = {
  selected: RandomPromptCandidate | null;
  candidateCount: number;
  scannedMediaCount: number;
  skippedDatasets: string[];
};

export function considerRandomPromptCandidate(
  state: RandomPromptState,
  candidate: RandomPromptCandidate,
  candidateCount = 1,
): void {
  state.candidateCount += candidateCount;
  // A worker's sample represents all of its candidates, preserving uniform selection.
  if (Math.random() < candidateCount / state.candidateCount) state.selected = candidate;
}

export function normalizeRandomPromptCaptionExt(captionExt: unknown) {
  const raw = typeof captionExt === 'string' && captionExt.trim() ? captionExt.trim() : 'txt';
  const normalized = raw.replace(/^\.+/, '').toLowerCase() || 'txt';
  const suffix = `.${normalized}`;
  return RANDOM_PROMPT_CAPTION_EXTENSIONS.has(suffix) ? suffix : '.txt';
}

export function getRandomPromptCaptionExtCandidates(captionExt: unknown) {
  const preferredExt = normalizeRandomPromptCaptionExt(captionExt);
  return Array.from(new Set(['.json', preferredExt, ...RANDOM_PROMPT_CAPTION_EXTENSION_PRIORITY]));
}

function isStructuredJsonCaption(parsed: Record<string, unknown>) {
  return (
    typeof parsed.high_level_description === 'string' ||
    (parsed.style_description !== null && typeof parsed.style_description === 'object') ||
    (parsed.compositional_deconstruction !== null && typeof parsed.compositional_deconstruction === 'object')
  );
}

function parseJsonPromptValue(value: unknown, depth = 0): string {
  if (typeof value === 'string') return parseJsonPromptString(value, depth + 1);
  if (!value || typeof value !== 'object' || Array.isArray(value)) return '';

  const caption = (value as { caption?: unknown }).caption;
  if (typeof caption === 'string') return parseJsonPromptString(caption, depth + 1);

  const parsedObject = value as Record<string, unknown>;
  return isStructuredJsonCaption(parsedObject) ? JSON.stringify(parsedObject) : '';
}

function parseJsonPromptText(jsonText: string, depth = 0): string {
  try {
    return parseJsonPromptValue(JSON.parse(jsonText) as unknown, depth + 1);
  } catch {
    return '';
  }
}

function parseEscapedJsonPromptText(jsonText: string, depth = 0): string {
  try {
    const unescapedText = JSON.parse(`"${jsonText}"`) as unknown;
    return typeof unescapedText === 'string' && unescapedText !== jsonText
      ? parseJsonPromptText(unescapedText, depth + 1)
      : '';
  } catch {
    return '';
  }
}

function parseJsonPromptString(value: string, depth = 0): string {
  const trimmed = value.trim();
  if (!trimmed) return '';
  if (depth > 4) return trimmed;

  return parseJsonPromptText(trimmed, depth + 1) || parseEscapedJsonPromptText(trimmed, depth + 1) || trimmed;
}

export function parseRandomPromptCaptionText(captionText: string, captionExt: string) {
  const normalizedExt = normalizeRandomPromptCaptionExt(captionExt);
  const trimmedCaption = captionText.trim();
  if (normalizedExt !== '.json') return trimmedCaption;

  return parseJsonPromptText(trimmedCaption) || parseEscapedJsonPromptText(trimmedCaption);
}

export function parseRandomPromptCaptionTextAuto(captionText: string) {
  const trimmedCaption = captionText.trim();
  if (!trimmedCaption) return '';

  const parsedJsonCaption = parseRandomPromptCaptionText(trimmedCaption, '.json');
  if (parsedJsonCaption) return parsedJsonCaption;

  try {
    JSON.parse(trimmedCaption);
    return '';
  } catch {
    return trimmedCaption;
  }
}
