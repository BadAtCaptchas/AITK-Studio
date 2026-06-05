export const RANDOM_PROMPT_CAPTION_EXTENSION_PRIORITY = ['.json', '.txt', '.caption', '.sdxl', '.md'];
export const RANDOM_PROMPT_CAPTION_EXTENSIONS = new Set(RANDOM_PROMPT_CAPTION_EXTENSION_PRIORITY);

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

export function parseRandomPromptCaptionText(captionText: string, captionExt: string) {
  const normalizedExt = normalizeRandomPromptCaptionExt(captionExt);
  const trimmedCaption = captionText.trim();
  if (normalizedExt !== '.json') return trimmedCaption;

  try {
    const parsed = JSON.parse(captionText) as unknown;
    if (typeof parsed === 'string') return parsed.trim();
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return '';

    const caption = (parsed as { caption?: unknown }).caption;
    if (typeof caption === 'string') return caption.trim();

    const parsedObject = parsed as Record<string, unknown>;
    return isStructuredJsonCaption(parsedObject) ? JSON.stringify(parsedObject) : '';
  } catch {
    return '';
  }
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
