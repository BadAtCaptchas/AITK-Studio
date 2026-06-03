export const RANDOM_PROMPT_CAPTION_EXTENSIONS = new Set(['.txt', '.caption', '.json', '.sdxl', '.md']);

export function normalizeRandomPromptCaptionExt(captionExt: unknown) {
  const raw = typeof captionExt === 'string' && captionExt.trim() ? captionExt.trim() : 'txt';
  const normalized = raw.replace(/^\.+/, '').toLowerCase() || 'txt';
  const suffix = `.${normalized}`;
  return RANDOM_PROMPT_CAPTION_EXTENSIONS.has(suffix) ? suffix : '.txt';
}

export function parseRandomPromptCaptionText(captionText: string, captionExt: string) {
  const normalizedExt = normalizeRandomPromptCaptionExt(captionExt);
  if (normalizedExt !== '.json') return captionText.trim();

  try {
    const parsed = JSON.parse(captionText) as { caption?: unknown } | null;
    return typeof parsed?.caption === 'string' ? parsed.caption.trim() : '';
  } catch {
    return '';
  }
}
