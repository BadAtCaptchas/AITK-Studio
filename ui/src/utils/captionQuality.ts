const FIRST_PERSON_REFUSAL = String.raw`\bi(?:\s+(?:can(?:not|'t)|won't(?:\s+be\s+able\s+to)?|will\s+not(?:\s+be\s+able\s+to)?|would\s+not\s+be\s+able\s+to)|(?:'m|\s+am)\s+(?:unable\s+to|not\s+able\s+to))`;

const REFUSAL_PATTERNS = [
  new RegExp(
    String.raw`${FIRST_PERSON_REFUSAL}\s+(?:fulfill|fulfil|comply\s+with|assist\s+with|help\s+with|process|complete|accommodate)\s+(?:this|that|the|your)\s+(?:request|prompt)\b`,
    'i',
  ),
  new RegExp(
    String.raw`${FIRST_PERSON_REFUSAL}\s+(?:assist|help|comply)\s+(?:with\s+)?(?:that|this|it|you)\b`,
    'i',
  ),
  new RegExp(
    String.raw`${FIRST_PERSON_REFUSAL}\s+(?:assist|help|comply)\s+with\s+(?:requests?|content|material|images?|prompts?)\b`,
    'i',
  ),
  new RegExp(
    String.raw`${FIRST_PERSON_REFUSAL}\s+(?:provide|generate|create|caption|describe|produce|write|answer|respond)\b`,
    'i',
  ),
  new RegExp(
    String.raw`${FIRST_PERSON_REFUSAL}\s+(?:help|assist)\s+(?:provide|generate|create|caption|describe|produce|write|answer|respond)\b`,
    'i',
  ),
  /\b(?:cannot|can't|unable\s+to|not\s+able\s+to)\s+(?:fulfill|fulfil|comply\s+with|assist\s+with|help\s+with|process|complete|accommodate)\s+(?:this|that|the|your)\s+(?:request|prompt)\b/i,
  /\b(?:(?:i'm|i\s+am)\s+)?sorry\b.{0,120}\b(?:can't|cannot|unable|not\s+able|won't|will\s+not)\b/i,
  /\b(?:i\s+apologi[sz]e|apologies)\b.{0,120}\b(?:can't|cannot|unable|not\s+able|won't|will\s+not)\b/i,
  /\b(?:i'm|i\s+am)\s+afraid\b.{0,120}\b(?:can't|cannot|unable|not\s+able|won't|will\s+not)\b/i,
  /\b(?:as\s+an?\s+(?:ai|language\s+model|assistant)|i\s+am\s+an?\s+(?:ai|language\s+model|assistant))\b.{0,160}\b(?:can't|cannot|unable|not\s+able|don't|do\s+not|won't|will\s+not)\b/i,
  /\b(?:against|violates?|breach(?:es)?)\s+(?:my\s+)?(?:policy|policies|guidelines?|safety\s+guidelines?|content\s+policy)\b/i,
  /\b(?:not\s+allowed|not\s+permitted|disallowed|prohibited)\s+(?:by|under)\s+(?:the\s+)?(?:policy|policies|guidelines?|safety\s+guidelines?|content\s+policy)\b/i,
  /\b(?:outside|beyond)\s+(?:my\s+)?(?:policy|policies|guidelines?|safety\s+guidelines?|content\s+policy|capabilities|scope)\b/i,
  /\brequest\s+(?:denied|rejected|refused|declined)\b/i,
  /\b(?:i\s+)?(?:must|have\s+to|need\s+to)\s+(?:refuse|decline)\b/i,
  /\b(?:(?:not|isn't)\s+(?:appropriate|safe)|would\s+be\s+(?:inappropriate|unsafe))\b.{0,80}\b(?:assist(?:ing)?|help(?:ing)?|provid(?:e|ing)|generat(?:e|ing)|creat(?:e|ing)|describ(?:e|ing)|caption(?:ing)?|comply(?:ing)?)\b/i,
  /\bi\s+(?:do\s+not|don't)\s+(?:feel\s+comfortable|feel\s+able)\b.{0,80}\b(?:assist(?:ing)?|help(?:ing)?|provid(?:e|ing)|generat(?:e|ing)|creat(?:e|ing)|describ(?:e|ing)|caption(?:ing)?|comply(?:ing)?)\b/i,
  new RegExp(
    String.raw`${FIRST_PERSON_REFUSAL}\s+(?:view|see|access|analy[sz]e|process|interpret|inspect)\s+(?:the|this|that)\s+(?:image|file|picture|photo)\b`,
    'i',
  ),
  /\b(?:no|without\s+an?)\s+(?:image|file|picture|photo)\s+(?:was\s+)?(?:provided|attached|uploaded|included|available)\b/i,
  /\b(?:image|file|picture|photo)\s+(?:is|was)\s+(?:not\s+)?(?:accessible|available|provided|attached|uploaded)\b/i,
  /\bplease\s+(?:provide|upload|attach|include|send|share)\s+(?:the|an?|your)?\s*(?:image|video|file|picture|photo)(?:\s+or\s+(?:image|video|file|picture|photo))*\s+(?:you\s+would\s+like\s+me\s+to|for\s+me\s+to|to)\s+(?:caption|describe|analy[sz]e|inspect|process)\b/i,
];

export function isRefusalCaption(caption: string) {
  const normalized = caption.trim().replace(/[\u2018\u2019]/g, "'").replace(/\s+/g, ' ');
  if (!normalized) return false;
  return REFUSAL_PATTERNS.some(pattern => pattern.test(normalized));
}

export function isFailedCaption(caption: string) {
  return !caption.trim() || isRefusalCaption(caption);
}

export function captionFailureReason(caption: string) {
  if (!caption.trim()) return 'Caption is empty.';
  if (isRefusalCaption(caption)) return 'Captioner returned a refusal instead of a usable caption.';
  return '';
}

export function assertUsableCaption(caption: string) {
  const reason = captionFailureReason(caption);
  if (reason) throw new Error(reason);
}
