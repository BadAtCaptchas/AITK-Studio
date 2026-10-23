import YAML from 'yaml';

/** Remove JSON comments without touching URLs, escapes or comment text in strings. */
function stripJsonComments(text: string): string {
  let result = '';
  let quoted = false;
  let escaped = false;
  let comment: 'line' | 'block' | null = null;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (comment === 'line') {
      if (char === '\n' || char === '\r') comment = null;
      result += char === '\n' || char === '\r' ? char : ' ';
    } else if (comment === 'block') {
      if (char === '*' && next === '/') {
        result += '  ';
        index += 1;
        comment = null;
      } else result += char === '\n' || char === '\r' ? char : ' ';
    } else if (quoted) {
      result += char;
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') quoted = false;
    } else if (char === '/' && (next === '/' || next === '*')) {
      comment = next === '/' ? 'line' : 'block';
      result += '  ';
      index += 1;
    } else {
      result += char;
      if (char === '"') quoted = true;
    }
  }
  if (comment === 'block') throw new Error('Unterminated JSON block comment.');
  return result;
}

export function parseTrainingConfigImport(text: string, filename: string): unknown {
  const source = text.replace(/^\uFEFF/, '');
  const parsed: unknown = /\.jsonc?$/i.test(filename)
    ? JSON.parse(stripJsonComments(source))
    : YAML.parse(source);
  // Draft tracking and the save API both serialize configuration as JSON. YAML
  // permits recursive aliases, which must be rejected before entering React state.
  try { JSON.stringify(parsed); } catch {
    throw new Error('Configuration cannot be serialized. Remove circular references or excessively nested values.');
  }
  return parsed;
}
