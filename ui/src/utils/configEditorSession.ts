import YAML from 'yaml';
import type { editor } from 'monaco-editor';

export type ConfigEditorSession = {
  text: string;
  appliedConfig: string;
  baselineText: string;
  dirty: boolean;
  valid: boolean;
  error: string;
  errorLine: number;
  viewState: editor.ICodeEditorViewState | null;
};

// Raw configuration may contain credentials. Keep it only in this tab's memory.
const sessions = new Map<string, ConfigEditorSession>();
export const readConfigEditorSession = (key: string): ConfigEditorSession | undefined => sessions.get(key);
export const clearConfigEditorSession = (key: string): void => { sessions.delete(key); };
export const rememberConfigEditorSession = (key: string, session: ConfigEditorSession): void => {
  sessions.set(key, session);
};

export function configToYaml(config: unknown): string {
  const options = { directives: true, indent: 2, lineWidth: 0, defaultStringType: 'QUOTE_DOUBLE' as const, defaultKeyType: 'PLAIN' as const };
  const doc = new YAML.Document(config);
  YAML.visit(doc, {
    Scalar(_key, node) {
      if (typeof node.value === 'string' && node.value.includes('\n')) node.type = YAML.Scalar.BLOCK_LITERAL;
    },
  });
  return doc.toString(options);
}

export function createConfigEditorSession(config: unknown): ConfigEditorSession {
  const text = configToYaml(config);
  return { text, baselineText: text, appliedConfig: JSON.stringify(config), dirty: false, valid: true, error: '', errorLine: 1, viewState: null };
}

export function syncConfigEditorSession(session: ConfigEditorSession, config: unknown): boolean {
  const serialized = JSON.stringify(config);
  // An unfinished edit must never disappear because a parent rerendered.
  if (!session.valid || serialized === session.appliedConfig) return false;
  session.text = configToYaml(config);
  session.baselineText = session.text;
  session.appliedConfig = serialized;
  session.dirty = false;
  return true;
}

export function updateConfigEditorSession<T>(
  session: ConfigEditorSession,
  text: string,
  transform: (parsed: unknown) => T,
): { valid: true; config: T } | { valid: false } {
  session.text = text;
  session.dirty = text !== session.baselineText;
  try {
    const parsed: unknown = YAML.parse(text);
    const config = transform(parsed);
    session.appliedConfig = JSON.stringify(config);
    session.valid = true;
    session.error = '';
    return { valid: true, config };
  } catch (error: unknown) {
    session.valid = false;
    session.error = error instanceof Error ? error.message : 'Invalid YAML';
    const positions = error instanceof YAML.YAMLParseError ? error.linePos : undefined;
    session.errorLine = positions?.[0]?.line ?? 1;
    return { valid: false };
  }
}
