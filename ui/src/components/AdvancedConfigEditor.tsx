'use client';
import { useEffect, useState, useRef } from 'react';
import Editor, { OnMount, type Monaco } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';
import { useTheme } from '@/components/ThemeProvider';
import { createConfigEditorSession, readConfigEditorSession, rememberConfigEditorSession, syncConfigEditorSession, updateConfigEditorSession } from '@/utils/configEditorSession';

type Props<T> = {
  config: T;
  setConfig: (value: T, key?: string) => void;
  transformOnParse?: (parsed: unknown) => T;
  onValidityChange?: (valid: boolean) => void;
  onDirtyChange?: (dirty: boolean) => void;
  sessionKey?: string;
  active?: boolean;
  readOnly?: boolean;
  ariaLabel?: string;
};

export default function AdvancedConfigEditor<T>({ config, setConfig, transformOnParse, onValidityChange, onDirtyChange, sessionKey, active = true, readOnly = false, ariaLabel = 'Configuration YAML' }: Props<T>) {
  const { theme } = useTheme();
  const [initialSession] = useState(() => (sessionKey && readConfigEditorSession(sessionKey)) || createConfigEditorSession(config));
  const sessionRef = useRef(initialSession);
  const [errorMessage, setErrorMessage] = useState(initialSession.error);
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);
  const changingModel = useRef(false);

  const updateMarkers = () => {
    const session = sessionRef.current;
    const monaco = monacoRef.current;
    const model = editorRef.current?.getModel();
    if (!monaco || !model) return;
    const line = Math.min(session.errorLine, model.getLineCount());
    monaco.editor.setModelMarkers(model, 'yaml', session.valid ? [] : [{
      severity: monaco.MarkerSeverity.Error, message: session.error,
      startLineNumber: line, startColumn: 1,
      endLineNumber: line, endColumn: model.getLineMaxColumn(line),
    }]);
  };
  const handleMount: OnMount = (instance, monaco) => {
    const session = sessionRef.current;
    editorRef.current = instance;
    monacoRef.current = monaco;
    instance.getModel()?.updateOptions({ tabSize: 2, insertSpaces: true });
    if (instance.getValue() !== session.text) {
      changingModel.current = true;
      instance.getModel()?.setValue(session.text);
      changingModel.current = false;
    }
    if (session.viewState) instance.restoreViewState(session.viewState);
    updateMarkers();
  };

  useEffect(() => {
    const session = sessionRef.current;
    onValidityChange?.(session.valid);
    onDirtyChange?.(session.dirty);
    if (sessionKey) rememberConfigEditorSession(sessionKey, session);
    return () => { session.viewState = editorRef.current?.saveViewState() ?? session.viewState; };
  }, [sessionKey, onValidityChange, onDirtyChange]);

  useEffect(() => {
    const session = sessionRef.current;
    if (!syncConfigEditorSession(session, config)) return;
    const instance = editorRef.current;
    if (instance) {
      const view = instance.saveViewState();
      changingModel.current = true;
      instance.getModel()?.setValue(session.text);
      changingModel.current = false;
      if (view) instance.restoreViewState(view);
    }
    onDirtyChange?.(false);
  }, [config, onDirtyChange]);

  useEffect(() => {
    const session = sessionRef.current;
    if (active) editorRef.current?.layout();
    else session.viewState = editorRef.current?.saveViewState() ?? session.viewState;
  }, [active]);

  const handleChange = (value: string | undefined) => {
    const session = sessionRef.current;
    if (value === undefined || changingModel.current || readOnly) return;
    const previous = session.appliedConfig;
    const result = updateConfigEditorSession(session, value, parsed => {
      if (transformOnParse) return transformOnParse(parsed);
      // Non-training callers retain their existing generic editor contract.
      if (parsed === null || typeof parsed !== 'object') throw new Error('Configuration must be an object.');
      return parsed as T;
    });
    setErrorMessage(session.error);
    onValidityChange?.(session.valid);
    onDirtyChange?.(session.dirty);
    updateMarkers();
    if (result.valid && session.appliedConfig !== previous) setConfig(result.config);
  };

  return (
    <div className="relative h-full min-h-80 w-full" data-config-editor>
      <Editor
        height="100%" width="100%" defaultLanguage="yaml" defaultValue={initialSession.text}
        theme={theme === 'dark' ? 'vs-dark' : 'light'} onChange={handleChange} onMount={handleMount}
        options={{ minimap: { enabled: false }, scrollBeyondLastLine: false, automaticLayout: true,
          tabSize: 2, insertSpaces: true, detectIndentation: false, fontSize: 13, padding: { top: 12 },
          ariaLabel, readOnly }}
      />
      {errorMessage && (
        <div role="alert" className="absolute bottom-3 left-3 right-3 z-20 rounded border border-rose-700 bg-gray-950 p-3 text-sm text-rose-700 dark:text-rose-300">
          {errorMessage} Your text is retained. Fix it in Config to edit the form or save.
        </div>
      )}
    </div>
  );
}
