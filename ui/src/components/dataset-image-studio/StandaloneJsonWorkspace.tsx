'use client';

import Editor, { type BeforeMount, type OnMount } from '@monaco-editor/react';
import classNames from 'classnames';
import type { editor as MonacoEditor } from 'monaco-editor';
import {
  Braces,
  CheckCircle2,
  Clipboard,
  ClipboardCheck,
  Image as ImageIcon,
  Layers3,
  LoaderCircle,
  Palette,
  PanelTop,
  Redo2,
  SlidersHorizontal,
  TriangleAlert,
  Undo2,
  Wrench,
  XCircle,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

export type CaptionJsonValidity = 'valid' | 'invalid' | 'checking';

export type CaptionJsonIssue = {
  id?: string;
  message: string;
  path?: string;
  line?: number;
  column?: number;
  endLine?: number;
  endColumn?: number;
};

export type CaptionJsonSectionKind = 'high-level' | 'style' | 'background' | 'elements';

export type CaptionJsonNavigatorSection = {
  id: string;
  label: string;
  path: string;
  kind: CaptionJsonSectionKind;
  count?: number;
};

export type CaptionJsonNavigatorElement = {
  index: number;
  label: string;
  type?: string;
  path: string;
  /** Optional source range used to reveal and highlight this element in the editor. */
  startLine?: number;
  endLine?: number;
};

export type StandaloneJsonWorkspaceProps = {
  value: string;
  validity: CaptionJsonValidity;
  errors: readonly CaptionJsonIssue[];
  warnings: readonly CaptionJsonIssue[];
  sections: readonly CaptionJsonNavigatorSection[];
  elements: readonly CaptionJsonNavigatorElement[];
  selectedSectionId: string;
  selectedPath: string;
  selectedElementIndex: number | null;
  disabled?: boolean;
  disabledReason?: string;
  isApplying?: boolean;
  canApply?: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onChange: (value: string) => void;
  onApply: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onFormat: () => void;
  onSelectSection: (sectionId: string, path: string) => void;
  onSelectElement: (elementIndex: number, path: string) => void;
};

const EDITOR_THEME = 'aitk-caption-json';

function SectionIcon({ kind }: { kind: CaptionJsonSectionKind }) {
  const iconClassName = 'h-3.5 w-3.5 flex-none';
  if (kind === 'style') return <Palette className={iconClassName} aria-hidden="true" />;
  if (kind === 'background') return <ImageIcon className={iconClassName} aria-hidden="true" />;
  if (kind === 'elements') return <Layers3 className={iconClassName} aria-hidden="true" />;
  return <PanelTop className={iconClassName} aria-hidden="true" />;
}

function issueTitle(errors: readonly CaptionJsonIssue[], warnings: readonly CaptionJsonIssue[]) {
  const messages = [
    ...errors.map(issue => `Error: ${issue.message}`),
    ...warnings.map(issue => `Warning: ${issue.message}`),
  ];
  return messages.length > 0 ? messages.join('\n') : 'No JSON issues';
}

function clampLine(value: number | undefined, lineCount: number) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 1;
  return Math.min(lineCount, Math.max(1, Math.floor(value)));
}

export function StandaloneJsonWorkspace({
  value,
  validity,
  errors,
  warnings,
  sections,
  elements,
  selectedSectionId,
  selectedPath,
  selectedElementIndex,
  disabled = false,
  disabledReason = '',
  isApplying = false,
  canApply = true,
  canUndo,
  canRedo,
  onChange,
  onApply,
  onUndo,
  onRedo,
  onFormat,
  onSelectSection,
  onSelectElement,
}: StandaloneJsonWorkspaceProps) {
  const editorRef = useRef<MonacoEditor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof import('monaco-editor') | null>(null);
  const decorationsRef = useRef<MonacoEditor.IEditorDecorationsCollection | null>(null);
  const cursorListenerRef = useRef<{ dispose: () => void } | null>(null);
  const copiedTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [editorMountVersion, setEditorMountVersion] = useState(0);
  const [cursor, setCursor] = useState({ line: 1, column: 1 });
  const [copied, setCopied] = useState(false);

  const selectedElement = useMemo(
    () => elements.find(element => element.index === selectedElementIndex) ?? null,
    [elements, selectedElementIndex],
  );
  const problemsTitle = useMemo(() => issueTitle(errors, warnings), [errors, warnings]);

  const handleBeforeMount: BeforeMount = monaco => {
    monaco.editor.defineTheme(EDITOR_THEME, {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'string.key.json', foreground: '9CDCF8' },
        { token: 'string.value.json', foreground: 'D6D694' },
        { token: 'number', foreground: 'A8D2A4' },
        { token: 'keyword', foreground: '55D9F2' },
        { token: 'delimiter.bracket.json', foreground: 'D7E0E8' },
      ],
      colors: {
        'editor.background': '#031019',
        'editor.foreground': '#D5DEE6',
        'editorLineNumber.foreground': '#70808E',
        'editorLineNumber.activeForeground': '#B9C6D0',
        'editorCursor.foreground': '#22D3EE',
        'editor.selectionBackground': '#123E59',
        'editor.inactiveSelectionBackground': '#0D2D41',
        'editor.lineHighlightBackground': '#071721',
        'editorIndentGuide.background1': '#17303F',
        'editorIndentGuide.activeBackground1': '#315063',
        'editorBracketHighlight.foreground1': '#9CDCF8',
        'editorBracketHighlight.foreground2': '#D6D694',
        'editorBracketHighlight.foreground3': '#C7A0E8',
        'editorOverviewRuler.border': '#00000000',
        'scrollbar.shadow': '#00000000',
        'scrollbarSlider.background': '#31404D66',
        'scrollbarSlider.hoverBackground': '#48596888',
        'scrollbarSlider.activeBackground': '#5B6E7D99',
      },
    });
  };

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    decorationsRef.current = editor.createDecorationsCollection();
    cursorListenerRef.current?.dispose();
    cursorListenerRef.current = editor.onDidChangeCursorPosition(event => {
      setCursor({ line: event.position.lineNumber, column: event.position.column });
    });
    const position = editor.getPosition();
    if (position) setCursor({ line: position.lineNumber, column: position.column });
    setEditorMountVersion(version => version + 1);
  };

  useEffect(() => {
    return () => {
      cursorListenerRef.current?.dispose();
      decorationsRef.current?.clear();
      if (copiedTimeoutRef.current) clearTimeout(copiedTimeoutRef.current);
    };
  }, []);

  useEffect(() => {
    const editor = editorRef.current;
    const monaco = monacoRef.current;
    const model = editor?.getModel();
    if (!editor || !monaco || !model) return;

    const lineCount = Math.max(1, model.getLineCount());
    const markers: MonacoEditor.IMarkerData[] = [];

    for (const issue of errors) {
      if (issue.line == null) continue;
      const startLineNumber = clampLine(issue.line, lineCount);
      const endLineNumber = clampLine(issue.endLine ?? issue.line, lineCount);
      markers.push({
        severity: monaco.MarkerSeverity.Error,
        message: issue.message,
        startLineNumber,
        startColumn: Math.max(1, issue.column ?? 1),
        endLineNumber,
        endColumn: Math.max(2, issue.endColumn ?? model.getLineMaxColumn(endLineNumber)),
      });
    }

    for (const issue of warnings) {
      if (issue.line == null) continue;
      const startLineNumber = clampLine(issue.line, lineCount);
      const endLineNumber = clampLine(issue.endLine ?? issue.line, lineCount);
      markers.push({
        severity: monaco.MarkerSeverity.Warning,
        message: issue.message,
        startLineNumber,
        startColumn: Math.max(1, issue.column ?? 1),
        endLineNumber,
        endColumn: Math.max(2, issue.endColumn ?? model.getLineMaxColumn(endLineNumber)),
      });
    }

    monaco.editor.setModelMarkers(model, 'caption-json', markers);
  }, [editorMountVersion, errors, value, warnings]);

  useEffect(() => {
    const editor = editorRef.current;
    const model = editor?.getModel();
    const decorations = decorationsRef.current;
    if (!editor || !model || !decorations) return;

    if (selectedElement?.startLine == null) {
      decorations.clear();
      const propertyName =
        selectedSectionId === 'high-level'
          ? selectedPath === 'caption'
            ? 'caption'
            : 'high_level_description'
          : selectedSectionId === 'background'
            ? 'background'
            : selectedSectionId === 'style'
              ? 'style_description'
              : 'elements';
      const matchingLine = model.getLinesContent().findIndex(line => line.includes(`"${propertyName}"`));
      if (matchingLine >= 0) editor.revealLineInCenterIfOutsideViewport(matchingLine + 1);
      return;
    }

    const lineCount = Math.max(1, model.getLineCount());
    const startLine = clampLine(selectedElement.startLine, lineCount);
    const endLine = clampLine(selectedElement.endLine ?? selectedElement.startLine, lineCount);
    decorations.set([
      {
        range: {
          startLineNumber: startLine,
          startColumn: 1,
          endLineNumber: endLine,
          endColumn: model.getLineMaxColumn(endLine),
        },
        options: {
          isWholeLine: true,
          className: 'aitk-caption-json-selected-element',
          linesDecorationsClassName: 'aitk-caption-json-selected-element-gutter',
        },
      },
    ]);
    editor.revealLineInCenterIfOutsideViewport(startLine);
  }, [editorMountVersion, selectedElement, selectedPath, selectedSectionId, value]);

  const copyValue = async () => {
    if (!navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      if (copiedTimeoutRef.current) clearTimeout(copiedTimeoutRef.current);
      copiedTimeoutRef.current = setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const runEditorHistoryAction = (action: 'undo' | 'redo', fallback: () => void) => {
    const editor = editorRef.current;
    if (!editor) {
      fallback();
      return;
    }
    editor.trigger('caption-json-toolbar', action, null);
  };

  const validityLabel = validity === 'valid' ? 'Valid JSON' : validity === 'invalid' ? 'Invalid JSON' : 'Checking JSON';
  const applyDisabled = disabled || isApplying || !canApply || validity !== 'valid';
  const selectedPathLabel = selectedPath.replace(/^compositional_deconstruction\./, '') || 'document';

  return (
    <section
      className="flex h-full min-h-0 min-w-0 overflow-hidden bg-[#031019] text-gray-100"
      aria-label="Caption JSON workspace"
      title={disabled ? disabledReason : undefined}
    >
      <nav
        className="flex w-[145px] flex-none flex-col border-r border-[#283640] bg-[#04111a]"
        aria-label="JSON sections"
      >
        <div className="flex h-12 flex-none items-center border-b border-[#27343e] px-4 text-gray-200">
          <Wrench className="h-4 w-4" aria-hidden="true" />
          <span className="sr-only">JSON structure</span>
        </div>

        <div className="operator-scrollbar-none min-h-0 flex-1 overflow-y-auto">
          {sections.map(section => {
            const selected = selectedSectionId === section.id;
            return (
              <div key={section.id}>
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => onSelectSection(section.id, section.path)}
                  aria-current={selected ? 'true' : undefined}
                  className={classNames(
                    'flex h-[33px] w-full items-center gap-2 px-3 text-left text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-cyan-300 disabled:cursor-not-allowed disabled:opacity-45',
                    selected
                      ? 'border-r-2 border-cyan-300 bg-[#082432] text-cyan-100'
                      : 'border-r-2 border-transparent text-gray-200 hover:bg-[#0a1a24] hover:text-white',
                  )}
                >
                  <SectionIcon kind={section.kind} />
                  <span className="min-w-0 flex-1 truncate">{section.label}</span>
                  {section.count != null ? (
                    <span className="inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-full bg-[#312a12] px-1 text-[10px] font-semibold tabular-nums text-amber-300">
                      {section.count}
                    </span>
                  ) : null}
                </button>

                {section.kind === 'elements' ? (
                  <div className="pb-1">
                    {elements.map(element => {
                      const elementSelected = element.index === selectedElementIndex;
                      return (
                        <button
                          key={`${element.index}-${element.path}`}
                          type="button"
                          disabled={disabled}
                          onClick={() => onSelectElement(element.index, element.path)}
                          aria-current={elementSelected ? 'true' : undefined}
                          className={classNames(
                            'grid min-h-[49px] w-full grid-cols-[16px_minmax(0,1fr)] items-start gap-1.5 border-r-2 px-2 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-cyan-300 disabled:cursor-not-allowed disabled:opacity-45',
                            elementSelected
                              ? 'border-cyan-300 bg-[#07364b] text-white'
                              : 'border-transparent text-gray-300 hover:bg-[#0a1a24] hover:text-white',
                          )}
                        >
                          <span className="pt-px text-[10px] tabular-nums text-gray-500">{element.index + 1}</span>
                          <span className="min-w-0">
                            <span className="block truncate text-[11px] font-medium leading-4" title={element.label}>
                              {element.label}
                            </span>
                            {element.type ? (
                              <span
                                className="mt-0.5 block truncate text-[10px] leading-4 text-gray-500"
                                title={element.type}
                              >
                                {element.type}
                              </span>
                            ) : null}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </nav>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col bg-[#031019]">
        <header className="flex h-12 flex-none items-center gap-2 border-b border-[#27343e] px-5">
          <h2 className="flex-none text-sm font-semibold tracking-tight text-gray-100">Caption JSON</h2>

          <span
            className={classNames(
              'ml-2 inline-flex h-7 flex-none items-center gap-1.5 rounded-[4px] border px-2.5 text-[10px] font-medium',
              validity === 'valid' && 'border-emerald-900/70 bg-emerald-950/20 text-emerald-300',
              validity === 'invalid' && 'border-red-900/80 bg-red-950/20 text-red-300',
              validity === 'checking' && 'border-cyan-900/70 bg-cyan-950/20 text-cyan-300',
            )}
            title={problemsTitle}
            role="status"
            aria-live="polite"
          >
            {validity === 'valid' ? (
              <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
            ) : validity === 'invalid' ? (
              <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
            ) : (
              <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
            )}
            {validityLabel}
          </span>

          <span
            className="min-w-0 flex-1 truncate border-l border-[#26333d] pl-3 text-[11px] text-gray-400"
            title={selectedPathLabel}
          >
            {selectedPathLabel}
          </span>

          <div className="flex flex-none items-center gap-1">
            <button
              type="button"
              disabled={disabled || !canUndo}
              onClick={() => runEditorHistoryAction('undo', onUndo)}
              className="flex h-8 w-8 items-center justify-center rounded-[4px] text-gray-400 transition-colors hover:bg-[#10202a] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:cursor-not-allowed disabled:opacity-30"
              title="Undo JSON change"
              aria-label="Undo JSON change"
            >
              <Undo2 className="h-4 w-4" aria-hidden="true" />
            </button>
            <button
              type="button"
              disabled={disabled || !canRedo}
              onClick={() => runEditorHistoryAction('redo', onRedo)}
              className="flex h-8 w-8 items-center justify-center rounded-[4px] text-gray-400 transition-colors hover:bg-[#10202a] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:cursor-not-allowed disabled:opacity-30"
              title="Redo JSON change"
              aria-label="Redo JSON change"
            >
              <Redo2 className="h-4 w-4" aria-hidden="true" />
            </button>
            <button
              type="button"
              disabled={disabled}
              onClick={() => void copyValue()}
              className="flex h-8 w-8 items-center justify-center rounded-[4px] text-gray-400 transition-colors hover:bg-[#10202a] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:cursor-not-allowed disabled:opacity-30"
              title={copied ? 'Copied JSON' : 'Copy JSON'}
              aria-label={copied ? 'Copied JSON' : 'Copy JSON'}
            >
              {copied ? (
                <ClipboardCheck className="h-4 w-4 text-emerald-300" aria-hidden="true" />
              ) : (
                <Clipboard className="h-4 w-4" aria-hidden="true" />
              )}
            </button>
            <button
              type="button"
              disabled={disabled}
              onClick={onFormat}
              className="flex h-8 w-8 items-center justify-center rounded-[4px] text-gray-300 transition-colors hover:bg-[#10202a] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:cursor-not-allowed disabled:opacity-30"
              title="Format JSON"
              aria-label="Format JSON"
            >
              <Braces className="h-4 w-4" aria-hidden="true" />
            </button>
            <button
              type="button"
              disabled={applyDisabled}
              onClick={onApply}
              className="ml-1 inline-flex h-8 min-w-[78px] items-center justify-center gap-1.5 rounded-[4px] border border-cyan-300/70 bg-cyan-400 px-2 text-[11px] font-semibold text-[#03121a] shadow-[0_1px_8px_rgba(34,211,238,0.12)] transition-colors hover:bg-cyan-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-100 disabled:cursor-not-allowed disabled:border-gray-700 disabled:bg-gray-800 disabled:text-gray-500"
              title={applyDisabled && disabledReason ? disabledReason : 'Apply JSON'}
            >
              {isApplying ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : null}
              Apply JSON
            </button>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-hidden" aria-label="Raw caption JSON editor">
          <Editor
            height="100%"
            width="100%"
            language="json"
            value={value}
            theme={EDITOR_THEME}
            beforeMount={handleBeforeMount}
            onMount={handleMount}
            onChange={nextValue => onChange(nextValue ?? '')}
            loading={
              <div className="flex h-full items-center justify-center bg-[#031019] text-xs text-gray-500">
                Loading JSON editor…
              </div>
            }
            options={{
              automaticLayout: true,
              bracketPairColorization: { enabled: true },
              cursorBlinking: 'smooth',
              cursorSmoothCaretAnimation: 'on',
              folding: true,
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
              fontLigatures: false,
              fontSize: 12,
              glyphMargin: false,
              guides: { bracketPairs: true, indentation: true },
              insertSpaces: true,
              lineHeight: 18,
              lineNumbersMinChars: 3,
              minimap: { enabled: false },
              padding: { top: 9, bottom: 9 },
              readOnly: disabled,
              renderLineHighlight: 'line',
              scrollBeyondLastLine: false,
              scrollbar: {
                horizontalScrollbarSize: 8,
                verticalScrollbarSize: 8,
                useShadows: false,
              },
              smoothScrolling: true,
              stickyScroll: { enabled: false },
              tabSize: 2,
              wordWrap: 'off',
            }}
          />
        </div>

        <footer className="flex h-7 flex-none items-center border-t border-[#27343e] bg-[#07131c] px-4 text-[10px] text-gray-400">
          <div className="flex items-center gap-3" title={problemsTitle}>
            <span
              className={classNames(
                'inline-flex items-center gap-1',
                errors.length > 0 ? 'text-red-300' : 'text-emerald-300',
              )}
            >
              {errors.length > 0 ? (
                <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
              ) : (
                <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
              )}
              {errors.length} {errors.length === 1 ? 'error' : 'errors'}
            </span>
            <span
              className={classNames(
                'inline-flex items-center gap-1',
                warnings.length > 0 ? 'text-amber-300' : 'text-gray-500',
              )}
            >
              <TriangleAlert className="h-3.5 w-3.5" aria-hidden="true" />
              {warnings.length} {warnings.length === 1 ? 'warning' : 'warnings'}
            </span>
          </div>

          <div className="ml-auto flex items-center gap-5 tabular-nums">
            <span>
              Ln {cursor.line}, Col {cursor.column}
            </span>
            <span>Spaces: 2</span>
            <span>UTF-8</span>
            <button
              type="button"
              disabled={disabled}
              onClick={onFormat}
              className="inline-flex h-6 items-center gap-1.5 px-1 text-gray-300 transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:cursor-not-allowed disabled:opacity-30"
            >
              <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
              Format
            </button>
          </div>
        </footer>
      </div>

      <style jsx global>{`
        .monaco-editor .aitk-caption-json-selected-element {
          background: rgba(8, 82, 118, 0.36);
          border-top: 1px solid rgba(34, 211, 238, 0.52);
          border-bottom: 1px solid rgba(34, 211, 238, 0.52);
        }

        .monaco-editor .aitk-caption-json-selected-element-gutter {
          border-left: 2px solid rgb(34 211 238);
        }
      `}</style>
    </section>
  );
}
