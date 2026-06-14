'use client';

import { useCallback, useEffect, useMemo, useRef, useState, type Ref } from 'react';
import classNames from 'classnames';
import {
  Clipboard,
  Code2,
  Copy,
  Download,
  ExternalLink,
  FileJson2,
  History,
  Image as ImageIcon,
  Loader2,
  MousePointer2,
  Move,
  Plus,
  RefreshCw,
  Save,
  Send,
  Sparkles,
  Square,
  Trash2,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import { ProgressBar, StatusBadge } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import {
  buildIdeogramComfyWorkflow,
  cloneIdeogramWorkflowState,
  DEFAULT_IDEOGRAM_WORKFLOW_STATE,
  closestIdeogramAspectRatio,
  IDEOGRAM_ASPECT_RATIOS,
  IDEOGRAM4_QUALITY_PRESETS,
  parseIdeogramAspectRatio,
  parseIdeogramComfyWorkflow,
  serializeIdeogramWorkflowPrompt,
  updateIdeogramWorkflowElementBox,
  type IdeogramImportResult,
  type IdeogramQualityPreset,
  type IdeogramWorkflowElement,
  type IdeogramWorkflowLora,
  type IdeogramWorkflowState,
} from '@/utils/ideogramWorkflow';
import { normalizeBox } from '@/utils/ideogramCaption';

type PanelTab = 'preview' | 'json' | 'comfy';
type ToolMode = 'select' | 'move' | 'object' | 'text';
type PreflightStatus = 'found' | 'missing' | 'unknown';

type PreflightItem = {
  id: string;
  label: string;
  status: PreflightStatus;
  detail: string;
};

type PreflightResult = {
  ok: boolean;
  serverUrl: string;
  connected: boolean;
  nodes: PreflightItem[];
  models: PreflightItem[];
  error: string | null;
};

type ComfyImageRef = {
  filename: string;
  subfolder: string;
  type: string;
};

type PreviewImageSource = 'imported' | 'result';

type PreviewImage = {
  id: string;
  objectUrl: string;
  filename: string;
  source: PreviewImageSource;
};

type ResultImage = ComfyImageRef & PreviewImage & {
  source: 'result';
};

type ToolkitLoraSummary = {
  id: string;
  label: string;
  path: string;
  filename: string;
  source: 'job' | 'uploaded';
  sizeBytes: number;
  updatedAt: string;
  triggerWords: string[];
};

type ImportedCanvasImage = PreviewImage & {
  source: 'imported';
  width: number;
  height: number;
  aspectRatio: string;
};

type GenerationState = {
  status: 'idle' | 'connecting' | 'queued' | 'executing' | 'completed' | 'error' | 'canceled';
  promptId: string;
  clientId: string;
  queuePosition: string;
  executingNode: string;
  step: number;
  maxStep: number;
  message: string;
  elapsedStart: number | null;
  error: string;
};

const DEFAULT_EXTERNAL_COMFY_URL = 'http://127.0.0.1:8188';

const QUALITY_OPTIONS: IdeogramQualityPreset[] = ['Turbo', 'Default', 'Quality'];

const EMPTY_GENERATION: GenerationState = {
  status: 'idle',
  promptId: '',
  clientId: '',
  queuePosition: '-',
  executingNode: '-',
  step: 0,
  maxStep: 0,
  message: 'Ready',
  elapsedStart: null,
  error: '',
};

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function shortText(value: string, max = 76) {
  const trimmed = value.trim().replace(/\s+/g, ' ');
  return trimmed.length > max ? `${trimmed.slice(0, max - 1)}...` : trimmed;
}

function normalizedWsUrl(serverUrl: string, clientId: string) {
  const url = new URL(serverUrl);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.pathname = `${url.pathname.replace(/\/+$/, '')}/ws`;
  url.search = '';
  url.searchParams.set('clientId', clientId);
  return url.toString();
}

function generationStatusLabel(status: GenerationState['status']) {
  if (status === 'connecting') return 'Connecting';
  if (status === 'queued') return 'Queued';
  if (status === 'executing') return 'Executing';
  if (status === 'completed') return 'Completed';
  if (status === 'error') return 'Error';
  if (status === 'canceled') return 'Canceled';
  return 'Idle';
}

function generationBadgeStatus(status: GenerationState['status']) {
  if (status === 'completed') return 'completed';
  if (status === 'error') return 'error';
  if (status === 'canceled') return 'stopped';
  if (status === 'idle') return 'stopped';
  return 'running';
}

function formatElapsed(start: number | null, nowTick: number) {
  if (!start) return '-';
  const seconds = Math.max(0, Math.floor((nowTick - start) / 1000));
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, '0')}`;
}

function Field({
  label,
  children,
  detail,
}: {
  label: string;
  children: React.ReactNode;
  detail?: string;
}) {
  return (
    <label className="block">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-gray-300">{label}</span>
        {detail ? <span className="truncate text-[11px] text-gray-500">{detail}</span> : null}
      </div>
      {children}
    </label>
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
  type = 'text',
  inputRef,
  list,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: string;
  inputRef?: Ref<HTMLInputElement>;
  list?: string;
}) {
  return (
    <input
      ref={inputRef}
      type={type}
      list={list}
      value={value}
      onChange={event => onChange(event.target.value)}
      placeholder={placeholder}
      className="h-9 w-full rounded-sm border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600 focus:border-cyan-700"
    />
  );
}

function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <input
      type="number"
      value={Number.isFinite(value) ? value : ''}
      min={min}
      max={max}
      step={step}
      onChange={event => {
        const next = Number(event.target.value);
        if (Number.isFinite(next)) onChange(next);
      }}
      className="h-9 w-full rounded-sm border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-700"
    />
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: string[];
}) {
  return (
    <select
      value={value}
      onChange={event => onChange(event.target.value)}
      className="h-9 w-full rounded-sm border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-700"
    >
      {options.map(option => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}

function IconButton({
  title,
  children,
  onClick,
  disabled,
  active,
  danger,
}: {
  title: string;
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  active?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      disabled={disabled}
      onClick={onClick}
      className={classNames(
        'inline-flex h-8 w-8 items-center justify-center rounded-sm border text-gray-300 transition-colors disabled:cursor-not-allowed disabled:opacity-45',
        active
          ? 'border-cyan-500/70 bg-cyan-500/15 text-cyan-100'
          : danger
            ? 'border-rose-900 bg-rose-950/15 text-rose-200 hover:bg-rose-950/35'
            : 'border-gray-800 bg-gray-950 hover:border-gray-700 hover:bg-gray-900 hover:text-gray-100',
      )}
    >
      {children}
    </button>
  );
}

function ActionButton({
  children,
  onClick,
  disabled,
  tone = 'neutral',
  type = 'button',
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  tone?: 'neutral' | 'cyan' | 'emerald' | 'rose';
  type?: 'button' | 'submit';
}) {
  const toneClass = {
    neutral: 'border-gray-800 bg-gray-950 text-gray-200 hover:bg-gray-900',
    cyan: 'border-cyan-700 bg-cyan-500 text-gray-950 hover:bg-cyan-400',
    emerald: 'border-emerald-800 bg-emerald-950/50 text-emerald-100 hover:bg-emerald-900',
    rose: 'border-rose-800 bg-rose-950/40 text-rose-100 hover:bg-rose-900/80',
  }[tone];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={classNames(
        'inline-flex h-9 items-center justify-center gap-2 rounded-sm border px-3 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50',
        toneClass,
      )}
    >
      {children}
    </button>
  );
}

function elementLabel(element: IdeogramWorkflowElement) {
  return element.type === 'text' ? element.text || element.desc || 'Text layer' : element.desc || 'Object layer';
}

function boxStyle(element: IdeogramWorkflowElement) {
  const [y1, x1, y2, x2] = element.bbox;
  return {
    left: `${x1 / 10}%`,
    top: `${y1 / 10}%`,
    width: `${Math.max(1, x2 - x1) / 10}%`,
    height: `${Math.max(1, y2 - y1) / 10}%`,
  };
}

function filenameBase(filename: string) {
  return filename.split(/[\\/]/).pop() || filename;
}

function clampZoom(value: number) {
  return Math.max(0.45, Math.min(1.2, Number(value.toFixed(2))));
}

export default function IdeogramWorkflowBuilderPage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const historyInputRef = useRef<HTMLInputElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const generationAbortRef = useRef(false);
  const objectUrlsRef = useRef<string[]>([]);
  const importedImageUrlRef = useRef<string>('');

  const [state, setState] = useState<IdeogramWorkflowState>(() => cloneIdeogramWorkflowState());
  const [serverUrl, setServerUrl] = useState(DEFAULT_EXTERNAL_COMFY_URL);
  const [serverUrlDraft, setServerUrlDraft] = useState(DEFAULT_EXTERNAL_COMFY_URL);
  const [loraDir, setLoraDir] = useState('');
  const [loraDirDraft, setLoraDirDraft] = useState('');
  const [externalLoras, setExternalLoras] = useState<string[]>([]);
  const [externalLoraSource, setExternalLoraSource] = useState('');
  const [toolkitLoras, setToolkitLoras] = useState<ToolkitLoraSummary[]>([]);
  const [copyToolkitPath, setCopyToolkitPath] = useState('');
  const [loraStatus, setLoraStatus] = useState<'idle' | 'loading' | 'copying' | 'error'>('idle');
  const [loraMessage, setLoraMessage] = useState('');
  const [settingsStatus, setSettingsStatus] = useState<'idle' | 'saving' | 'error'>('idle');
  const [settingsMessage, setSettingsMessage] = useState('');
  const [activeTab, setActiveTab] = useState<PanelTab>('preview');
  const [activeTool, setActiveTool] = useState<ToolMode>('select');
  const [selectedElementIndex, setSelectedElementIndex] = useState(0);
  const [zoom, setZoom] = useState(0.72);
  const [preflight, setPreflight] = useState<PreflightResult | null>(null);
  const [isPreflighting, setIsPreflighting] = useState(false);
  const [importPromptId, setImportPromptId] = useState('');
  const [importMessage, setImportMessage] = useState('');
  const [generation, setGeneration] = useState<GenerationState>(EMPTY_GENERATION);
  const [results, setResults] = useState<ResultImage[]>([]);
  const [importedImage, setImportedImage] = useState<ImportedCanvasImage | null>(null);
  const [canvasImage, setCanvasImage] = useState<PreviewImage | null>(null);
  const [lightboxImage, setLightboxImage] = useState<PreviewImage | null>(null);
  const [nowTick, setNowTick] = useState(Date.now());

  const workflow = useMemo(() => buildIdeogramComfyWorkflow(state), [state]);
  const workflowJson = useMemo(() => JSON.stringify(workflow, null, 2), [workflow]);
  const promptJson = useMemo(() => serializeIdeogramWorkflowPrompt(state), [state]);
  const selectedElement = state.elements[selectedElementIndex] || null;
  const selectedPreset = IDEOGRAM4_QUALITY_PRESETS[state.qualityPreset];
  const canvasAspect = parseIdeogramAspectRatio(state.aspectRatio);
  const canGenerate = Boolean(serverUrl && preflight?.ok && generation.status !== 'connecting' && generation.status !== 'queued' && generation.status !== 'executing');
  const stepPercent = generation.maxStep > 0 ? Math.round((generation.step / generation.maxStep) * 100) : 0;
  const externalLoraSet = useMemo(() => new Set(externalLoras), [externalLoras]);
  const missingLoras = useMemo(
    () =>
      state.loras.filter(lora => {
        const name = lora.loraName.trim();
        return name && externalLoras.length > 0 && !externalLoraSet.has(name);
      }),
    [externalLoraSet, externalLoras.length, state.loras],
  );

  const clearResultObjectUrls = useCallback(() => {
    objectUrlsRef.current.forEach(url => URL.revokeObjectURL(url));
    objectUrlsRef.current = [];
  }, []);

  const clearImportedImage = useCallback(() => {
    const objectUrl = importedImageUrlRef.current;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    importedImageUrlRef.current = '';
    setImportedImage(null);
    setCanvasImage(current => (current?.source === 'imported' ? null : current));
    setLightboxImage(current => (current?.source === 'imported' && current.objectUrl === objectUrl ? null : current));
  }, []);

  const refreshLoras = useCallback(
    async (urlOverride?: string) => {
      const url = urlOverride || serverUrl;
      if (!url) return;
      setLoraStatus('loading');
      setLoraMessage('');
      try {
        const response = await apiClient.get('/api/comfy/external/loras', { params: { server_url: url } });
        const nextExternalLoras = Array.isArray(response.data?.externalLoras) ? response.data.externalLoras.map(String) : [];
        const nextToolkitLoras = Array.isArray(response.data?.toolkitLoras) ? response.data.toolkitLoras : [];
        setExternalLoras(nextExternalLoras);
        setExternalLoraSource(response.data?.externalSource || '');
        setToolkitLoras(nextToolkitLoras);
        if (typeof response.data?.loraDir === 'string') {
          setLoraDir(response.data.loraDir);
          setLoraDirDraft(response.data.loraDir);
        }
        setCopyToolkitPath(current => current || nextToolkitLoras[0]?.path || '');
        setLoraStatus('idle');
        setLoraMessage(nextExternalLoras.length > 0 ? `${nextExternalLoras.length} external LoRAs found.` : 'No external LoRAs reported by ComfyUI.');
      } catch (error: any) {
        setLoraStatus('error');
        setLoraMessage(error.response?.data?.error || 'Could not refresh external LoRAs.');
      }
    },
    [serverUrl],
  );

  useEffect(() => {
    apiClient
      .get('/api/comfy/external/settings')
      .then(response => {
        const nextUrl = response.data?.serverUrl || '';
        setServerUrl(nextUrl || DEFAULT_EXTERNAL_COMFY_URL);
        setServerUrlDraft(nextUrl || DEFAULT_EXTERNAL_COMFY_URL);
        setLoraDir(response.data?.loraDir || '');
        setLoraDirDraft(response.data?.loraDir || '');
      })
      .catch(error => {
        setSettingsStatus('error');
        setSettingsMessage(error.response?.data?.error || 'Could not load external ComfyUI settings.');
      });
  }, []);

  useEffect(() => {
    const interval = window.setInterval(() => setNowTick(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
      clearResultObjectUrls();
      if (importedImageUrlRef.current) URL.revokeObjectURL(importedImageUrlRef.current);
    };
  }, [clearResultObjectUrls]);

  const updateState = useCallback(
    (updater: (current: IdeogramWorkflowState) => IdeogramWorkflowState) => {
      setState(current => updater(cloneIdeogramWorkflowState(current)));
    },
    [],
  );

  const saveServerUrl = useCallback(async () => {
    setSettingsStatus('saving');
    setSettingsMessage('');
    try {
      const response = await apiClient.post('/api/comfy/external/settings', { server_url: serverUrlDraft, lora_dir: loraDirDraft });
      const nextUrl = response.data?.serverUrl || '';
      const nextLoraDir = response.data?.loraDir || '';
      setServerUrl(nextUrl || DEFAULT_EXTERNAL_COMFY_URL);
      setServerUrlDraft(nextUrl || DEFAULT_EXTERNAL_COMFY_URL);
      setLoraDir(nextLoraDir);
      setLoraDirDraft(nextLoraDir);
      setSettingsStatus('idle');
      setSettingsMessage('External ComfyUI settings saved.');
      setPreflight(null);
      void refreshLoras(nextUrl || DEFAULT_EXTERNAL_COMFY_URL);
    } catch (error: any) {
      setSettingsStatus('error');
      setSettingsMessage(error.response?.data?.error || 'Failed to save external ComfyUI settings.');
    }
  }, [loraDirDraft, refreshLoras, serverUrlDraft]);

  const runPreflight = useCallback(async () => {
    if (!serverUrl) {
      setPreflight({
        ok: false,
        serverUrl: '',
        connected: false,
        nodes: [],
        models: [],
        error: 'External ComfyUI URL is not configured.',
      });
      return;
    }
    setIsPreflighting(true);
    try {
      const [response] = await Promise.all([
        apiClient.post('/api/comfy/external/preflight', { server_url: serverUrl, state, workflow }),
        refreshLoras(serverUrl),
      ]);
      setPreflight(response.data);
    } catch (error: any) {
      setPreflight({
        ok: false,
        serverUrl,
        connected: false,
        nodes: [],
        models: [],
        error: error.response?.data?.error || 'External ComfyUI preflight failed.',
      });
    } finally {
      setIsPreflighting(false);
    }
  }, [refreshLoras, serverUrl, state, workflow]);

  useEffect(() => {
    if (!serverUrl || preflight) return;
    void runPreflight();
  }, [preflight, runPreflight, serverUrl]);

  const setStyleField = (field: keyof IdeogramWorkflowState['style'], value: string | string[]) => {
    updateState(current => {
      current.style = { ...current.style, [field]: value };
      return current;
    });
  };

  const updateSelectedElement = (updater: (element: IdeogramWorkflowElement) => IdeogramWorkflowElement) => {
    updateState(current => {
      if (!current.elements[selectedElementIndex]) return current;
      current.elements[selectedElementIndex] = updater({ ...current.elements[selectedElementIndex] });
      return current;
    });
  };

  const addElement = (type: 'obj' | 'text') => {
    updateState(current => {
      const index = current.elements.length;
      const offset = Math.min(120 + index * 45, 360);
      current.elements.push(
        type === 'text'
          ? {
              type: 'text',
              bbox: [offset, 180, offset + 150, 820],
              text: 'NEW TEXT',
              desc: 'Visible text layer',
              color_palette: ['#22D3EE'],
            }
          : {
              type: 'obj',
              bbox: [offset + 120, 220, offset + 420, 800],
              desc: 'Visible object',
              color_palette: ['#F59E0B'],
            },
      );
      setSelectedElementIndex(index);
      return current;
    });
    setActiveTool('select');
  };

  const deleteSelectedElement = useCallback(() => {
    updateState(current => {
      if (!current.elements[selectedElementIndex]) return current;
      current.elements.splice(selectedElementIndex, 1);
      setSelectedElementIndex(index => Math.max(0, Math.min(index, current.elements.length - 1)));
      return current;
    });
  }, [selectedElementIndex, updateState]);

  const randomSeed = () => {
    updateState(current => {
      current.seed = Math.floor(Math.random() * 999999999999);
      return current;
    });
  };

  const addLora = () => {
    updateState(current => {
      current.loras = [
        ...(current.loras || []),
        {
          loraName: externalLoras[0] || '',
          strengthModel: 1,
          strengthClip: 1,
        },
      ];
      return current;
    });
    setPreflight(null);
  };

  const updateLora = (index: number, patch: Partial<IdeogramWorkflowLora>) => {
    updateState(current => {
      const loras = [...(current.loras || [])];
      while (loras.length <= index) {
        loras.push({ loraName: '', strengthModel: 1, strengthClip: 1 });
      }
      loras[index] = {
        loraName: loras[index]?.loraName || '',
        strengthModel: loras[index]?.strengthModel ?? 1,
        strengthClip: loras[index]?.strengthClip ?? 1,
        ...patch,
      };
      current.loras = loras;
      return current;
    });
    setPreflight(null);
  };

  const removeLora = (index: number) => {
    updateState(current => {
      current.loras = (current.loras || []).filter((_, loraIndex) => loraIndex !== index);
      return current;
    });
    setPreflight(null);
  };

  const copyToolkitLora = async () => {
    if (!copyToolkitPath) {
      setLoraStatus('error');
      setLoraMessage('Select a Toolkit LoRA to copy.');
      return;
    }
    setLoraStatus('copying');
    setLoraMessage('');
    try {
      const response = await apiClient.post('/api/comfy/external/loras/copy', {
        toolkitPath: copyToolkitPath,
        loraDir: loraDirDraft || loraDir,
      });
      const filename = response.data?.filename || filenameBase(copyToolkitPath);
      updateState(current => {
        const existing = (current.loras || []).some(lora => lora.loraName === filename);
        if (!existing) {
          current.loras = [
            ...(current.loras || []),
            {
              loraName: filename,
              strengthModel: 1,
              strengthClip: 1,
              toolkitPath: copyToolkitPath,
            },
          ];
        }
        return current;
      });
      setPreflight(null);
      setLoraStatus('idle');
      setLoraMessage(`${filename} copied to external ComfyUI.`);
      void refreshLoras(serverUrl);
    } catch (error: any) {
      setLoraStatus('error');
      setLoraMessage(error.response?.data?.error || 'Failed to copy Toolkit LoRA.');
    }
  };

  const exportJson = () => {
    const blob = new Blob([workflowJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${state.filenamePrefix || 'ideogram4_fp8_workflow'}.json`.replace(/[^\w.-]+/g, '_');
    link.click();
    URL.revokeObjectURL(url);
  };

  const copyText = async (value: string, message: string) => {
    await navigator.clipboard.writeText(value);
    setImportMessage(message);
  };

  const applyImportedWorkflow = (result: IdeogramImportResult, message: string) => {
    setState(result.state);
    setPreflight(null);
    setSelectedElementIndex(0);
    setImportMessage(result.warnings.length > 0 ? `${message} Warnings: ${result.warnings.join(' ')}` : message);
  };

  const importWorkflowObject = (raw: unknown, message: string) => {
    const imported = parseIdeogramComfyWorkflow(raw);
    applyImportedWorkflow(imported, message);
  };

  const handleJsonFile = async (file: File) => {
    try {
      importWorkflowObject(JSON.parse(await file.text()), `Imported ${file.name}.`);
    } catch (error) {
      setImportMessage(error instanceof Error ? error.message : 'Failed to import workflow JSON.');
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleImageFile = (file: File) => {
    const objectUrl = URL.createObjectURL(file);
    const image = new window.Image();
    image.onload = () => {
      if (importedImageUrlRef.current) URL.revokeObjectURL(importedImageUrlRef.current);
      importedImageUrlRef.current = objectUrl;
      const aspectRatio = closestIdeogramAspectRatio(image.naturalWidth, image.naturalHeight);
      const imported: ImportedCanvasImage = {
        id: `imported:${objectUrl}`,
        filename: file.name,
        objectUrl,
        source: 'imported',
        width: image.naturalWidth,
        height: image.naturalHeight,
        aspectRatio,
      };
      setImportedImage(imported);
      setCanvasImage(imported);
      updateState(current => ({ ...current, aspectRatio }));
      if (imageInputRef.current) imageInputRef.current.value = '';
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setImportMessage('Could not import that image.');
      if (imageInputRef.current) imageInputRef.current.value = '';
    };
    image.src = objectUrl;
  };

  const fetchResultImages = useCallback(
    async (images: ComfyImageRef[]) => {
      setCanvasImage(current => (current?.source === 'result' ? null : current));
      setLightboxImage(current => (current?.source === 'result' ? null : current));
      clearResultObjectUrls();
      const loaded: ResultImage[] = [];
      for (const image of images) {
        const response = await apiClient.get('/api/comfy/external/view', {
          responseType: 'blob',
          params: {
            server_url: serverUrl,
            filename: image.filename,
            subfolder: image.subfolder,
            type: image.type,
          },
        });
        const objectUrl = URL.createObjectURL(response.data as Blob);
        objectUrlsRef.current.push(objectUrl);
        loaded.push({
          ...image,
          id: `${image.type}:${image.subfolder}:${image.filename}`,
          objectUrl,
          source: 'result',
        });
      }
      setResults(loaded);
      if (loaded[0]) {
        setCanvasImage(loaded[0]);
      }
    },
    [clearResultObjectUrls, serverUrl],
  );

  const importHistory = useCallback(
    async (promptId: string, options: { applyWorkflow?: boolean; quietNotFound?: boolean } = {}) => {
      try {
        const response = await apiClient.post('/api/comfy/external/import', { server_url: serverUrl, promptId });
        if (options.applyWorkflow) {
          applyImportedWorkflow(response.data, `Imported ComfyUI history ${promptId}.`);
        }
        if (Array.isArray(response.data?.images) && response.data.images.length > 0) {
          await fetchResultImages(response.data.images);
        }
        return response.data;
      } catch (error: any) {
        if (!options.quietNotFound || error.response?.status !== 404) {
          setImportMessage(error.response?.data?.error || 'Failed to import ComfyUI history.');
        }
        throw error;
      }
    },
    [fetchResultImages, serverUrl],
  );

  const pollHistoryUntilComplete = useCallback(
    async (promptId: string) => {
      for (let attempt = 0; attempt < 240; attempt += 1) {
        if (generationAbortRef.current) return;
        try {
          const result = await importHistory(promptId, { quietNotFound: true });
          const images = Array.isArray(result?.images) ? result.images : [];
          const status = isRecord(result?.status) ? result.status : {};
          if (status.status_str === 'error') {
            setGeneration(current => ({
              ...current,
              status: 'error',
              error: 'ComfyUI prompt failed.',
              message: 'ComfyUI prompt failed.',
            }));
            return;
          }
          if (images.length > 0) {
            setGeneration(current => ({
              ...current,
              status: 'completed',
              step: current.maxStep || current.step,
              message: 'Generation completed.',
            }));
            return;
          }
        } catch (error: any) {
          if (error.response?.status !== 404) {
            setGeneration(current => ({
              ...current,
              status: 'error',
              error: error.response?.data?.error || 'Could not read ComfyUI history.',
              message: 'Could not read ComfyUI history.',
            }));
            return;
          }
        }
        await new Promise(resolve => window.setTimeout(resolve, 1500));
      }
      setGeneration(current => ({
        ...current,
        status: 'error',
        error: 'Timed out waiting for ComfyUI history outputs.',
        message: 'Timed out waiting for ComfyUI history outputs.',
      }));
    },
    [importHistory],
  );

  const connectProgressSocket = useCallback(
    (clientId: string) =>
      new Promise<void>(resolve => {
        if (!serverUrl) {
          resolve();
          return;
        }
        try {
          wsRef.current?.close();
          const socket = new WebSocket(normalizedWsUrl(serverUrl, clientId));
          wsRef.current = socket;
          const fallback = window.setTimeout(resolve, 1200);
          socket.onopen = () => {
            window.clearTimeout(fallback);
            resolve();
          };
          socket.onerror = () => {
            window.clearTimeout(fallback);
            resolve();
          };
          socket.onmessage = event => {
            try {
              const payload = JSON.parse(event.data);
              const type = payload?.type;
              const data = payload?.data || {};
              if (type === 'status') {
                const remaining = data?.status?.exec_info?.queue_remaining;
                setGeneration(current => ({
                  ...current,
                  queuePosition: typeof remaining === 'number' ? String(remaining) : current.queuePosition,
                }));
              }
              if (type === 'progress') {
                setGeneration(current => ({
                  ...current,
                  status: 'executing',
                  step: Number(data.value || current.step || 0),
                  maxStep: Number(data.max || current.maxStep || 0),
                  message: 'Generating image',
                }));
              }
              if (type === 'executing') {
                setGeneration(current => ({
                  ...current,
                  status: data.node ? 'executing' : current.status,
                  executingNode: data.node ? String(data.node) : current.executingNode,
                  message: data.node ? `Executing node ${data.node}` : current.message,
                }));
              }
              if (type === 'execution_error') {
                setGeneration(current => ({
                  ...current,
                  status: 'error',
                  error: data?.exception_message || 'ComfyUI execution failed.',
                  message: 'ComfyUI execution failed.',
                }));
              }
            } catch {
              // Ignore non-JSON socket frames.
            }
          };
        } catch {
          resolve();
        }
      }),
    [serverUrl],
  );

  const generate = useCallback(async () => {
    if (!serverUrl) {
      setSettingsMessage('External ComfyUI URL is required.');
      return;
    }
    generationAbortRef.current = false;
    const clientId = crypto.randomUUID();
    setCanvasImage(current => (current?.source === 'result' ? null : current));
    setLightboxImage(current => (current?.source === 'result' ? null : current));
    clearResultObjectUrls();
    setResults([]);
    setGeneration({
      ...EMPTY_GENERATION,
      status: 'connecting',
      clientId,
      elapsedStart: Date.now(),
      message: 'Connecting to ComfyUI progress stream',
    });
    await connectProgressSocket(clientId);
    if (generationAbortRef.current) return;
    try {
      const response = await apiClient.post('/api/comfy/external/prompt', {
        server_url: serverUrl,
        state,
        workflow,
        clientId,
      });
      const promptId = response.data?.promptId || '';
      setGeneration(current => ({
        ...current,
        status: 'queued',
        promptId,
        queuePosition: response.data?.queueNumber == null ? current.queuePosition : String(response.data.queueNumber),
        message: 'Queued in ComfyUI',
      }));
      void pollHistoryUntilComplete(promptId);
    } catch (error: any) {
      const preflightResult = error.response?.data?.preflight;
      if (preflightResult) setPreflight(preflightResult);
      setGeneration(current => ({
        ...current,
        status: 'error',
        error: error.response?.data?.error || 'Failed to queue ComfyUI prompt.',
        message: 'Failed to queue prompt.',
      }));
    }
  }, [clearResultObjectUrls, connectProgressSocket, pollHistoryUntilComplete, serverUrl, state, workflow]);

  const cancelGeneration = async () => {
    generationAbortRef.current = true;
    wsRef.current?.close();
    try {
      await apiClient.post('/api/comfy/external/interrupt', { server_url: serverUrl });
    } catch {
      // Ignore interrupt failures; the UI still exits the local wait state.
    }
    setGeneration(current => ({
      ...current,
      status: 'canceled',
      message: 'Generation canceled.',
    }));
  };

  const onCanvasPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (activeTool !== 'object' && activeTool !== 'text') return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 1000;
    const y = ((event.clientY - rect.top) / rect.height) * 1000;
    updateState(current => {
      const index = current.elements.length;
      current.elements.push(
        activeTool === 'text'
          ? {
              type: 'text',
              bbox: [Math.max(0, y - 60), Math.max(0, x - 180), Math.min(1000, y + 40), Math.min(1000, x + 180)],
              text: 'TEXT',
              desc: 'Visible text',
              color_palette: ['#22D3EE'],
            }
          : {
              type: 'obj',
              bbox: [Math.max(0, y - 100), Math.max(0, x - 140), Math.min(1000, y + 160), Math.min(1000, x + 140)],
              desc: 'Visible object',
              color_palette: ['#F59E0B'],
            },
      );
      setSelectedElementIndex(index);
      return current;
    });
    setActiveTool('select');
  };

  const nudgeSelectedElement = useCallback(
    (dx: number, dy: number) => {
      const element = state.elements[selectedElementIndex];
      if (!element) return;
      const [y1, x1, y2, x2] = element.bbox;
      const spanX = x2 - x1;
      const spanY = y2 - y1;
      const nextX1 = Math.max(0, Math.min(1000 - spanX, x1 + dx));
      const nextY1 = Math.max(0, Math.min(1000 - spanY, y1 + dy));
      setState(current =>
        updateIdeogramWorkflowElementBox(current, selectedElementIndex, {
          y1: nextY1,
          x1: nextX1,
          y2: nextY1 + spanY,
          x2: nextX1 + spanX,
        }),
      );
    },
    [selectedElementIndex, state.elements],
  );

  const handleCanvasWheel = useCallback((event: React.WheelEvent<HTMLDivElement>) => {
    if (event.deltaY === 0) return;
    event.preventDefault();
    const direction = event.deltaY < 0 ? 1 : -1;
    const increment = event.shiftKey ? 0.04 : 0.08;
    setZoom(value => clampZoom(value + direction * increment));
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tagName = target?.tagName;
      const isTyping = tagName === 'INPUT' || tagName === 'TEXTAREA' || target?.isContentEditable;
      if (lightboxImage) return;
      if (isTyping) return;
      if (event.key === 'Delete' || event.key === 'Backspace') {
        if (!state.elements[selectedElementIndex]) return;
        event.preventDefault();
        deleteSelectedElement();
        return;
      }
      const isArrowKey = event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowUp' || event.key === 'ArrowDown';
      if (!isArrowKey) return;
      event.preventDefault();
      const step = event.shiftKey ? 20 : 5;
      const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0;
      const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0;
      nudgeSelectedElement(dx, dy);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [deleteSelectedElement, lightboxImage, nudgeSelectedElement, selectedElementIndex, state.elements]);

  useEffect(() => {
    if (!lightboxImage) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setLightboxImage(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [lightboxImage]);

  const startBoxDrag = (event: React.PointerEvent<HTMLElement>, elementIndex: number, mode: 'move' | 'resize') => {
    event.preventDefault();
    event.stopPropagation();
    const canvas = event.currentTarget.closest('[data-composition-canvas="true"]') as HTMLDivElement | null;
    const startElement = state.elements[elementIndex];
    if (!canvas || !startElement) return;
    setSelectedElementIndex(elementIndex);
    const canvasRect = canvas.getBoundingClientRect();
    const startX = event.clientX;
    const startY = event.clientY;
    const [y1, x1, y2, x2] = startElement.bbox;
    const pointerId = event.pointerId;
    event.currentTarget.setPointerCapture(pointerId);

    const onMove = (moveEvent: PointerEvent) => {
      const dx = ((moveEvent.clientX - startX) / canvasRect.width) * 1000;
      const dy = ((moveEvent.clientY - startY) / canvasRect.height) * 1000;
      const nextBox =
        mode === 'resize'
          ? normalizeBox({ y1, x1, y2: y2 + dy, x2: x2 + dx })
          : normalizeBox({ y1: y1 + dy, x1: x1 + dx, y2: y2 + dy, x2: x2 + dx });
      const spanX = nextBox.x2 - nextBox.x1;
      const spanY = nextBox.y2 - nextBox.y1;
      if (mode === 'move') {
        const adjusted = {
          y1: Math.max(0, Math.min(1000 - spanY, nextBox.y1)),
          x1: Math.max(0, Math.min(1000 - spanX, nextBox.x1)),
          y2: Math.max(spanY, Math.min(1000, nextBox.y2)),
          x2: Math.max(spanX, Math.min(1000, nextBox.x2)),
        };
        setState(current => updateIdeogramWorkflowElementBox(current, elementIndex, adjusted));
      } else {
        setState(current => updateIdeogramWorkflowElementBox(current, elementIndex, nextBox));
      }
    };

    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp, { once: true });
  };

  const renderGenerationProgressPanel = () => (
    <section className="rounded-sm border border-gray-800 bg-gray-950 p-3">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-100">Generation Progress</h2>
        <StatusBadge status={generationBadgeStatus(generation.status)} label={generationStatusLabel(generation.status)} />
      </div>
      <div className="grid grid-cols-[7rem_1fr] gap-y-3 text-xs">
        <span className="text-gray-500">Prompt ID</span>
        <span className="truncate text-gray-300">{generation.promptId || '-'}</span>
        <span className="text-gray-500">Queue</span>
        <span className="text-gray-300">{generation.queuePosition}</span>
        <span className="text-gray-500">Executing</span>
        <span className="truncate text-gray-300">{generation.executingNode}</span>
        <span className="text-gray-500">Step</span>
        <span className="text-gray-300">{generation.maxStep ? `${generation.step} / ${generation.maxStep}` : '-'}</span>
        <span className="text-gray-500">Elapsed</span>
        <span className="text-gray-300">{formatElapsed(generation.elapsedStart, nowTick)}</span>
      </div>
      <ProgressBar value={stepPercent} className="mt-4" tone={generation.status === 'error' ? 'danger' : generation.status === 'completed' ? 'success' : 'info'} />
      <div className="mt-3 text-xs text-gray-400">{generation.error || generation.message}</div>
      {(generation.status === 'connecting' || generation.status === 'queued' || generation.status === 'executing') && (
        <div className="mt-4 flex justify-end">
          <ActionButton tone="rose" onClick={cancelGeneration}>
            <X className="h-4 w-4" />
            Cancel
          </ActionButton>
        </div>
      )}
    </section>
  );

  return (
    <>
      <TopBar className="h-14 border-gray-900 bg-[#02060a] px-4">
        <div className="flex min-w-0 flex-1 items-center gap-3">
          <h1 className="truncate text-lg font-semibold text-gray-100">Ideogram Workflow Builder</h1>
          <button
            type="button"
            onClick={() => {
              if (serverUrl) {
                void runPreflight();
              } else {
                setActiveTab('comfy');
              }
            }}
            disabled={isPreflighting}
            title={serverUrl ? `External ComfyUI: ${serverUrl}` : 'Configure external ComfyUI'}
            className={classNames(
              'hidden min-w-0 items-center gap-2 rounded-sm border px-2.5 py-1 text-xs transition-colors disabled:cursor-wait disabled:opacity-70 lg:inline-flex',
              preflight?.ok
                ? 'border-emerald-700/50 bg-emerald-950/25 text-emerald-200'
                : serverUrl
                  ? 'border-amber-700/50 bg-amber-950/20 text-amber-200'
                  : 'border-gray-800 bg-transparent text-gray-400 hover:border-gray-700 hover:text-gray-200',
            )}
          >
            {isPreflighting ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <span className={classNames('h-2 w-2 rounded-full', preflight?.ok ? 'bg-emerald-400' : 'bg-amber-300')} />
            )}
            <span>{isPreflighting ? 'Checking Comfy' : preflight?.ok ? 'Comfy ready' : serverUrl ? 'Check Comfy' : 'Set Comfy URL'}</span>
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          className="hidden"
          onChange={event => {
            const file = event.target.files?.[0];
            if (file) void handleJsonFile(file);
          }}
        />

        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={event => {
            const file = event.target.files?.[0];
            if (file) handleImageFile(file);
          }}
        />

        <div className="flex flex-none items-center gap-2">
          <ActionButton onClick={() => imageInputRef.current?.click()}>
            <ImageIcon className="h-4 w-4" />
            <span className="hidden sm:inline">Import Image</span>
          </ActionButton>
          <ActionButton onClick={() => historyInputRef.current?.focus()}>
            <History className="h-4 w-4" />
            <span className="hidden sm:inline">Import History</span>
          </ActionButton>
          <ActionButton onClick={exportJson}>
            <Code2 className="h-4 w-4" />
            <span className="hidden sm:inline">Export JSON</span>
          </ActionButton>
          <ActionButton onClick={generate} disabled={!canGenerate}>
            <Send className="h-4 w-4" />
            <span className="hidden md:inline">Send to Comfy</span>
          </ActionButton>
          <ActionButton onClick={generate} disabled={!canGenerate} tone="cyan">
            {generation.status === 'connecting' || generation.status === 'queued' || generation.status === 'executing' ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            Generate
          </ActionButton>
        </div>
      </TopBar>

      <MainContent className="operator-scrollbar-none bg-[#02060a] px-2 pt-16 text-gray-100 xl:!overflow-hidden sm:px-3">
        <div className="grid min-h-[calc(100dvh-4rem)] grid-cols-1 gap-2 xl:h-[calc(100dvh-4.5rem)] xl:min-h-0 xl:grid-cols-[315px_minmax(0,1fr)_360px]">
          <aside className="operator-panel flex min-h-0 flex-col overflow-hidden">
            <div className="operator-panel-header">
              <span>Prompt Structure</span>
              <button
                type="button"
                onClick={() => {
                  setState(cloneIdeogramWorkflowState(DEFAULT_IDEOGRAM_WORKFLOW_STATE));
                  setPreflight(null);
                }}
                className="text-xs text-gray-400 hover:text-gray-100"
              >
                Reset
              </button>
            </div>
            <div className="operator-scrollbar-none min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
              <Field label="Global Prompt" detail={`${state.highLevelDescription.length} chars`}>
                <textarea
                  value={state.highLevelDescription}
                  onChange={event => updateState(current => ({ ...current, highLevelDescription: event.target.value }))}
                  rows={5}
                  className="w-full resize-none rounded-sm border border-gray-800 bg-gray-950 px-3 py-2 text-sm leading-5 text-gray-100 outline-none placeholder:text-gray-600 focus:border-cyan-700"
                />
              </Field>

              <Field label="Style / Aesthetic">
                <TextInput value={state.style.aesthetics} onChange={value => setStyleField('aesthetics', value)} />
              </Field>
              <Field label="Camera / Medium">
                <div className="grid grid-cols-1 gap-2">
                  <TextInput value={state.style.photo} onChange={value => setStyleField('photo', value)} />
                  <TextInput value={state.style.medium} onChange={value => setStyleField('medium', value)} />
                </div>
              </Field>
              <Field label="Lighting">
                <TextInput value={state.style.lighting} onChange={value => setStyleField('lighting', value)} />
              </Field>
              <Field label="Background">
                <textarea
                  value={state.background}
                  onChange={event => updateState(current => ({ ...current, background: event.target.value }))}
                  rows={3}
                  className="w-full resize-none rounded-sm border border-gray-800 bg-gray-950 px-3 py-2 text-sm leading-5 text-gray-100 outline-none focus:border-cyan-700"
                />
              </Field>

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-medium text-gray-300">Palette</span>
                  <button
                    type="button"
                    className="text-xs text-cyan-300 hover:text-cyan-100"
                    onClick={() => setStyleField('colorPalette', [...state.style.colorPalette, '#64748B'])}
                  >
                    Add
                  </button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {state.style.colorPalette.map((color, index) => (
                    <input
                      key={`${color}-${index}`}
                      type="color"
                      value={color}
                      title={color}
                      onChange={event => {
                        const colors = [...state.style.colorPalette];
                        colors[index] = event.target.value.toUpperCase();
                        setStyleField('colorPalette', colors);
                      }}
                      className="h-8 w-8 rounded-sm border border-gray-700 bg-gray-900 p-0.5"
                    />
                  ))}
                </div>
              </div>

              <Field label="Quality Preset" detail={`default ${selectedPreset.steps} steps`}>
                <div className="grid grid-cols-3 overflow-hidden rounded-sm border border-gray-800">
                  {QUALITY_OPTIONS.map(option => (
                    <button
                      key={option}
                      type="button"
                      onClick={() =>
                        updateState(current => ({
                          ...current,
                          qualityPreset: option,
                          steps: IDEOGRAM4_QUALITY_PRESETS[option].steps,
                        }))
                      }
                      className={classNames(
                        'h-9 border-r border-gray-800 px-2 text-xs last:border-r-0',
                        state.qualityPreset === option ? 'bg-cyan-500/15 text-cyan-100' : 'bg-gray-950 text-gray-400 hover:bg-gray-900',
                      )}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              </Field>

              <div className="grid grid-cols-2 gap-3">
                <Field label="Aspect">
                  <Select
                    value={state.aspectRatio}
                    onChange={value => updateState(current => ({ ...current, aspectRatio: value }))}
                    options={IDEOGRAM_ASPECT_RATIOS}
                  />
                </Field>
                <Field label="Megapixels">
                  <NumberInput value={state.megapixels} min={0.25} max={4} step={0.25} onChange={value => updateState(current => ({ ...current, megapixels: value }))} />
                </Field>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <Field label="Steps">
                  <NumberInput value={state.steps} min={1} max={120} step={1} onChange={value => updateState(current => ({ ...current, steps: Math.max(1, Math.round(value)) }))} />
                </Field>
                <Field label="CFG">
                  <NumberInput value={state.guiderCfg} min={0} max={20} step={0.1} onChange={value => updateState(current => ({ ...current, guiderCfg: value }))} />
                </Field>
              </div>

              <div className="grid grid-cols-[1fr_auto] gap-2">
                <Field label="Seed">
                  <NumberInput value={state.seed} onChange={value => updateState(current => ({ ...current, seed: value }))} />
                </Field>
                <div className="pt-5">
                  <ActionButton onClick={randomSeed}>
                    <RefreshCw className="h-4 w-4" />
                  </ActionButton>
                </div>
              </div>

              <section className="border-t border-gray-900 pt-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <div>
                    <h2 className="text-xs font-semibold uppercase tracking-wide text-gray-300">LoRAs</h2>
                    <div className="mt-1 text-[11px] text-gray-500">
                      {externalLoras.length > 0
                        ? `${externalLoras.length} external${externalLoraSource ? ` via ${externalLoraSource}` : ''}`
                        : 'External choices load on Check Comfy'}
                    </div>
                  </div>
                  <button type="button" onClick={addLora} className="text-xs font-medium text-cyan-300 hover:text-cyan-100">
                    Add
                  </button>
                </div>

                <datalist id="external-comfy-lora-options">
                  {externalLoras.map(name => (
                    <option key={name} value={name} />
                  ))}
                </datalist>

                <div className="space-y-2">
                  {(state.loras || []).length === 0 ? (
                    <div className="rounded-sm border border-dashed border-gray-800 bg-gray-950 px-3 py-3 text-xs text-gray-500">
                      No LoRAs selected.
                    </div>
                  ) : (
                    state.loras.map((lora, index) => {
                      const name = lora.loraName.trim();
                      const missing = Boolean(name && externalLoras.length > 0 && !externalLoraSet.has(name));
                      return (
                        <div key={`lora-${index}`} className="rounded-sm border border-gray-800 bg-gray-950 p-2">
                          <div className="grid grid-cols-[1fr_auto] gap-2">
                            <TextInput
                              value={lora.loraName}
                              onChange={value => updateLora(index, { loraName: value })}
                              placeholder="filename.safetensors"
                              list="external-comfy-lora-options"
                            />
                            <IconButton title="Remove LoRA" danger onClick={() => removeLora(index)}>
                              <Trash2 className="h-4 w-4" />
                            </IconButton>
                          </div>
                          <div className="mt-2 grid grid-cols-2 gap-2">
                            <Field label="Model">
                              <NumberInput value={lora.strengthModel} min={-10} max={10} step={0.05} onChange={value => updateLora(index, { strengthModel: value })} />
                            </Field>
                            <Field label="CLIP">
                              <NumberInput value={lora.strengthClip} min={-10} max={10} step={0.05} onChange={value => updateLora(index, { strengthClip: value })} />
                            </Field>
                          </div>
                          {missing ? <div className="mt-2 text-xs text-amber-300">Missing in external ComfyUI.</div> : null}
                        </div>
                      );
                    })
                  )}
                </div>

                {missingLoras.length > 0 ? (
                  <div className="mt-2 rounded-sm border border-amber-900/70 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">
                    {missingLoras.length} selected LoRA{missingLoras.length === 1 ? '' : 's'} need to be installed in external ComfyUI.
                  </div>
                ) : null}

                <div className="mt-3 rounded-sm border border-gray-800 bg-gray-950 p-2">
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <span className="font-medium text-gray-300">Copy Toolkit LoRA</span>
                    <button
                      type="button"
                      disabled={!serverUrl || loraStatus === 'loading'}
                      onClick={() => void refreshLoras(serverUrl)}
                      className="text-cyan-300 hover:text-cyan-100 disabled:text-gray-600"
                    >
                      Refresh
                    </button>
                  </div>
                  <div className="grid grid-cols-[1fr_auto] gap-2">
                    <select
                      value={copyToolkitPath}
                      onChange={event => setCopyToolkitPath(event.target.value)}
                      className="h-9 min-w-0 rounded-sm border border-gray-800 bg-gray-950 px-2 text-xs text-gray-100 outline-none focus:border-cyan-700"
                    >
                      {toolkitLoras.length === 0 ? <option value="">No Toolkit LoRAs found</option> : null}
                      {toolkitLoras.map(lora => (
                        <option key={lora.id} value={lora.path}>
                          {lora.label}
                        </option>
                      ))}
                    </select>
                    <ActionButton
                      onClick={() => void copyToolkitLora()}
                      disabled={!copyToolkitPath || loraStatus === 'copying'}
                      tone={loraStatus === 'copying' ? 'neutral' : 'emerald'}
                    >
                      {loraStatus === 'copying' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Copy className="h-4 w-4" />}
                      Copy
                    </ActionButton>
                  </div>
                  {loraMessage ? (
                    <div className={classNames('mt-2 text-xs', loraStatus === 'error' ? 'text-rose-300' : 'text-gray-400')}>
                      {loraMessage}
                    </div>
                  ) : null}
                </div>
              </section>
            </div>
          </aside>

          <main className="flex min-h-[780px] min-w-0 flex-col gap-2 overflow-hidden xl:min-h-0">
            <section className="operator-panel flex min-h-0 flex-1 flex-col overflow-hidden">
              <div className="operator-panel-header">
                <span>Composition Canvas</span>
                <div className="flex items-center gap-1">
                  <IconButton title="Zoom out" onClick={() => setZoom(value => clampZoom(value - 0.08))}>
                    <ZoomOut className="h-4 w-4" />
                  </IconButton>
                  <span className="w-12 text-center text-xs text-gray-400">{Math.round(zoom * 100)}%</span>
                  <IconButton title="Zoom in" onClick={() => setZoom(value => clampZoom(value + 0.08))}>
                    <ZoomIn className="h-4 w-4" />
                  </IconButton>
                </div>
              </div>

              <div
                onWheel={handleCanvasWheel}
                className="operator-scrollbar-none relative flex min-h-[480px] flex-1 items-center justify-center overflow-auto bg-[#03070b] p-4"
              >
                <div className="absolute right-3 top-3 z-10 flex flex-col gap-2 rounded-sm border border-gray-800 bg-gray-950/90 p-2">
                  <IconButton title="Select" active={activeTool === 'select'} onClick={() => setActiveTool('select')}>
                    <MousePointer2 className="h-4 w-4" />
                  </IconButton>
                  <IconButton title="Move" active={activeTool === 'move'} onClick={() => setActiveTool('move')}>
                    <Move className="h-4 w-4" />
                  </IconButton>
                  <IconButton title="Add object" active={activeTool === 'object'} onClick={() => setActiveTool('object')}>
                    <Square className="h-4 w-4" />
                  </IconButton>
                  <IconButton title="Add text" active={activeTool === 'text'} onClick={() => setActiveTool('text')}>
                    <span className="text-sm font-semibold">T</span>
                  </IconButton>
                  <IconButton title="Delete selected" danger disabled={!selectedElement} onClick={deleteSelectedElement}>
                    <Trash2 className="h-4 w-4" />
                  </IconButton>
                </div>

                <div className="flex flex-col items-center gap-3" style={{ transform: `scale(${zoom})`, transformOrigin: 'center' }}>
                  <div className="flex w-full items-center justify-between text-xs text-gray-500">
                    <span>{canvasAspect.width}:{canvasAspect.height}</span>
                    <span>{state.aspectRatio}</span>
                  </div>
                  <div
                    data-composition-canvas="true"
                    onPointerDown={onCanvasPointerDown}
                    className={classNames(
                      'relative overflow-hidden border border-gray-700 bg-[#071017] shadow-[0_0_0_1px_rgba(34,211,238,0.06)]',
                      activeTool === 'object' || activeTool === 'text' ? 'cursor-crosshair' : 'cursor-default',
                    )}
                    style={{
                      aspectRatio: canvasAspect.css,
                      width: canvasAspect.ratio >= 1 ? 820 : 460,
                      maxWidth: 'calc(100vw - 4rem)',
                      backgroundImage:
                        'linear-gradient(rgba(148,163,184,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.08) 1px, transparent 1px)',
                      backgroundSize: '38px 38px',
                    }}
                  >
                    <div className="absolute inset-5 border border-dashed border-gray-700/70" />
                    {canvasImage ? (
                      <>
                        <button
                          type="button"
                          aria-label={`Open ${canvasImage.filename}`}
                          onPointerDown={event => {
                            if (activeTool === 'select') event.stopPropagation();
                          }}
                          onClick={event => {
                            if (activeTool !== 'select') return;
                            event.stopPropagation();
                            setLightboxImage(canvasImage);
                          }}
                          className="absolute inset-0 cursor-zoom-in"
                        >
                          <img
                            src={canvasImage.objectUrl}
                            alt={canvasImage.filename}
                            className="h-full w-full object-contain"
                          />
                        </button>
                      </>
                    ) : null}
                    {state.elements.map((element, index) => {
                      const selected = index === selectedElementIndex;
                      const color = element.color_palette?.[0] || (element.type === 'text' ? '#22D3EE' : '#F59E0B');
                      return (
                        <div
                          key={`${element.type}-${index}`}
                          className={classNames('absolute select-none border-2', selected ? 'z-20' : 'z-10')}
                          style={{ ...boxStyle(element), borderColor: selected ? color : `${color}B8` }}
                          onPointerDown={event => startBoxDrag(event, index, activeTool === 'move' ? 'move' : 'move')}
                        >
                          <button
                            type="button"
                            title={elementLabel(element)}
                            onClick={event => {
                              event.stopPropagation();
                              setSelectedElementIndex(index);
                            }}
                            className={classNames(
                              'absolute -left-0.5 -top-7 flex max-w-[95%] items-center gap-1.5 rounded-sm border bg-[#05080c] px-2 py-1 text-left text-xs font-semibold text-gray-50 shadow-[0_1px_3px_rgba(0,0,0,0.75)]',
                              selected ? 'border-cyan-300' : 'border-gray-500',
                            )}
                          >
                            <span className="h-1.5 w-1.5 flex-none rounded-full ring-1 ring-white/25" style={{ backgroundColor: color }} />
                            <span className="truncate">{element.type === 'text' ? 'Text' : 'Object'}</span>
                          </button>
                          {selected ? (
                            <>
                              <span className="absolute -left-1.5 -top-1.5 h-3 w-3 border border-gray-950" style={{ backgroundColor: color }} />
                              <span className="absolute -right-1.5 -top-1.5 h-3 w-3 border border-gray-950" style={{ backgroundColor: color }} />
                              <span className="absolute -bottom-1.5 -left-1.5 h-3 w-3 border border-gray-950" style={{ backgroundColor: color }} />
                              <span
                                role="button"
                                tabIndex={0}
                                title="Resize"
                                onPointerDown={event => startBoxDrag(event, index, 'resize')}
                                className="absolute -bottom-1.5 -right-1.5 h-3 w-3 cursor-nwse-resize border border-gray-950"
                                style={{ backgroundColor: color }}
                              />
                            </>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span>Normalized bbox grid</span>
                    {canvasImage ? (
                      <IconButton title="Open canvas image" onClick={() => setLightboxImage(canvasImage)}>
                        <ZoomIn className="h-4 w-4" />
                      </IconButton>
                    ) : null}
                    {canvasImage?.source === 'imported' ? (
                      <button type="button" onClick={clearImportedImage} className="text-cyan-300 hover:text-cyan-100">
                        Clear image
                      </button>
                    ) : null}
                    {canvasImage?.source === 'result' ? (
                      <button type="button" onClick={() => setCanvasImage(importedImage)} className="text-cyan-300 hover:text-cyan-100">
                        Clear preview
                      </button>
                    ) : null}
                  </div>
                </div>
              </div>
            </section>

            <section className="operator-panel flex min-h-[245px] flex-col overflow-hidden">
              <div className="operator-panel-header">
                <div className="flex items-center gap-2">
                  <span>Results</span>
                  <span className="rounded-sm border border-gray-800 bg-gray-950 px-2 py-0.5 text-[11px] text-gray-400">
                    {results.length} completed
                  </span>
                </div>
                <div className="text-xs text-gray-500">{generation.promptId ? `Prompt ${generation.promptId}` : 'No active prompt'}</div>
              </div>
              <div className="grid min-h-0 flex-1 gap-2 p-3 lg:grid-cols-[minmax(0,1fr)_260px]">
                <div className="grid grid-cols-4 gap-2">
                  {[0, 1, 2, 3].map(index => {
                    const image = results[index];
                    return image ? (
                      <button
                        key={image.id}
                        type="button"
                        onClick={() => {
                          setCanvasImage(image);
                          setLightboxImage(image);
                        }}
                        className="relative overflow-hidden rounded-sm border border-gray-800 bg-gray-950 text-left transition-colors hover:border-cyan-700 focus:border-cyan-600 focus:outline-none"
                      >
                        <div className="flex h-44 w-full items-center justify-center bg-[#050a0f]">
                          <img src={image.objectUrl} alt={image.filename} className="h-full w-full object-contain" />
                        </div>
                        <div className="truncate border-t border-gray-800 px-2 py-1 text-xs text-gray-400">{filenameBase(image.filename)}</div>
                      </button>
                    ) : (
                      <div key={index} className="flex h-full min-h-44 flex-col items-center justify-center rounded-sm border border-dashed border-gray-800 bg-gray-950 text-center text-xs text-gray-500">
                        <Sparkles className="mb-3 h-5 w-5" />
                        Waiting
                      </div>
                    );
                  })}
                </div>
                <div className="rounded-sm border border-gray-800 bg-gray-950 p-3 text-xs text-gray-400">
                  <div className="grid grid-cols-[6rem_1fr] gap-y-2">
                    <span className="text-gray-500">Prompt ID</span>
                    <span className="truncate text-gray-300">{generation.promptId || '-'}</span>
                    <span className="text-gray-500">Filename</span>
                    <span className="truncate text-gray-300">{results[0]?.filename || '-'}</span>
                    <span className="text-gray-500">Seed</span>
                    <span className="text-gray-300">{state.seed}</span>
                    <span className="text-gray-500">Steps</span>
                    <span className="text-gray-300">{state.steps}</span>
                    <span className="text-gray-500">CFG</span>
                    <span className="text-gray-300">{state.guiderCfg}</span>
                  </div>
                  <div className="mt-4 flex gap-2">
                    <IconButton title="Copy prompt ID" disabled={!generation.promptId} onClick={() => void copyText(generation.promptId, 'Prompt ID copied.')}>
                      <Copy className="h-4 w-4" />
                    </IconButton>
                    <IconButton title="Download JSON" onClick={exportJson}>
                      <Download className="h-4 w-4" />
                    </IconButton>
                    <IconButton title="Open external ComfyUI" disabled={!serverUrl} onClick={() => window.open(serverUrl, '_blank', 'noopener,noreferrer')}>
                      <ExternalLink className="h-4 w-4" />
                    </IconButton>
                  </div>
                </div>
              </div>
            </section>
          </main>

          <aside className="operator-panel flex min-h-[760px] min-w-0 flex-col overflow-hidden xl:min-h-0">
            <div className="flex h-12 flex-none items-center border-b border-gray-800 px-3">
              {(['preview', 'json', 'comfy'] as PanelTab[]).map(tab => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setActiveTab(tab)}
                  className={classNames('mr-5 h-12 border-b-2 text-sm font-semibold capitalize', {
                    'border-cyan-400 text-cyan-100': activeTab === tab,
                    'border-transparent text-gray-400 hover:text-gray-200': activeTab !== tab,
                  })}
                >
                  {tab}
                </button>
              ))}
            </div>
            <div className="operator-scrollbar-none min-h-0 flex-1 overflow-y-auto p-3">
              {activeTab === 'preview' && (
                <div className="space-y-3">
                  {selectedElement ? (
                    <section className="rounded-sm border border-gray-800 bg-gray-950 p-3">
                      <div className="mb-3 flex items-center justify-between gap-2">
                        <h2 className="text-sm font-semibold text-gray-100">Selected Element</h2>
                        <span className="rounded-sm border border-gray-800 bg-gray-900 px-2 py-0.5 text-[11px] uppercase tracking-wide text-gray-400">
                          {selectedElement.type === 'text' ? 'Text' : 'Object'}
                        </span>
                      </div>
                      <div className="space-y-3">
                        {selectedElement.type === 'text' ? (
                          <Field label="Text">
                            <TextInput
                              value={selectedElement.text || ''}
                              onChange={value =>
                                updateSelectedElement(element => ({
                                  ...element,
                                  text: value,
                                  desc: element.desc || value,
                                }))
                              }
                            />
                          </Field>
                        ) : null}
                        <Field label="Description" detail={`${selectedElement.desc.length} chars`}>
                          <textarea
                            value={selectedElement.desc}
                            onChange={event =>
                              updateSelectedElement(element => ({
                                ...element,
                                desc: event.target.value,
                              }))
                            }
                            rows={4}
                            className="w-full resize-none rounded-sm border border-gray-800 bg-[#050a0f] px-3 py-2 text-sm leading-5 text-gray-100 outline-none placeholder:text-gray-600 focus:border-cyan-700"
                          />
                        </Field>
                      </div>
                    </section>
                  ) : null}
                  <section className="rounded-sm border border-gray-800 bg-gray-950 p-3">
                    <h2 className="mb-2 text-sm font-semibold text-gray-100">Ideogram Prompt JSON</h2>
                    <pre className="operator-scrollbar-none max-h-[360px] overflow-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-gray-300">
                      {promptJson}
                    </pre>
                  </section>
                  <section className="rounded-sm border border-gray-800 bg-gray-950 p-3">
                    <h2 className="mb-2 text-sm font-semibold text-gray-100">Elements</h2>
                    <div className="divide-y divide-gray-900">
                      {state.elements.map((element, index) => (
                        <button
                          key={`${element.type}-${index}`}
                          type="button"
                          onClick={() => setSelectedElementIndex(index)}
                          className={classNames('flex w-full min-w-0 items-center gap-2 px-1 py-2 text-left text-xs', index === selectedElementIndex ? 'text-cyan-100' : 'text-gray-400')}
                        >
                          <span className="h-2.5 w-2.5 flex-none rounded-full" style={{ backgroundColor: element.color_palette?.[0] || '#64748B' }} />
                          <span className="min-w-0 flex-1 truncate">{element.type}: {elementLabel(element)}</span>
                        </button>
                      ))}
                    </div>
                    <div className="mt-3 flex gap-2">
                      <ActionButton onClick={() => addElement('obj')}>
                        <Plus className="h-4 w-4" />
                        Object
                      </ActionButton>
                      <ActionButton onClick={() => addElement('text')}>
                        <Plus className="h-4 w-4" />
                        Text
                      </ActionButton>
                    </div>
                  </section>
                </div>
              )}

              {activeTab === 'json' && (
                <div className="space-y-3">
                  <section className="rounded-sm border border-gray-800 bg-gray-950 p-3">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <h2 className="text-sm font-semibold text-gray-100">Workflow JSON (ComfyUI API)</h2>
                      <div className="flex gap-1">
                        <IconButton title="Copy JSON" onClick={() => void copyText(workflowJson, 'Workflow JSON copied.')}>
                          <Clipboard className="h-4 w-4" />
                        </IconButton>
                        <IconButton title="Import JSON file" onClick={() => fileInputRef.current?.click()}>
                          <FileJson2 className="h-4 w-4" />
                        </IconButton>
                        <IconButton title="Download JSON" onClick={exportJson}>
                          <Download className="h-4 w-4" />
                        </IconButton>
                      </div>
                    </div>
                    <textarea
                      value={workflowJson}
                      readOnly
                      rows={24}
                      className="operator-scrollbar-none h-[340px] w-full resize-none rounded-sm border border-gray-800 bg-[#050a0f] p-3 font-mono text-xs leading-relaxed text-cyan-50 outline-none"
                    />
                    {importMessage ? <div className="mt-2 rounded-sm border border-gray-800 bg-gray-900 px-3 py-2 text-xs text-gray-300">{importMessage}</div> : null}
                  </section>
                  {renderGenerationProgressPanel()}
                </div>
              )}

              {activeTab === 'comfy' && (
                <div className="space-y-3">
                  <section className="rounded-sm border border-gray-800 bg-gray-950 p-3">
                    <h2 className="mb-3 text-sm font-semibold text-gray-100">External ComfyUI</h2>
                    <Field label="Server URL">
                      <div className="grid grid-cols-[1fr_auto] gap-2">
                        <TextInput value={serverUrlDraft} onChange={setServerUrlDraft} placeholder="http://127.0.0.1:8188" />
                        <ActionButton onClick={saveServerUrl} disabled={settingsStatus === 'saving'}>
                          {settingsStatus === 'saving' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                          Save
                        </ActionButton>
                      </div>
                    </Field>
                    <div className="mt-3">
                      <Field label="LoRA Folder" detail="external models/loras">
                        <TextInput value={loraDirDraft} onChange={setLoraDirDraft} placeholder="E:\\ComfyUI\\models\\loras" />
                      </Field>
                    </div>
                    {settingsMessage ? (
                      <div className={classNames('mt-2 rounded-sm border px-3 py-2 text-xs', settingsStatus === 'error' ? 'border-rose-900 bg-rose-950/20 text-rose-200' : 'border-gray-800 bg-gray-900 text-gray-300')}>
                        {settingsMessage}
                      </div>
                    ) : null}
                  </section>
                  <section className="rounded-sm border border-gray-800 bg-gray-950 p-3">
                    <h2 className="mb-3 text-sm font-semibold text-gray-100">Import From History</h2>
                    <div className="grid grid-cols-[1fr_auto] gap-2">
                      <TextInput
                        value={importPromptId}
                        onChange={setImportPromptId}
                        placeholder="ComfyUI prompt_id"
                        inputRef={historyInputRef}
                      />
                      <ActionButton
                        onClick={() => {
                          if (importPromptId.trim()) void importHistory(importPromptId.trim(), { applyWorkflow: true });
                        }}
                        disabled={!serverUrl || !importPromptId.trim()}
                      >
                        <History className="h-4 w-4" />
                        Import
                      </ActionButton>
                    </div>
                  </section>

                  {renderGenerationProgressPanel()}
                </div>
              )}
            </div>
          </aside>
        </div>
      </MainContent>
      {lightboxImage ? (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={lightboxImage.filename}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 p-4 backdrop-blur-sm"
          onMouseDown={() => setLightboxImage(null)}
        >
          <div
            className="flex max-h-[92dvh] w-full max-w-6xl flex-col overflow-hidden rounded-sm border border-gray-800 bg-[#05080c] shadow-2xl"
            onMouseDown={event => event.stopPropagation()}
          >
            <div className="flex h-12 flex-none items-center justify-between gap-3 border-b border-gray-800 px-3">
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold text-gray-100">{filenameBase(lightboxImage.filename)}</div>
                <div className="text-xs capitalize text-gray-500">{lightboxImage.source}</div>
              </div>
              <div className="flex items-center gap-2">
                <IconButton
                  title="Use on canvas"
                  onClick={() => {
                    setCanvasImage(lightboxImage);
                    setLightboxImage(null);
                  }}
                >
                  <ImageIcon className="h-4 w-4" />
                </IconButton>
                <IconButton title="Open image in new tab" onClick={() => window.open(lightboxImage.objectUrl, '_blank', 'noopener,noreferrer')}>
                  <ExternalLink className="h-4 w-4" />
                </IconButton>
                <IconButton title="Close preview" onClick={() => setLightboxImage(null)}>
                  <X className="h-4 w-4" />
                </IconButton>
              </div>
            </div>
            <div className="flex min-h-0 flex-1 items-center justify-center bg-black p-3">
              <img src={lightboxImage.objectUrl} alt={lightboxImage.filename} className="max-h-[calc(92dvh-4.5rem)] max-w-full object-contain" />
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
