'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Loader2, RotateCcw } from 'lucide-react';
import { apiClient } from '@/utils/api';
import { Modal } from '@/components/Modal';
import useRemoteOllamaWorkers from '@/hooks/useRemoteOllamaWorkers';
import { defaultIdeogramJsonCaptionPrompt, defaultImageCaptionPrompt } from '@/helpers/captionOptions';
import {
  captionObjectPath,
  decryptEncryptedObjectBlob,
  encryptCaptionObject,
  randomId,
} from '@/utils/encryptedDatasets';
import {
  addIdeogramElement,
  appendGeneratedIdeogramElements,
  applyGeneratedBoxPatches,
  arrayToBox,
  boxToRect,
  cloneIdeogramData,
  deleteIdeogramElement,
  duplicateIdeogramElement,
  normalizeGeneratedElementBoxes,
  normalizeGeneratedBoxPatches,
  parseIdeogramCaption,
  serializeIdeogramCaption,
  type GeneratedBoxPatch,
  type GeneratedElementBox,
  type IdeogramElementType,
  type NormalizedBox,
  updateIdeogramElementBox,
  updateIdeogramElementField,
  updateIdeogramElementPalette,
  updateIdeogramElementType,
  updateIdeogramHighLevelDescription,
} from '@/utils/ideogramCaption';
import { resizeOrMoveBox } from '@/utils/annotationGeometry';
import { AnnotationLayer } from './AnnotationLayer';
import {
  AUTO_BOX_PROVIDERS,
  BOX_COLORS,
  DEFAULT_OLLAMA_VISION_MODEL,
  DEFAULT_OPENROUTER_BOX_MODEL,
  MAX_HISTORY,
  MIN_BOX_SPAN,
  OLLAMA_VISION_MODELS,
  OPENROUTER_BOX_MODELS,
} from './constants';
import { ImageNavigator } from './ImageNavigator';
import { CaptionEditorPanel, ObjectDetailsPanel } from './InspectorPanels';
import { LayersPanel } from './LayersPanel';
import { StudioMedia } from './StudioMedia';
import { StudioToolbar } from './StudioToolbar';
import { ToolRail } from './ToolRail';
import { appendImageSizeFields, createEncryptedImageFormData } from './openRouterMedia';
import type {
  BulkCaptionActionRequest,
  BulkCaptionActionResult,
  CaptionCacheEntry,
  CaptionTab,
  DatasetImageStudioProps,
  DatasetStudioItem,
  DeleteImagesResult,
  ImageSize,
  ToolMode,
} from './types';
import {
  captionResponseToText,
  clampIndex,
  isLayerCaptionRequestForItem,
  itemKey,
  itemKind,
  itemName,
  isPlainTextCaptionItem,
  layerCaptionRequestKey,
  layerCaptionTargetText,
  normalizeHexColor,
  pendingCaptionLayerStillMatches,
  reindexLayerIndexSetAfterDelete,
  reindexLayerIndexSetAfterInsert,
  responseErrorMessage,
  statusForCaption,
} from './utils';

type StudioBoxProvider = (typeof AUTO_BOX_PROVIDERS)[number]['value'];
type RecaptionProvider = StudioBoxProvider;
type RecaptionOutputFormat = 'text' | 'ideogram_json';
type RefusalCaptionAuditResponse = {
  refusals?: Record<string, unknown>;
};
type RecaptionModelOption = { value: string; label: string };
type RecaptionSettingsPreset = {
  provider: RecaptionProvider;
  model: string;
  outputFormat: RecaptionOutputFormat;
  prompt: string;
  systemPrompt: string;
  remoteWorkerId: string;
  maxNewTokens: number;
};
type RecaptionQueueEntry = {
  id: string;
  item: DatasetStudioItem;
  key: string;
  name: string;
  existingCaption: string;
  settings: RecaptionSettingsPreset;
};
type OllamaModelListItem = {
  name?: string;
  model?: string;
  details?: Record<string, unknown>;
};

const RECAPTION_SETTINGS_STORAGE_PREFIX = 'aitk.datasetEditor.recaptionSettings.v1';

function ollamaModelName(model: OllamaModelListItem) {
  return (typeof model.model === 'string' && model.model.trim()) || (typeof model.name === 'string' && model.name.trim()) || '';
}

function ollamaModelOptions(models: OllamaModelListItem[]): RecaptionModelOption[] {
  return models.flatMap(model => {
    const value = ollamaModelName(model);
    if (!value) return [];
    const detail = [model.details?.parameter_size, model.details?.quantization_level]
      .filter((part): part is string => typeof part === 'string' && part.trim().length > 0)
      .join(' ');
    return [{ value, label: detail ? `${value} (${detail})` : value }];
  });
}

function recaptionSettingsStorageKey({
  datasetName,
  projectID,
  datasetPath,
  workerID,
}: {
  datasetName: string;
  projectID?: string | null;
  datasetPath?: string | null;
  workerID: string;
}) {
  const dataset = datasetName.trim();
  if (!dataset) return '';
  const scope = (projectID || datasetPath || workerID || 'global').trim();
  return `${RECAPTION_SETTINGS_STORAGE_PREFIX}:${encodeURIComponent(scope)}:${encodeURIComponent(dataset)}`;
}

function normalizeRecaptionProvider(value: unknown): RecaptionProvider {
  return AUTO_BOX_PROVIDERS.some(provider => provider.value === value) ? (value as RecaptionProvider) : 'openrouter';
}

function normalizeRecaptionOutputFormat(value: unknown): RecaptionOutputFormat {
  return value === 'ideogram_json' ? 'ideogram_json' : 'text';
}

function readRecaptionSettingsPreset(storageKey: string): RecaptionSettingsPreset | null {
  if (!storageKey || typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<RecaptionSettingsPreset> | null;
    if (!parsed || typeof parsed !== 'object') return null;
    const provider = normalizeRecaptionProvider(parsed.provider);
    const fallbackModel = provider === 'openrouter' ? DEFAULT_OPENROUTER_BOX_MODEL : DEFAULT_OLLAMA_VISION_MODEL;
    const maxNewTokens = Number(parsed.maxNewTokens);
    return {
      provider,
      model: typeof parsed.model === 'string' && parsed.model.trim() ? parsed.model : fallbackModel,
      outputFormat: normalizeRecaptionOutputFormat(parsed.outputFormat),
      prompt: typeof parsed.prompt === 'string' && parsed.prompt.trim() ? parsed.prompt : defaultImageCaptionPrompt,
      systemPrompt: typeof parsed.systemPrompt === 'string' ? parsed.systemPrompt : '',
      remoteWorkerId: typeof parsed.remoteWorkerId === 'string' ? parsed.remoteWorkerId : '',
      maxNewTokens: Number.isFinite(maxNewTokens) && maxNewTokens > 0 ? Math.floor(maxNewTokens) : 256,
    };
  } catch (error) {
    console.warn('Could not load saved recaption settings:', error);
    return null;
  }
}

function scheduleIdleTask(callback: () => void) {
  const idleWindow = window as Window & {
    requestIdleCallback?: (callback: () => void, options?: { timeout?: number }) => number;
    cancelIdleCallback?: (handle: number) => void;
  };
  if (idleWindow.requestIdleCallback) {
    const handle = idleWindow.requestIdleCallback(callback, { timeout: 2500 });
    return () => idleWindow.cancelIdleCallback?.(handle);
  }
  const handle = window.setTimeout(callback, 700);
  return () => window.clearTimeout(handle);
}

export default function DatasetImageStudio({
  datasetName,
  workerID,
  projectID,
  datasetPath,
  items,
  isAutoCaptioning,
  encryptedKey,
  encryptedRawKeyB64,
  rootCaption,
  onRefresh,
  onAddImages,
  onConvertDatasetToJson,
  onDeleteImages,
  onBulkEncryptedCaptionAction,
  onSaveEncryptedCaption,
}: DatasetImageStudioProps) {
  const { workers } = useRemoteOllamaWorkers();
  const projectPayload = useMemo(() => (projectID ? { project_id: projectID } : {}), [projectID]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [captionText, setCaptionText] = useState('');
  const [savedCaption, setSavedCaption] = useState('');
  const [isCaptionLoaded, setIsCaptionLoaded] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isDeletingImages, setIsDeletingImages] = useState(false);
  const [deleteMessage, setDeleteMessage] = useState('');
  const [activeTool, setActiveTool] = useState<ToolMode>('select');
  const [selectedElementIndex, setSelectedElementIndex] = useState<number | null>(null);
  const [captionTab, setCaptionTab] = useState<CaptionTab>('caption');
  const [zoom, setZoom] = useState(1);
  const [undoStack, setUndoStack] = useState<string[]>([]);
  const [redoStack, setRedoStack] = useState<string[]>([]);
  const [autoBoxProvider, setAutoBoxProvider] = useState<StudioBoxProvider>('openrouter');
  const [autoBoxModel, setAutoBoxModel] = useState(DEFAULT_OPENROUTER_BOX_MODEL);
  const [recaptionProvider, setRecaptionProvider] = useState<RecaptionProvider>('openrouter');
  const [recaptionModel, setRecaptionModel] = useState(DEFAULT_OPENROUTER_BOX_MODEL);
  const [recaptionOutputFormat, setRecaptionOutputFormat] = useState<RecaptionOutputFormat>('text');
  const [recaptionPrompt, setRecaptionPrompt] = useState(defaultImageCaptionPrompt);
  const [recaptionSystemPrompt, setRecaptionSystemPrompt] = useState('');
  const [recaptionMaxNewTokens, setRecaptionMaxNewTokens] = useState(256);
  const [isRecaptionModalOpen, setIsRecaptionModalOpen] = useState(false);
  const [isRecaptioning, setIsRecaptioning] = useState(false);
  const [recaptionMessage, setRecaptionMessage] = useState('');
  const [hasRecaptionSettingsForDataset, setHasRecaptionSettingsForDataset] = useState(false);
  const [recaptionQueue, setRecaptionQueue] = useState<RecaptionQueueEntry[]>([]);
  const [activeRecaptionLabel, setActiveRecaptionLabel] = useState('');
  const [activeRecaptionKey, setActiveRecaptionKey] = useState('');
  const [recaptionRootPrompt, setRecaptionRootPrompt] = useState('');
  const [recaptionRootPromptStatus, setRecaptionRootPromptStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [recaptionRemoteModelOptions, setRecaptionRemoteModelOptions] = useState<RecaptionModelOption[]>([]);
  const [recaptionRemoteModelStatus, setRecaptionRemoteModelStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [recaptionRemoteModelError, setRecaptionRemoteModelError] = useState('');
  const [remoteOllamaWorkerId, setRemoteOllamaWorkerId] = useState('');
  const [recaptionRemoteWorkerId, setRecaptionRemoteWorkerId] = useState('');
  const [autoBoxRefine, setAutoBoxRefine] = useState(false);
  const [isGeneratingBoxes, setIsGeneratingBoxes] = useState(false);
  const [autoBoxMessage, setAutoBoxMessage] = useState('');
  const [captioningLayerKeys, setCaptioningLayerKeys] = useState<Set<string>>(() => new Set());
  const [layerCaptionMessages, setLayerCaptionMessages] = useState<Record<string, string>>({});
  const [encryptedProviderConfirmations, setEncryptedProviderConfirmations] = useState<Record<string, boolean>>({});
  const [selectedImageSize, setSelectedImageSize] = useState<ImageSize | null>(null);
  const [hiddenLayerIndexes, setHiddenLayerIndexes] = useState<Set<number>>(() => new Set());
  const [lockedLayerIndexes, setLockedLayerIndexes] = useState<Set<number>>(() => new Set());
  const [overlapElementStack, setOverlapElementStack] = useState<number[]>([]);
  const [activePaletteSamplerIndex, setActivePaletteSamplerIndex] = useState<number | null>(null);
  const [encryptedCaptionPaths, setEncryptedCaptionPaths] = useState<Record<string, string>>({});
  const [captionCacheVersion, setCaptionCacheVersion] = useState(0);
  const captionCacheRef = useRef(new Map<string, CaptionCacheEntry>());
  const saveCaptionRef = useRef<() => Promise<void>>(async () => undefined);
  const autoSelectKeyRef = useRef('');
  const latestCaptionRef = useRef('');
  const savedCaptionRef = useRef('');
  const isDirtyRef = useRef(false);
  const selectedKeyRef = useRef('');
  const isRecaptioningRef = useRef(false);
  const recaptionQueueRef = useRef<RecaptionQueueEntry[]>([]);
  const recaptionModelRef = useRef(DEFAULT_OPENROUTER_BOX_MODEL);
  const recaptionSystemPromptRef = useRef('');
  const recaptionSystemPromptTouchedRef = useRef(false);

  const writeCaptionCache = useCallback((key: string, entry: CaptionCacheEntry) => {
    captionCacheRef.current.set(key, entry);
    setCaptionCacheVersion(version => version + 1);
  }, []);

  const bumpCaptionCacheVersion = useCallback(() => {
    setCaptionCacheVersion(version => version + 1);
  }, []);

  useEffect(() => {
    setSelectedIndex(index => clampIndex(index, items.length));
  }, [items.length]);

  const selectedItem = items[selectedIndex] || null;
  const selectedKey = selectedItem ? itemKey(selectedItem) : '';
  const selectedName = selectedItem ? itemName(selectedItem) : '';
  const selectedKind = selectedItem ? itemKind(selectedItem) : 'image';
  const plainAuditItemPaths = useMemo(
    () => items.flatMap(item => (item.kind === 'plain' ? [item.path] : [])),
    [items],
  );
  const plainAuditKey = useMemo(() => plainAuditItemPaths.join('\n'), [plainAuditItemPaths]);
  const remoteWorkerOptions = useMemo(
    () => workers.filter(worker => worker.enabled).map(worker => ({ value: worker.id, label: worker.name })),
    [workers],
  );
  const recaptionStorageKey = useMemo(
    () => recaptionSettingsStorageKey({ datasetName, projectID, datasetPath, workerID }),
    [datasetName, datasetPath, projectID, workerID],
  );
  const autoBoxProviderLabel = AUTO_BOX_PROVIDERS.find(provider => provider.value === autoBoxProvider)?.label || 'OpenRouter';
  const recaptionModelOptions = useMemo(() => {
    if (recaptionProvider === 'openrouter') return OPENROUTER_BOX_MODELS;
    if (recaptionProvider === 'remote_ollama' && recaptionRemoteModelOptions.length > 0) return recaptionRemoteModelOptions;
    return OLLAMA_VISION_MODELS;
  }, [recaptionProvider, recaptionRemoteModelOptions]);
  const captionParse = useMemo(
    () => parseIdeogramCaption(captionText, selectedImageSize ?? undefined),
    [captionText, selectedImageSize],
  );
  const isIdeogram = captionParse.kind === 'ideogram';
  const boxes = isIdeogram ? captionParse.boxes : [];
  const selectedElement =
    isIdeogram && selectedElementIndex != null ? captionParse.elements[selectedElementIndex] ?? null : null;
  const selectedBox = boxes.find(box => box.elementIndex === selectedElementIndex) || null;
  const selectedLayerCaptionKey =
    selectedKey && selectedElementIndex != null ? layerCaptionRequestKey(selectedKey, selectedElementIndex) : '';
  const selectedLayerIsCaptioning = Boolean(selectedLayerCaptionKey && captioningLayerKeys.has(selectedLayerCaptionKey));
  const hasCurrentImageCaptioningLayer = Boolean(
    selectedKey && Array.from(captioningLayerKeys).some(requestKey => isLayerCaptionRequestForItem(requestKey, selectedKey)),
  );
  const selectedLayerCaptionMessage = selectedLayerCaptionKey ? layerCaptionMessages[selectedLayerCaptionKey] || '' : '';
  const selectedPalette = Array.isArray(selectedElement?.color_palette) ? selectedElement.color_palette : [];
  const isDirty = captionText.trim() !== savedCaption.trim();
  const captionStatus = statusForCaption(captionText, isCaptionLoaded);
  const isPlainTextItem = isPlainTextCaptionItem(selectedItem);
  const canAnnotate = isIdeogram && selectedKind === 'image' && isCaptionLoaded;
  const canConvertDataset = Boolean(datasetPath && onConvertDatasetToJson);
  const autoBoxDisabledReason = !isCaptionLoaded
    ? 'Load the caption first.'
    : selectedKind !== 'image'
      ? 'Auto Boxes works on images only.'
      : !isIdeogram
        ? 'Auto Boxes requires Ideogram JSON.'
        : selectedItem?.kind === 'encrypted' && !encryptedKey
          ? 'Unlock the encrypted dataset first.'
          : autoBoxProvider === 'remote_ollama' && !remoteOllamaWorkerId
            ? 'Select a Remote Ollama endpoint.'
            : !selectedImageSize
              ? 'Image size pending.'
              : '';
  const canGenerateAutoBoxes = !autoBoxDisabledReason && !isGeneratingBoxes && !hasCurrentImageCaptioningLayer && !isAutoCaptioning;
  const selectedLayerHasCaptionTarget = Boolean(selectedBox || layerCaptionTargetText(selectedElement));
  const layerCaptionDisabledReason = !isCaptionLoaded
    ? 'Load the caption first.'
    : selectedKind !== 'image'
      ? 'Caption Layer works on images only.'
      : !isIdeogram
        ? 'Caption Layer requires Ideogram JSON.'
        : selectedItem?.kind === 'encrypted' && !encryptedKey
          ? 'Unlock the encrypted dataset first.'
          : autoBoxProvider === 'remote_ollama' && !remoteOllamaWorkerId
            ? 'Select a Remote Ollama endpoint.'
            : !selectedImageSize
              ? 'Image size pending.'
              : !selectedElement || selectedElementIndex == null
                ? 'Select a layer.'
                : !selectedLayerHasCaptionTarget
                  ? 'Add a layer label or draw a box first.'
                  : '';
  const canCaptionSelectedLayer =
    !layerCaptionDisabledReason && !selectedLayerIsCaptioning && !isGeneratingBoxes && !isAutoCaptioning;
  const canRecaptionSelectedImage =
    Boolean(selectedItem) && selectedKind === 'image' && isCaptionLoaded && !isAutoCaptioning && !isSaving;
  const selectedRecaptionIsRunning = Boolean(selectedKey && isRecaptioning && activeRecaptionKey === selectedKey);
  const selectedRecaptionIsQueued = Boolean(selectedKey && recaptionQueue.some(entry => entry.key === selectedKey));
  const hasPendingRecaptions = isRecaptioning || recaptionQueue.length > 0;
  const canQueueSelectedRecaption = canRecaptionSelectedImage && !selectedRecaptionIsRunning && !selectedRecaptionIsQueued;
  const recaptionFeedback = isRecaptioning
    ? `Recaptioning${activeRecaptionLabel ? ` ${activeRecaptionLabel}` : ''}${
        recaptionQueue.length ? `, ${recaptionQueue.length} queued` : ''
      }.`
    : recaptionQueue.length
      ? `${recaptionQueue.length} recaption${recaptionQueue.length === 1 ? '' : 's'} queued.`
      : recaptionMessage;

  useEffect(() => {
    latestCaptionRef.current = captionText;
  }, [captionText]);

  useEffect(() => {
    recaptionSystemPromptRef.current = recaptionSystemPrompt;
  }, [recaptionSystemPrompt]);

  useEffect(() => {
    recaptionModelRef.current = recaptionModel;
  }, [recaptionModel]);

  useEffect(() => {
    isRecaptioningRef.current = isRecaptioning;
  }, [isRecaptioning]);

  useEffect(() => {
    recaptionQueueRef.current = recaptionQueue;
  }, [recaptionQueue]);

  useEffect(() => {
    savedCaptionRef.current = savedCaption;
    isDirtyRef.current = isDirty;
  }, [isDirty, savedCaption]);

  useEffect(() => {
    selectedKeyRef.current = selectedKey;
    if (isPlainTextCaptionItem(selectedItem)) setCaptionTab('caption');
    setAutoBoxMessage('');
    setSelectedImageSize(null);
    setHiddenLayerIndexes(new Set());
    setLockedLayerIndexes(new Set());
    setOverlapElementStack([]);
    setActivePaletteSamplerIndex(null);
  }, [selectedItem, selectedKey]);

  useEffect(() => {
    if (autoBoxProvider !== 'remote_ollama' || remoteOllamaWorkerId || remoteWorkerOptions.length === 0) return;
    setRemoteOllamaWorkerId(remoteWorkerOptions[0].value);
  }, [autoBoxProvider, remoteOllamaWorkerId, remoteWorkerOptions]);

  useEffect(() => {
    if (recaptionProvider !== 'remote_ollama' || recaptionRemoteWorkerId || remoteWorkerOptions.length === 0) return;
    setRecaptionRemoteWorkerId(remoteWorkerOptions[0].value);
  }, [recaptionProvider, recaptionRemoteWorkerId, remoteWorkerOptions]);

  const applyRecaptionRootPrompt = useCallback((value: string) => {
    const trimmed = value.trim();
    setRecaptionRootPrompt(trimmed);
    if (!trimmed || recaptionSystemPromptTouchedRef.current || recaptionSystemPromptRef.current.trim()) return;
    recaptionSystemPromptRef.current = trimmed;
    setRecaptionSystemPrompt(trimmed);
  }, []);

  useEffect(() => {
    let cancelled = false;
    recaptionSystemPromptTouchedRef.current = false;
    recaptionSystemPromptRef.current = '';
    setRecaptionRootPrompt('');
    setRecaptionSystemPrompt('');

    if (rootCaption !== undefined) {
      const value = rootCaption || '';
      applyRecaptionRootPrompt(value);
      setRecaptionRootPromptStatus(value.trim() ? 'success' : 'idle');
      return;
    }

    if (!datasetName || workerID !== 'local') {
      setRecaptionRootPromptStatus('idle');
      return;
    }

    setRecaptionRootPromptStatus('loading');
    apiClient
      .post('/api/datasets/root-caption', {
        datasetName,
        ...projectPayload,
      })
      .then(response => {
        if (cancelled) return;
        if (response.data?.found) {
          applyRecaptionRootPrompt(response.data.systemPrompt || '');
          setRecaptionRootPromptStatus('success');
        } else {
          setRecaptionRootPromptStatus('idle');
        }
      })
      .catch(error => {
        if (cancelled) return;
        console.warn('Could not load dataset root prompt for recaption:', error);
        setRecaptionRootPromptStatus('error');
      });

    return () => {
      cancelled = true;
    };
  }, [applyRecaptionRootPrompt, datasetName, projectPayload, rootCaption, workerID]);

  useEffect(() => {
    setHasRecaptionSettingsForDataset(false);
    const preset = readRecaptionSettingsPreset(recaptionStorageKey);
    if (!preset) return;
    setRecaptionProvider(preset.provider);
    setRecaptionModel(preset.model);
    setRecaptionOutputFormat(preset.outputFormat);
    setRecaptionPrompt(preset.prompt);
    setRecaptionSystemPrompt(preset.systemPrompt);
    recaptionSystemPromptRef.current = preset.systemPrompt;
    recaptionSystemPromptTouchedRef.current = true;
    setRecaptionRemoteWorkerId(preset.remoteWorkerId);
    setRecaptionMaxNewTokens(preset.maxNewTokens);
    setHasRecaptionSettingsForDataset(true);
  }, [recaptionStorageKey, rootCaption]);

  const useRecaptionRootPrompt = useCallback(() => {
    recaptionSystemPromptTouchedRef.current = true;
    recaptionSystemPromptRef.current = recaptionRootPrompt;
    setRecaptionSystemPrompt(recaptionRootPrompt);
  }, [recaptionRootPrompt]);

  const loadRecaptionRemoteModels = useCallback(
    async (workerId = recaptionRemoteWorkerId, preferredModel = recaptionModelRef.current) => {
      if (!workerId) return;
      const preferred = preferredModel.trim();
      setRecaptionRemoteModelStatus('loading');
      setRecaptionRemoteModelError('');
      setRecaptionRemoteModelOptions([]);
      try {
        const response = await apiClient.get(`/api/ollama-workers/${encodeURIComponent(workerId)}/models`);
        const options = ollamaModelOptions(response.data?.models || []);
        setRecaptionRemoteModelOptions(options);
        setRecaptionRemoteModelStatus('success');
        setRecaptionModel(currentModel => {
          const trimmed = preferred || currentModel.trim();
          if (!options.length) return '';
          return options.some(option => option.value === trimmed) ? trimmed : options[0].value;
        });
      } catch (error: any) {
        setRecaptionRemoteModelOptions([]);
        setRecaptionRemoteModelStatus('error');
        setRecaptionRemoteModelError(error?.response?.data?.error || error?.message || 'Could not load Remote Ollama models.');
      }
    },
    [recaptionRemoteWorkerId],
  );

  useEffect(() => {
    if (recaptionProvider !== 'remote_ollama' || !recaptionRemoteWorkerId) return;
    void loadRecaptionRemoteModels(recaptionRemoteWorkerId);
  }, [loadRecaptionRemoteModels, recaptionProvider, recaptionRemoteWorkerId]);

  const handleAutoBoxProviderChange = useCallback((value: string) => {
    const nextProvider = AUTO_BOX_PROVIDERS.some(provider => provider.value === value)
      ? (value as StudioBoxProvider)
      : 'openrouter';
    setAutoBoxProvider(nextProvider);
    setAutoBoxModel(currentModel => {
      const trimmed = currentModel.trim();
      if (nextProvider === 'openrouter') {
        return !trimmed || trimmed.startsWith('qwen3.5:') ? DEFAULT_OPENROUTER_BOX_MODEL : trimmed;
      }
      return !trimmed || trimmed === DEFAULT_OPENROUTER_BOX_MODEL ? DEFAULT_OLLAMA_VISION_MODEL : trimmed;
    });
  }, []);

  const handleRecaptionProviderChange = useCallback((value: string) => {
    const nextProvider = AUTO_BOX_PROVIDERS.some(provider => provider.value === value)
      ? (value as RecaptionProvider)
      : 'openrouter';
    setRecaptionProvider(nextProvider);
    if (nextProvider !== 'remote_ollama') {
      setRecaptionRemoteModelStatus('idle');
      setRecaptionRemoteModelError('');
    }
    setRecaptionModel(currentModel => {
      const trimmed = currentModel.trim();
      if (nextProvider === 'openrouter') {
        return !trimmed || trimmed.startsWith('qwen3.5:') || trimmed.startsWith('gemma4:') ? DEFAULT_OPENROUTER_BOX_MODEL : trimmed;
      }
      if (nextProvider === 'remote_ollama') {
        return recaptionRemoteModelOptions.some(option => option.value === trimmed)
          ? trimmed
          : recaptionRemoteModelOptions[0]?.value || '';
      }
      return !trimmed || trimmed === DEFAULT_OPENROUTER_BOX_MODEL ? DEFAULT_OLLAMA_VISION_MODEL : trimmed;
    });
  }, [recaptionRemoteModelOptions]);

  const handleRecaptionOutputFormatChange = useCallback((value: RecaptionOutputFormat) => {
    setRecaptionOutputFormat(value);
    setRecaptionPrompt(currentPrompt => {
      const trimmed = currentPrompt.trim();
      if (value === 'ideogram_json') {
        return !trimmed || trimmed === defaultImageCaptionPrompt ? defaultIdeogramJsonCaptionPrompt : currentPrompt;
      }
      return !trimmed || trimmed === defaultIdeogramJsonCaptionPrompt ? defaultImageCaptionPrompt : currentPrompt;
    });
    setRecaptionMaxNewTokens(current => {
      if (value === 'ideogram_json') return current < 2048 ? 2048 : current;
      return current >= 2048 ? 256 : current;
    });
  }, []);

  const currentRecaptionSettings = useCallback(
    (): RecaptionSettingsPreset => ({
      provider: recaptionProvider,
      model: recaptionModel.trim(),
      outputFormat: recaptionOutputFormat,
      prompt: recaptionPrompt,
      systemPrompt: recaptionSystemPrompt,
      remoteWorkerId: recaptionRemoteWorkerId,
      maxNewTokens: Math.max(1, Math.floor(Number(recaptionMaxNewTokens) || 1)),
    }),
    [
      recaptionMaxNewTokens,
      recaptionModel,
      recaptionOutputFormat,
      recaptionPrompt,
      recaptionProvider,
      recaptionRemoteWorkerId,
      recaptionSystemPrompt,
    ],
  );

  const persistRecaptionSettingsForDataset = useCallback(
    (settings: RecaptionSettingsPreset) => {
      if (!recaptionStorageKey || typeof window === 'undefined') {
        setHasRecaptionSettingsForDataset(true);
        return;
      }
      try {
        window.localStorage.setItem(recaptionStorageKey, JSON.stringify(settings));
      } catch (error) {
        console.warn('Could not save recaption settings:', error);
      }
      setHasRecaptionSettingsForDataset(true);
    },
    [recaptionStorageKey],
  );

  const appendRecaptionFields = useCallback(
    (formData: FormData, settings: RecaptionSettingsPreset, existingCaption: string) => {
      formData.append('provider', settings.provider);
      formData.append('model', settings.model);
      formData.append('outputFormat', settings.outputFormat);
      formData.append('prompt', settings.prompt);
      formData.append('systemPrompt', settings.systemPrompt);
      formData.append('existingCaption', existingCaption);
      formData.append('datasetName', datasetName);
      formData.append('remoteWorkerId', settings.remoteWorkerId);
      formData.append('maxNewTokens', String(settings.maxNewTokens));
    },
    [datasetName],
  );

  const readCaptionForItem = useCallback(
    async (item: DatasetStudioItem, signal?: AbortSignal) => {
      let text = '';
      if (item.kind === 'plain') {
        const direct = isPlainTextCaptionItem(item);
        const response = await apiClient.post(
          '/api/caption/get',
          {
            imgPath: item.path,
            ...(direct ? { direct: true } : {}),
            ...projectPayload,
          },
          signal ? { signal } : undefined,
        );
        text = captionResponseToText(response.data);
      } else if (encryptedKey) {
        const captionPath = item.item.captionObjectPath;
        if (captionPath) {
          const response = await apiClient.post(
            '/api/datasets/encrypted/object',
            { datasetName, worker_id: workerID, objectPath: captionPath, ...projectPayload },
            { responseType: 'blob', ...(signal ? { signal } : {}) },
          );
          const decrypted = await decryptEncryptedObjectBlob(encryptedKey, captionPath, response.data as Blob);
          text = new TextDecoder().decode(decrypted);
        }
      }
      return text;
    },
    [datasetName, encryptedKey, projectPayload, workerID],
  );

  useEffect(() => {
    if (!selectedKey) {
      setCaptionText('');
      setSavedCaption('');
      setIsCaptionLoaded(false);
      return;
    }

    const cached = captionCacheRef.current.get(selectedKey);
    if (cached?.loaded) {
      setCaptionText(cached.caption);
      setSavedCaption(cached.saved);
      setIsCaptionLoaded(true);
      return;
    }

    let cancelled = false;
    setCaptionText('');
    setSavedCaption('');
    setIsCaptionLoaded(false);

    async function loadCaption() {
      try {
        if (!selectedItem) return;
        const text = await readCaptionForItem(selectedItem);
        if (cancelled) return;
        setCaptionText(text);
        setSavedCaption(text);
        setIsCaptionLoaded(true);
        writeCaptionCache(selectedKey, { caption: text, saved: text, loaded: true });
      } catch (error) {
        if (!cancelled) {
          console.error('Caption load failed:', error);
          setIsCaptionLoaded(true);
          writeCaptionCache(selectedKey, { caption: '', saved: '', loaded: true });
        }
      }
    }

    void loadCaption();
    return () => {
      cancelled = true;
    };
  }, [readCaptionForItem, selectedItem, selectedKey, writeCaptionCache]);

  useEffect(() => {
    if (!selectedKey) return;
    writeCaptionCache(selectedKey, { caption: captionText, saved: savedCaption, loaded: isCaptionLoaded });
  }, [captionText, isCaptionLoaded, savedCaption, selectedKey, writeCaptionCache]);

  useEffect(() => {
    if (plainAuditItemPaths.length === 0) return;
    const controller = new AbortController();
    let cancelled = false;

    const runAudit = async () => {
      try {
        const response = await apiClient.post<RefusalCaptionAuditResponse>(
          '/api/datasets/refusal-caption-audit',
          {
            datasetName,
            worker_id: workerID,
            itemPaths: plainAuditItemPaths,
            ...projectPayload,
          },
          { signal: controller.signal },
        );
        if (cancelled || controller.signal.aborted) return;
        const refusals = response.data?.refusals || {};
        let cacheChanged = false;

        Object.entries(refusals).forEach(([key, captionValue]) => {
          if (typeof captionValue !== 'string') return;
          const previous = captionCacheRef.current.get(key);
          if (previous?.loaded && previous.caption === captionValue && previous.saved === captionValue) return;

          captionCacheRef.current.set(key, { caption: captionValue, saved: captionValue, loaded: true });
          cacheChanged = true;

          if (selectedKeyRef.current === key && !isDirtyRef.current) {
            latestCaptionRef.current = captionValue;
            setCaptionText(captionValue);
            setSavedCaption(captionValue);
            setIsCaptionLoaded(true);
            setUndoStack([]);
            setRedoStack([]);
          }
        });

        if (cacheChanged) {
          setCaptionCacheVersion(version => version + 1);
        }
      } catch (error: any) {
        if (error?.name !== 'CanceledError' && error?.name !== 'AbortError' && !controller.signal.aborted) {
          console.warn('Caption refusal audit failed:', error);
        }
      }
    };

    const cancelScheduledAudit = scheduleIdleTask(() => void runAudit());
    return () => {
      cancelled = true;
      controller.abort();
      cancelScheduledAudit();
    };
  }, [datasetName, plainAuditItemPaths, plainAuditKey, projectPayload, workerID]);

  useEffect(() => {
    if (!isAutoCaptioning || !selectedItem || !selectedKey) return;
    const requestKey = selectedKey;
    const controller = new AbortController();
    let busy = false;

    const pollSelectedCaption = async () => {
      if (busy || controller.signal.aborted) return;
      busy = true;
      try {
        const text = await readCaptionForItem(selectedItem, controller.signal);
        if (controller.signal.aborted || selectedKeyRef.current !== requestKey) return;
        const cached = captionCacheRef.current.get(requestKey);
        if (cached?.caption === text && cached.saved === text && cached.loaded) return;
        captionCacheRef.current.set(requestKey, { caption: text, saved: text, loaded: true });
        setCaptionCacheVersion(version => version + 1);
        if (!isDirtyRef.current && savedCaptionRef.current !== text) {
          latestCaptionRef.current = text;
          setCaptionText(text);
          setSavedCaption(text);
          setIsCaptionLoaded(true);
          setUndoStack([]);
          setRedoStack([]);
        }
      } catch (error: any) {
        if (error?.name !== 'AbortError') {
          console.error('Caption refresh failed:', error);
        }
      } finally {
        busy = false;
      }
    };

    void pollSelectedCaption();
    const interval = window.setInterval(pollSelectedCaption, 5000);
    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, [isAutoCaptioning, readCaptionForItem, selectedItem, selectedKey]);

  useEffect(() => {
    if (!isAutoCaptioning || !encryptedKey || !onRefresh) return;
    const interval = window.setInterval(() => onRefresh(), 5000);
    return () => window.clearInterval(interval);
  }, [encryptedKey, isAutoCaptioning, onRefresh]);

  useEffect(() => {
    if (!isIdeogram || selectedElementIndex == null) return;
    if (!captionParse.elements[selectedElementIndex]) setSelectedElementIndex(null);
  }, [captionParse, isIdeogram, selectedElementIndex]);

  useEffect(() => {
    if (!isIdeogram) {
      setHiddenLayerIndexes(new Set());
      setLockedLayerIndexes(new Set());
      setOverlapElementStack([]);
      return;
    }
    const elementCount = captionParse.elements.length;
    setHiddenLayerIndexes(previous => {
      const next = new Set([...previous].filter(elementIndex => elementIndex < elementCount));
      return next.size === previous.size ? previous : next;
    });
    setLockedLayerIndexes(previous => {
      const next = new Set([...previous].filter(elementIndex => elementIndex < elementCount));
      return next.size === previous.size ? previous : next;
    });
    setOverlapElementStack(previous => previous.filter(elementIndex => elementIndex < elementCount));
  }, [captionParse, isIdeogram]);

  useEffect(() => {
    if (!selectedKey || autoSelectKeyRef.current === selectedKey) return;
    if (!isIdeogram || boxes.length === 0) return;
    autoSelectKeyRef.current = selectedKey;
    setSelectedElementIndex(boxes[0].elementIndex);
  }, [boxes, isIdeogram, selectedKey]);

  const saveCaptionForItem = useCallback(
    async (targetItem: DatasetStudioItem, targetKey: string, captionValue: string) => {
      const value = isPlainTextCaptionItem(targetItem) ? captionValue : captionValue.trim();
      if (targetItem.kind === 'plain') {
        await apiClient.post('/api/img/caption', {
          imgPath: targetItem.path,
          caption: value,
          direct: isPlainTextCaptionItem(targetItem),
          ...projectPayload,
        });
      } else if (encryptedKey && onSaveEncryptedCaption) {
        const targetCaptionPath =
          encryptedCaptionPaths[targetKey] || targetItem.item.captionObjectPath || captionObjectPath(randomId());
        const encryptedCaption = await encryptCaptionObject(encryptedKey, targetCaptionPath, value);
        await onSaveEncryptedCaption(targetItem.item, targetCaptionPath, JSON.stringify(encryptedCaption));
        setEncryptedCaptionPaths(previous => ({ ...previous, [targetKey]: targetCaptionPath }));
      } else {
        throw new Error('Encrypted dataset is locked.');
      }

      if (selectedKeyRef.current === targetKey) {
        latestCaptionRef.current = value;
        setCaptionText(value);
        setSavedCaption(value);
      }
      writeCaptionCache(targetKey, { caption: value, saved: value, loaded: true });
    },
    [encryptedCaptionPaths, encryptedKey, onSaveEncryptedCaption, projectPayload, writeCaptionCache],
  );

  const saveCaption = useCallback(async (captionOverride?: string) => {
    if (!selectedItem || !isCaptionLoaded || isSaving) return;
    const sourceCaption = captionOverride ?? captionText;
    const value = isPlainTextCaptionItem(selectedItem) ? sourceCaption : sourceCaption.trim();
    if (captionOverride === undefined && !isDirty) return;
    if (captionOverride !== undefined && value.trim() === savedCaption.trim()) return;
    setIsSaving(true);
    try {
      await saveCaptionForItem(selectedItem, selectedKey, value);
    } catch (error) {
      console.error('Caption save failed:', error);
      alert('Failed to save caption. Please try again.');
      if (captionOverride !== undefined) throw error;
    } finally {
      setIsSaving(false);
    }
  }, [
    captionText,
    isCaptionLoaded,
    isDirty,
    isSaving,
    savedCaption,
    saveCaptionForItem,
    selectedItem,
    selectedKey,
  ]);

  useEffect(() => {
    saveCaptionRef.current = saveCaption;
  }, [saveCaption]);

  const runRecaptionQueueEntry = useCallback(async (entry: RecaptionQueueEntry) => {
    const { item, key, name, existingCaption, settings } = entry;
    if (!settings.model.trim()) {
      setRecaptionMessage('Select a model before recaptioning.');
      return;
    }
    if (settings.provider === 'remote_ollama' && !settings.remoteWorkerId) {
      setRecaptionMessage('Select a Remote Ollama endpoint.');
      return;
    }
    if (item.kind === 'encrypted' && !encryptedKey) {
      setRecaptionMessage('Unlock the encrypted dataset first.');
      return;
    }

    const providerLabel = AUTO_BOX_PROVIDERS.find(provider => provider.value === settings.provider)?.label || settings.provider;
    setIsRecaptioning(true);
    setActiveRecaptionLabel(name);
    setActiveRecaptionKey(key);
    setRecaptionMessage(`Recaptioning ${name}${recaptionQueueRef.current.length ? ` (${recaptionQueueRef.current.length} queued)` : ''}.`);
    try {
      let response;
      if (item.kind === 'plain') {
        response = await apiClient.post(
          '/api/datasets/recaption-single',
          {
            imgPath: item.path,
            provider: settings.provider,
            model: settings.model,
            outputFormat: settings.outputFormat,
            prompt: settings.prompt,
            systemPrompt: settings.systemPrompt,
            existingCaption,
            datasetName,
            worker_id: workerID,
            remoteWorkerId: settings.remoteWorkerId,
            maxNewTokens: settings.maxNewTokens,
            ...projectPayload,
          },
          { timeout: 0 },
        );
      } else {
        if (!encryptedProviderConfirmations[`recaption:${settings.provider}`]) {
          const confirmed = window.confirm(
            `Recaption will send this decrypted image to ${providerLabel}. Continue?`,
          );
          if (!confirmed) {
            setRecaptionMessage('Recaption canceled.');
            return;
          }
          setEncryptedProviderConfirmations(previous => ({ ...previous, [`recaption:${settings.provider}`]: true }));
        }
        const formData = await createEncryptedImageFormData({
          datasetName,
          workerID,
          projectID,
          encryptedKey: encryptedKey as CryptoKey,
          item: item.item,
        });
        appendRecaptionFields(formData, settings, existingCaption);
        response = await apiClient.post('/api/datasets/recaption-single', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 0,
        });
      }

      const caption = String(response.data?.caption || '').trim();
      if (!caption) throw new Error('Recaption returned an empty caption.');
      if (selectedKeyRef.current === key) {
        setUndoStack(previous => [...previous.slice(Math.max(0, previous.length - MAX_HISTORY + 1)), latestCaptionRef.current]);
        setRedoStack([]);
        if (settings.outputFormat === 'ideogram_json') setCaptionTab('json');
      }
      await saveCaptionForItem(item, key, caption);
      setRecaptionMessage(`${name} recaptioned${recaptionQueueRef.current.length ? `, ${recaptionQueueRef.current.length} queued` : ''}.`);
    } catch (error) {
      setRecaptionMessage(`${name}: ${responseErrorMessage(error, 'Recaption failed. Please try again.')}`);
    } finally {
      setIsRecaptioning(false);
      setActiveRecaptionLabel('');
      setActiveRecaptionKey('');
    }
  }, [
    appendRecaptionFields,
    datasetName,
    encryptedKey,
    encryptedProviderConfirmations,
    projectID,
    projectPayload,
    saveCaptionForItem,
    workerID,
  ]);

  useEffect(() => {
    if (isRecaptioning || recaptionQueue.length === 0) return;
    const [nextEntry] = recaptionQueue;
    setRecaptionQueue(previous => previous.slice(1));
    void runRecaptionQueueEntry(nextEntry);
  }, [isRecaptioning, recaptionQueue, runRecaptionQueueEntry]);

  const queueSelectedRecaption = useCallback(() => {
    if (!selectedItem || !canRecaptionSelectedImage) return;
    if (selectedRecaptionIsRunning) {
      setRecaptionMessage(`${selectedName} is already being recaptioned.`);
      return;
    }
    if (selectedRecaptionIsQueued) {
      setRecaptionMessage(`${selectedName} is already queued.`);
      return;
    }
    const settings = currentRecaptionSettings();
    if (!settings.model.trim()) {
      setRecaptionMessage('Select a model before recaptioning.');
      setIsRecaptionModalOpen(true);
      return;
    }
    if (settings.provider === 'remote_ollama' && !settings.remoteWorkerId) {
      setRecaptionMessage('Select a Remote Ollama endpoint.');
      setIsRecaptionModalOpen(true);
      return;
    }
    if (selectedItem.kind === 'encrypted' && !encryptedKey) {
      setRecaptionMessage('Unlock the encrypted dataset first.');
      setIsRecaptionModalOpen(true);
      return;
    }

    const entry: RecaptionQueueEntry = {
      id: randomId(),
      item: selectedItem,
      key: selectedKey,
      name: selectedName,
      existingCaption: captionText,
      settings,
    };
    persistRecaptionSettingsForDataset(settings);
    setRecaptionQueue(previous => [...previous, entry]);
    const queuedTotal = recaptionQueueRef.current.length + (isRecaptioningRef.current ? 1 : 0) + 1;
    setRecaptionMessage(
      queuedTotal > 1 ? `${selectedName} queued (${queuedTotal} total).` : `${selectedName} queued.`,
    );
    setIsRecaptionModalOpen(false);
  }, [
    canRecaptionSelectedImage,
    captionText,
    currentRecaptionSettings,
    encryptedKey,
    persistRecaptionSettingsForDataset,
    selectedItem,
    selectedKey,
    selectedName,
    selectedRecaptionIsQueued,
    selectedRecaptionIsRunning,
  ]);

  const openRecaptionSettings = useCallback(() => {
    setRecaptionMessage('');
    setIsRecaptionModalOpen(true);
  }, []);

  const handleRecaptionClick = useCallback(() => {
    setRecaptionMessage('');
    if (hasRecaptionSettingsForDataset) {
      queueSelectedRecaption();
      return;
    }
    setIsRecaptionModalOpen(true);
  }, [hasRecaptionSettingsForDataset, queueSelectedRecaption]);

  useEffect(() => {
    return () => {
      void saveCaptionRef.current();
    };
  }, []);

  const selectIndex = useCallback(
    (nextIndex: number) => {
      void saveCaption();
      setSelectedIndex(clampIndex(nextIndex, items.length));
      setSelectedElementIndex(null);
      setActiveTool('select');
      setUndoStack([]);
      setRedoStack([]);
    },
    [items.length, saveCaption],
  );

  const imageDeleteResultMessage = useCallback((result: DeleteImagesResult) => {
    const deleted = result.deleted.toLocaleString();
    const requested = result.requested.toLocaleString();
    const failed = result.failed || 0;
    const skipped = result.skipped || 0;
    if (failed > 0) {
      return `${deleted} of ${requested} deleted, ${failed.toLocaleString()} failed.`;
    }
    if (skipped > 0) {
      return `${deleted} deleted, ${skipped.toLocaleString()} already missing.`;
    }
    return `${deleted} deleted.`;
  }, []);

  const applyImageDeleteResult = useCallback(
    (result: DeleteImagesResult, requestedItems: DatasetStudioItem[]) => {
      const requestedKeys = requestedItems.map(item => itemKey(item));
      const removedKeys =
        result.removedKeys && result.removedKeys.length > 0
          ? result.removedKeys
          : result.deleted > 0 && (result.failed || 0) === 0
            ? requestedKeys
            : [];
      if (removedKeys.length === 0) return;

      const removedKeySet = new Set(removedKeys);
      let cacheChanged = false;
      removedKeySet.forEach(key => {
        cacheChanged = captionCacheRef.current.delete(key) || cacheChanged;
      });
      if (cacheChanged) bumpCaptionCacheVersion();

      const remainingItems = items.filter(item => !removedKeySet.has(itemKey(item)));
      const currentSelectedKey = selectedKeyRef.current;
      const currentWasDeleted = removedKeySet.has(currentSelectedKey);
      const nextSelectedIndex = currentWasDeleted
        ? clampIndex(Math.min(selectedIndex, remainingItems.length - 1), remainingItems.length)
        : Math.max(
            0,
            remainingItems.findIndex(item => itemKey(item) === currentSelectedKey),
          );

      if (remainingItems.length === 0 || currentWasDeleted) {
        latestCaptionRef.current = '';
        setCaptionText('');
        setSavedCaption('');
        setSelectedElementIndex(null);
        setUndoStack([]);
        setRedoStack([]);
      }
      if (nextSelectedIndex >= 0) setSelectedIndex(nextSelectedIndex);
    },
    [bumpCaptionCacheVersion, items, selectedIndex],
  );

  const handleDeleteImages = useCallback(
    async (targetItems: DatasetStudioItem[], label = 'selected image(s)'): Promise<DeleteImagesResult> => {
      if (!onDeleteImages || isDeletingImages) {
        return { requested: targetItems.length, deleted: 0, failed: targetItems.length };
      }
      const uniqueItems = Array.from(new Map(targetItems.map(item => [itemKey(item), item] as const)).values());
      if (uniqueItems.length === 0) return { requested: 0, deleted: 0 };

      const includesTextFiles = uniqueItems.some(isPlainTextCaptionItem);
      const suffix = includesTextFiles
        ? 'Text files will be deleted directly.'
        : 'Associated captions will be removed too.';
      const confirmed = window.confirm(`Delete ${uniqueItems.length.toLocaleString()} ${label}? ${suffix}`);
      if (!confirmed) return { requested: uniqueItems.length, deleted: 0 };

      setIsDeletingImages(true);
      setDeleteMessage('');
      try {
        await saveCaption();
        const result = await onDeleteImages(uniqueItems);
        applyImageDeleteResult(result, uniqueItems);
        const message = result.message || imageDeleteResultMessage(result);
        setDeleteMessage(message);
        return result;
      } catch (error) {
        const message = responseErrorMessage(error, 'Failed to delete image(s).');
        setDeleteMessage(message);
        alert(message);
        throw error;
      } finally {
        setIsDeletingImages(false);
      }
    },
    [applyImageDeleteResult, imageDeleteResultMessage, isDeletingImages, onDeleteImages, saveCaption],
  );

  const handleDeleteCurrentImage = useCallback(() => {
    if (!selectedItem) return;
    void handleDeleteImages([selectedItem], isPlainTextCaptionItem(selectedItem) ? 'current text file' : 'current image');
  }, [handleDeleteImages, selectedItem]);

  const applyBulkCaptionResult = useCallback(
    (result: BulkCaptionActionResult) => {
      let cacheChanged = false;
      if (result.updatedCaptions) {
        Object.entries(result.updatedCaptions).forEach(([key, caption]) => {
          captionCacheRef.current.set(key, { caption, saved: caption, loaded: true });
          cacheChanged = true;
          if (selectedKeyRef.current === key) {
            latestCaptionRef.current = caption;
            setCaptionText(caption);
            setSavedCaption(caption);
            setIsCaptionLoaded(true);
            setUndoStack([]);
            setRedoStack([]);
          }
        });
      }
      if (result.removedKeys) {
        result.removedKeys.forEach(key => {
          cacheChanged = captionCacheRef.current.delete(key) || cacheChanged;
          if (selectedKeyRef.current === key) {
            latestCaptionRef.current = '';
            setCaptionText('');
            setSavedCaption('');
            setSelectedElementIndex(null);
            setUndoStack([]);
            setRedoStack([]);
          }
        });
      }
      if (cacheChanged) bumpCaptionCacheVersion();
    },
    [bumpCaptionCacheVersion],
  );

  const handleBulkCaptionAction = useCallback(
    async (request: BulkCaptionActionRequest): Promise<BulkCaptionActionResult> => {
      if (request.matches.length === 0) {
        return { action: request.action, found: 0, affected: 0 };
      }

      await saveCaption();
      const firstItem = request.matches[0]?.item;
      let result: BulkCaptionActionResult;

      if (firstItem?.kind === 'encrypted') {
        if (!onBulkEncryptedCaptionAction) {
          throw new Error('Encrypted bulk actions are not available for this dataset.');
        }
        result = await onBulkEncryptedCaptionAction(request);
      } else {
        const plainImgPaths = request.matches.flatMap(match => (match.item.kind === 'plain' ? [match.item.path] : []));
        const response = await apiClient.post('/api/datasets/caption-bulk', {
          datasetName,
          worker_id: workerID,
          ...projectPayload,
          action: request.action,
          query: request.query,
          matchMode: request.matchMode,
          destinationName: request.destinationName,
          imgPaths: plainImgPaths,
        });
        const data = response.data || {};
        result = {
          action: request.action,
          found: Number(data.found || 0),
          affected: Number(data.affected || 0),
          deleted: data.deleted,
          moved: data.moved,
          updated: data.updated,
          removedWords: data.removedWords,
          destinationName: data.destinationName,
          updatedCaptions: data.updatedCaptions,
          removedKeys: Array.isArray(data.removedPaths) ? data.removedPaths : undefined,
        };
      }

      applyBulkCaptionResult(result);
      if (request.action === 'delete' || request.action === 'move') {
        onRefresh?.();
      }
      return result;
    },
    [applyBulkCaptionResult, datasetName, onBulkEncryptedCaptionAction, onRefresh, projectPayload, saveCaption, workerID],
  );

  const mutateCaption = useCallback(
    (mutator: (data: Record<string, any>) => void, nextSelectedElementIndex?: number | null) => {
      const parsed = parseIdeogramCaption(captionText, selectedImageSize ?? undefined);
      if (parsed.kind !== 'ideogram') return;
      const data = cloneIdeogramData(parsed.data);
      mutator(data);
      const next = serializeIdeogramCaption(data);
      if (next === captionText) return;
      setUndoStack(previous => [...previous.slice(Math.max(0, previous.length - MAX_HISTORY + 1)), captionText]);
      setRedoStack([]);
      latestCaptionRef.current = next;
      setCaptionText(next);
      if (nextSelectedElementIndex !== undefined) setSelectedElementIndex(nextSelectedElementIndex);
    },
    [captionText, selectedImageSize],
  );

  const mutateLatestCaption = useCallback(
    (mutator: (data: Record<string, any>) => void, nextSelectedElementIndex?: number | null) => {
      const currentCaption = latestCaptionRef.current;
      const parsed = parseIdeogramCaption(currentCaption, selectedImageSize ?? undefined);
      if (parsed.kind !== 'ideogram') return false;
      const data = cloneIdeogramData(parsed.data);
      mutator(data);
      const next = serializeIdeogramCaption(data);
      if (next === currentCaption) return false;
      setUndoStack(previous => [...previous.slice(Math.max(0, previous.length - MAX_HISTORY + 1)), currentCaption]);
      setRedoStack([]);
      latestCaptionRef.current = next;
      setCaptionText(next);
      if (nextSelectedElementIndex !== undefined) setSelectedElementIndex(nextSelectedElementIndex);
      return true;
    },
    [selectedImageSize],
  );

  const setLayerCaptionMessageForKey = useCallback((requestLayerKey: string, message: string) => {
    setLayerCaptionMessages(previous => {
      if (!requestLayerKey) return previous;
      if (!message) {
        if (!Object.prototype.hasOwnProperty.call(previous, requestLayerKey)) return previous;
        const next = { ...previous };
        delete next[requestLayerKey];
        return next;
      }
      return { ...previous, [requestLayerKey]: message };
    });
  }, []);

  const setLayerCaptioningForKey = useCallback((requestLayerKey: string, isPending: boolean) => {
    setCaptioningLayerKeys(previous => {
      if (!requestLayerKey) return previous;
      const next = new Set(previous);
      if (isPending) {
        next.add(requestLayerKey);
      } else {
        next.delete(requestLayerKey);
      }
      return next;
    });
  }, []);

  const handleGenerateAutoBoxes = useCallback(async () => {
    if (!selectedItem || autoBoxDisabledReason || isGeneratingBoxes) return;

    const requestCaption = captionText;
    const requestKey = selectedKey;
    const imageWidth = selectedImageSize?.width || null;
    const imageHeight = selectedImageSize?.height || null;

    setIsGeneratingBoxes(true);
    setAutoBoxMessage('');
    try {
      let response;
      if (selectedItem.kind === 'plain') {
        response = await apiClient.post(
          '/api/datasets/auto-boxes',
          {
            imgPath: selectedItem.path,
            caption: requestCaption,
            provider: autoBoxProvider,
            model: autoBoxModel,
            remoteWorkerId: remoteOllamaWorkerId,
            refine: autoBoxRefine,
            imageWidth,
            imageHeight,
            ...projectPayload,
          },
          { timeout: 0 },
        );
      } else {
        if (!encryptedKey) throw new Error('Unlock the encrypted dataset first.');
        if (!encryptedProviderConfirmations[autoBoxProvider]) {
          const confirmed = window.confirm(
            `Auto Boxes will send this decrypted image to ${autoBoxProviderLabel} to generate bounding boxes. Continue?`,
          );
          if (!confirmed) {
            setAutoBoxMessage('Auto Boxes canceled.');
            return;
          }
          setEncryptedProviderConfirmations(previous => ({ ...previous, [autoBoxProvider]: true }));
        }

        const formData = await createEncryptedImageFormData({
          datasetName,
          workerID,
          projectID,
          encryptedKey,
          item: selectedItem.item,
        });
        formData.append('caption', requestCaption);
        formData.append('provider', autoBoxProvider);
        formData.append('model', autoBoxModel);
        formData.append('remoteWorkerId', remoteOllamaWorkerId);
        formData.append('refine', autoBoxRefine ? 'true' : 'false');
        appendImageSizeFields(formData, imageWidth, imageHeight);

        response = await apiClient.post('/api/datasets/auto-boxes', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 0,
        });
      }

      if (selectedKeyRef.current !== requestKey || latestCaptionRef.current !== requestCaption) {
        setAutoBoxMessage('Caption changed while Auto Boxes was running. Rerun Auto Boxes to apply fresh boxes.');
        return;
      }

      const elementCount = captionParse.kind === 'ideogram' ? captionParse.elements.length : 0;
      const patches =
        elementCount > 0 ? normalizeGeneratedBoxPatches({ boxes: response.data?.boxes }, elementCount, 2) : [];
      const generatedElements =
        elementCount === 0 ? normalizeGeneratedElementBoxes({ generatedElements: response.data?.generatedElements }, 2, 20) : [];
      if (patches.length === 0 && generatedElements.length === 0) {
        throw new Error(`${autoBoxProviderLabel} did not return any usable boxes.`);
      }

      let appliedCount = 0;
      const nextSelection =
        generatedElements.length > 0 ? elementCount : selectedElementIndex ?? patches[0]?.elementIndex ?? null;
      mutateCaption(data => {
        if (generatedElements.length > 0) {
          const result = appendGeneratedIdeogramElements(data, generatedElements as GeneratedElementBox[]);
          appliedCount = result.count;
        } else {
          appliedCount = applyGeneratedBoxPatches(data, patches as GeneratedBoxPatch[]);
        }
      }, nextSelection);
      const count = appliedCount || patches.length || generatedElements.length;
      setAutoBoxMessage(`${count} box${count === 1 ? '' : 'es'} ${response.data?.refined ? 'refined' : 'generated'}.`);
    } catch (error) {
      console.error('Auto Boxes failed:', error);
      setAutoBoxMessage(responseErrorMessage(error, 'Auto Boxes failed. Please try again.'));
    } finally {
      setIsGeneratingBoxes(false);
    }
  }, [
    autoBoxDisabledReason,
    autoBoxProvider,
    autoBoxProviderLabel,
    autoBoxModel,
    autoBoxRefine,
    captionParse,
    captionText,
    datasetName,
    encryptedKey,
    encryptedProviderConfirmations,
    isGeneratingBoxes,
    mutateCaption,
    projectID,
    projectPayload,
    remoteOllamaWorkerId,
    selectedElementIndex,
    selectedImageSize,
    selectedItem,
    selectedKey,
    workerID,
  ]);

  const handleCaptionSelectedLayer = useCallback(async () => {
    if (!selectedItem || layerCaptionDisabledReason || selectedLayerIsCaptioning || selectedElementIndex == null || !selectedElement) {
      return;
    }

    const requestCaption = captionText;
    const requestKey = selectedKey;
    const requestElementIndex = selectedElementIndex;
    const requestLayerKey = layerCaptionRequestKey(requestKey, requestElementIndex);
    const requestElement = selectedElement;
    const requestHadBox = Boolean(selectedBox);
    const imageWidth = selectedImageSize?.width || null;
    const imageHeight = selectedImageSize?.height || null;

    setLayerCaptioningForKey(requestLayerKey, true);
    setLayerCaptionMessageForKey(requestLayerKey, '');
    try {
      let response;
      if (selectedItem.kind === 'plain') {
        response = await apiClient.post(
          '/api/datasets/layer-caption',
          {
            imgPath: selectedItem.path,
            caption: requestCaption,
            elementIndex: requestElementIndex,
            provider: autoBoxProvider,
            model: autoBoxModel,
            remoteWorkerId: remoteOllamaWorkerId,
            imageWidth,
            imageHeight,
            ...projectPayload,
          },
          { timeout: 0 },
        );
      } else {
        if (!encryptedKey) throw new Error('Unlock the encrypted dataset first.');
        if (!encryptedProviderConfirmations[autoBoxProvider]) {
          const confirmed = window.confirm(
            `Caption Layer will send this decrypted image to ${autoBoxProviderLabel} to caption the selected layer. Continue?`,
          );
          if (!confirmed) {
            setLayerCaptionMessageForKey(requestLayerKey, 'Caption Layer canceled.');
            return;
          }
          setEncryptedProviderConfirmations(previous => ({ ...previous, [autoBoxProvider]: true }));
        }

        const formData = await createEncryptedImageFormData({
          datasetName,
          workerID,
          projectID,
          encryptedKey,
          item: selectedItem.item,
        });
        formData.append('caption', requestCaption);
        formData.append('elementIndex', String(requestElementIndex));
        formData.append('provider', autoBoxProvider);
        formData.append('model', autoBoxModel);
        formData.append('remoteWorkerId', remoteOllamaWorkerId);
        appendImageSizeFields(formData, imageWidth, imageHeight);

        response = await apiClient.post('/api/datasets/layer-caption', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 0,
        });
      }

      if (selectedKeyRef.current !== requestKey) {
        return;
      }
      const latestParsed = parseIdeogramCaption(latestCaptionRef.current, selectedImageSize ?? undefined);
      const currentElement =
        latestParsed.kind === 'ideogram' ? latestParsed.elements[requestElementIndex] ?? null : null;
      if (!pendingCaptionLayerStillMatches(currentElement, requestElement)) {
        setLayerCaptionMessageForKey(requestLayerKey, 'Layer changed while Caption Layer was running. Rerun it.');
        return;
      }

      const desc = typeof response.data?.desc === 'string' ? response.data.desc.trim() : '';
      const text = typeof response.data?.text === 'string' ? response.data.text.trim() : '';
      const colorPalette = Array.isArray(response.data?.color_palette)
        ? response.data.color_palette.flatMap((color: unknown) => {
            const normalized = normalizeHexColor(color);
            return normalized ? [normalized] : [];
          })
        : [];
      const currentHasBox = Boolean(arrayToBox(currentElement?.bbox));
      const generatedBox = currentHasBox ? null : arrayToBox(response.data?.bbox);
      if (!desc) throw new Error(`${autoBoxProviderLabel} did not return a usable layer caption.`);
      if (!currentHasBox && !generatedBox) throw new Error(`${autoBoxProviderLabel} did not return a usable layer box.`);

      const updated = mutateLatestCaption(data => {
        updateIdeogramElementField(data, requestElementIndex, 'desc', desc);
        if (requestElement?.type === 'text' && text && !String(currentElement?.text || '').trim()) {
          updateIdeogramElementField(data, requestElementIndex, 'text', text);
        }
        if (!currentHasBox && generatedBox) {
          updateIdeogramElementBox(data, requestElementIndex, generatedBox);
        }
        if (colorPalette.length > 0) {
          updateIdeogramElementPalette(data, requestElementIndex, colorPalette);
        }
      });
      setLayerCaptionMessageForKey(
        requestLayerKey,
        updated ? (requestHadBox || currentHasBox ? 'Layer caption updated.' : 'Layer caption and box updated.') : '',
      );
    } catch (error) {
      console.error('Caption Layer failed:', error);
      setLayerCaptionMessageForKey(requestLayerKey, responseErrorMessage(error, 'Caption Layer failed. Please try again.'));
    } finally {
      setLayerCaptioningForKey(requestLayerKey, false);
    }
  }, [
    autoBoxModel,
    autoBoxProvider,
    autoBoxProviderLabel,
    captionText,
    datasetName,
    encryptedKey,
    encryptedProviderConfirmations,
    layerCaptionDisabledReason,
    mutateLatestCaption,
    projectID,
    projectPayload,
    remoteOllamaWorkerId,
    selectedElement,
    selectedElementIndex,
    selectedImageSize,
    selectedBox,
    selectedItem,
    selectedKey,
    selectedLayerIsCaptioning,
    setLayerCaptioningForKey,
    setLayerCaptionMessageForKey,
    workerID,
  ]);

  const undo = useCallback(() => {
    setUndoStack(previous => {
      const nextCaption = previous[previous.length - 1];
      if (!nextCaption) return previous;
      setRedoStack(redo => [captionText, ...redo].slice(0, MAX_HISTORY));
      latestCaptionRef.current = nextCaption;
      setCaptionText(nextCaption);
      return previous.slice(0, -1);
    });
  }, [captionText]);

  const redo = useCallback(() => {
    setRedoStack(previous => {
      const nextCaption = previous[0];
      if (!nextCaption) return previous;
      setUndoStack(undoStackValue => [...undoStackValue.slice(Math.max(0, undoStackValue.length - MAX_HISTORY + 1)), captionText]);
      latestCaptionRef.current = nextCaption;
      setCaptionText(nextCaption);
      return previous.slice(1);
    });
  }, [captionText]);

  const handleCreateBox = useCallback(
    (type: IdeogramElementType, box: NormalizedBox) => {
      let createdIndex: number | null = null;
      mutateCaption(data => {
        createdIndex = addIdeogramElement(data, type, box);
      }, createdIndex);
      if (createdIndex != null) {
        setSelectedElementIndex(createdIndex);
        setActiveTool('select');
      }
    },
    [mutateCaption],
  );

  const handleChangeBox = useCallback(
    (elementIndex: number, box: NormalizedBox) => {
      mutateCaption(data => updateIdeogramElementBox(data, elementIndex, box));
    },
    [mutateCaption],
  );

  const handleToggleLayerHidden = useCallback((elementIndex: number) => {
    setHiddenLayerIndexes(previous => {
      const next = new Set(previous);
      if (next.has(elementIndex)) {
        next.delete(elementIndex);
      } else {
        next.add(elementIndex);
      }
      return next;
    });
  }, []);

  const handleToggleLayerLocked = useCallback((elementIndex: number) => {
    setLockedLayerIndexes(previous => {
      const next = new Set(previous);
      if (next.has(elementIndex)) {
        next.delete(elementIndex);
      } else {
        next.add(elementIndex);
      }
      return next;
    });
  }, []);

  const cycleOverlapSelection = useCallback(
    (direction: 1 | -1) => {
      if (overlapElementStack.length === 0) return;
      setSelectedElementIndex(current => {
        const currentIndex = current == null ? -1 : overlapElementStack.indexOf(current);
        const nextIndex =
          currentIndex < 0
            ? direction > 0
              ? 0
              : overlapElementStack.length - 1
            : (currentIndex + direction + overlapElementStack.length) % overlapElementStack.length;
        return overlapElementStack[nextIndex] ?? current;
      });
    },
    [overlapElementStack],
  );

  const handleDuplicateElement = useCallback(
    (elementIndex: number) => {
      const elementCount = captionParse.kind === 'ideogram' ? captionParse.elements.length : 0;
      if (elementIndex < 0 || elementIndex >= elementCount) return;
      const duplicateIndex = elementIndex + 1;
      mutateCaption(data => {
        duplicateIdeogramElement(data, elementIndex);
      }, duplicateIndex);
      setHiddenLayerIndexes(previous => reindexLayerIndexSetAfterInsert(previous, duplicateIndex));
      setLockedLayerIndexes(previous => reindexLayerIndexSetAfterInsert(previous, duplicateIndex));
      setOverlapElementStack([]);
    },
    [captionParse, mutateCaption],
  );

  const handleDeleteElement = useCallback(
    (elementIndex: number) => {
      const elementCount = captionParse.kind === 'ideogram' ? captionParse.elements.length : 0;
      if (elementIndex < 0 || elementIndex >= elementCount) return;
      const nextSelection =
        selectedElementIndex == null
          ? null
          : selectedElementIndex === elementIndex
            ? elementCount > 1
              ? Math.min(elementIndex, elementCount - 2)
              : null
            : selectedElementIndex > elementIndex
              ? selectedElementIndex - 1
              : selectedElementIndex;
      mutateCaption(data => deleteIdeogramElement(data, elementIndex), nextSelection);
      setHiddenLayerIndexes(previous => reindexLayerIndexSetAfterDelete(previous, elementIndex));
      setLockedLayerIndexes(previous => reindexLayerIndexSetAfterDelete(previous, elementIndex));
      setOverlapElementStack(previous =>
        previous.flatMap(index => (index === elementIndex ? [] : [index > elementIndex ? index - 1 : index])),
      );
    },
    [captionParse, mutateCaption, selectedElementIndex],
  );

  const handleDeleteSelectedElement = useCallback(() => {
    if (selectedElementIndex == null) return;
    handleDeleteElement(selectedElementIndex);
  }, [handleDeleteElement, selectedElementIndex]);

  const handleSelectedFieldChange = useCallback(
    (field: 'desc' | 'text', value: string) => {
      if (selectedElementIndex == null) return;
      mutateCaption(data => updateIdeogramElementField(data, selectedElementIndex, field, value));
    },
    [mutateCaption, selectedElementIndex],
  );

  const handleSelectedTypeChange = useCallback(
    (type: IdeogramElementType) => {
      if (selectedElementIndex == null) return;
      mutateCaption(data => updateIdeogramElementType(data, selectedElementIndex, type));
    },
    [mutateCaption, selectedElementIndex],
  );

  const handleSelectedPaletteChange = useCallback(
    (colors: string[]) => {
      if (selectedElementIndex == null) return;
      mutateCaption(data => updateIdeogramElementPalette(data, selectedElementIndex, colors));
    },
    [mutateCaption, selectedElementIndex],
  );

  const handleSelectedPaletteColorChange = useCallback(
    (index: number, color: string) => {
      const normalized = normalizeHexColor(color);
      if (!normalized) return;
      const nextPalette = [...selectedPalette];
      nextPalette[index] = normalized;
      handleSelectedPaletteChange(nextPalette);
    },
    [handleSelectedPaletteChange, selectedPalette],
  );

  const handleStartPaletteSample = useCallback((index: number) => {
    setActivePaletteSamplerIndex(index);
    setActiveTool('select');
  }, []);

  const handleCancelPaletteSample = useCallback(() => {
    setActivePaletteSamplerIndex(null);
  }, []);

  const handleSamplePaletteColor = useCallback(
    (color: string) => {
      if (activePaletteSamplerIndex == null) return;
      handleSelectedPaletteColorChange(activePaletteSamplerIndex, color);
      setActivePaletteSamplerIndex(null);
    },
    [activePaletteSamplerIndex, handleSelectedPaletteColorChange],
  );

  const handleCaptionDescriptionChange = useCallback(
    (value: string) => {
      if (!isIdeogram) {
        latestCaptionRef.current = value;
        setCaptionText(value);
        return;
      }
      mutateCaption(data => updateIdeogramHighLevelDescription(data, value));
    },
    [isIdeogram, mutateCaption],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      const isTyping = tag === 'TEXTAREA' || tag === 'INPUT' || (target?.isContentEditable ?? false);
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
        event.preventDefault();
        void saveCaption();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        void saveCaption();
        return;
      }
      if (isTyping) return;
      const isArrowKey = event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowUp' || event.key === 'ArrowDown';
      if (isArrowKey) {
        // Nudge the selected (unlocked) box with arrow keys; otherwise navigate images.
        if (selectedBox && selectedElementIndex != null && !lockedLayerIndexes.has(selectedElementIndex)) {
          event.preventDefault();
          const step = event.shiftKey ? 20 : 5;
          const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0;
          const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0;
          handleChangeBox(selectedElementIndex, resizeOrMoveBox(selectedBox, dx, dy, 'move', MIN_BOX_SPAN));
          return;
        }
        if (event.key === 'ArrowLeft') selectIndex(selectedIndex - 1);
        if (event.key === 'ArrowRight') selectIndex(selectedIndex + 1);
        return;
      }
      if (event.key === 'Escape') {
        if (activePaletteSamplerIndex != null) {
          handleCancelPaletteSample();
        } else {
          setSelectedElementIndex(null);
        }
      }
      if (event.key === '[') cycleOverlapSelection(-1);
      if (event.key === ']') cycleOverlapSelection(1);
      if (event.key === 'Delete' || event.key === 'Backspace') handleDeleteSelectedElement();
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'd') {
        if (selectedElementIndex != null) {
          event.preventDefault();
          handleDuplicateElement(selectedElementIndex);
        }
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (event.shiftKey) {
          redo();
        } else {
          undo();
        }
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
        event.preventDefault();
        redo();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    activePaletteSamplerIndex,
    cycleOverlapSelection,
    handleCancelPaletteSample,
    handleChangeBox,
    handleDeleteSelectedElement,
    handleDuplicateElement,
    lockedLayerIndexes,
    redo,
    saveCaption,
    selectIndex,
    selectedBox,
    selectedElementIndex,
    selectedIndex,
    undo,
  ]);

  const highLevelDescription =
    isIdeogram && typeof captionParse.data.high_level_description === 'string'
      ? captionParse.data.high_level_description
      : captionText;
  const selectedLayerColor =
    selectedBox?.color || (selectedElementIndex != null ? BOX_COLORS[selectedElementIndex % BOX_COLORS.length] : BOX_COLORS[0]);
  const layerCaptionStatus =
    (selectedLayerIsCaptioning ? 'Captioning layer...' : selectedLayerCaptionMessage) ||
    (selectedElement && layerCaptionDisabledReason ? layerCaptionDisabledReason : '');
  const selectedRect = selectedBox ? boxToRect(selectedBox) : null;
  const handleCaptionTextChange = useCallback((value: string) => {
    latestCaptionRef.current = value;
    setCaptionText(value);
  }, []);

  if (items.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-gray-400">
        <div className="border border-dashed border-gray-700 bg-gray-900/60 px-6 py-5 text-sm">No editable media or text files found.</div>
      </div>
    );
  }

  if (!selectedItem) return null;

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-[#02060a] text-gray-100">
      <StudioToolbar
        selectedIndex={selectedIndex}
        itemCount={items.length}
        isSaving={isSaving}
        isDirty={isDirty}
        zoom={zoom}
        isDeletingCurrent={isDeletingImages}
        canDeleteCurrent={Boolean(onDeleteImages && selectedItem)}
        onPrevious={() => selectIndex(selectedIndex - 1)}
        onNext={() => selectIndex(selectedIndex + 1)}
        onCycleZoom={() => setZoom(value => (value >= 2 ? 1 : Number((value + 0.25).toFixed(2))))}
        onPan={() => setActiveTool('pan')}
        onFit={() => setZoom(1)}
        onDeleteCurrent={handleDeleteCurrentImage}
      />

      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden md:flex-row">
        <ToolRail
          activeTool={activeTool}
          canAnnotate={canAnnotate}
          hasSelection={selectedElementIndex != null}
          canUndo={undoStack.length > 0}
          canRedo={redoStack.length > 0}
          canShowJson={!isPlainTextItem}
          onToolChange={setActiveTool}
          onDelete={handleDeleteSelectedElement}
          onUndo={undo}
          onRedo={redo}
          onShowJson={() => {
            if (!isPlainTextItem) setCaptionTab('json');
          }}
        />

        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden xl:flex-row">
          <main className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-[#03070b]">
            <div className="relative flex min-h-[260px] min-w-0 flex-1 items-stretch justify-stretch overflow-hidden">
              <div className="absolute left-3 top-3 z-10 max-w-[calc(50%-1rem)] truncate rounded-md border border-gray-800 bg-gray-950/80 px-2 py-1 text-xs text-gray-300 backdrop-blur">
                {selectedName}
              </div>
              {deleteMessage && (
                <div className="absolute right-3 top-3 z-10 max-w-[calc(50%-1rem)] truncate rounded-md border border-gray-800 bg-gray-950/80 px-2 py-1 text-xs text-gray-300 backdrop-blur">
                  {deleteMessage}
                </div>
              )}
              <StudioMedia
                item={selectedItem}
                datasetName={datasetName}
                workerID={workerID}
                projectID={projectID}
                cryptoKey={encryptedKey}
                zoom={zoom}
                onNaturalSizeChange={setSelectedImageSize}
                isSamplingColor={activePaletteSamplerIndex != null}
                onSampleColor={handleSamplePaletteColor}
                onCancelColorSample={handleCancelPaletteSample}
              >
                {canAnnotate && (
                  <AnnotationLayer
                    boxes={boxes}
                    activeTool={activeTool}
                    selectedElementIndex={selectedElementIndex}
                    hiddenElementIndexes={hiddenLayerIndexes}
                    lockedElementIndexes={lockedLayerIndexes}
                    imageSize={selectedImageSize}
                    onSelect={setSelectedElementIndex}
                    onCreate={handleCreateBox}
                    onChangeBox={handleChangeBox}
                    onOverlapStackChange={setOverlapElementStack}
                  />
                )}
              </StudioMedia>
            </div>
            <ImageNavigator
              items={items}
              selectedIndex={selectedIndex}
              datasetName={datasetName}
              workerID={workerID}
              projectID={projectID}
              encryptedKey={encryptedKey}
              isAutoCaptioning={isAutoCaptioning}
              captionCache={captionCacheRef.current}
              captionCacheVersion={captionCacheVersion}
              onCaptionCacheChange={bumpCaptionCacheVersion}
              onSelectIndex={selectIndex}
              onBulkCaptionAction={handleBulkCaptionAction}
              onDeleteImages={handleDeleteImages}
            />
          </main>

          <aside className="flex w-full max-h-[34dvh] min-h-[190px] min-w-0 flex-shrink-0 flex-col overflow-hidden border-t border-gray-900 bg-[#080d12] xl:max-h-none xl:min-h-0 xl:w-[410px] xl:flex-none xl:border-l xl:border-t-0">
            <div className="operator-scrollbar-none min-h-0 flex-1 overflow-y-auto p-2 md:p-3">
              {canAnnotate && captionParse.kind === 'ideogram' && (
                <LayersPanel
                  elements={captionParse.elements}
                  boxes={boxes}
                  selectedElementIndex={selectedElementIndex}
                  hiddenElementIndexes={hiddenLayerIndexes}
                  lockedElementIndexes={lockedLayerIndexes}
                  onSelect={setSelectedElementIndex}
                  onToggleHidden={handleToggleLayerHidden}
                  onToggleLocked={handleToggleLayerLocked}
                  onDuplicate={handleDuplicateElement}
                  onDelete={handleDeleteElement}
                />
              )}
              <ObjectDetailsPanel
                canAnnotate={canAnnotate}
                isCaptionLoaded={isCaptionLoaded}
                canConvertDataset={canConvertDataset}
                isPlainTextItem={isPlainTextItem}
                selectedImageSize={selectedImageSize}
                canGenerateAutoBoxes={canGenerateAutoBoxes}
                autoBoxDisabledReason={autoBoxDisabledReason}
                autoBoxProvider={autoBoxProvider}
                autoBoxProviderLabel={autoBoxProviderLabel}
                autoBoxModel={autoBoxModel}
                remoteWorkerId={remoteOllamaWorkerId}
                remoteWorkerOptions={remoteWorkerOptions}
                autoBoxRefine={autoBoxRefine}
                isGeneratingBoxes={isGeneratingBoxes}
                autoBoxMessage={autoBoxMessage}
                selectedElement={selectedElement}
                selectedElementIndex={selectedElementIndex}
                selectedLayerColor={selectedLayerColor}
                selectedRect={selectedRect}
                selectedPalette={selectedPalette}
                activePaletteSamplerIndex={activePaletteSamplerIndex}
                layerCaptionStatus={layerCaptionStatus}
                selectedLayerIsCaptioning={selectedLayerIsCaptioning}
                canCaptionSelectedLayer={canCaptionSelectedLayer}
                layerCaptionDisabledReason={layerCaptionDisabledReason}
                onConvertDatasetToJson={onConvertDatasetToJson}
                onGenerateAutoBoxes={() => void handleGenerateAutoBoxes()}
                onAutoBoxProviderChange={handleAutoBoxProviderChange}
                onAutoBoxModelChange={setAutoBoxModel}
                onRemoteWorkerChange={setRemoteOllamaWorkerId}
                onAutoBoxRefineChange={setAutoBoxRefine}
                onSelectedFieldChange={handleSelectedFieldChange}
                onSelectedTypeChange={handleSelectedTypeChange}
                onChangeBox={handleChangeBox}
                onSelectedPaletteChange={handleSelectedPaletteChange}
                onStartPaletteSample={handleStartPaletteSample}
                onCancelPaletteSample={handleCancelPaletteSample}
                onCaptionSelectedLayer={() => void handleCaptionSelectedLayer()}
              />

              <CaptionEditorPanel
                captionTab={captionTab}
                captionStatus={captionStatus}
                captionText={captionText}
                highLevelDescription={highLevelDescription}
                isIdeogram={isIdeogram}
                isPlainTextItem={isPlainTextItem}
                isAutoCaptioning={isAutoCaptioning}
                isCaptionLoaded={isCaptionLoaded}
                isDirty={isDirty}
                isSaving={isSaving}
                isRecaptioning={selectedRecaptionIsRunning}
                canRecaption={canQueueSelectedRecaption}
                isSelectedRecaptionQueued={selectedRecaptionIsQueued}
                hasActiveRecaptions={hasPendingRecaptions}
                hasQueuedRecaptions={recaptionQueue.length > 0}
                recaptionFeedback={recaptionFeedback}
                onCaptionTabChange={setCaptionTab}
                onCaptionDescriptionChange={handleCaptionDescriptionChange}
                onCaptionTextChange={handleCaptionTextChange}
                onRecaption={handleRecaptionClick}
                onRecaptionSettings={openRecaptionSettings}
                onSave={() => void saveCaption()}
              />
            </div>
          </aside>
        </div>
      </div>
      <Modal
        isOpen={isRecaptionModalOpen}
        onClose={() => {
          if (!isRecaptioning) setIsRecaptionModalOpen(false);
        }}
        title="Recaption Image"
        size="lg"
        closeOnOverlayClick={!isRecaptioning}
      >
        <form
          className="space-y-4 text-gray-200"
          onSubmit={event => {
            event.preventDefault();
            queueSelectedRecaption();
          }}
        >
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <label className="block text-sm">
              <span className="mb-1 block text-xs font-medium text-gray-400">Provider</span>
              <select
                value={recaptionProvider}
                onChange={event => handleRecaptionProviderChange(event.target.value)}
                disabled={isRecaptioning}
                className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
              >
                {AUTO_BOX_PROVIDERS.map(provider => (
                  <option key={provider.value} value={provider.value}>
                    {provider.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-xs font-medium text-gray-400">Output</span>
              <select
                value={recaptionOutputFormat}
                onChange={event => handleRecaptionOutputFormatChange(event.target.value as RecaptionOutputFormat)}
                disabled={isRecaptioning}
                className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
              >
                <option value="text">Text caption</option>
                <option value="ideogram_json">Ideogram JSON</option>
              </select>
            </label>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <label className="block text-sm">
              <span className="mb-1 block text-xs font-medium text-gray-400">Model</span>
              {recaptionProvider === 'remote_ollama' ? (
                <select
                  value={recaptionModel}
                  onChange={event => setRecaptionModel(event.target.value)}
                  disabled={isRecaptioning || recaptionRemoteModelStatus === 'loading' || recaptionRemoteModelOptions.length === 0}
                  className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {recaptionRemoteModelStatus === 'loading' && <option value="">Loading models...</option>}
                  {recaptionRemoteModelStatus !== 'loading' && recaptionRemoteModelOptions.length === 0 && (
                    <option value="">No server models loaded</option>
                  )}
                  {recaptionRemoteModelOptions.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              ) : (
                <>
                  <input
                    list="recaption-model-options"
                    value={recaptionModel}
                    onChange={event => setRecaptionModel(event.target.value)}
                    disabled={isRecaptioning}
                    className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
                  />
                  <datalist id="recaption-model-options">
                    {recaptionModelOptions.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </datalist>
                </>
              )}
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-xs font-medium text-gray-400">Max tokens</span>
              <input
                type="number"
                min={1}
                value={recaptionMaxNewTokens}
                onChange={event => setRecaptionMaxNewTokens(Math.max(1, Number(event.target.value) || 1))}
                disabled={isRecaptioning}
                className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
              />
            </label>
          </div>

          {recaptionProvider === 'remote_ollama' && (
            <div className="space-y-2">
              <div className="grid grid-cols-[1fr_auto] gap-2">
                <label className="block text-sm">
                  <span className="mb-1 block text-xs font-medium text-gray-400">Remote Ollama</span>
                  <select
                    value={recaptionRemoteWorkerId}
                    onChange={event => setRecaptionRemoteWorkerId(event.target.value)}
                    disabled={isRecaptioning || remoteWorkerOptions.length === 0}
                    className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
                  >
                    {remoteWorkerOptions.length === 0 && <option value="">No enabled workers</option>}
                    {remoteWorkerOptions.map(worker => (
                      <option key={worker.value} value={worker.value}>
                        {worker.label}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  disabled={isRecaptioning || !recaptionRemoteWorkerId || recaptionRemoteModelStatus === 'loading'}
                  onClick={() => void loadRecaptionRemoteModels()}
                  className="mt-5 inline-flex h-10 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-200 hover:bg-gray-900 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {recaptionRemoteModelStatus === 'loading' ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RotateCcw className="h-4 w-4" />
                  )}
                  Models
                </button>
              </div>
              {recaptionRemoteModelStatus === 'success' && (
                <div className="text-xs text-gray-500">
                  {recaptionRemoteModelOptions.length.toLocaleString()} model
                  {recaptionRemoteModelOptions.length === 1 ? '' : 's'} loaded.
                </div>
              )}
              {recaptionRemoteModelStatus === 'error' && (
                <div className="text-xs text-red-400">{recaptionRemoteModelError}</div>
              )}
            </div>
          )}

          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-gray-400">Prompt</span>
            <textarea
              value={recaptionPrompt}
              onChange={event => setRecaptionPrompt(event.target.value)}
              disabled={isRecaptioning}
              rows={6}
              className="w-full resize-none rounded-md border border-gray-800 bg-gray-950 p-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
            />
          </label>

          <label className="block text-sm">
            <span className="mb-1 flex items-center justify-between gap-3 text-xs font-medium text-gray-400">
              <span>System prompt</span>
              {recaptionRootPrompt && (
                <button
                  type="button"
                  disabled={isRecaptioning}
                  onClick={useRecaptionRootPrompt}
                  className="text-cyan-300 hover:text-cyan-200 disabled:opacity-45"
                >
                  Use ROOT_CAPTION.txt
                </button>
              )}
            </span>
            <textarea
              value={recaptionSystemPrompt}
              onChange={event => {
                recaptionSystemPromptTouchedRef.current = true;
                setRecaptionSystemPrompt(event.target.value);
              }}
              disabled={isRecaptioning}
              rows={3}
              className="w-full resize-none rounded-md border border-gray-800 bg-gray-950 p-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
            />
          </label>
          {recaptionRootPromptStatus === 'loading' && <div className="text-xs text-gray-500">Loading ROOT_CAPTION.txt</div>}
          {recaptionRootPromptStatus === 'error' && <div className="text-xs text-red-400">Could not load ROOT_CAPTION.txt.</div>}

          {recaptionMessage && <div className="text-sm text-gray-400">{recaptionMessage}</div>}

          <div className="flex justify-end gap-3">
            <button
              type="button"
              disabled={isRecaptioning}
              onClick={() => setIsRecaptionModalOpen(false)}
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!canQueueSelectedRecaption}
              className="rounded-md bg-cyan-600 px-4 py-2 font-medium text-white hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {selectedRecaptionIsRunning
                ? 'Recaptioning'
                : selectedRecaptionIsQueued
                  ? 'Queued'
                  : hasPendingRecaptions
                    ? 'Add to Queue'
                    : 'Recaption'}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
