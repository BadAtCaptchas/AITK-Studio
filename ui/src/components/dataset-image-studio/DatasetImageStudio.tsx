'use client';

import classNames from 'classnames';
import {
  Braces,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  GripHorizontal,
  Layers3,
  Loader2,
  Maximize2,
  Minus,
  Palette,
  Pipette,
  Plus,
  Save,
  ScanLine,
  Settings2,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { createPortal } from 'react-dom';
import { apiClient } from '@/utils/api';
import useRemoteOllamaWorkers from '@/hooks/useRemoteOllamaWorkers';
import { defaultImageCaptionPrompt } from '@/helpers/captionOptions';
import {
  captionObjectPath,
  decryptEncryptedObjectBlob,
  encryptCaptionObject,
  randomId,
} from '@/utils/encryptedDatasets';
import { buildEncryptedObjectRequestBody } from '@/utils/encryptedObjectMediaCache';
import {
  addIdeogramLayer,
  addIdeogramElement,
  appendGeneratedIdeogramElements,
  applyGeneratedBoxPatches,
  arrayToBox,
  boxToRect,
  cloneIdeogramData,
  deleteIdeogramElement,
  duplicateIdeogramElement,
  moveIdeogramElement,
  mergeIdeogramPaletteColors,
  normalizeGeneratedElementBoxes,
  normalizeGeneratedBoxPatches,
  normalizeIdeogramColorPalette,
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
  updateIdeogramImagePalette,
} from '@/utils/ideogramCaption';
import {
  extractPaletteFromImage,
  findNearDuplicatePalettePairs,
  PALETTE_DELTA_E_THRESHOLDS,
  paletteColorDistance,
  type ImagePaletteExtraction,
} from '@/utils/imagePalette';
import { analyzeCaptionJsonDraft, formatCaptionJsonDraft } from '@/utils/captionJsonWorkspace';
import { resizeOrMoveBox } from '@/utils/annotationGeometry';
import { AnnotationLayer } from './AnnotationLayer';
import { AdvancedFilmstrip } from './AdvancedFilmstrip';
import { BOX_COLORS, DEFAULT_OPENROUTER_BOX_MODEL, MAX_HISTORY, MIN_BOX_SPAN } from './constants';
import { ImageNavigator } from './ImageNavigator';
import { CaptionEditorPanel, ObjectDetailsPanel } from './InspectorPanels';
import { LayersPanel } from './LayersPanel';
import { RecaptionSettingsModal } from './RecaptionSettingsModal';
import { EncryptedThumb, PlainThumb, StudioMedia } from './StudioMedia';
import { StudioToolbar } from './StudioToolbar';
import { StandaloneBoxDetailsPanel } from './StandaloneBoxDetailsPanel';
import { StandaloneBoxesPanel } from './StandaloneBoxesPanel';
import { StandaloneBoxesToolRail } from './StandaloneBoxesToolRail';
import { StandaloneLayerDetailsPanel } from './StandaloneLayerDetailsPanel';
import {
  StandaloneImagePalettePanel,
  type PaletteLayer,
  type PaletteReviewIssue,
  type PaletteSwatch,
} from './StandaloneImagePalettePanel';
import { StandaloneJsonWorkspace } from './StandaloneJsonWorkspace';
import { ToolRail } from './ToolRail';
import { appendImageSizeFields, createEncryptedImageFormData } from './openRouterMedia';
import {
  appendRecaptionFields,
  maxNewTokensForRecaptionOutputFormat,
  modelOptionsForRecaptionProvider,
  nextAutoBoxModelForProvider,
  nextRecaptionModelForProvider,
  normalizeRecaptionProvider,
  normalizeStudioBoxProvider,
  normalizedRecaptionMaxNewTokens,
  ollamaModelOptions,
  persistedRecaptionQueueEntry,
  promptForRecaptionOutputFormat,
  providerLabel,
  purgeLegacyPersistedRecaptionQueues,
  readPersistedRecaptionQueue,
  readRecaptionSettingsPreset,
  recaptionQueueStorageKey,
  recaptionSettingsStorageKey,
  type PersistedRecaptionQueue,
  type RecaptionLoadStatus,
  type RecaptionModelOption,
  type RecaptionOutputFormat,
  type RecaptionProvider,
  type RecaptionQueueEntry,
  type RecaptionSettingsPreset,
  type StudioBoxProvider,
} from './recaption';
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
  remapLayerIndexArrayForMove,
  remapLayerIndexForMove,
  remapLayerIndexSetForMove,
  responseErrorMessage,
  scheduleIdleTask,
  statusForCaption,
} from './utils';

type RefusalCaptionAuditResponse = {
  refusals?: Record<string, unknown>;
};

type CaptionHistoryEntry = {
  caption: string;
  selectedElementIndex: number | null;
  hiddenLayerIndexes: number[];
  lockedLayerIndexes: number[];
  overlapElementStack: number[];
};

const ADVANCED_WORKSPACE_TOOLS = [
  { value: 'boxes', label: 'Boxes', icon: ScanLine },
  { value: 'layers', label: 'Layers', icon: Layers3 },
  { value: 'palettes', label: 'Palettes', icon: Palette },
  { value: 'json', label: 'JSON', icon: Braces },
] as const;

function paletteLayerLabel(element: unknown, index: number) {
  if (typeof element !== 'object' || element === null) return `Layer ${index + 1}`;
  const record = element as Record<string, unknown>;
  const source = record.type === 'text' ? record.text || record.desc : record.desc;
  if (typeof source !== 'string' || !source.trim()) return `${record.type === 'text' ? 'Text' : 'Layer'} ${index + 1}`;
  const firstClause = source.trim().split(/[,.]/, 1)[0].trim();
  if (firstClause.length <= 34) return firstClause;
  return `${firstClause.slice(0, 31).trimEnd()}…`;
}

function formatByteSize(bytes?: number | null) {
  if (!bytes || bytes < 1) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatStudioDate(value?: string | null) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', year: 'numeric' }).format(date);
}

export default function DatasetImageStudio({
  datasetName,
  workerID,
  projectID,
  datasetPath,
  items,
  isAutoCaptioning,
  liveCaptionRefresh = false,
  encryptedKey,
  rootCaption,
  captionAction,
  watcherAction,
  presentation = 'studio',
  onUnsavedChange,
  onRefresh,
  onAddImages,
  onConvertDatasetToJson,
  onDeleteImages,
  onPlainItemCaptioned,
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
  const [undoStack, setUndoStack] = useState<CaptionHistoryEntry[]>([]);
  const [redoStack, setRedoStack] = useState<CaptionHistoryEntry[]>([]);
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
  const [activeRecaptionEntry, setActiveRecaptionEntry] = useState<RecaptionQueueEntry | null>(null);
  const [activeRecaptionLabel, setActiveRecaptionLabel] = useState('');
  const [activeRecaptionKey, setActiveRecaptionKey] = useState('');
  const [recaptionRootPrompt, setRecaptionRootPrompt] = useState('');
  const [recaptionRootPromptStatus, setRecaptionRootPromptStatus] = useState<RecaptionLoadStatus>('idle');
  const [recaptionRemoteModelOptions, setRecaptionRemoteModelOptions] = useState<RecaptionModelOption[]>([]);
  const [recaptionRemoteModelStatus, setRecaptionRemoteModelStatus] = useState<RecaptionLoadStatus>('idle');
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
  const [activeImagePaletteSamplerIndex, setActiveImagePaletteSamplerIndex] = useState<number | null>(null);
  const [paletteImageElement, setPaletteImageElement] = useState<HTMLImageElement | null>(null);
  const [paletteExtraction, setPaletteExtraction] = useState<ImagePaletteExtraction | null>(null);
  const [appliedPaletteExtractionKey, setAppliedPaletteExtractionKey] = useState<string | null>(null);
  const [isExtractingPalette, setIsExtractingPalette] = useState(false);
  const [paletteExtractionMessage, setPaletteExtractionMessage] = useState('');
  const [selectedImagePaletteColor, setSelectedImagePaletteColor] = useState<string | null>(null);
  const [encryptedCaptionPaths, setEncryptedCaptionPaths] = useState<Record<string, string>>({});
  const [captionCacheVersion, setCaptionCacheVersion] = useState(0);
  const [approvedKeys, setApprovedKeys] = useState<Set<string>>(() => new Set());
  const [workbenchHeight, setWorkbenchHeight] = useState(254);
  const [workbenchCollapsed, setWorkbenchCollapsed] = useState(false);
  const [toolsMenuOpen, setToolsMenuOpen] = useState(false);
  const [toolsMenuPosition, setToolsMenuPosition] = useState({ left: 0, top: 0 });
  const toolsButtonRef = useRef<HTMLButtonElement | null>(null);
  const toolsMenuRef = useRef<HTMLDivElement | null>(null);
  const [autoBoxesPanelOpen, setAutoBoxesPanelOpen] = useState(false);
  const [advancedMode, setAdvancedMode] = useState<'boxes' | 'layers' | 'palettes' | 'json' | null>(null);
  const [jsonDraft, setJsonDraft] = useState('');
  const [jsonDraftBaseCaption, setJsonDraftBaseCaption] = useState('');
  const [jsonDraftKey, setJsonDraftKey] = useState('');
  const [jsonSelectedSectionId, setJsonSelectedSectionId] = useState('elements');
  const [jsonSelectedPath, setJsonSelectedPath] = useState('compositional_deconstruction.elements');
  const captionCacheRef = useRef(new Map<string, CaptionCacheEntry>());
  const saveCaptionRef = useRef<() => Promise<boolean>>(async () => true);
  const isSavingRef = useRef(false);
  const autoSelectKeyRef = useRef('');
  const latestCaptionRef = useRef('');
  const savedCaptionRef = useRef('');
  const isDirtyRef = useRef(false);
  const selectedKeyRef = useRef('');
  const isRecaptioningRef = useRef(false);
  const recaptionQueueRef = useRef<RecaptionQueueEntry[]>([]);
  const hydratedRecaptionQueueKeyRef = useRef('');
  const recaptionModelRef = useRef(DEFAULT_OPENROUTER_BOX_MODEL);
  const recaptionSystemPromptRef = useRef('');
  const recaptionSystemPromptTouchedRef = useRef(false);

  const captureHistoryEntry = useCallback(
    (caption: string): CaptionHistoryEntry => ({
      caption,
      selectedElementIndex,
      hiddenLayerIndexes: [...hiddenLayerIndexes],
      lockedLayerIndexes: [...lockedLayerIndexes],
      overlapElementStack: [...overlapElementStack],
    }),
    [hiddenLayerIndexes, lockedLayerIndexes, overlapElementStack, selectedElementIndex],
  );

  const restoreHistoryEntry = useCallback((entry: CaptionHistoryEntry) => {
    latestCaptionRef.current = entry.caption;
    setCaptionText(entry.caption);
    setJsonDraft(entry.caption);
    setJsonDraftBaseCaption(entry.caption);
    setJsonDraftKey(selectedKeyRef.current);
    setSelectedElementIndex(entry.selectedElementIndex);
    setHiddenLayerIndexes(new Set(entry.hiddenLayerIndexes));
    setLockedLayerIndexes(new Set(entry.lockedLayerIndexes));
    setOverlapElementStack(entry.overlapElementStack);
    setActivePaletteSamplerIndex(null);
  }, []);

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
  const hasEncryptedItems = useMemo(() => items.some(item => item.kind === 'encrypted'), [items]);
  const canPersistRecaptionState = !hasEncryptedItems;
  const plainAuditItemPaths = useMemo(() => items.flatMap(item => (item.kind === 'plain' ? [item.path] : [])), [items]);
  const plainAuditKey = useMemo(() => plainAuditItemPaths.join('\n'), [plainAuditItemPaths]);
  const remoteWorkerOptions = useMemo(
    () => workers.filter(worker => worker.enabled).map(worker => ({ value: worker.id, label: worker.name })),
    [workers],
  );
  const remoteWorkerOptionValues = useMemo(
    () => new Set(remoteWorkerOptions.map(option => option.value)),
    [remoteWorkerOptions],
  );
  const recaptionStorageKey = useMemo(
    () => recaptionSettingsStorageKey({ datasetName, projectID, datasetPath, workerID }),
    [datasetName, datasetPath, projectID, workerID],
  );
  const recaptionQueueStorageKeyValue = useMemo(
    () => recaptionQueueStorageKey({ datasetName, projectID, datasetPath, workerID }),
    [datasetName, datasetPath, projectID, workerID],
  );
  const approvalStorageKey = useMemo(
    () => `caption-studio:approvals:${projectID || 'global'}:${workerID}:${datasetName}`,
    [datasetName, projectID, workerID],
  );
  const autoBoxProviderLabel = providerLabel(autoBoxProvider);
  const recaptionModelOptions = useMemo(() => {
    return modelOptionsForRecaptionProvider(recaptionProvider, recaptionRemoteModelOptions);
  }, [recaptionProvider, recaptionRemoteModelOptions]);
  const captionParse = useMemo(
    () => parseIdeogramCaption(captionText, selectedImageSize ?? undefined),
    [captionText, selectedImageSize],
  );
  const isIdeogram = captionParse.kind === 'ideogram';
  const imagePaletteColors = useMemo(() => {
    if (captionParse.kind !== 'ideogram') return [];
    const style = captionParse.data.style_description;
    if (typeof style !== 'object' || style === null || Array.isArray(style)) return [];
    return normalizeIdeogramColorPalette((style as Record<string, unknown>).color_palette, 16);
  }, [captionParse]);
  const paletteLayers = useMemo<PaletteLayer[]>(() => {
    if (captionParse.kind !== 'ideogram') return [];
    return captionParse.elements.map((element, elementIndex) => {
      const record = typeof element === 'object' && element !== null ? (element as Record<string, unknown>) : {};
      return {
        elementIndex,
        label: paletteLayerLabel(element, elementIndex),
        colorPalette: normalizeIdeogramColorPalette(record.color_palette),
      };
    });
  }, [captionParse]);
  const boxes = isIdeogram ? captionParse.boxes : [];
  const selectedElement =
    isIdeogram && selectedElementIndex != null ? (captionParse.elements[selectedElementIndex] ?? null) : null;
  const selectedBox = boxes.find(box => box.elementIndex === selectedElementIndex) || null;
  const selectedLayerCaptionKey =
    selectedKey && selectedElementIndex != null ? layerCaptionRequestKey(selectedKey, selectedElementIndex) : '';
  const selectedLayerIsCaptioning = Boolean(
    selectedLayerCaptionKey && captioningLayerKeys.has(selectedLayerCaptionKey),
  );
  const hasCurrentImageCaptioningLayer = Boolean(
    selectedKey &&
      Array.from(captioningLayerKeys).some(requestKey => isLayerCaptionRequestForItem(requestKey, selectedKey)),
  );
  const selectedLayerCaptionMessage = selectedLayerCaptionKey
    ? layerCaptionMessages[selectedLayerCaptionKey] || ''
    : '';
  const selectedPalette = Array.isArray(selectedElement?.color_palette) ? selectedElement.color_palette : [];
  const isDirty = captionText.trim() !== savedCaption.trim();
  const jsonDraftDirty = advancedMode === 'json' && jsonDraftKey === selectedKey && jsonDraft !== jsonDraftBaseCaption;
  const hasUnsavedChanges = isDirty || jsonDraftDirty;
  const jsonDraftAnalysis = useMemo(() => analyzeCaptionJsonDraft(jsonDraft), [jsonDraft]);
  const appliedJsonAnalysis = useMemo(() => analyzeCaptionJsonDraft(captionText), [captionText]);
  const jsonNavigatorAnalysis = jsonDraftAnalysis.validity === 'valid' ? jsonDraftAnalysis : appliedJsonAnalysis;
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
  const canGenerateAutoBoxes =
    !autoBoxDisabledReason && !isGeneratingBoxes && !hasCurrentImageCaptioningLayer && !isAutoCaptioning;
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
  const layerStructureDisabledReason = !isCaptionLoaded
    ? 'Load the caption first.'
    : !isIdeogram
      ? 'Layers require an Ideogram JSON caption.'
      : selectedKind !== 'image'
        ? 'Layers are available for images only.'
        : isSaving
          ? 'Wait for the current save to finish.'
          : isAutoCaptioning
            ? 'Dataset auto-captioning is editing this caption.'
            : hasCurrentImageCaptioningLayer
              ? 'Wait for the active layer caption request to finish.'
              : isGeneratingBoxes
                ? 'Wait for Auto Boxes to finish.'
                : selectedRecaptionIsRunning || selectedRecaptionIsQueued
                  ? "Wait for this image's recaption request to finish."
                  : '';
  const jsonWorkspaceDisabledReason = !isCaptionLoaded
    ? 'Load the caption before editing JSON.'
    : isPlainTextItem
      ? 'This asset uses a plain-text caption.'
      : isAutoCaptioning
        ? 'Dataset auto-captioning is editing this caption.'
        : selectedItem?.kind === 'encrypted' && !encryptedKey
          ? 'Unlock the encrypted dataset before editing JSON.'
          : isSaving
            ? 'Wait for the current save to finish.'
            : '';
  const jsonWorkspaceDisabled = Boolean(jsonWorkspaceDisabledReason);
  const hasPendingRecaptions = isRecaptioning || recaptionQueue.length > 0;
  const canQueueSelectedRecaption =
    canRecaptionSelectedImage && !selectedRecaptionIsRunning && !selectedRecaptionIsQueued;
  const recaptionFeedback = isRecaptioning
    ? `Recaptioning${activeRecaptionLabel ? ` ${activeRecaptionLabel}` : ''}${
        recaptionQueue.length ? `, ${recaptionQueue.length} queued` : ''
      }.`
    : recaptionQueue.length
      ? `${recaptionQueue.length} recaption${recaptionQueue.length === 1 ? '' : 's'} queued.`
      : recaptionMessage;

  useEffect(() => {
    try {
      const stored: unknown = JSON.parse(window.localStorage.getItem(approvalStorageKey) || '[]');
      setApprovedKeys(
        new Set(Array.isArray(stored) ? stored.filter((value): value is string => typeof value === 'string') : []),
      );
    } catch {
      setApprovedKeys(new Set());
    }
  }, [approvalStorageKey]);

  useEffect(() => {
    const availableKeys = new Set(items.map(item => itemKey(item)));
    setApprovedKeys(previous => {
      const next = new Set([...previous].filter(key => availableKeys.has(key)));
      return next.size === previous.size ? previous : next;
    });
  }, [items]);

  const persistApprovedKeys = useCallback(
    (next: Set<string>) => {
      setApprovedKeys(next);
      try {
        window.localStorage.setItem(approvalStorageKey, JSON.stringify([...next]));
      } catch {
        // Approval state is a convenience layer; caption saves remain authoritative.
      }
    },
    [approvalStorageKey],
  );

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
    hydratedRecaptionQueueKeyRef.current = '';
  }, [canPersistRecaptionState, recaptionQueueStorageKeyValue]);

  useEffect(() => {
    purgeLegacyPersistedRecaptionQueues();
  }, []);

  useEffect(() => {
    if (!hasEncryptedItems || typeof window === 'undefined') return;
    if (recaptionQueueStorageKeyValue) window.localStorage.removeItem(recaptionQueueStorageKeyValue);
    if (recaptionStorageKey) window.localStorage.removeItem(recaptionStorageKey);
  }, [hasEncryptedItems, recaptionQueueStorageKeyValue, recaptionStorageKey]);

  useEffect(() => {
    const storageKey = recaptionQueueStorageKeyValue;
    if (!canPersistRecaptionState) return;
    if (!storageKey || hydratedRecaptionQueueKeyRef.current === storageKey) return;
    if (isRecaptioning || recaptionQueue.length > 0) return;

    const persisted = readPersistedRecaptionQueue(storageKey);
    if (!persisted || (!persisted.active && persisted.queue.length === 0)) {
      hydratedRecaptionQueueKeyRef.current = storageKey;
      return;
    }

    const itemByKey = new Map(items.map(item => [itemKey(item), item] as const));
    if (itemByKey.size === 0) return;

    const seen = new Set<string>();
    const restored = [persisted.active, ...persisted.queue].flatMap(persistedEntry => {
      if (!persistedEntry || seen.has(persistedEntry.key)) return [];
      const item = itemByKey.get(persistedEntry.key);
      if (!item) return [];
      seen.add(persistedEntry.key);
      return [
        {
          id: persistedEntry.id || randomId(),
          item,
          key: persistedEntry.key,
          name: persistedEntry.name || itemName(item),
          existingCaption: null,
          settings: persistedEntry.settings,
        },
      ];
    });

    hydratedRecaptionQueueKeyRef.current = storageKey;
    if (restored.length === 0) {
      window.localStorage.removeItem(storageKey);
      return;
    }

    setRecaptionQueue(restored);
    setRecaptionMessage(`Restored ${restored.length} recaption${restored.length === 1 ? '' : 's'} after refresh.`);
  }, [canPersistRecaptionState, isRecaptioning, items, recaptionQueue.length, recaptionQueueStorageKeyValue]);

  useEffect(() => {
    const storageKey = recaptionQueueStorageKeyValue;
    if (!canPersistRecaptionState) return;
    if (!storageKey || hydratedRecaptionQueueKeyRef.current !== storageKey || typeof window === 'undefined') return;

    if (!activeRecaptionEntry && recaptionQueue.length === 0) {
      window.localStorage.removeItem(storageKey);
      return;
    }

    const snapshot: PersistedRecaptionQueue = {
      version: 2,
      active: activeRecaptionEntry ? persistedRecaptionQueueEntry(activeRecaptionEntry, 'running') : null,
      queue: recaptionQueue.map(entry => persistedRecaptionQueueEntry(entry, 'queued')),
      updatedAt: new Date().toISOString(),
    };
    window.localStorage.setItem(storageKey, JSON.stringify(snapshot));
  }, [activeRecaptionEntry, canPersistRecaptionState, recaptionQueue, recaptionQueueStorageKeyValue]);

  useEffect(() => {
    savedCaptionRef.current = savedCaption;
    isDirtyRef.current = hasUnsavedChanges;
  }, [hasUnsavedChanges, savedCaption]);

  useEffect(() => {
    onUnsavedChange?.(hasUnsavedChanges);
  }, [hasUnsavedChanges, onUnsavedChange]);

  useEffect(() => {
    return () => onUnsavedChange?.(false);
  }, [onUnsavedChange]);

  useEffect(() => {
    if (!hasUnsavedChanges) return;
    const warnBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', warnBeforeUnload);
    return () => window.removeEventListener('beforeunload', warnBeforeUnload);
  }, [hasUnsavedChanges]);

  useEffect(() => {
    if (!hasUnsavedChanges) return;

    type NavigationEventLike = Event & { navigationType?: string };
    const navigation = (window as Window & { navigation?: EventTarget }).navigation;
    const confirmTraversal = (event: Event) => {
      if ((event as NavigationEventLike).navigationType !== 'traverse') return;
      const shouldLeave = window.confirm(
        'You have unapplied or unsaved caption changes. Leave Caption Studio and discard them?',
      );
      if (!shouldLeave) event.preventDefault();
    };

    if (navigation) {
      navigation.addEventListener('navigate', confirmTraversal);
      return () => navigation.removeEventListener('navigate', confirmTraversal);
    }

    const restoreAfterCancelledTraversal = () => {
      const shouldLeave = window.confirm(
        'You have unapplied or unsaved caption changes. Leave Caption Studio and discard them?',
      );
      if (!shouldLeave) window.history.forward();
    };
    window.addEventListener('popstate', restoreAfterCancelledTraversal);
    return () => window.removeEventListener('popstate', restoreAfterCancelledTraversal);
  }, [hasUnsavedChanges]);

  useEffect(() => {
    if (!hasUnsavedChanges) return;
    const confirmInternalLink = (event: MouseEvent) => {
      if (
        event.defaultPrevented ||
        event.button !== 0 ||
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      ) {
        return;
      }
      const target = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[href]') : null;
      if (!target || target.target === '_blank' || target.hasAttribute('download')) return;
      const destination = new URL(target.href, window.location.href);
      if (destination.origin !== window.location.origin) return;
      if (destination.href === window.location.href) return;
      const shouldLeave = window.confirm(
        'You have unapplied or unsaved caption changes. Leave Caption Studio and discard them?',
      );
      if (shouldLeave) return;
      event.preventDefault();
      event.stopImmediatePropagation();
    };
    document.addEventListener('click', confirmInternalLink, true);
    return () => document.removeEventListener('click', confirmInternalLink, true);
  }, [hasUnsavedChanges]);

  useEffect(() => {
    selectedKeyRef.current = selectedKey;
    if (isPlainTextCaptionItem(selectedItem)) setCaptionTab('caption');
    setAutoBoxMessage('');
    setSelectedImageSize(null);
    setHiddenLayerIndexes(new Set());
    setLockedLayerIndexes(new Set());
    setOverlapElementStack([]);
    setActivePaletteSamplerIndex(null);
    setActiveImagePaletteSamplerIndex(null);
    setPaletteImageElement(null);
    setPaletteExtraction(null);
    setAppliedPaletteExtractionKey(null);
    setPaletteExtractionMessage('');
    setSelectedImagePaletteColor(null);
  }, [selectedItem, selectedKey]);

  useEffect(() => {
    if (advancedMode !== 'json' || !selectedKey || !isCaptionLoaded || jsonDraftKey === selectedKey) return;
    setJsonDraft(captionText);
    setJsonDraftBaseCaption(captionText);
    setJsonDraftKey(selectedKey);
    setJsonSelectedSectionId('elements');
    setJsonSelectedPath(
      selectedElementIndex == null
        ? 'compositional_deconstruction.elements'
        : `compositional_deconstruction.elements[${selectedElementIndex}]`,
    );
  }, [advancedMode, captionText, isCaptionLoaded, jsonDraftKey, selectedElementIndex, selectedKey]);

  useEffect(() => {
    if (
      advancedMode !== 'json' ||
      jsonDraftKey !== selectedKey ||
      jsonDraft !== jsonDraftBaseCaption ||
      jsonDraftBaseCaption === captionText
    ) {
      return;
    }
    setJsonDraft(captionText);
    setJsonDraftBaseCaption(captionText);
  }, [advancedMode, captionText, jsonDraft, jsonDraftBaseCaption, jsonDraftKey, selectedKey]);

  useEffect(() => {
    if (advancedMode !== 'json' || selectedElementIndex == null) return;
    const elementPath =
      jsonNavigatorAnalysis.elements.find(element => element.index === selectedElementIndex)?.path ||
      `compositional_deconstruction.elements[${selectedElementIndex}]`;
    setJsonSelectedSectionId('elements');
    setJsonSelectedPath(elementPath);
  }, [advancedMode, jsonNavigatorAnalysis.elements, selectedElementIndex]);

  useEffect(() => {
    if (imagePaletteColors.length === 0) {
      if (selectedImagePaletteColor !== null) setSelectedImagePaletteColor(null);
      return;
    }
    if (selectedImagePaletteColor && !imagePaletteColors.includes(selectedImagePaletteColor)) {
      setSelectedImagePaletteColor(null);
    }
  }, [imagePaletteColors, selectedImagePaletteColor]);

  useEffect(() => {
    setActivePaletteSamplerIndex(null);
  }, [selectedElementIndex]);

  useEffect(() => {
    if (advancedMode !== 'palettes' || !paletteImageElement) return;
    let cancelled = false;
    const animationFrame = window.requestAnimationFrame(() => {
      try {
        const analysis = extractPaletteFromImage(paletteImageElement, 8);
        if (!cancelled) setPaletteExtraction(analysis);
      } catch {
        if (!cancelled) setPaletteExtraction(null);
      }
    });
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(animationFrame);
    };
  }, [advancedMode, paletteImageElement, selectedKey]);

  useEffect(() => {
    if (autoBoxProvider !== 'remote_ollama' || remoteWorkerOptions.length === 0) return;
    if (remoteOllamaWorkerId && remoteWorkerOptionValues.has(remoteOllamaWorkerId)) return;
    setRemoteOllamaWorkerId(remoteWorkerOptions[0].value);
  }, [autoBoxProvider, remoteOllamaWorkerId, remoteWorkerOptionValues, remoteWorkerOptions]);

  useEffect(() => {
    if (recaptionProvider !== 'remote_ollama' || remoteWorkerOptions.length === 0) return;
    if (recaptionRemoteWorkerId && remoteWorkerOptionValues.has(recaptionRemoteWorkerId)) return;
    setRecaptionRemoteWorkerId(remoteWorkerOptions[0].value);
  }, [recaptionProvider, recaptionRemoteWorkerId, remoteWorkerOptionValues, remoteWorkerOptions]);

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
    if (!canPersistRecaptionState) return;
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
  }, [canPersistRecaptionState, recaptionStorageKey, rootCaption]);

  const useRecaptionRootPrompt = useCallback(() => {
    recaptionSystemPromptTouchedRef.current = true;
    recaptionSystemPromptRef.current = recaptionRootPrompt;
    setRecaptionSystemPrompt(recaptionRootPrompt);
  }, [recaptionRootPrompt]);

  const handleRecaptionSystemPromptChange = useCallback((value: string) => {
    recaptionSystemPromptTouchedRef.current = true;
    recaptionSystemPromptRef.current = value;
    setRecaptionSystemPrompt(value);
  }, []);

  const loadRecaptionRemoteModels = useCallback(
    async (workerId = recaptionRemoteWorkerId, preferredModel = recaptionModelRef.current) => {
      if (!workerId) return;
      const preferred = preferredModel.trim();
      setRecaptionRemoteModelStatus('loading');
      setRecaptionRemoteModelError('');
      setRecaptionRemoteModelOptions([]);
      try {
        const response = await apiClient.get(`/api/ollama-workers/${encodeURIComponent(workerId)}/models`);
        const options = ollamaModelOptions(response.data?.models);
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
        setRecaptionRemoteModelError(
          error?.response?.data?.error || error?.message || 'Could not load Remote Ollama models.',
        );
      }
    },
    [recaptionRemoteWorkerId],
  );

  useEffect(() => {
    if (recaptionProvider !== 'remote_ollama' || !recaptionRemoteWorkerId) return;
    void loadRecaptionRemoteModels(recaptionRemoteWorkerId);
  }, [loadRecaptionRemoteModels, recaptionProvider, recaptionRemoteWorkerId]);

  const handleAutoBoxProviderChange = useCallback((value: string) => {
    const nextProvider = normalizeStudioBoxProvider(value);
    setAutoBoxProvider(nextProvider);
    setAutoBoxModel(currentModel => nextAutoBoxModelForProvider(nextProvider, currentModel));
  }, []);

  const handleRecaptionProviderChange = useCallback(
    (value: string) => {
      const nextProvider = normalizeRecaptionProvider(value);
      setRecaptionProvider(nextProvider);
      if (nextProvider !== 'remote_ollama') {
        setRecaptionRemoteModelStatus('idle');
        setRecaptionRemoteModelError('');
      }
      setRecaptionModel(currentModel =>
        nextRecaptionModelForProvider(nextProvider, currentModel, recaptionRemoteModelOptions),
      );
    },
    [recaptionRemoteModelOptions],
  );

  const handleRecaptionOutputFormatChange = useCallback((value: RecaptionOutputFormat) => {
    setRecaptionOutputFormat(value);
    setRecaptionPrompt(currentPrompt => promptForRecaptionOutputFormat(value, currentPrompt));
    setRecaptionMaxNewTokens(current => maxNewTokensForRecaptionOutputFormat(value, current));
  }, []);

  const currentRecaptionSettings = useCallback(
    (): RecaptionSettingsPreset => ({
      provider: recaptionProvider,
      model: recaptionModel.trim(),
      outputFormat: recaptionOutputFormat,
      prompt: recaptionPrompt,
      systemPrompt: recaptionSystemPrompt,
      remoteWorkerId: recaptionRemoteWorkerId,
      maxNewTokens: normalizedRecaptionMaxNewTokens(recaptionMaxNewTokens),
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
      if (!canPersistRecaptionState) {
        setHasRecaptionSettingsForDataset(true);
        return;
      }
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
    [canPersistRecaptionState, recaptionStorageKey],
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
            buildEncryptedObjectRequestBody({ datasetName, workerID, projectID, objectPath: captionPath }),
            { responseType: 'blob', ...(signal ? { signal } : {}) },
          );
          const decrypted = await decryptEncryptedObjectBlob(encryptedKey, captionPath, response.data as Blob);
          text = new TextDecoder().decode(decrypted);
        }
      }
      return text;
    },
    [datasetName, encryptedKey, projectID, projectPayload, workerID],
  );

  useEffect(() => {
    const activeKeys = new Set(items.map(itemKey));
    let cacheChanged = false;

    for (const key of Array.from(captionCacheRef.current.keys())) {
      if (activeKeys.has(key)) continue;
      cacheChanged = captionCacheRef.current.delete(key) || cacheChanged;
    }

    const selectedKey = selectedKeyRef.current;
    if (selectedKey && !activeKeys.has(selectedKey) && !isDirtyRef.current) {
      latestCaptionRef.current = '';
      setCaptionText('');
      setSavedCaption('');
      setIsCaptionLoaded(false);
    }

    if (cacheChanged) bumpCaptionCacheVersion();
  }, [bumpCaptionCacheVersion, items]);

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
    const shouldLiveRefreshCaption = isAutoCaptioning || liveCaptionRefresh;
    if (!shouldLiveRefreshCaption || !selectedItem || !selectedKey) return;
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
  }, [isAutoCaptioning, liveCaptionRefresh, readCaptionForItem, selectedItem, selectedKey]);

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
        const response = await apiClient.post('/api/img/caption', {
          imgPath: targetItem.path,
          caption: value,
          direct: isPlainTextCaptionItem(targetItem),
          ...projectPayload,
        });
        onPlainItemCaptioned?.(
          targetItem.path,
          typeof response.data?.captioned_at === 'string' ? response.data.captioned_at : null,
        );
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
    [
      encryptedCaptionPaths,
      encryptedKey,
      onPlainItemCaptioned,
      onSaveEncryptedCaption,
      projectPayload,
      writeCaptionCache,
    ],
  );

  const saveCaption = useCallback(
    async (captionOverride?: string) => {
      if (!selectedItem || !isCaptionLoaded || isSavingRef.current) return false;
      const sourceCaption = captionOverride ?? captionText;
      const value = isPlainTextCaptionItem(selectedItem) ? sourceCaption : sourceCaption.trim();
      if (captionOverride === undefined && !isDirty) return true;
      if (captionOverride !== undefined && value.trim() === savedCaption.trim()) return true;
      isSavingRef.current = true;
      setIsSaving(true);
      try {
        await saveCaptionForItem(selectedItem, selectedKey, value);
        return true;
      } catch (error) {
        console.error('Caption save failed:', error);
        alert('Failed to save caption. Please try again.');
        if (captionOverride !== undefined) throw error;
        return false;
      } finally {
        isSavingRef.current = false;
        setIsSaving(false);
      }
    },
    [captionText, isCaptionLoaded, isDirty, savedCaption, saveCaptionForItem, selectedItem, selectedKey],
  );

  useEffect(() => {
    saveCaptionRef.current = saveCaption;
  }, [saveCaption]);

  const runRecaptionQueueEntry = useCallback(
    async (entry: RecaptionQueueEntry) => {
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

      const recaptionProviderLabel = providerLabel(settings.provider);
      setActiveRecaptionEntry(entry);
      setIsRecaptioning(true);
      setActiveRecaptionLabel(name);
      setActiveRecaptionKey(key);
      setRecaptionMessage(
        `Recaptioning ${name}${recaptionQueueRef.current.length ? ` (${recaptionQueueRef.current.length} queued)` : ''}.`,
      );
      try {
        const requestExistingCaption = existingCaption ?? (await readCaptionForItem(item));
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
              existingCaption: requestExistingCaption,
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
              `Recaption will send this decrypted image to ${recaptionProviderLabel}. Continue?`,
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
          appendRecaptionFields(formData, settings, requestExistingCaption, datasetName);
          response = await apiClient.post('/api/datasets/recaption-single', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 0,
          });
        }

        const caption = String(response.data?.caption || '').trim();
        if (!caption) throw new Error('Recaption returned an empty caption.');
        if (selectedKeyRef.current === key) {
          setUndoStack(previous => [
            ...previous.slice(Math.max(0, previous.length - MAX_HISTORY + 1)),
            captureHistoryEntry(latestCaptionRef.current),
          ]);
          setRedoStack([]);
          if (settings.outputFormat === 'ideogram_json') setCaptionTab('json');
        }
        await saveCaptionForItem(item, key, caption);
        setRecaptionMessage(
          `${name} recaptioned${recaptionQueueRef.current.length ? `, ${recaptionQueueRef.current.length} queued` : ''}.`,
        );
      } catch (error) {
        setRecaptionMessage(`${name}: ${responseErrorMessage(error, 'Recaption failed. Please try again.')}`);
      } finally {
        setActiveRecaptionEntry(null);
        setIsRecaptioning(false);
        setActiveRecaptionLabel('');
        setActiveRecaptionKey('');
      }
    },
    [
      captureHistoryEntry,
      datasetName,
      encryptedKey,
      encryptedProviderConfirmations,
      projectID,
      projectPayload,
      readCaptionForItem,
      saveCaptionForItem,
      workerID,
    ],
  );

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
    setRecaptionMessage(queuedTotal > 1 ? `${selectedName} queued (${queuedTotal} total).` : `${selectedName} queued.`);
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

  const commitSelectedIndex = useCallback(
    (nextIndex: number) => {
      setSelectedIndex(clampIndex(nextIndex, items.length));
      setSelectedElementIndex(null);
      setActiveTool('select');
      setUndoStack([]);
      setRedoStack([]);
    },
    [items.length],
  );

  const selectIndex = useCallback(
    async (nextIndex: number) => {
      const targetIndex = clampIndex(nextIndex, items.length);
      if (targetIndex === selectedIndex) return true;
      const sourceKey = selectedKey;
      const saved = await saveCaption();
      if (!saved || selectedKeyRef.current !== sourceKey) return false;
      commitSelectedIndex(targetIndex);
      return true;
    },
    [commitSelectedIndex, items.length, saveCaption, selectedIndex, selectedKey],
  );

  const saveAndNext = useCallback(async () => {
    const sourceKey = selectedKey;
    const saved = await saveCaption();
    if (!saved || selectedKeyRef.current !== sourceKey) return;
    if (selectedIndex < items.length - 1) commitSelectedIndex(selectedIndex + 1);
  }, [commitSelectedIndex, items.length, saveCaption, selectedIndex, selectedKey]);

  const toggleSelectedApproval = useCallback(async () => {
    if (!selectedKey || !isCaptionLoaded) return;
    const saved = await saveCaption();
    if (!saved || selectedKeyRef.current !== selectedKey) return;
    const next = new Set(approvedKeys);
    if (next.has(selectedKey)) {
      next.delete(selectedKey);
    } else {
      next.add(selectedKey);
    }
    persistApprovedKeys(next);
  }, [approvedKeys, isCaptionLoaded, persistApprovedKeys, saveCaption, selectedKey]);

  const openAdvancedMode = useCallback(
    (mode: 'boxes' | 'layers' | 'palettes' | 'json') => {
      if (mode === 'json') {
        const currentCaption = latestCaptionRef.current;
        setJsonDraft(currentCaption);
        setJsonDraftBaseCaption(currentCaption);
        setJsonDraftKey(selectedKeyRef.current);
        setJsonSelectedSectionId('elements');
        setJsonSelectedPath(
          selectedElementIndex == null
            ? 'compositional_deconstruction.elements'
            : `compositional_deconstruction.elements[${selectedElementIndex}]`,
        );
      }
      setAdvancedMode(mode);
      setCaptionTab('caption');
      if (mode === 'boxes') setActiveTool('select');
      if (mode !== 'boxes') setAutoBoxesPanelOpen(false);
      setToolsMenuOpen(false);
    },
    [selectedElementIndex],
  );

  const handleWorkbenchResizeStart = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      const startY = event.clientY;
      const startHeight = workbenchHeight;
      setWorkbenchCollapsed(false);
      const onPointerMove = (pointerEvent: PointerEvent) => {
        setWorkbenchHeight(Math.max(196, Math.min(430, startHeight + startY - pointerEvent.clientY)));
      };
      const onPointerUp = () => {
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', onPointerUp);
      };
      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp);
    },
    [workbenchHeight],
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
    void handleDeleteImages(
      [selectedItem],
      isPlainTextCaptionItem(selectedItem) ? 'current text file' : 'current image',
    );
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
    [
      applyBulkCaptionResult,
      datasetName,
      onBulkEncryptedCaptionAction,
      onRefresh,
      projectPayload,
      saveCaption,
      workerID,
    ],
  );

  const mutateCaption = useCallback(
    (mutator: (data: Record<string, any>) => void, nextSelectedElementIndex?: number | null) => {
      const parsed = parseIdeogramCaption(captionText, selectedImageSize ?? undefined);
      if (parsed.kind !== 'ideogram') return;
      const data = cloneIdeogramData(parsed.data);
      mutator(data);
      const next = serializeIdeogramCaption(data);
      if (next === captionText) return;
      setUndoStack(previous => [
        ...previous.slice(Math.max(0, previous.length - MAX_HISTORY + 1)),
        captureHistoryEntry(captionText),
      ]);
      setRedoStack([]);
      latestCaptionRef.current = next;
      setCaptionText(next);
      if (nextSelectedElementIndex !== undefined) setSelectedElementIndex(nextSelectedElementIndex);
    },
    [captionText, captureHistoryEntry, selectedImageSize],
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
      setUndoStack(previous => [
        ...previous.slice(Math.max(0, previous.length - MAX_HISTORY + 1)),
        captureHistoryEntry(currentCaption),
      ]);
      setRedoStack([]);
      latestCaptionRef.current = next;
      setCaptionText(next);
      if (nextSelectedElementIndex !== undefined) setSelectedElementIndex(nextSelectedElementIndex);
      return true;
    },
    [captureHistoryEntry, selectedImageSize],
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
        elementCount === 0
          ? normalizeGeneratedElementBoxes({ generatedElements: response.data?.generatedElements }, 2, 20)
          : [];
      if (patches.length === 0 && generatedElements.length === 0) {
        throw new Error(`${autoBoxProviderLabel} did not return any usable boxes.`);
      }

      let appliedCount = 0;
      const nextSelection =
        generatedElements.length > 0 ? elementCount : (selectedElementIndex ?? patches[0]?.elementIndex ?? null);
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
    if (
      !selectedItem ||
      layerCaptionDisabledReason ||
      selectedLayerIsCaptioning ||
      selectedElementIndex == null ||
      !selectedElement
    ) {
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
        latestParsed.kind === 'ideogram' ? (latestParsed.elements[requestElementIndex] ?? null) : null;
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
      if (!currentHasBox && !generatedBox)
        throw new Error(`${autoBoxProviderLabel} did not return a usable layer box.`);

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
      setLayerCaptionMessageForKey(
        requestLayerKey,
        responseErrorMessage(error, 'Caption Layer failed. Please try again.'),
      );
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
      const nextEntry = previous[previous.length - 1];
      if (!nextEntry) return previous;
      setRedoStack(redo => [captureHistoryEntry(captionText), ...redo].slice(0, MAX_HISTORY));
      restoreHistoryEntry(nextEntry);
      return previous.slice(0, -1);
    });
  }, [captionText, captureHistoryEntry, restoreHistoryEntry]);

  const redo = useCallback(() => {
    setRedoStack(previous => {
      const nextEntry = previous[0];
      if (!nextEntry) return previous;
      setUndoStack(undoStackValue => [
        ...undoStackValue.slice(Math.max(0, undoStackValue.length - MAX_HISTORY + 1)),
        captureHistoryEntry(captionText),
      ]);
      restoreHistoryEntry(nextEntry);
      return previous.slice(1);
    });
  }, [captionText, captureHistoryEntry, restoreHistoryEntry]);

  const replaceCaptionWithHistory = useCallback(
    (nextCaption: string) => {
      if (nextCaption !== captionText) {
        setUndoStack(previous => [
          ...previous.slice(Math.max(0, previous.length - MAX_HISTORY + 1)),
          captureHistoryEntry(captionText),
        ]);
        setRedoStack([]);
        latestCaptionRef.current = nextCaption;
        setCaptionText(nextCaption);
      }
      setJsonDraft(nextCaption);
      setJsonDraftBaseCaption(nextCaption);
      setJsonDraftKey(selectedKey);
      return nextCaption;
    },
    [captionText, captureHistoryEntry, selectedKey],
  );

  const applyJsonDraft = useCallback(() => {
    if (jsonDraftKey !== selectedKey) return null;
    const formatted = formatCaptionJsonDraft(jsonDraft);
    if (!formatted.ok) return null;
    return replaceCaptionWithHistory(formatted.value);
  }, [jsonDraft, jsonDraftKey, replaceCaptionWithHistory, selectedKey]);

  const saveJsonDraft = useCallback(async () => {
    const nextCaption = applyJsonDraft();
    if (nextCaption == null) return false;
    return saveCaption(nextCaption);
  }, [applyJsonDraft, saveCaption]);

  const saveJsonAndNext = useCallback(async () => {
    const sourceKey = selectedKey;
    const saved = await saveJsonDraft();
    if (!saved || selectedKeyRef.current !== sourceKey) return;
    if (selectedIndex < items.length - 1) commitSelectedIndex(selectedIndex + 1);
  }, [commitSelectedIndex, items.length, saveJsonDraft, selectedIndex, selectedKey]);

  const updateToolsMenuPosition = useCallback(() => {
    const button = toolsButtonRef.current;
    if (!button) return;
    const rect = button.getBoundingClientRect();
    const menuWidth = toolsMenuRef.current?.offsetWidth || 190;
    const menuHeight = toolsMenuRef.current?.offsetHeight || 176;
    const viewportPadding = 8;
    const gap = 8;
    const left = Math.max(viewportPadding, Math.min(rect.left, window.innerWidth - menuWidth - viewportPadding));
    const top =
      rect.top >= menuHeight + gap + viewportPadding
        ? rect.top - menuHeight - gap
        : Math.min(window.innerHeight - menuHeight - viewportPadding, rect.bottom + gap);
    setToolsMenuPosition({ left, top: Math.max(viewportPadding, top) });
  }, []);

  useEffect(() => {
    if (!toolsMenuOpen) return;
    updateToolsMenuPosition();
    const animationFrame = window.requestAnimationFrame(updateToolsMenuPosition);
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (toolsButtonRef.current?.contains(target) || toolsMenuRef.current?.contains(target)) return;
      setToolsMenuOpen(false);
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setToolsMenuOpen(false);
      toolsButtonRef.current?.focus();
    };
    window.addEventListener('resize', updateToolsMenuPosition);
    window.addEventListener('scroll', updateToolsMenuPosition, true);
    document.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.cancelAnimationFrame(animationFrame);
      window.removeEventListener('resize', updateToolsMenuPosition);
      window.removeEventListener('scroll', updateToolsMenuPosition, true);
      document.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [toolsMenuOpen, updateToolsMenuPosition]);

  const selectJsonIndex = useCallback(
    async (nextIndex: number) => {
      const targetIndex = clampIndex(nextIndex, items.length);
      if (targetIndex === selectedIndex) return true;
      const sourceKey = selectedKey;
      const saved = await saveJsonDraft();
      if (!saved || selectedKeyRef.current !== sourceKey) return false;
      commitSelectedIndex(targetIndex);
      return true;
    },
    [commitSelectedIndex, items.length, saveJsonDraft, selectedIndex, selectedKey],
  );

  const toggleJsonApproval = useCallback(async () => {
    if (!selectedKey || !isCaptionLoaded) return;
    const saved = await saveJsonDraft();
    if (!saved || selectedKeyRef.current !== selectedKey) return;
    const next = new Set(approvedKeys);
    if (next.has(selectedKey)) next.delete(selectedKey);
    else next.add(selectedKey);
    persistApprovedKeys(next);
  }, [approvedKeys, isCaptionLoaded, persistApprovedKeys, saveJsonDraft, selectedKey]);

  const formatJsonDraft = useCallback(() => {
    const formatted = formatCaptionJsonDraft(jsonDraft);
    if (!formatted.ok) return;
    setJsonDraft(formatted.value);
  }, [jsonDraft]);

  const switchAdvancedMode = useCallback(
    (mode: 'boxes' | 'layers' | 'palettes' | 'json') => {
      if (advancedMode === 'json' && jsonDraftDirty && applyJsonDraft() == null) return;
      openAdvancedMode(mode);
    },
    [advancedMode, applyJsonDraft, jsonDraftDirty, openAdvancedMode],
  );

  const closeAdvancedMode = useCallback(() => {
    if (advancedMode === 'json' && jsonDraftDirty && applyJsonDraft() == null) return;
    setAdvancedMode(null);
    setToolsMenuOpen(false);
  }, [advancedMode, applyJsonDraft, jsonDraftDirty]);

  useEffect(() => {
    saveCaptionRef.current = advancedMode === 'json' ? saveJsonDraft : saveCaption;
  }, [advancedMode, saveCaption, saveJsonDraft]);

  const handleAddLayer = useCallback(
    (type: IdeogramElementType = 'obj') => {
      if (layerStructureDisabledReason) return false;
      const parsed = parseIdeogramCaption(latestCaptionRef.current, selectedImageSize ?? undefined);
      if (parsed.kind !== 'ideogram') return false;

      let createdIndex: number | null = null;
      const changed = mutateLatestCaption(data => {
        createdIndex = addIdeogramLayer(data, type);
      });
      if (!changed || createdIndex == null) return false;

      autoSelectKeyRef.current = selectedKey;
      setSelectedElementIndex(createdIndex);
      setActiveTool('select');
      setOverlapElementStack([]);
      setActivePaletteSamplerIndex(null);
      return true;
    },
    [layerStructureDisabledReason, mutateLatestCaption, selectedImageSize, selectedKey],
  );

  const handleReorderLayer = useCallback(
    (fromIndex: number, toIndex: number) => {
      if (layerStructureDisabledReason) return false;
      const parsed = parseIdeogramCaption(latestCaptionRef.current, selectedImageSize ?? undefined);
      if (parsed.kind !== 'ideogram') return false;
      const elementCount = parsed.elements.length;
      if (
        fromIndex < 0 ||
        toIndex < 0 ||
        fromIndex >= elementCount ||
        toIndex >= elementCount ||
        fromIndex === toIndex
      ) {
        return false;
      }

      const nextSelectedIndex =
        selectedElementIndex == null ? null : remapLayerIndexForMove(selectedElementIndex, fromIndex, toIndex);
      const changed = mutateLatestCaption(data => {
        moveIdeogramElement(data, fromIndex, toIndex);
      }, nextSelectedIndex);
      if (!changed) return false;

      autoSelectKeyRef.current = selectedKey;
      setHiddenLayerIndexes(previous => remapLayerIndexSetForMove(previous, fromIndex, toIndex));
      setLockedLayerIndexes(previous => remapLayerIndexSetForMove(previous, fromIndex, toIndex));
      setOverlapElementStack(previous => remapLayerIndexArrayForMove(previous, fromIndex, toIndex));
      setLayerCaptionMessages(previous =>
        Object.fromEntries(
          Object.entries(previous).filter(([requestKey]) => !isLayerCaptionRequestForItem(requestKey, selectedKey)),
        ),
      );
      setActivePaletteSamplerIndex(null);
      return true;
    },
    [layerStructureDisabledReason, mutateLatestCaption, selectedElementIndex, selectedImageSize, selectedKey],
  );

  const handleCreateBox = useCallback(
    (type: IdeogramElementType, box: NormalizedBox) => {
      if (layerStructureDisabledReason) return;
      const parsed = parseIdeogramCaption(latestCaptionRef.current, selectedImageSize ?? undefined);
      if (parsed.kind !== 'ideogram') return;
      const selected = selectedElementIndex == null ? null : (parsed.elements[selectedElementIndex] ?? null);
      const selectedType: IdeogramElementType = selected?.type === 'text' ? 'text' : 'obj';
      const selectedHasBox = parsed.boxes.some(candidate => candidate.elementIndex === selectedElementIndex);

      if (selected && selectedElementIndex != null && selectedType === type && !selectedHasBox) {
        mutateLatestCaption(data => updateIdeogramElementBox(data, selectedElementIndex, box), selectedElementIndex);
        setActiveTool('select');
        return;
      }

      let createdIndex: number | null = null;
      const changed = mutateLatestCaption(data => {
        createdIndex = addIdeogramElement(data, type, box);
      });
      if (changed && createdIndex != null) {
        autoSelectKeyRef.current = selectedKey;
        setSelectedElementIndex(createdIndex);
        setActiveTool('select');
      }
    },
    [layerStructureDisabledReason, mutateLatestCaption, selectedElementIndex, selectedImageSize, selectedKey],
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
      if (layerStructureDisabledReason) return;
      const parsed = parseIdeogramCaption(latestCaptionRef.current, selectedImageSize ?? undefined);
      const elementCount = parsed.kind === 'ideogram' ? parsed.elements.length : 0;
      if (elementIndex < 0 || elementIndex >= elementCount) return;
      const duplicateIndex = elementIndex + 1;
      mutateLatestCaption(data => {
        duplicateIdeogramElement(data, elementIndex);
      }, duplicateIndex);
      setHiddenLayerIndexes(previous => reindexLayerIndexSetAfterInsert(previous, duplicateIndex));
      setLockedLayerIndexes(previous => reindexLayerIndexSetAfterInsert(previous, duplicateIndex));
      setOverlapElementStack([]);
    },
    [layerStructureDisabledReason, mutateLatestCaption, selectedImageSize],
  );

  const handleDeleteElement = useCallback(
    (elementIndex: number) => {
      if (layerStructureDisabledReason) return;
      const parsed = parseIdeogramCaption(latestCaptionRef.current, selectedImageSize ?? undefined);
      const elementCount = parsed.kind === 'ideogram' ? parsed.elements.length : 0;
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
      mutateLatestCaption(data => deleteIdeogramElement(data, elementIndex), nextSelection);
      setHiddenLayerIndexes(previous => reindexLayerIndexSetAfterDelete(previous, elementIndex));
      setLockedLayerIndexes(previous => reindexLayerIndexSetAfterDelete(previous, elementIndex));
      setOverlapElementStack(previous =>
        previous.flatMap(index => (index === elementIndex ? [] : [index > elementIndex ? index - 1 : index])),
      );
    },
    [layerStructureDisabledReason, mutateLatestCaption, selectedElementIndex, selectedImageSize],
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

  const handleImagePaletteChange = useCallback(
    (colors: string[]) => {
      mutateCaption(data => updateIdeogramImagePalette(data, colors));
    },
    [mutateCaption],
  );

  const handleElementPaletteChange = useCallback(
    (elementIndex: number, colors: string[]) => {
      mutateCaption(data => updateIdeogramElementPalette(data, elementIndex, colors));
    },
    [mutateCaption],
  );

  const handleStartPaletteSample = useCallback((index: number) => {
    setActiveImagePaletteSamplerIndex(null);
    setActivePaletteSamplerIndex(index);
    setActiveTool('select');
  }, []);

  const handleStartImagePaletteSample = useCallback(
    (index: number) => {
      if (index < 0 || index >= imagePaletteColors.length) return;
      setActivePaletteSamplerIndex(null);
      setActiveImagePaletteSamplerIndex(index);
      setSelectedImagePaletteColor(imagePaletteColors[index]);
      setPaletteExtractionMessage('Click the image to replace this color. Double-click to cancel.');
      setActiveTool('select');
    },
    [imagePaletteColors],
  );

  const handleCancelPaletteSample = useCallback(() => {
    setActivePaletteSamplerIndex(null);
    setActiveImagePaletteSamplerIndex(null);
    setPaletteExtractionMessage('');
  }, []);

  const handleSamplePaletteColor = useCallback(
    (color: string) => {
      if (activeImagePaletteSamplerIndex != null) {
        const normalized = normalizeHexColor(color);
        if (!normalized) return;
        const nextPalette = [...imagePaletteColors];
        nextPalette[activeImagePaletteSamplerIndex] = normalized;
        handleImagePaletteChange(nextPalette);
        setSelectedImagePaletteColor(normalized);
        setActiveImagePaletteSamplerIndex(null);
        setPaletteExtractionMessage(`Updated palette color to ${normalized}.`);
        return;
      }
      if (activePaletteSamplerIndex == null) return;
      handleSelectedPaletteColorChange(activePaletteSamplerIndex, color);
      setActivePaletteSamplerIndex(null);
    },
    [
      activeImagePaletteSamplerIndex,
      activePaletteSamplerIndex,
      handleImagePaletteChange,
      handleSelectedPaletteColorChange,
      imagePaletteColors,
    ],
  );

  const handleExtractImagePalette = useCallback(() => {
    if (!paletteImageElement || isExtractingPalette) return;
    setIsExtractingPalette(true);
    setPaletteExtractionMessage('Analyzing image colors locally…');
    window.requestAnimationFrame(() => {
      try {
        const extraction = extractPaletteFromImage(paletteImageElement, 8);
        const colors = extraction.swatches.map(swatch => swatch.color);
        if (colors.length === 0) throw new Error('No opaque image colors were found.');
        setPaletteExtraction(extraction);
        handleImagePaletteChange(colors);
        setAppliedPaletteExtractionKey(colors.join('|'));
        setSelectedImagePaletteColor(colors[0]);
        setPaletteExtractionMessage(`Extracted ${colors.length} colors locally.`);
      } catch (error) {
        setPaletteExtractionMessage(error instanceof Error ? error.message : 'Palette extraction failed.');
      } finally {
        setIsExtractingPalette(false);
      }
    });
  }, [handleImagePaletteChange, isExtractingPalette, paletteImageElement]);

  const handleMergePaletteColors = useCallback(
    (keepColor: string, removeColor: string) => {
      mutateCaption(data => mergeIdeogramPaletteColors(data, keepColor, removeColor));
      setSelectedImagePaletteColor(keepColor);
      setPaletteExtractionMessage(`Merged ${removeColor} into ${keepColor}.`);
    },
    [mutateCaption],
  );

  const handleToggleLayerPaletteColor = useCallback(
    (elementIndex: number, color: string, assigned: boolean) => {
      const layer = paletteLayers.find(candidate => candidate.elementIndex === elementIndex);
      if (!layer) return;
      const current = normalizeIdeogramColorPalette(layer.colorPalette);
      const next = assigned
        ? normalizeIdeogramColorPalette([...current, color])
        : current.filter(candidate => candidate !== color);
      if (assigned && !current.includes(color) && current.length >= 5) {
        setPaletteExtractionMessage(`${layer.label} already has the maximum of 5 colors.`);
        return;
      }
      handleElementPaletteChange(elementIndex, next);
      setPaletteExtractionMessage(
        `${assigned ? 'Assigned' : 'Removed'} ${color} ${assigned ? 'to' : 'from'} ${layer.label}.`,
      );
    },
    [handleElementPaletteChange, paletteLayers],
  );

  const handleAssignPaletteColor = useCallback(
    (color: string) => {
      const preferredLayer =
        selectedElementIndex == null
          ? null
          : (paletteLayers.find(layer => layer.elementIndex === selectedElementIndex) ?? null);
      const targetLayer =
        preferredLayer && !preferredLayer.colorPalette.includes(color) && preferredLayer.colorPalette.length < 5
          ? preferredLayer
          : (paletteLayers.find(layer => !layer.colorPalette.includes(color) && layer.colorPalette.length < 5) ?? null);
      if (!targetLayer) {
        setPaletteExtractionMessage('Every layer already has this color or has reached the 5-color limit.');
        return;
      }
      setSelectedElementIndex(targetLayer.elementIndex);
      handleToggleLayerPaletteColor(targetLayer.elementIndex, color, true);
    },
    [handleToggleLayerPaletteColor, paletteLayers, selectedElementIndex],
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
        void (advancedMode === 'json' ? saveJsonDraft() : saveCaption());
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        void (advancedMode === 'json' ? saveJsonAndNext() : saveAndNext());
        return;
      }
      if (isTyping) return;
      const isArrowKey =
        event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowUp' || event.key === 'ArrowDown';
      if (isArrowKey) {
        if (activeTool === 'pan') return;
        // Nudge the selected (unlocked) box with arrow keys; otherwise navigate images.
        if (selectedBox && selectedElementIndex != null && !lockedLayerIndexes.has(selectedElementIndex)) {
          event.preventDefault();
          if (advancedMode === 'json' && jsonDraftDirty) return;
          const step = event.shiftKey ? 20 : 5;
          const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0;
          const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0;
          handleChangeBox(selectedElementIndex, resizeOrMoveBox(selectedBox, dx, dy, 'move', MIN_BOX_SPAN));
          return;
        }
        if (event.key === 'ArrowLeft') {
          event.preventDefault();
          void (advancedMode === 'json' ? selectJsonIndex(selectedIndex - 1) : selectIndex(selectedIndex - 1));
        }
        if (event.key === 'ArrowRight') {
          event.preventDefault();
          void (advancedMode === 'json' ? selectJsonIndex(selectedIndex + 1) : selectIndex(selectedIndex + 1));
        }
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
      if (event.key === 'Delete' || event.key === 'Backspace') {
        if (advancedMode === 'json' && jsonDraftDirty) {
          event.preventDefault();
          return;
        }
        handleDeleteSelectedElement();
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'd') {
        if (advancedMode === 'json' && jsonDraftDirty) {
          event.preventDefault();
          return;
        }
        if (selectedElementIndex != null) {
          event.preventDefault();
          handleDuplicateElement(selectedElementIndex);
        }
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (advancedMode === 'json' && jsonDraftDirty) return;
        if (event.shiftKey) {
          redo();
        } else {
          undo();
        }
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
        event.preventDefault();
        if (advancedMode === 'json' && jsonDraftDirty) return;
        redo();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    advancedMode,
    activeTool,
    activePaletteSamplerIndex,
    cycleOverlapSelection,
    handleCancelPaletteSample,
    handleChangeBox,
    handleDeleteSelectedElement,
    handleDuplicateElement,
    jsonDraftDirty,
    lockedLayerIndexes,
    redo,
    saveCaption,
    saveJsonAndNext,
    saveJsonDraft,
    saveAndNext,
    selectIndex,
    selectJsonIndex,
    selectedBox,
    selectedElementIndex,
    selectedIndex,
    undo,
  ]);

  const highLevelDescription =
    isIdeogram && typeof captionParse.data.high_level_description === 'string'
      ? captionParse.data.high_level_description
      : captionText;
  const selectedApproved = approvedKeys.has(selectedKey);
  const captionWordCount = highLevelDescription.trim() ? highLevelDescription.trim().split(/\s+/).length : 0;
  const captionCharacterCount = highLevelDescription.length;
  const selectedSizeBytes = selectedItem?.kind === 'plain' ? selectedItem.sizeBytes : selectedItem?.item.size;
  const selectedDate =
    selectedItem?.kind === 'plain'
      ? selectedItem.captionedAt || selectedItem.addedAt
      : selectedItem?.item.updatedAt || selectedItem?.item.createdAt;
  const selectedDateLabel = formatStudioDate(selectedDate);
  const selectedAddedDateLabel = formatStudioDate(
    selectedItem?.kind === 'plain' ? selectedItem.addedAt : selectedItem?.item.createdAt,
  );
  const selectedLayerColor =
    selectedBox?.color ||
    (selectedElementIndex != null ? BOX_COLORS[selectedElementIndex % BOX_COLORS.length] : BOX_COLORS[0]);
  const assignedPaletteColors = useMemo(
    () => new Set(paletteLayers.flatMap(layer => normalizeIdeogramColorPalette(layer.colorPalette))),
    [paletteLayers],
  );
  const paletteSwatches = useMemo<PaletteSwatch[]>(() => {
    return imagePaletteColors.map(color => {
      const direct = paletteExtraction?.swatches.find(swatch => swatch.color === color);
      let nearest = direct ?? null;
      let nearestDistance = direct ? 0 : Number.POSITIVE_INFINITY;
      if (!direct && paletteExtraction) {
        for (const swatch of paletteExtraction.swatches) {
          const distance = paletteColorDistance(color, swatch.color);
          if (distance <= PALETTE_DELTA_E_THRESHOLDS.swatchMatch && distance < nearestDistance) {
            nearest = swatch;
            nearestDistance = distance;
          }
        }
      }
      return {
        color,
        coverage: nearest ? nearest.coverage / 100 : paletteExtraction ? 0 : null,
        confidence: paletteExtraction?.confidence ?? null,
      };
    });
  }, [imagePaletteColors, paletteExtraction]);
  const paletteCoverage = useMemo(() => {
    if (imagePaletteColors.length === 0) return null;
    if (paletteExtraction && appliedPaletteExtractionKey === imagePaletteColors.join('|')) {
      return paletteExtraction.coverage / 100;
    }
    const assignedRatio =
      imagePaletteColors.filter(color => assignedPaletteColors.has(color)).length / imagePaletteColors.length;
    return Math.min(0.99, 0.72 + assignedRatio * 0.24);
  }, [appliedPaletteExtractionKey, assignedPaletteColors, imagePaletteColors, paletteExtraction]);
  const paletteConfidence = useMemo(() => {
    if (imagePaletteColors.length === 0 || paletteCoverage == null) return null;
    if (paletteExtraction && appliedPaletteExtractionKey === imagePaletteColors.join('|')) {
      return paletteExtraction.confidence;
    }
    const diversity = Math.min(1, imagePaletteColors.length / 8);
    return Math.min(0.99, 0.55 + paletteCoverage * 0.35 + diversity * 0.04);
  }, [appliedPaletteExtractionKey, imagePaletteColors, paletteCoverage, paletteExtraction]);
  const paletteReviewIssues = useMemo<PaletteReviewIssue[]>(() => {
    const duplicateIssues: PaletteReviewIssue[] = findNearDuplicatePalettePairs(imagePaletteColors).map(pair => {
      const idColors = [...pair.colors].sort();
      return {
        id: `duplicate-${idColors[0]}-${idColors[1]}`,
        kind: 'near-duplicate',
        colors: pair.colors,
        tone: pair.distance <= PALETTE_DELTA_E_THRESHOLDS.indistinguishable ? 'warning' : 'info',
      };
    });

    const assignedSwatches = paletteSwatches
      .filter(swatch => assignedPaletteColors.has(swatch.color) && swatch.coverage != null)
      .sort((left, right) => {
        const coverageDifference = (left.coverage ?? 0) - (right.coverage ?? 0);
        if (coverageDifference !== 0) return coverageDifference;
        return imagePaletteColors.indexOf(right.color) - imagePaletteColors.indexOf(left.color);
      });
    const lowCoverageSwatch = assignedSwatches.find(swatch => (swatch.coverage ?? 0) <= 0.035) ?? null;
    const unassignedColor = imagePaletteColors.find(color => !assignedPaletteColors.has(color)) ?? null;
    const issues: PaletteReviewIssue[] = [];
    if (duplicateIssues[0]) issues.push(duplicateIssues[0]);
    if (lowCoverageSwatch) {
      issues.push({
        id: `coverage-${lowCoverageSwatch.color}`,
        kind: 'low-coverage',
        color: lowCoverageSwatch.color,
        coverage: lowCoverageSwatch.coverage ?? 0,
      });
    }
    if (unassignedColor) {
      issues.push({ id: `unassigned-${unassignedColor}`, kind: 'unassigned', color: unassignedColor });
    }
    if (duplicateIssues[1]) issues.push(duplicateIssues[1]);
    return issues.slice(0, 4);
  }, [assignedPaletteColors, imagePaletteColors, paletteSwatches]);
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
        <div className="border border-dashed border-gray-700 bg-gray-900/60 px-6 py-5 text-sm">
          No editable media or text files found.
        </div>
      </div>
    );
  }

  if (!selectedItem) return null;

  if (presentation === 'legacy')
    return (
      <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-gray-950 text-gray-100">
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
            <main className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-gray-950">
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
                liveCaptionRefresh={liveCaptionRefresh}
                captionCache={captionCacheRef.current}
                captionCacheVersion={captionCacheVersion}
                approvedKeys={approvedKeys}
                captionAction={captionAction}
                watcherAction={watcherAction}
                onAddImages={onAddImages}
                presentation="legacy"
                onCaptionCacheChange={bumpCaptionCacheVersion}
                onSelectIndex={selectIndex}
                onBulkCaptionAction={handleBulkCaptionAction}
                onDeleteImages={handleDeleteImages}
              />
            </main>

            <aside className="flex w-full max-h-[34dvh] min-h-[190px] min-w-0 flex-shrink-0 flex-col overflow-hidden border-t border-gray-900 bg-gray-950 xl:max-h-none xl:min-h-0 xl:w-[410px] xl:flex-none xl:border-l xl:border-t-0">
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
        <RecaptionSettingsModal
          isOpen={isRecaptionModalOpen}
          provider={recaptionProvider}
          outputFormat={recaptionOutputFormat}
          model={recaptionModel}
          modelOptions={recaptionModelOptions}
          maxNewTokens={recaptionMaxNewTokens}
          prompt={recaptionPrompt}
          systemPrompt={recaptionSystemPrompt}
          rootPrompt={recaptionRootPrompt}
          rootPromptStatus={recaptionRootPromptStatus}
          remoteWorkerId={recaptionRemoteWorkerId}
          remoteWorkerOptions={remoteWorkerOptions}
          remoteModelOptions={recaptionRemoteModelOptions}
          remoteModelStatus={recaptionRemoteModelStatus}
          remoteModelError={recaptionRemoteModelError}
          message={recaptionMessage}
          isRecaptioning={isRecaptioning}
          canQueueSelectedRecaption={canQueueSelectedRecaption}
          selectedRecaptionIsRunning={selectedRecaptionIsRunning}
          selectedRecaptionIsQueued={selectedRecaptionIsQueued}
          hasPendingRecaptions={hasPendingRecaptions}
          onClose={() => setIsRecaptionModalOpen(false)}
          onSubmit={queueSelectedRecaption}
          onProviderChange={handleRecaptionProviderChange}
          onOutputFormatChange={handleRecaptionOutputFormatChange}
          onModelChange={setRecaptionModel}
          onMaxNewTokensChange={setRecaptionMaxNewTokens}
          onPromptChange={setRecaptionPrompt}
          onSystemPromptChange={handleRecaptionSystemPromptChange}
          onUseRootPrompt={useRecaptionRootPrompt}
          onRemoteWorkerChange={setRecaptionRemoteWorkerId}
          onLoadRemoteModels={() => void loadRecaptionRemoteModels()}
        />
      </div>
    );

  return (
    <>
      <div className="caption-studio relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-gray-950 text-gray-100">
        <ImageNavigator
          items={items}
          selectedIndex={selectedIndex}
          datasetName={datasetName}
          workerID={workerID}
          projectID={projectID}
          encryptedKey={encryptedKey}
          isAutoCaptioning={isAutoCaptioning}
          liveCaptionRefresh={liveCaptionRefresh}
          captionCache={captionCacheRef.current}
          captionCacheVersion={captionCacheVersion}
          approvedKeys={approvedKeys}
          captionAction={captionAction}
          watcherAction={watcherAction}
          onAddImages={onAddImages}
          onCaptionCacheChange={bumpCaptionCacheVersion}
          onSelectIndex={selectIndex}
          onBulkCaptionAction={handleBulkCaptionAction}
          onDeleteImages={handleDeleteImages}
        />

        {!workbenchCollapsed && (
          <section
            className="relative flex flex-shrink-0 flex-col border-t border-brand-400/80 bg-gray-950"
            style={{ height: `${workbenchHeight}px` }}
            aria-label="Caption workbench"
          >
            <button
              type="button"
              onPointerDown={handleWorkbenchResizeStart}
              className="absolute -top-2 left-1/2 z-20 flex h-4 w-20 -translate-x-1/2 cursor-row-resize items-center justify-center text-brand-300/70 hover:text-brand-200"
              title="Resize caption workbench"
            >
              <GripHorizontal className="h-4 w-4" />
            </button>

            <div className="operator-scrollbar-none flex min-h-0 flex-1 overflow-x-auto overflow-y-hidden">
              <button
                type="button"
                onClick={() => setWorkbenchCollapsed(true)}
                className="sticky left-0 z-10 flex w-[42px] flex-shrink-0 items-center justify-center border-r border-gray-800 bg-gray-950 text-gray-500 hover:bg-gray-950 hover:text-gray-200"
                title="Collapse caption editor"
                aria-label="Collapse caption editor"
              >
                <ChevronDown className="h-5 w-5" />
              </button>
              <div
                className="grid min-h-0 min-w-[1280px] flex-1 gap-4 px-4 py-4"
                style={{ gridTemplateColumns: '353fr 566fr 320fr 121fr' }}
              >
                <div className="flex min-w-[300px] gap-4 overflow-hidden">
                  <div className="h-full w-[158px] flex-shrink-0 overflow-hidden rounded-[4px] border border-gray-800 bg-gray-950">
                    {selectedItem.kind === 'plain' ? (
                      <PlainThumb
                        path={selectedItem.path}
                        mediaUrl={selectedItem.mediaUrl}
                        alt={selectedName}
                        onNaturalSizeChange={setSelectedImageSize}
                      />
                    ) : (
                      <EncryptedThumb
                        datasetName={datasetName}
                        workerID={workerID}
                        projectID={projectID}
                        cryptoKey={encryptedKey}
                        item={selectedItem.item}
                        onNaturalSizeChange={setSelectedImageSize}
                      />
                    )}
                  </div>
                  <div className="min-w-0 flex-1 py-1">
                    <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-gray-500">
                      Selected asset
                      <span
                        className={classNames('h-1.5 w-1.5 rounded-full', {
                          'bg-emerald-400': selectedApproved,
                          'bg-amber-300': !selectedApproved && captionText.trim(),
                          'bg-rose-400': !captionText.trim(),
                        })}
                      />
                    </div>
                    <h2
                      className="mt-2 truncate text-[18px] font-semibold leading-tight text-gray-100"
                      title={selectedName}
                    >
                      {selectedName}
                    </h2>
                    <dl className="mt-4 grid grid-cols-[72px_1fr] gap-x-2 gap-y-2 text-[11px] leading-tight">
                      <dt className="text-gray-600">Dimensions</dt>
                      <dd className="text-gray-300">
                        {selectedImageSize ? `${selectedImageSize.width} × ${selectedImageSize.height}` : '—'}
                      </dd>
                      <dt className="text-gray-600">File size</dt>
                      <dd className="text-gray-300">{formatByteSize(selectedSizeBytes)}</dd>
                      <dt className="text-gray-600">Modified</dt>
                      <dd className="text-gray-300">{selectedDateLabel}</dd>
                      <dt className="text-gray-600">Type</dt>
                      <dd className="capitalize text-gray-300">{selectedKind}</dd>
                    </dl>
                    <div className="mt-4 flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleRecaptionClick}
                        disabled={!canQueueSelectedRecaption}
                        className="inline-flex h-8 items-center gap-1.5 rounded-[4px] border border-gray-800 bg-gray-950 px-2 text-[11px] font-medium text-gray-300 hover:border-gray-600 hover:text-white disabled:opacity-40"
                      >
                        {selectedRecaptionIsRunning ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Sparkles className="h-3.5 w-3.5" />
                        )}
                        {selectedRecaptionIsQueued ? 'Queued' : 'Recaption'}
                      </button>
                      <button
                        type="button"
                        onClick={openRecaptionSettings}
                        className="flex h-8 w-8 items-center justify-center rounded-[4px] border border-gray-800 bg-gray-950 text-gray-400 hover:text-white"
                        title="Recaption settings"
                      >
                        <Settings2 className="h-3.5 w-3.5" />
                      </button>
                      {onDeleteImages && (
                        <button
                          type="button"
                          onClick={handleDeleteCurrentImage}
                          disabled={isDeletingImages}
                          className="flex h-8 w-8 items-center justify-center rounded-[4px] border border-rose-950 bg-rose-950/30 text-rose-400 hover:bg-rose-950/60"
                          title="Delete selected asset"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex min-w-[410px] flex-col overflow-hidden rounded-[4px] border border-gray-800 bg-gray-950">
                  <div className="flex h-10 flex-shrink-0 items-center gap-2 border-b border-gray-800 px-3">
                    <span className={classNames('h-2 w-2 rounded-full', captionStatus.dot)} />
                    <span className="text-[12px] font-semibold text-gray-200">Caption</span>
                    {isIdeogram && (
                      <span className="rounded-[3px] bg-brand-950 px-1.5 py-0.5 text-[10px] text-brand-300">JSON</span>
                    )}
                    <span className="ml-auto text-[11px] text-gray-600">
                      {isSaving ? 'Saving…' : isDirty ? 'Unsaved changes' : 'Saved'}
                    </span>
                  </div>
                  <textarea
                    value={highLevelDescription}
                    onChange={event => handleCaptionDescriptionChange(event.target.value)}
                    disabled={!isCaptionLoaded || isAutoCaptioning}
                    placeholder={isCaptionLoaded ? 'Write a precise caption for this image…' : 'Loading caption…'}
                    className="min-h-0 flex-1 resize-none bg-transparent px-4 py-3 text-[15px] leading-7 text-gray-100 outline-none placeholder:text-gray-650 disabled:opacity-60"
                    spellCheck
                    aria-label="Image caption"
                  />
                  <div className="flex h-8 flex-shrink-0 items-center border-t border-gray-800 px-3 text-[10px] text-gray-600">
                    <span>{captionWordCount.toLocaleString()} words</span>
                    <span className="mx-2 text-gray-800">•</span>
                    <span>{captionCharacterCount.toLocaleString()} characters</span>
                    <span className="ml-auto">⌘ Enter — save & next</span>
                  </div>
                </div>

                <div className="grid min-w-[320px] grid-cols-2 grid-rows-[56px_50px_51px_1fr] gap-2">
                  <button
                    type="button"
                    onClick={() => void saveAndNext()}
                    disabled={!isCaptionLoaded || isAutoCaptioning || isSaving}
                    className="col-span-2 inline-flex items-center justify-center gap-2 rounded-[4px] bg-brand-400 text-[16px] font-semibold text-[var(--brand-ink)] hover:bg-brand-300 disabled:cursor-not-allowed disabled:opacity-45"
                  >
                    {isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                    Save & Next
                  </button>
                  <button
                    type="button"
                    onClick={() => void toggleSelectedApproval()}
                    disabled={!isCaptionLoaded || !captionText.trim()}
                    className={classNames(
                      'col-span-2 inline-flex items-center justify-center gap-2 rounded-[4px] border text-[14px] font-semibold',
                      selectedApproved
                        ? 'border-emerald-500/70 bg-emerald-950/40 text-emerald-300'
                        : 'border-gray-800 bg-gray-950 text-gray-200 hover:border-emerald-700 hover:text-emerald-300',
                    )}
                  >
                    <Check className="h-4 w-4" />
                    {selectedApproved ? 'Approved' : 'Approve'}
                  </button>
                  <button
                    type="button"
                    onClick={() => selectIndex(selectedIndex - 1)}
                    disabled={selectedIndex === 0}
                    className="inline-flex items-center justify-center gap-1 rounded-[4px] border border-gray-800 bg-gray-950 text-[12px] text-gray-300 hover:bg-gray-900 disabled:opacity-35"
                  >
                    <ChevronLeft className="h-4 w-4" /> Previous
                  </button>
                  <button
                    type="button"
                    onClick={() => selectIndex(selectedIndex + 1)}
                    disabled={selectedIndex >= items.length - 1}
                    className="inline-flex items-center justify-center gap-1 rounded-[4px] border border-gray-800 bg-gray-950 text-[12px] text-gray-300 hover:bg-gray-900 disabled:opacity-35"
                  >
                    Next <ChevronRight className="h-4 w-4" />
                  </button>
                  <div className="col-span-2 flex items-end justify-center pb-1 text-[11px] tabular-nums text-gray-600">
                    {selectedIndex + 1} of {items.length.toLocaleString()}
                    {recaptionFeedback ? (
                      <span className="ml-2 max-w-[190px] truncate text-brand-700">{recaptionFeedback}</span>
                    ) : null}
                  </div>
                </div>

                <div className="relative min-w-[110px] overflow-hidden rounded-[4px] border border-gray-800 bg-gray-950">
                  <div className="flex h-[42px] w-full items-center border-b border-gray-800 px-3 text-left text-[12px] font-semibold text-gray-200">
                    <Settings2 className="mr-2 h-3.5 w-3.5" />
                    Tools
                  </div>
                  {ADVANCED_WORKSPACE_TOOLS.map(tool => {
                    const Icon = tool.icon;
                    return (
                      <button
                        key={tool.value}
                        type="button"
                        onClick={() => openAdvancedMode(tool.value)}
                        className={classNames(
                          'flex h-[41px] w-full items-center gap-2 border-b border-gray-800 px-3 text-[12px] last:border-b-0 hover:bg-gray-900',
                          advancedMode === tool.value ? 'text-brand-300' : 'text-gray-400',
                        )}
                      >
                        <Icon className="h-3.5 w-3.5" />
                        {tool.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>
        )}

        <button
          type="button"
          onClick={() => setWorkbenchCollapsed(collapsed => !collapsed)}
          className="flex h-[46px] flex-shrink-0 items-center justify-center gap-2 border-t border-gray-800 bg-gray-950 text-[11px] font-medium text-gray-500 hover:bg-gray-950 hover:text-gray-200"
        >
          {workbenchCollapsed ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          {workbenchCollapsed ? 'Open caption editor' : 'Collapse caption editor'}
        </button>

        {(advancedMode === 'layers' ||
          advancedMode === 'boxes' ||
          advancedMode === 'palettes' ||
          advancedMode === 'json') && (
          <section
            className={classNames(
              'absolute inset-x-0 bottom-0 z-40 flex min-h-0 flex-col bg-gray-950',
              advancedMode === 'palettes' || advancedMode === 'json' ? 'top-[46px]' : 'top-[61px]',
            )}
            aria-label={
              advancedMode === 'boxes'
                ? 'Boxes editor'
                : advancedMode === 'layers'
                  ? 'Layers editor'
                  : advancedMode === 'palettes'
                    ? 'Image palette editor'
                    : 'Caption JSON editor'
            }
          >
            <AdvancedFilmstrip
              items={items}
              selectedIndex={selectedIndex}
              datasetName={datasetName}
              workerID={workerID}
              projectID={projectID}
              encryptedKey={encryptedKey}
              onSelectIndex={advancedMode === 'json' ? index => void selectJsonIndex(index) : selectIndex}
            />

            <div className="flex min-h-0 flex-1 border-b border-gray-800">
              {advancedMode === 'boxes' ? (
                <StandaloneBoxesToolRail
                  activeTool={activeTool}
                  canAnnotate={canAnnotate}
                  canAutoBoxes={canAnnotate}
                  canUndo={undoStack.length > 0}
                  canRedo={redoStack.length > 0}
                  onToolChange={setActiveTool}
                  onAutoBoxes={() => setAutoBoxesPanelOpen(true)}
                  onZoomIn={() => setZoom(value => Math.min(2, Number((value + 0.25).toFixed(2))))}
                  onFit={() => setZoom(1)}
                  onUndo={undo}
                  onRedo={redo}
                />
              ) : advancedMode === 'layers' ? (
                <ToolRail
                  presentation="standalone"
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
                  onShowJson={() => setAdvancedMode('json')}
                />
              ) : null}

              <main className="relative min-h-0 min-w-0 flex-1 overflow-hidden bg-gray-950">
                <StudioMedia
                  item={selectedItem}
                  datasetName={datasetName}
                  workerID={workerID}
                  projectID={projectID}
                  cryptoKey={encryptedKey}
                  zoom={zoom}
                  onNaturalSizeChange={setSelectedImageSize}
                  isSamplingColor={advancedMode === 'palettes' && activeImagePaletteSamplerIndex != null}
                  onSampleColor={handleSamplePaletteColor}
                  onCancelColorSample={handleCancelPaletteSample}
                  onImageElementChange={advancedMode === 'palettes' ? setPaletteImageElement : undefined}
                >
                  {canAnnotate && advancedMode !== 'palettes' && (
                    <AnnotationLayer
                      presentation="standalone"
                      boxes={boxes}
                      activeTool={advancedMode === 'json' ? 'select' : activeTool}
                      selectedElementIndex={selectedElementIndex}
                      hiddenElementIndexes={hiddenLayerIndexes}
                      lockedElementIndexes={
                        advancedMode === 'json' && jsonDraftDirty
                          ? new Set(boxes.map(box => box.elementIndex))
                          : lockedLayerIndexes
                      }
                      imageSize={selectedImageSize}
                      onSelect={setSelectedElementIndex}
                      onCreate={handleCreateBox}
                      onChangeBox={jsonDraftDirty ? () => undefined : handleChangeBox}
                      onOverlapStackChange={setOverlapElementStack}
                    />
                  )}
                </StudioMedia>

                {advancedMode === 'palettes' ? (
                  <button
                    type="button"
                    disabled={imagePaletteColors.length === 0}
                    aria-pressed={activeImagePaletteSamplerIndex != null}
                    onClick={() => {
                      if (activeImagePaletteSamplerIndex != null) {
                        handleCancelPaletteSample();
                        return;
                      }
                      const color = selectedImagePaletteColor || imagePaletteColors[0];
                      const colorIndex = imagePaletteColors.indexOf(color);
                      if (colorIndex >= 0) handleStartImagePaletteSample(colorIndex);
                    }}
                    className={classNames(
                      'absolute left-3 top-3 z-20 flex h-8 w-8 items-center justify-center rounded-[4px] border shadow-lg shadow-black/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-300 disabled:cursor-not-allowed disabled:opacity-35',
                      activeImagePaletteSamplerIndex != null
                        ? 'border-brand-300 bg-brand-400 text-[var(--brand-ink)]'
                        : 'border-gray-800 bg-gray-950/95 text-gray-200 hover:border-brand-700 hover:text-brand-200',
                    )}
                    title={
                      activeImagePaletteSamplerIndex != null
                        ? 'Cancel image color sampling'
                        : 'Sample the selected palette color from the image'
                    }
                    aria-label={
                      activeImagePaletteSamplerIndex != null
                        ? 'Cancel image color sampling'
                        : 'Sample selected palette color from image'
                    }
                  >
                    <Pipette className="h-4 w-4" />
                  </button>
                ) : advancedMode === 'json' ? null : (
                  <div className="absolute bottom-4 left-4 z-20 flex h-9 items-center overflow-hidden rounded-[4px] border border-gray-800 bg-gray-950/95 text-xs text-gray-200 shadow-lg shadow-black/30">
                    <button
                      type="button"
                      onClick={() => setZoom(value => Math.max(0.5, Number((value - 0.25).toFixed(2))))}
                      className="flex h-9 w-9 items-center justify-center border-r border-gray-800 hover:bg-white/5"
                      title="Zoom out"
                    >
                      <Minus className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setZoom(value => (value >= 2 ? 1 : Number((value + 0.25).toFixed(2))))}
                      className="h-9 min-w-[54px] border-r border-gray-800 px-2 tabular-nums hover:bg-white/5"
                      title="Cycle zoom"
                    >
                      {Math.round(zoom * 100)}%
                    </button>
                    <button
                      type="button"
                      onClick={() => setZoom(value => Math.min(2, Number((value + 0.25).toFixed(2))))}
                      className="flex h-9 w-9 items-center justify-center border-r border-gray-800 hover:bg-white/5"
                      title="Zoom in"
                    >
                      <Plus className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setZoom(1)}
                      className="flex h-9 w-9 items-center justify-center hover:bg-white/5"
                      title="Fit image"
                    >
                      <Maximize2 className="h-4 w-4" />
                    </button>
                  </div>
                )}
              </main>

              <aside
                className={classNames(
                  'flex flex-none flex-col bg-gray-950',
                  advancedMode === 'boxes'
                    ? 'w-[clamp(410px,30.8vw,458px)] border-l border-gray-800'
                    : advancedMode === 'layers'
                      ? 'w-[clamp(410px,32.5vw,483px)] border-l border-gray-800'
                      : advancedMode === 'palettes'
                        ? 'w-[clamp(546px,40.7vw,586px)]'
                        : 'w-[clamp(700px,53.5vw,770px)] border-l border-gray-800',
                )}
              >
                {advancedMode === 'json' ? (
                  <StandaloneJsonWorkspace
                    value={jsonDraft}
                    validity={jsonDraftAnalysis.validity}
                    errors={jsonDraftAnalysis.errors}
                    warnings={jsonDraftAnalysis.warnings}
                    sections={jsonNavigatorAnalysis.sections}
                    elements={jsonNavigatorAnalysis.elements}
                    selectedSectionId={jsonSelectedSectionId}
                    selectedPath={jsonSelectedPath}
                    selectedElementIndex={selectedElementIndex}
                    disabled={jsonWorkspaceDisabled}
                    disabledReason={jsonWorkspaceDisabledReason}
                    canApply={jsonDraftAnalysis.validity === 'valid'}
                    canUndo={jsonDraft.length > 0}
                    canRedo={jsonDraft.length > 0}
                    onChange={setJsonDraft}
                    onApply={() => void applyJsonDraft()}
                    onUndo={undo}
                    onRedo={redo}
                    onFormat={formatJsonDraft}
                    onSelectSection={(sectionId, path) => {
                      setJsonSelectedSectionId(sectionId);
                      setJsonSelectedPath(path);
                      if (sectionId !== 'elements') setSelectedElementIndex(null);
                    }}
                    onSelectElement={(elementIndex, path) => {
                      setJsonSelectedSectionId('elements');
                      setJsonSelectedPath(path);
                      setSelectedElementIndex(elementIndex);
                    }}
                  />
                ) : advancedMode === 'palettes' ? (
                  <div className="h-full w-[546px] max-w-full flex-none">
                    <StandaloneImagePalettePanel
                      palette={paletteSwatches}
                      layers={paletteLayers}
                      reviewIssues={paletteReviewIssues}
                      confidence={paletteConfidence}
                      coverage={paletteCoverage}
                      selectedColor={selectedImagePaletteColor}
                      isExtracting={isExtractingPalette}
                      disabled={!canAnnotate}
                      disabledReason={
                        layerStructureDisabledReason || 'Image palettes require an Ideogram JSON image caption.'
                      }
                      extractionMessage={paletteExtractionMessage}
                      onExtractPalette={handleExtractImagePalette}
                      onSelectColor={color => {
                        setSelectedImagePaletteColor(color);
                        setPaletteExtractionMessage('');
                      }}
                      onMergeColors={issue => handleMergePaletteColors(issue.colors[0], issue.colors[1])}
                      onReviewColor={issue => {
                        const colorIndex = imagePaletteColors.indexOf(issue.color);
                        if (colorIndex >= 0) handleStartImagePaletteSample(colorIndex);
                      }}
                      onAssignColor={issue => handleAssignPaletteColor(issue.color)}
                      onToggleLayerColor={handleToggleLayerPaletteColor}
                    />
                  </div>
                ) : advancedMode === 'boxes' ? (
                  canAnnotate && captionParse.kind === 'ideogram' ? (
                    <>
                      <div className="flex h-[301px] flex-none flex-col overflow-hidden">
                        <StandaloneBoxesPanel
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
                          onAddBox={
                            layerStructureDisabledReason
                              ? undefined
                              : () => {
                                  setSelectedElementIndex(null);
                                  setActiveTool('box');
                                }
                          }
                          onReorderBox={layerStructureDisabledReason ? undefined : handleReorderLayer}
                        />
                      </div>
                      <StandaloneBoxDetailsPanel
                        selectedElement={selectedElement}
                        selectedElementIndex={selectedElementIndex}
                        selectedLayerColor={selectedLayerColor}
                        selectedRect={selectedRect}
                        selectedLocked={selectedElementIndex != null && lockedLayerIndexes.has(selectedElementIndex)}
                        mutationDisabledReason={layerStructureDisabledReason}
                        autoBoxesOpen={autoBoxesPanelOpen}
                        selectedImageSizeLabel={
                          selectedImageSize
                            ? `${selectedImageSize.width} × ${selectedImageSize.height}`
                            : 'Image size pending'
                        }
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
                        onSelectedFieldChange={handleSelectedFieldChange}
                        onSelectedTypeChange={handleSelectedTypeChange}
                        onChangeBox={handleChangeBox}
                        onToggleLocked={handleToggleLayerLocked}
                        onAutoBoxesOpenChange={setAutoBoxesPanelOpen}
                        onGenerateAutoBoxes={() => void handleGenerateAutoBoxes()}
                        onAutoBoxProviderChange={handleAutoBoxProviderChange}
                        onAutoBoxModelChange={setAutoBoxModel}
                        onRemoteWorkerChange={setRemoteOllamaWorkerId}
                        onAutoBoxRefineChange={setAutoBoxRefine}
                      />
                    </>
                  ) : (
                    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-4 px-8 text-center text-xs leading-5 text-gray-500">
                      <Braces className="h-6 w-6 text-gray-600" />
                      <span>{layerStructureDisabledReason || 'Boxes are unavailable for this asset.'}</span>
                      {canConvertDataset ? (
                        <button
                          type="button"
                          onClick={onConvertDatasetToJson}
                          className="h-9 rounded-[4px] border border-brand-800 bg-brand-950/20 px-3 text-xs text-brand-200 hover:bg-brand-950/40"
                        >
                          Convert dataset to JSON
                        </button>
                      ) : null}
                    </div>
                  )
                ) : (
                  <>
                    {canAnnotate && captionParse.kind === 'ideogram' ? (
                      <div className="flex h-[280px] flex-none flex-col overflow-hidden">
                        <LayersPanel
                          presentation="standalone"
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
                          onAddLayer={layerStructureDisabledReason ? undefined : () => void handleAddLayer()}
                          onReorderLayer={layerStructureDisabledReason ? undefined : handleReorderLayer}
                        />
                      </div>
                    ) : (
                      <div className="flex min-h-[220px] items-center justify-center border-b border-gray-800 px-6 text-center text-xs leading-5 text-gray-500">
                        {layerStructureDisabledReason || 'Layers are unavailable for this asset.'}
                      </div>
                    )}
                    <StandaloneLayerDetailsPanel
                      selectedElement={selectedElement}
                      selectedElementIndex={selectedElementIndex}
                      selectedLayerColor={selectedLayerColor}
                      selectedRect={selectedRect}
                      layerCaptionStatus={layerCaptionStatus}
                      selectedLayerIsCaptioning={selectedLayerIsCaptioning}
                      canCaptionSelectedLayer={canCaptionSelectedLayer}
                      layerCaptionDisabledReason={layerCaptionDisabledReason}
                      onSelectedFieldChange={handleSelectedFieldChange}
                      onSelectedTypeChange={handleSelectedTypeChange}
                      onChangeBox={handleChangeBox}
                      onCaptionSelectedLayer={() => void handleCaptionSelectedLayer()}
                    />
                  </>
                )}
              </aside>
            </div>

            <section
              className={classNames(
                'flex flex-none border-t border-gray-800 bg-gray-950',
                advancedMode === 'json' ? 'h-[174px]' : 'h-[203px]',
              )}
              aria-label={
                advancedMode === 'boxes'
                  ? 'Box caption workbench'
                  : advancedMode === 'layers'
                    ? 'Layer caption workbench'
                    : advancedMode === 'palettes'
                      ? 'Palette caption workbench'
                      : 'JSON caption workbench'
              }
            >
              <button
                type="button"
                onClick={closeAdvancedMode}
                className={classNames(
                  'flex flex-none items-center justify-center border-r border-gray-800 text-gray-500 hover:bg-gray-950 hover:text-gray-100',
                  advancedMode === 'json' ? 'w-[27px]' : 'w-[30px]',
                )}
                title="Return to gallery"
                aria-label="Return to gallery"
              >
                <ChevronDown className="h-4 w-4" />
              </button>

              <div className="operator-scrollbar-none min-w-0 flex-1 overflow-x-auto">
                <div
                  className={classNames(
                    'grid h-full py-3',
                    advancedMode === 'palettes'
                      ? 'min-w-[1320px] grid-cols-[320px_572px_minmax(454px,1fr)] gap-[15px] pl-[17px] pr-4'
                      : advancedMode === 'json'
                        ? 'min-w-[1386px] grid-cols-[308px_587px_minmax(456px,1fr)] gap-4 pl-[13px] pr-[17px]'
                        : 'min-w-[1320px] grid-cols-[337px_591px_minmax(420px,1fr)] gap-[17px] px-4',
                  )}
                >
                  <div
                    className={classNames(
                      'flex min-w-0 overflow-hidden',
                      advancedMode === 'palettes' ? 'gap-5' : advancedMode === 'json' ? 'gap-3' : 'gap-4',
                    )}
                  >
                    <div
                      className={classNames(
                        'h-full flex-none overflow-hidden rounded-[4px] border border-gray-800 bg-gray-950',
                        advancedMode === 'json' ? 'w-[98px]' : 'w-[126px]',
                      )}
                    >
                      {selectedItem.kind === 'plain' ? (
                        <PlainThumb
                          path={selectedItem.path}
                          mediaUrl={selectedItem.mediaUrl}
                          alt={selectedName}
                          onNaturalSizeChange={setSelectedImageSize}
                        />
                      ) : (
                        <EncryptedThumb
                          datasetName={datasetName}
                          workerID={workerID}
                          projectID={projectID}
                          cryptoKey={encryptedKey}
                          item={selectedItem.item}
                          onNaturalSizeChange={setSelectedImageSize}
                        />
                      )}
                    </div>
                    <div className="min-w-0 flex-1 py-1">
                      <h2 className="truncate text-[17px] font-semibold text-gray-100" title={selectedName}>
                        {selectedName}
                      </h2>
                      <div className="mt-2 text-xs tabular-nums text-gray-500">
                        {selectedImageSize
                          ? `${selectedImageSize.width} × ${selectedImageSize.height}`
                          : 'Dimensions pending'}
                      </div>
                      <div className="mt-3 flex items-center gap-2 text-xs">
                        <span
                          className={classNames('h-2 w-2 rounded-full', {
                            'bg-emerald-400': selectedApproved,
                            'bg-amber-300': !selectedApproved && captionText.trim(),
                            'bg-rose-400': !captionText.trim(),
                          })}
                        />
                        <span className={selectedApproved ? 'text-emerald-300' : 'text-gray-400'}>
                          {selectedApproved ? 'Approved' : captionText.trim() ? 'Captioned' : 'Missing caption'}
                        </span>
                      </div>
                      {advancedMode === 'json' ? (
                        <dl className="mt-3 grid grid-cols-[54px_1fr] gap-x-2 gap-y-1.5 text-[10px] text-gray-600">
                          <dt>Added</dt>
                          <dd className="truncate text-gray-400">{selectedAddedDateLabel}</dd>
                          <dt>Modified</dt>
                          <dd className="truncate text-gray-400">{selectedDateLabel}</dd>
                          <dt>File size</dt>
                          <dd className="text-gray-400">{formatByteSize(selectedSizeBytes)}</dd>
                        </dl>
                      ) : (
                        <dl className="mt-5 grid grid-cols-[60px_1fr] gap-x-2 gap-y-2 text-[10px] text-gray-600">
                          <dt>Modified</dt>
                          <dd className="truncate text-gray-400">{selectedDateLabel}</dd>
                          <dt>File size</dt>
                          <dd className="text-gray-400">{formatByteSize(selectedSizeBytes)}</dd>
                        </dl>
                      )}
                    </div>
                  </div>

                  <div className="flex min-w-0 flex-col overflow-hidden rounded-[4px] border border-gray-800 bg-gray-950">
                    <div className="flex h-10 flex-none items-center border-b border-gray-800 px-3">
                      <button
                        type="button"
                        onClick={() => setCaptionTab('caption')}
                        className={classNames('mr-4 h-10 border-b-2 text-xs font-medium', {
                          'border-brand-400 text-gray-100': advancedMode === 'json' || captionTab === 'caption',
                          'border-transparent text-gray-500 hover:text-gray-200':
                            advancedMode !== 'json' && captionTab !== 'caption',
                        })}
                      >
                        <span className={classNames('mr-2 inline-block h-2 w-2 rounded-full', captionStatus.dot)} />
                        Caption
                      </button>
                      {!isPlainTextItem ? (
                        <button
                          type="button"
                          onClick={() => switchAdvancedMode('json')}
                          className="h-10 border-b-2 border-transparent text-xs font-medium text-gray-500 hover:text-gray-200"
                        >
                          JSON
                        </button>
                      ) : null}
                      <span className="ml-auto text-[10px] text-gray-600">
                        {isSaving ? 'Saving…' : hasUnsavedChanges ? 'Unsaved changes' : 'Saved'}
                      </span>
                    </div>
                    <textarea
                      value={highLevelDescription}
                      onChange={event => handleCaptionDescriptionChange(event.target.value)}
                      disabled={!isCaptionLoaded || isAutoCaptioning || (advancedMode === 'json' && jsonDraftDirty)}
                      className="min-h-0 flex-1 resize-none bg-transparent px-4 py-3 text-sm leading-6 text-gray-100 outline-none disabled:opacity-50"
                      aria-label="Image caption"
                    />
                    <div className="flex h-7 flex-none items-center border-t border-gray-800 px-3 text-[10px] text-gray-600">
                      {captionWordCount.toLocaleString()} words
                      <span className="ml-auto">{hasUnsavedChanges ? 'Unsaved' : 'Saved just now'}</span>
                    </div>
                  </div>

                  <div
                    className={classNames(
                      'relative grid',
                      advancedMode === 'palettes'
                        ? 'min-w-[454px] grid-cols-[134px_40px_minmax(0,1fr)] grid-rows-[42px_41px_47px_1fr] gap-x-0 gap-y-[7px]'
                        : advancedMode === 'json'
                          ? 'min-w-[456px] grid-cols-[141px_66px_minmax(0,1fr)] grid-rows-[38px_37px_41px_1fr] gap-x-0 gap-y-[7px]'
                          : 'min-w-[420px] grid-cols-2 grid-rows-[44px_42px_46px_1fr] gap-2',
                    )}
                  >
                    <button
                      type="button"
                      onClick={() => void (advancedMode === 'json' ? saveJsonAndNext() : saveAndNext())}
                      disabled={
                        !isCaptionLoaded ||
                        isAutoCaptioning ||
                        isSaving ||
                        (advancedMode === 'json' && jsonDraftAnalysis.validity !== 'valid')
                      }
                      className={classNames(
                        'relative inline-flex items-center justify-center rounded-[4px] bg-brand-400 px-3 text-sm font-semibold text-[var(--brand-ink)] hover:bg-brand-300 disabled:cursor-not-allowed disabled:opacity-45',
                        advancedMode === 'palettes' || advancedMode === 'json' ? 'col-span-3' : 'col-span-2',
                      )}
                    >
                      <span className="inline-flex items-center gap-2">
                        {isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                        Save & Next
                      </span>
                      <span aria-hidden="true" className="absolute right-3 text-[10px] font-medium text-brand-950/70">
                        Ctrl + Enter
                      </span>
                    </button>

                    <div className="relative min-w-0">
                      <button
                        ref={toolsButtonRef}
                        type="button"
                        onClick={() => {
                          if (!toolsMenuOpen) updateToolsMenuPosition();
                          setToolsMenuOpen(open => !open);
                        }}
                        className="inline-flex h-full w-full items-center justify-center gap-2 rounded-[4px] border border-gray-800 bg-gray-950 text-xs text-gray-200 hover:bg-gray-950"
                        aria-expanded={toolsMenuOpen}
                        aria-haspopup="menu"
                        aria-controls="caption-studio-workspace-tools"
                      >
                        <Settings2 className="h-4 w-4" /> Tools
                        {toolsMenuOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                      </button>

                      {toolsMenuOpen && typeof document !== 'undefined'
                        ? createPortal(
                            <div
                              ref={toolsMenuRef}
                              id="caption-studio-workspace-tools"
                              role="menu"
                              aria-label="Workspace tools"
                              className="fixed z-[100] w-[190px] overflow-hidden rounded-[4px] border border-gray-800 bg-gray-900 p-1 shadow-xl shadow-black/50"
                              style={{ left: toolsMenuPosition.left, top: toolsMenuPosition.top }}
                            >
                              {ADVANCED_WORKSPACE_TOOLS.map(tool => {
                                const Icon = tool.icon;
                                return (
                                  <button
                                    key={tool.value}
                                    type="button"
                                    role="menuitem"
                                    onClick={() => switchAdvancedMode(tool.value)}
                                    className={classNames(
                                      'flex h-8 w-full items-center gap-2 rounded-[3px] px-2 text-left text-xs hover:bg-gray-800',
                                      tool.value === advancedMode ? 'text-brand-300' : 'text-gray-300',
                                    )}
                                  >
                                    <Icon className="h-4 w-4" /> {tool.label}
                                  </button>
                                );
                              })}
                              <button
                                type="button"
                                role="menuitem"
                                onClick={closeAdvancedMode}
                                className="mt-1 flex h-8 w-full items-center gap-2 border-t border-gray-800 px-2 pt-1 text-left text-xs text-gray-400 hover:text-white"
                              >
                                <X className="h-4 w-4" /> Return to gallery
                              </button>
                            </div>,
                            document.body,
                          )
                        : null}
                    </div>
                    <button
                      type="button"
                      onClick={() => void (advancedMode === 'json' ? toggleJsonApproval() : toggleSelectedApproval())}
                      disabled={!isCaptionLoaded || !captionText.trim()}
                      className={classNames(
                        'inline-flex items-center justify-center gap-2 rounded-[4px] border text-xs font-medium',
                        (advancedMode === 'palettes' || advancedMode === 'json') && 'col-span-2',
                        selectedApproved
                          ? 'border-emerald-500/70 bg-emerald-950/40 text-emerald-300'
                          : 'border-gray-800 bg-gray-950 text-gray-200 hover:text-emerald-300',
                      )}
                    >
                      <Check className="h-4 w-4" /> {selectedApproved ? 'Approved' : 'Approve'}
                    </button>

                    <button
                      type="button"
                      onClick={() =>
                        void (advancedMode === 'json'
                          ? selectJsonIndex(selectedIndex - 1)
                          : selectIndex(selectedIndex - 1))
                      }
                      disabled={selectedIndex === 0}
                      className={classNames(
                        'inline-flex items-center justify-center gap-2 rounded-[4px] border border-gray-800 bg-gray-950 text-xs text-gray-300 hover:bg-gray-950 disabled:opacity-35',
                        (advancedMode === 'palettes' || advancedMode === 'json') && 'col-span-2',
                      )}
                    >
                      <ChevronLeft className="h-4 w-4" /> Previous
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        void (advancedMode === 'json'
                          ? selectJsonIndex(selectedIndex + 1)
                          : selectIndex(selectedIndex + 1))
                      }
                      disabled={selectedIndex >= items.length - 1}
                      className={classNames(
                        'inline-flex items-center justify-center gap-2 rounded-[4px] border border-gray-800 bg-gray-950 text-xs text-gray-300 hover:bg-gray-950 disabled:opacity-35',
                        advancedMode === 'palettes' && 'ml-[9px]',
                        advancedMode === 'json' && 'ml-[10px]',
                      )}
                    >
                      Next <ChevronRight className="h-4 w-4" />
                    </button>
                    <div
                      className={classNames(
                        'flex items-start justify-center pt-1 text-[10px] tabular-nums text-gray-600',
                        advancedMode === 'palettes' || advancedMode === 'json' ? 'col-span-3' : 'col-span-2',
                      )}
                    >
                      {selectedIndex + 1} of {items.length.toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            </section>
            {advancedMode === 'palettes' ? (
              <div className="h-[15px] flex-none bg-gray-950" />
            ) : advancedMode === 'json' ? (
              <div className="h-[18px] flex-none bg-gray-950" />
            ) : null}
          </section>
        )}
      </div>

      <RecaptionSettingsModal
        isOpen={isRecaptionModalOpen}
        provider={recaptionProvider}
        outputFormat={recaptionOutputFormat}
        model={recaptionModel}
        modelOptions={recaptionModelOptions}
        maxNewTokens={recaptionMaxNewTokens}
        prompt={recaptionPrompt}
        systemPrompt={recaptionSystemPrompt}
        rootPrompt={recaptionRootPrompt}
        rootPromptStatus={recaptionRootPromptStatus}
        remoteWorkerId={recaptionRemoteWorkerId}
        remoteWorkerOptions={remoteWorkerOptions}
        remoteModelOptions={recaptionRemoteModelOptions}
        remoteModelStatus={recaptionRemoteModelStatus}
        remoteModelError={recaptionRemoteModelError}
        message={recaptionMessage}
        isRecaptioning={isRecaptioning}
        canQueueSelectedRecaption={canQueueSelectedRecaption}
        selectedRecaptionIsRunning={selectedRecaptionIsRunning}
        selectedRecaptionIsQueued={selectedRecaptionIsQueued}
        hasPendingRecaptions={hasPendingRecaptions}
        onClose={() => setIsRecaptionModalOpen(false)}
        onSubmit={queueSelectedRecaption}
        onProviderChange={handleRecaptionProviderChange}
        onOutputFormatChange={handleRecaptionOutputFormatChange}
        onModelChange={setRecaptionModel}
        onMaxNewTokensChange={setRecaptionMaxNewTokens}
        onPromptChange={setRecaptionPrompt}
        onSystemPromptChange={handleRecaptionSystemPromptChange}
        onUseRootPrompt={useRecaptionRootPrompt}
        onRemoteWorkerChange={setRecaptionRemoteWorkerId}
        onLoadRemoteModels={() => void loadRecaptionRemoteModels()}
      />
    </>
  );
}
