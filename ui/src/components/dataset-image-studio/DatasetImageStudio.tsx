'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
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
import { AnnotationLayer } from './AnnotationLayer';
import { BOX_COLORS, MAX_HISTORY } from './constants';
import { ImageNavigator } from './ImageNavigator';
import { CaptionEditorPanel, ObjectDetailsPanel } from './InspectorPanels';
import { LayersPanel } from './LayersPanel';
import { StudioMedia } from './StudioMedia';
import { StudioToolbar } from './StudioToolbar';
import { ToolRail } from './ToolRail';
import { appendImageSizeFields, createEncryptedImageFormData } from './openRouterMedia';
import type { CaptionCacheEntry, CaptionTab, DatasetImageStudioProps, ImageSize, ToolMode } from './types';
import {
  captionResponseToText,
  clampIndex,
  isLayerCaptionRequestForItem,
  itemKey,
  itemKind,
  itemName,
  layerCaptionRequestKey,
  layerCaptionTargetText,
  normalizeHexColor,
  pendingCaptionLayerStillMatches,
  reindexLayerIndexSetAfterDelete,
  reindexLayerIndexSetAfterInsert,
  responseErrorMessage,
  statusForCaption,
} from './utils';

export default function DatasetImageStudio({
  datasetName,
  workerID,
  datasetPath,
  items,
  isAutoCaptioning,
  encryptedKey,
  encryptedRawKeyB64,
  onRefresh,
  onAddImages,
  onConvertDatasetToJson,
  onSaveEncryptedCaption,
}: DatasetImageStudioProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [captionText, setCaptionText] = useState('');
  const [savedCaption, setSavedCaption] = useState('');
  const [isCaptionLoaded, setIsCaptionLoaded] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [activeTool, setActiveTool] = useState<ToolMode>('select');
  const [selectedElementIndex, setSelectedElementIndex] = useState<number | null>(null);
  const [captionTab, setCaptionTab] = useState<CaptionTab>('caption');
  const [zoom, setZoom] = useState(1);
  const [undoStack, setUndoStack] = useState<string[]>([]);
  const [redoStack, setRedoStack] = useState<string[]>([]);
  const [autoBoxModel, setAutoBoxModel] = useState('x-ai/grok-4.3');
  const [autoBoxRefine, setAutoBoxRefine] = useState(false);
  const [isGeneratingBoxes, setIsGeneratingBoxes] = useState(false);
  const [autoBoxMessage, setAutoBoxMessage] = useState('');
  const [captioningLayerKeys, setCaptioningLayerKeys] = useState<Set<string>>(() => new Set());
  const [layerCaptionMessages, setLayerCaptionMessages] = useState<Record<string, string>>({});
  const [encryptedOpenRouterConfirmed, setEncryptedOpenRouterConfirmed] = useState(false);
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
  const selectedKeyRef = useRef('');

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
  const captionParse = useMemo(() => parseIdeogramCaption(captionText), [captionText]);
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
          : !selectedImageSize
            ? 'Image size pending.'
            : !selectedElement || selectedElementIndex == null
              ? 'Select a layer.'
              : !selectedLayerHasCaptionTarget
                ? 'Add a layer label or draw a box first.'
                : '';
  const canCaptionSelectedLayer =
    !layerCaptionDisabledReason && !selectedLayerIsCaptioning && !isGeneratingBoxes && !isAutoCaptioning;

  useEffect(() => {
    latestCaptionRef.current = captionText;
  }, [captionText]);

  useEffect(() => {
    selectedKeyRef.current = selectedKey;
    setAutoBoxMessage('');
    setSelectedImageSize(null);
    setHiddenLayerIndexes(new Set());
    setLockedLayerIndexes(new Set());
    setOverlapElementStack([]);
    setActivePaletteSamplerIndex(null);
  }, [selectedKey]);

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
        let text = '';
        if (!selectedItem) return;
        if (selectedItem.kind === 'plain') {
          const response = await apiClient.post('/api/caption/get', { imgPath: selectedItem.path });
          text = captionResponseToText(response.data);
        } else if (encryptedKey) {
          const captionPath = selectedItem.item.captionObjectPath;
          if (captionPath) {
            const response = await apiClient.post(
              '/api/datasets/encrypted/object',
              { datasetName, worker_id: workerID, objectPath: captionPath },
              { responseType: 'blob' },
            );
            const decrypted = await decryptEncryptedObjectBlob(encryptedKey, captionPath, response.data as Blob);
            text = new TextDecoder().decode(decrypted);
          }
        }
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
  }, [datasetName, encryptedKey, selectedItem, selectedKey, workerID, writeCaptionCache]);

  useEffect(() => {
    if (!selectedKey) return;
    writeCaptionCache(selectedKey, { caption: captionText, saved: savedCaption, loaded: isCaptionLoaded });
  }, [captionText, isCaptionLoaded, savedCaption, selectedKey, writeCaptionCache]);

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

  const saveCaption = useCallback(async () => {
    if (!selectedItem || !isCaptionLoaded || isSaving || !isDirty) return;
    const value = captionText.trim();
    setIsSaving(true);
    try {
      if (selectedItem.kind === 'plain') {
        await apiClient.post('/api/img/caption', { imgPath: selectedItem.path, caption: value });
      } else if (encryptedKey && onSaveEncryptedCaption) {
        const key = itemKey(selectedItem);
        const targetCaptionPath =
          encryptedCaptionPaths[key] || selectedItem.item.captionObjectPath || captionObjectPath(randomId());
        const encryptedCaption = await encryptCaptionObject(encryptedKey, targetCaptionPath, value);
        await onSaveEncryptedCaption(selectedItem.item, targetCaptionPath, JSON.stringify(encryptedCaption));
        setEncryptedCaptionPaths(previous => ({ ...previous, [key]: targetCaptionPath }));
      }
      setSavedCaption(value);
      writeCaptionCache(selectedKey, { caption: value, saved: value, loaded: true });
    } catch (error) {
      console.error('Caption save failed:', error);
      alert('Failed to save caption. Please try again.');
    } finally {
      setIsSaving(false);
    }
  }, [
    captionText,
    encryptedCaptionPaths,
    encryptedKey,
    isCaptionLoaded,
    isDirty,
    isSaving,
    onSaveEncryptedCaption,
    selectedItem,
    selectedKey,
    writeCaptionCache,
  ]);

  useEffect(() => {
    saveCaptionRef.current = saveCaption;
  }, [saveCaption]);

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

  const mutateCaption = useCallback(
    (mutator: (data: Record<string, any>) => void, nextSelectedElementIndex?: number | null) => {
      const parsed = parseIdeogramCaption(captionText);
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
    [captionText],
  );

  const mutateLatestCaption = useCallback(
    (mutator: (data: Record<string, any>) => void, nextSelectedElementIndex?: number | null) => {
      const currentCaption = latestCaptionRef.current;
      const parsed = parseIdeogramCaption(currentCaption);
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
    [],
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
          '/api/datasets/openrouter-boxes',
          {
            imgPath: selectedItem.path,
            caption: requestCaption,
            model: autoBoxModel,
            refine: autoBoxRefine,
            imageWidth,
            imageHeight,
          },
          { timeout: 0 },
        );
      } else {
        if (!encryptedKey) throw new Error('Unlock the encrypted dataset first.');
        if (!encryptedOpenRouterConfirmed) {
          const confirmed = window.confirm(
            'Auto Boxes will send this decrypted image to OpenRouter to generate bounding boxes. Continue?',
          );
          if (!confirmed) {
            setAutoBoxMessage('Auto Boxes canceled.');
            return;
          }
          setEncryptedOpenRouterConfirmed(true);
        }

        const formData = await createEncryptedImageFormData({ datasetName, workerID, encryptedKey, item: selectedItem.item });
        formData.append('caption', requestCaption);
        formData.append('model', autoBoxModel);
        formData.append('refine', autoBoxRefine ? 'true' : 'false');
        appendImageSizeFields(formData, imageWidth, imageHeight);

        response = await apiClient.post('/api/datasets/openrouter-boxes', formData, {
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
        throw new Error('OpenRouter did not return any usable boxes.');
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
    autoBoxModel,
    autoBoxRefine,
    captionParse,
    captionText,
    datasetName,
    encryptedKey,
    encryptedOpenRouterConfirmed,
    isGeneratingBoxes,
    mutateCaption,
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
          '/api/datasets/openrouter-layer-caption',
          {
            imgPath: selectedItem.path,
            caption: requestCaption,
            elementIndex: requestElementIndex,
            model: autoBoxModel,
            imageWidth,
            imageHeight,
          },
          { timeout: 0 },
        );
      } else {
        if (!encryptedKey) throw new Error('Unlock the encrypted dataset first.');
        if (!encryptedOpenRouterConfirmed) {
          const confirmed = window.confirm(
            'Caption Layer will send this decrypted image to OpenRouter to caption the selected layer. Continue?',
          );
          if (!confirmed) {
            setLayerCaptionMessageForKey(requestLayerKey, 'Caption Layer canceled.');
            return;
          }
          setEncryptedOpenRouterConfirmed(true);
        }

        const formData = await createEncryptedImageFormData({ datasetName, workerID, encryptedKey, item: selectedItem.item });
        formData.append('caption', requestCaption);
        formData.append('elementIndex', String(requestElementIndex));
        formData.append('model', autoBoxModel);
        appendImageSizeFields(formData, imageWidth, imageHeight);

        response = await apiClient.post('/api/datasets/openrouter-layer-caption', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 0,
        });
      }

      if (selectedKeyRef.current !== requestKey) {
        return;
      }
      const latestParsed = parseIdeogramCaption(latestCaptionRef.current);
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
      if (!desc) throw new Error('OpenRouter did not return a usable layer caption.');
      if (!currentHasBox && !generatedBox) throw new Error('OpenRouter did not return a usable layer box.');

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
    captionText,
    datasetName,
    encryptedKey,
    encryptedOpenRouterConfirmed,
    layerCaptionDisabledReason,
    mutateLatestCaption,
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
      if (isTyping) return;
      if (event.key === 'ArrowLeft') selectIndex(selectedIndex - 1);
      if (event.key === 'ArrowRight') selectIndex(selectedIndex + 1);
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
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        undo();
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
    handleDeleteSelectedElement,
    redo,
    saveCaption,
    selectIndex,
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
        <div className="border border-dashed border-gray-700 bg-gray-900/60 px-6 py-5 text-sm">No media found.</div>
      </div>
    );
  }

  if (!selectedItem) return null;

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#02060a] text-gray-100">
      <StudioToolbar
        selectedIndex={selectedIndex}
        itemCount={items.length}
        isSaving={isSaving}
        isDirty={isDirty}
        zoom={zoom}
        onPrevious={() => selectIndex(selectedIndex - 1)}
        onNext={() => selectIndex(selectedIndex + 1)}
        onCycleZoom={() => setZoom(value => (value >= 1.5 ? 1 : Number((value + 0.25).toFixed(2))))}
        onPan={() => setActiveTool('pan')}
        onFit={() => setZoom(1)}
      />

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <ToolRail
          activeTool={activeTool}
          canAnnotate={canAnnotate}
          hasSelection={selectedElementIndex != null}
          canUndo={undoStack.length > 0}
          canRedo={redoStack.length > 0}
          onToolChange={setActiveTool}
          onDelete={handleDeleteSelectedElement}
          onUndo={undo}
          onRedo={redo}
          onShowJson={() => setCaptionTab('json')}
        />

        <div className="flex min-h-0 flex-1 flex-col xl:flex-row">
          <main className="relative flex min-h-0 flex-1 flex-col bg-[#03070b]">
            <div className="relative flex min-h-[260px] flex-1 items-stretch justify-stretch overflow-hidden">
              <div className="absolute left-3 top-3 z-10 max-w-[calc(100%-1.5rem)] truncate rounded-md border border-gray-800 bg-gray-950/80 px-2 py-1 text-xs text-gray-300 backdrop-blur">
                {selectedName}
              </div>
              <StudioMedia
                item={selectedItem}
                datasetName={datasetName}
                workerID={workerID}
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
              encryptedKey={encryptedKey}
              captionCache={captionCacheRef.current}
              captionCacheVersion={captionCacheVersion}
              onCaptionCacheChange={bumpCaptionCacheVersion}
              onSelectIndex={selectIndex}
            />
          </main>

          <aside className="flex max-h-[34dvh] min-h-[190px] flex-shrink-0 flex-col overflow-hidden border-t border-gray-900 bg-[#080d12] xl:max-h-none xl:min-h-0 xl:w-[410px] xl:border-l xl:border-t-0">
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
                selectedImageSize={selectedImageSize}
                canGenerateAutoBoxes={canGenerateAutoBoxes}
                autoBoxDisabledReason={autoBoxDisabledReason}
                autoBoxModel={autoBoxModel}
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
                onAutoBoxModelChange={setAutoBoxModel}
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
                isAutoCaptioning={isAutoCaptioning}
                isCaptionLoaded={isCaptionLoaded}
                isDirty={isDirty}
                isSaving={isSaving}
                onCaptionTabChange={setCaptionTab}
                onCaptionDescriptionChange={handleCaptionDescriptionChange}
                onCaptionTextChange={handleCaptionTextChange}
                onSave={() => void saveCaption()}
              />
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
