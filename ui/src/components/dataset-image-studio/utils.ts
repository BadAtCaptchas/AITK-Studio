import { isAudio, isVideo } from '@/utils/basic';
import { getDisplayPath } from '@/utils/media';
import {
  arrayToBox,
  boxToArray,
  parseIdeogramCaption,
  type IdeogramBox,
  type IdeogramElementType,
} from '@/utils/ideogramCaption';
import { BOX_COLORS, HEX_COLOR_PATTERN } from './constants';
import type { CaptionStatus, DatasetStudioItem } from './types';

export function itemKey(item: DatasetStudioItem) {
  return item.kind === 'plain' ? item.path : item.item.id;
}

export function itemName(item: DatasetStudioItem) {
  if (item.kind === 'encrypted') return item.item.name;
  const displayPath = getDisplayPath(item.path);
  if (displayPath !== item.path) return displayPath;
  return item.path.split(/[\\/]/).pop() || item.path;
}

export function itemKind(item: DatasetStudioItem) {
  if (item.kind === 'encrypted') return item.item.mediaKind;
  if (isAudio(item.path)) return 'audio';
  if (isVideo(item.path)) return 'video';
  return 'image';
}

export function clampIndex(value: number, length: number) {
  if (length <= 0) return 0;
  return Math.max(0, Math.min(length - 1, value));
}

export function statusForCaption(caption: string, loaded: boolean): CaptionStatus {
  if (!loaded) return { dot: 'bg-gray-500', label: '...', title: 'Caption not loaded' };
  if (!caption.trim()) return { dot: 'bg-rose-400', label: '0%', title: 'Missing caption' };
  const parsed = parseIdeogramCaption(caption);
  if (parsed.kind === 'ideogram') {
    return {
      dot: parsed.boxes.length > 0 ? 'bg-emerald-400' : 'bg-blue-400',
      label: parsed.boxes.length > 0 ? '100%' : 'JSON',
      title:
        parsed.boxes.length > 0
          ? `${parsed.boxes.length} box${parsed.boxes.length === 1 ? '' : 'es'}`
          : 'JSON caption',
    };
  }
  if (parsed.kind === 'json') return { dot: 'bg-amber-400', label: 'JSON', title: parsed.error };
  return { dot: 'bg-amber-400', label: 'TXT', title: 'Plain text caption' };
}

export function captionResponseToText(value: unknown) {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}

export function responseErrorMessage(error: any, fallback: string) {
  const responseError = error?.response?.data?.error;
  if (typeof responseError === 'string' && responseError.trim()) return responseError;
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}

export function normalizeHexColor(value: unknown) {
  const color = typeof value === 'string' ? value.trim().toUpperCase() : '';
  return HEX_COLOR_PATTERN.test(color) ? color : null;
}

function componentToHex(value: number) {
  return Math.max(0, Math.min(255, value)).toString(16).padStart(2, '0').toUpperCase();
}

export function sampleImageColorAt(image: HTMLImageElement, clientX: number, clientY: number) {
  const rect = image.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0 || image.naturalWidth <= 0 || image.naturalHeight <= 0) return null;

  const x = Math.max(0, Math.min(image.naturalWidth - 1, Math.floor(((clientX - rect.left) / rect.width) * image.naturalWidth)));
  const y = Math.max(0, Math.min(image.naturalHeight - 1, Math.floor(((clientY - rect.top) / rect.height) * image.naturalHeight)));
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) return null;

  context.drawImage(image, x, y, 1, 1, 0, 0, 1, 1);
  const [red, green, blue] = context.getImageData(0, 0, 1, 1).data;
  return `#${componentToHex(red)}${componentToHex(green)}${componentToHex(blue)}`;
}

export function resolveBoxColor(box: IdeogramBox, index: number, selected: boolean) {
  if (selected) return '#22D3EE';
  if (box.type === 'text') return '#F59E0B';
  return box.color || BOX_COLORS[index % BOX_COLORS.length];
}

export function layerLabelForElement(element: any, index: number) {
  const type = element?.type === 'text' ? 'text' : 'obj';
  const value = type === 'text' ? element?.text || element?.desc : element?.desc;
  const label = value == null ? '' : String(value).trim();
  return label || (type === 'text' ? `Text ${index + 1}` : `Object ${index + 1}`);
}

export function layerCaptionTargetText(element: any) {
  const desc = typeof element?.desc === 'string' ? element.desc.trim() : '';
  const text = typeof element?.text === 'string' ? element.text.trim() : '';
  return element?.type === 'text' ? text || desc : desc || text;
}

const LAYER_CAPTION_KEY_SEPARATOR = '\u0000';

export function layerCaptionRequestKey(itemKey: string, elementIndex: number) {
  return `${itemKey}${LAYER_CAPTION_KEY_SEPARATOR}${elementIndex}`;
}

export function isLayerCaptionRequestForItem(requestKey: string, itemKey: string) {
  return requestKey.startsWith(`${itemKey}${LAYER_CAPTION_KEY_SEPARATOR}`);
}

function layerFieldValue(value: unknown) {
  return typeof value === 'string' ? value.trim() : '';
}

function layerTypeForElement(element: any): IdeogramElementType {
  return element?.type === 'text' ? 'text' : 'obj';
}

function layerBoxKey(value: unknown) {
  const box = arrayToBox(value);
  return box ? boxToArray(box).join(',') : '';
}

export function pendingCaptionLayerStillMatches(currentElement: any, requestElement: any) {
  if (!currentElement || !requestElement) return false;
  if (layerTypeForElement(currentElement) !== layerTypeForElement(requestElement)) return false;
  if (layerFieldValue(currentElement.desc) !== layerFieldValue(requestElement.desc)) return false;
  if (layerFieldValue(currentElement.text) !== layerFieldValue(requestElement.text)) return false;
  const requestBoxKey = layerBoxKey(requestElement.bbox);
  return !requestBoxKey || layerBoxKey(currentElement.bbox) === requestBoxKey;
}

export function reindexLayerIndexSetAfterDelete(indexes: Set<number>, deletedIndex: number) {
  const next = new Set<number>();
  indexes.forEach(index => {
    if (index < deletedIndex) next.add(index);
    if (index > deletedIndex) next.add(index - 1);
  });
  return next;
}
