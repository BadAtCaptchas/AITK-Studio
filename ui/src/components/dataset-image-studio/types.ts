import type { EncryptedDatasetItem } from '@/types';

export type DatasetStudioItem =
  | {
      kind: 'plain';
      path: string;
    }
  | {
      kind: 'encrypted';
      item: EncryptedDatasetItem;
    };

export type ToolMode = 'box' | 'text' | 'select' | 'move' | 'pan';
export type CaptionTab = 'caption' | 'json';
export type ImageSize = { width: number; height: number };
export type BoxRect = { x: number; y: number; w: number; h: number };
export type CaptionStatus = { dot: string; label: string; title: string };
export type CaptionCacheEntry = { caption: string; saved: string; loaded: boolean };
export type CaptionKeywordMatchMode = 'whole-word' | 'partial';
export type BulkCaptionAction = 'delete' | 'move' | 'remove_words';

export type BulkCaptionMatch = {
  key: string;
  index: number;
  item: DatasetStudioItem;
  caption: string;
};

export type BulkCaptionActionRequest = {
  action: BulkCaptionAction;
  query: string;
  matchMode: CaptionKeywordMatchMode;
  destinationName?: string;
  matches: BulkCaptionMatch[];
};

export type BulkCaptionActionResult = {
  action: BulkCaptionAction;
  found: number;
  affected: number;
  deleted?: number;
  moved?: number;
  updated?: number;
  removedWords?: number;
  destinationName?: string;
  updatedCaptions?: Record<string, string>;
  removedKeys?: string[];
};

export type DatasetImageStudioProps = {
  datasetName: string;
  workerID: string;
  datasetPath?: string | null;
  items: DatasetStudioItem[];
  isAutoCaptioning: boolean;
  encryptedKey?: CryptoKey | null;
  encryptedRawKeyB64?: string | null;
  onRefresh?: () => void;
  onAddImages: () => void;
  onConvertDatasetToJson?: () => void;
  onBulkEncryptedCaptionAction?: (request: BulkCaptionActionRequest) => Promise<BulkCaptionActionResult>;
  onSaveEncryptedCaption?: (
    item: EncryptedDatasetItem,
    captionObjectPath: string,
    encryptedCaptionJson: string,
  ) => Promise<void>;
};
