'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import useDatasetWatcherLiveRefresh from '@/hooks/useDatasetWatcherLiveRefresh';
import { Button } from '@headlessui/react';
import {
  AlertTriangle,
  CheckCircle2,
  CloudDownload,
  Database,
  Download,
  Grid2X2,
  FolderPlus,
  HelpCircle,
  KeyRound,
  Layers,
  List,
  Loader2,
  LockKeyhole,
  MinusCircle,
  Pencil,
  Plus,
  Search,
  Trash2,
} from 'lucide-react';
import { openConfirm } from '@/components/ConfirmModal';
import DatasetFolderIcon from '@/components/DatasetFolderIcon';
import DatasetWatchFoldersButton from '@/components/DatasetWatchFoldersButton';
import DatasetWatcherProgressBadge from '@/components/DatasetWatcherProgressBadge';
import { TopBar, MainContent } from '@/components/layout';
import { PageNotice } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';
import { aggregateAutoCaptionProgressByDataset } from '@/utils/datasetWatcherStatus';
import {
  buildEncryptedDatasetItem,
  createEmptyEncryptedManifest,
  decryptCatalog,
  encryptCatalog,
  getMediaKind,
  getRememberedEncryptedDatasetKey,
  readRootCaptionFile,
  rememberEncryptedDatasetKey,
  unlockEncryptedDatasetKey,
  type DatasetCredentialMode,
} from '@/utils/encryptedDatasets';
import {
  createFlattenedFileNameAllocator,
  folderImportCaptionKey,
  folderImportExtension,
  folderImportRootName,
  FOLDER_IMPORT_SUPPORTED_EXTENSIONS,
  isFolderImportCaptionSidecarPath,
  stripFolderImportRoot,
} from '@/utils/folderImport';
import { isImage } from '@/utils/basic';
import { getDisplayPath, getMediaUrl } from '@/utils/media';
import { makeRemoteDatasetRef, remoteDatasetRememberKey } from '@/utils/remoteDatasetRefs';
import type { DatasetSummary, EncryptedDatasetCatalog, EncryptedDatasetManifest } from '@/types';

type DatasetExplorerView = 'details' | 'icons';

type DatasetExplorerRow = {
  dataset: DatasetSummary;
  name: string;
  encrypted: boolean;
  unlocked: boolean;
  source: 'local' | 'remote';
  worker: string;
  captions: string;
  ref: string;
  worker_id: string;
};

const DATASET_VIEW_STORAGE_KEY = 'aitk.datasets.view';

function isPreviewableImagePath(value: string) {
  return isImage(value) || isImage(getDisplayPath(value));
}

function parseDatasetExplorerView(value: string | null | undefined): DatasetExplorerView | null {
  return value === 'details' || value === 'icons' ? value : null;
}

function readDatasetViewCookie() {
  if (typeof document === 'undefined') return null;
  const cookie = document.cookie
    .split('; ')
    .find(value => value.startsWith(`${DATASET_VIEW_STORAGE_KEY}=`));
  if (!cookie) return null;
  return parseDatasetExplorerView(decodeURIComponent(cookie.split('=').slice(1).join('=')));
}

function readStoredDatasetView() {
  if (typeof window === 'undefined') return null;
  try {
    return parseDatasetExplorerView(window.localStorage.getItem(DATASET_VIEW_STORAGE_KEY)) || readDatasetViewCookie();
  } catch {
    return readDatasetViewCookie();
  }
}

function writeStoredDatasetView(view: DatasetExplorerView) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(DATASET_VIEW_STORAGE_KEY, view);
  } catch {
    // Keep the cookie fallback below for private or restricted storage modes.
  }
  try {
    document.cookie = `${DATASET_VIEW_STORAGE_KEY}=${encodeURIComponent(view)}; Max-Age=31536000; Path=/; SameSite=Lax`;
  } catch {
    // If both storage mechanisms are blocked, the current in-memory state still updates.
  }
}

function datasetRowKey(dataset: DatasetSummary) {
  return dataset.ref || `${dataset.worker_id || 'local'}:${dataset.name}`;
}

function datasetWorkerID(dataset: DatasetSummary) {
  return dataset.worker_id || 'local';
}

function getRememberedDatasetKey(dataset: DatasetSummary) {
  const workerID = datasetWorkerID(dataset);
  return (
    getRememberedEncryptedDatasetKey(datasetRowKey(dataset)) ||
    getRememberedEncryptedDatasetKey(dataset.name) ||
    (dataset.path ? getRememberedEncryptedDatasetKey(dataset.path) : null) ||
    (workerID !== 'local' ? getRememberedEncryptedDatasetKey(remoteDatasetRememberKey(workerID, dataset.name)) : null)
  );
}

function rememberDatasetKey(dataset: DatasetSummary, rawKeyB64: string) {
  const workerID = datasetWorkerID(dataset);
  rememberEncryptedDatasetKey(datasetRowKey(dataset), rawKeyB64);
  rememberEncryptedDatasetKey(dataset.name, rawKeyB64);
  if (dataset.path) rememberEncryptedDatasetKey(dataset.path, rawKeyB64);
  if (workerID !== 'local') {
    rememberEncryptedDatasetKey(makeRemoteDatasetRef(workerID, dataset.name), rawKeyB64);
    rememberEncryptedDatasetKey(remoteDatasetRememberKey(workerID, dataset.name), rawKeyB64);
  }
}

function captionStatusSearchText(dataset: DatasetSummary, unlocked = false) {
  if (dataset.encrypted) return unlocked ? 'captions unlocked encrypted' : 'captions locked encrypted';
  if (typeof dataset.itemCount !== 'number' || typeof dataset.missingCaptionCount !== 'number') {
    return 'captions not scanned unknown';
  }
  if (dataset.itemCount === 0) return 'captions no media empty';
  if (dataset.missingCaptionCount > 0) return `${dataset.missingCaptionCount} missing captions`;
  return 'captions complete captioned';
}

function CaptionStatusBadge({ dataset, unlocked = false }: { dataset: DatasetSummary; unlocked?: boolean }) {
  const itemCount = typeof dataset.itemCount === 'number' ? dataset.itemCount : null;
  const missingCount = typeof dataset.missingCaptionCount === 'number' ? dataset.missingCaptionCount : null;
  const captionedCount =
    typeof dataset.captionedItemCount === 'number'
      ? dataset.captionedItemCount
      : itemCount !== null && missingCount !== null
        ? Math.max(itemCount - missingCount, 0)
        : null;

  if (dataset.encrypted && unlocked) {
    return (
      <span
        className="inline-flex w-fit items-center gap-1.5 rounded-md border border-emerald-500/35 bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-100"
        title="This encrypted dataset is unlocked in the current browser session"
      >
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
        Unlocked
      </span>
    );
  }

  if (dataset.encrypted) {
    return (
      <span
        className="inline-flex w-fit items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900/70 px-2 py-0.5 text-xs font-medium text-slate-300"
        title="Caption counts are encrypted until the dataset is unlocked"
      >
        <LockKeyhole className="h-3.5 w-3.5 text-slate-400" />
        Locked
      </span>
    );
  }

  if (itemCount === null || missingCount === null || captionedCount === null) {
    return (
      <span
        className="inline-flex w-fit items-center gap-1.5 rounded-md border border-gray-700 bg-gray-900 px-2 py-0.5 text-xs font-medium text-gray-300"
        title="Caption status is not available from this worker"
      >
        <HelpCircle className="h-3.5 w-3.5 text-gray-400" />
        Not scanned
      </span>
    );
  }

  if (itemCount === 0) {
    return (
      <span className="inline-flex w-fit items-center gap-1.5 rounded-md border border-gray-700 bg-gray-900 px-2 py-0.5 text-xs font-medium text-gray-300">
        <MinusCircle className="h-3.5 w-3.5 text-gray-400" />
        No media
      </span>
    );
  }

  const hasMissingCaptions = missingCount > 0;
  const coveragePercent = Math.max(0, Math.min(100, (captionedCount / itemCount) * 100));

  return (
    <div
      className="flex min-w-[9.5rem] flex-col gap-1"
      title={
        hasMissingCaptions
          ? `${missingCount} of ${itemCount} media item${itemCount === 1 ? '' : 's'} are missing caption sidecars`
          : `All ${itemCount} media item${itemCount === 1 ? '' : 's'} have caption sidecars`
      }
    >
      <span
        className={
          hasMissingCaptions
            ? 'inline-flex w-fit items-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs font-medium text-amber-100'
            : 'inline-flex w-fit items-center gap-1.5 rounded-md border border-emerald-500/35 bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-100'
        }
      >
        {hasMissingCaptions ? (
          <AlertTriangle className="h-3.5 w-3.5 text-amber-300" />
        ) : (
          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
        )}
        <span>{hasMissingCaptions ? `${missingCount} missing` : 'Complete'}</span>
        <span className={hasMissingCaptions ? 'text-amber-200/70' : 'text-emerald-200/70'}>of {itemCount}</span>
      </span>
      <span className="block h-1.5 w-28 overflow-hidden rounded-full bg-gray-800" aria-hidden="true">
        <span
          className={`block h-full rounded-full ${hasMissingCaptions ? 'bg-amber-400' : 'bg-emerald-400'}`}
          style={{ width: `${hasMissingCaptions ? Math.max(coveragePercent, 6) : 100}%` }}
        />
      </span>
    </div>
  );
}

type FolderImportEntry = {
  id: string;
  file: File;
  relativePath: string;
  rootName: string;
};

type HfDatasetPreview = {
  datasetID: string;
  configs: string[];
  splits: string[];
  selectedConfig: string;
  selectedSplit: string;
  rowCount?: number | null;
  features: Array<{ name: string; kind: string }>;
  imageColumns: string[];
  textColumns: string[];
  suggestedImageColumn: string | null;
  suggestedCaptionColumn: string | null;
  samples: Array<Record<string, unknown>>;
};

type HfCaptionMode = 'auto' | 'none' | 'column';

type BulkUnlockStatus = 'loading' | 'locked' | 'unlocking' | 'unlocked' | 'error';

function cleanClientDatasetName(name: string) {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function relativePathForFile(file: File) {
  return ((file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name).replace(/\\/g, '/');
}

function fileSystemPathForFile(file: File) {
  const value = (file as File & { path?: unknown }).path;
  return typeof value === 'string' ? value.trim() : '';
}

function dirnameFromClientPath(filePath: string) {
  const trimmed = filePath.trim().replace(/[\\/]+$/, '');
  const lastSlash = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
  return lastSlash > 0 ? trimmed.slice(0, lastSlash) : '';
}

function restoreClientPathSeparators(normalizedPath: string, sourcePath: string) {
  if (sourcePath.includes('\\')) return normalizedPath.replace(/\//g, '\\');
  return normalizedPath;
}

function sourceFolderPathForEntry(entry: FolderImportEntry) {
  const filePath = fileSystemPathForFile(entry.file);
  if (!filePath) return '';
  const relativeInsideSource = stripFolderImportRoot(entry.relativePath, entry.file.name).replace(/^\/+/, '');
  const normalizedFilePath = filePath.replace(/\\/g, '/').replace(/\/+$/, '');
  const normalizedRelativePath = relativeInsideSource.replace(/\\/g, '/').replace(/^\/+/, '');
  const suffix = normalizedRelativePath ? `/${normalizedRelativePath}` : '';

  if (suffix && normalizedFilePath.toLowerCase().endsWith(suffix.toLowerCase())) {
    return restoreClientPathSeparators(normalizedFilePath.slice(0, -suffix.length), filePath);
  }

  return dirnameFromClientPath(filePath);
}

function sourceFolderPathForEntries(entries: FolderImportEntry[]) {
  const sourcePaths = Array.from(
    new Set(
      entries
        .map(sourceFolderPathForEntry)
        .map(value => value.trim())
        .filter(Boolean),
    ),
  );
  return sourcePaths.length === 1 ? sourcePaths[0] : '';
}

function nextClientDatasetName(preferredName: string, usedNames: Set<string>) {
  const baseName = cleanClientDatasetName(preferredName) || 'imported_folder';
  let candidate = baseName;
  let suffix = 2;
  while (usedNames.has(candidate.toLowerCase())) {
    candidate = `${baseName}_${suffix}`;
    suffix += 1;
  }
  usedNames.add(candidate.toLowerCase());
  return candidate;
}

function hfOutputNameFromDataset(datasetID: string, split?: string) {
  const base = cleanClientDatasetName(datasetID.replace(/^https?:\/\/(?:www\.)?huggingface\.co\/datasets\//i, ''));
  const splitSuffix = split && split !== 'train' ? `_${cleanClientDatasetName(split)}` : '';
  return `${base || 'hf_dataset'}${splitSuffix}`;
}

export default function Datasets() {
  const router = useRouter();
  const folderImportInputRef = useRef<HTMLInputElement | null>(null);
  const setFolderImportInputRef = useCallback((node: HTMLInputElement | null) => {
    folderImportInputRef.current = node;
    if (!node) return;
    node.setAttribute('webkitdirectory', '');
    node.setAttribute('directory', '');
  }, []);
  const { datasets, errors, status, refreshDatasets } = useDatasetList({ includeRemote: true });
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [datasetFilter, setDatasetFilter] = useState('');
  const [datasetView, setDatasetView] = useState<DatasetExplorerView>('details');
  const [datasetPreviewUrls, setDatasetPreviewUrls] = useState<Record<string, string | null>>({});
  const datasetPreviewRequestsRef = useRef<Set<string>>(new Set());
  const [newDatasetMode, setNewDatasetMode] = useState<'plain' | 'encrypted'>('plain');
  const [credentialMode, setCredentialMode] = useState<DatasetCredentialMode>('password');
  const [datasetPassword, setDatasetPassword] = useState('');
  const [datasetPasswordConfirm, setDatasetPasswordConfirm] = useState('');
  const [datasetKeyFile, setDatasetKeyFile] = useState<File | null>(null);
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);
  const [renameDataset, setRenameDataset] = useState<DatasetSummary | null>(null);
  const [renameDatasetName, setRenameDatasetName] = useState('');
  const [isRenamingDataset, setIsRenamingDataset] = useState(false);
  const [renameDatasetError, setRenameDatasetError] = useState('');
  const [importingRef, setImportingRef] = useState<string | null>(null);
  const [selectedDatasetRefs, setSelectedDatasetRefs] = useState<Set<string>>(() => new Set());
  const [unlockedDatasetRefs, setUnlockedDatasetRefs] = useState<Set<string>>(() => new Set());
  const [isBulkUnlockModalOpen, setIsBulkUnlockModalOpen] = useState(false);
  const [bulkUnlockTargets, setBulkUnlockTargets] = useState<DatasetSummary[]>([]);
  const [bulkUnlockManifests, setBulkUnlockManifests] = useState<Record<string, EncryptedDatasetManifest>>({});
  const [bulkUnlockStatus, setBulkUnlockStatus] = useState<Record<string, BulkUnlockStatus>>({});
  const [bulkUnlockErrors, setBulkUnlockErrors] = useState<Record<string, string>>({});
  const [bulkSharedPassword, setBulkSharedPassword] = useState('');
  const [bulkRowPasswords, setBulkRowPasswords] = useState<Record<string, string>>({});
  const [bulkRowKeyFiles, setBulkRowKeyFiles] = useState<Record<string, File | null>>({});
  const [isBulkUnlocking, setIsBulkUnlocking] = useState(false);
  const [isCombineModalOpen, setIsCombineModalOpen] = useState(false);
  const [combineSources, setCombineSources] = useState<DatasetSummary[]>([]);
  const [combineOutputName, setCombineOutputName] = useState('');
  const [combineOutputMode, setCombineOutputMode] = useState<'plain' | 'encrypted'>('plain');
  const [combineCredentialMode, setCombineCredentialMode] = useState<DatasetCredentialMode>('password');
  const [combinePassword, setCombinePassword] = useState('');
  const [combinePasswordConfirm, setCombinePasswordConfirm] = useState('');
  const [combineKeyFile, setCombineKeyFile] = useState<File | null>(null);
  const [combineSourceKeys, setCombineSourceKeys] = useState<Record<string, string>>({});
  const [combineSourceManifests, setCombineSourceManifests] = useState<Record<string, EncryptedDatasetManifest>>({});
  const [combineSourcePasswords, setCombineSourcePasswords] = useState<Record<string, string>>({});
  const [combineSourceKeyFiles, setCombineSourceKeyFiles] = useState<Record<string, File | null>>({});
  const [combineSourceErrors, setCombineSourceErrors] = useState<Record<string, string>>({});
  const [combineSourceLoading, setCombineSourceLoading] = useState<Record<string, boolean>>({});
  const [isCombiningDatasets, setIsCombiningDatasets] = useState(false);
  const [isFolderImportModalOpen, setIsFolderImportModalOpen] = useState(false);
  const [folderImportEntries, setFolderImportEntries] = useState<FolderImportEntry[]>([]);
  const [folderImportMode, setFolderImportMode] = useState<'separate' | 'combined'>('separate');
  const [folderImportOutputMode, setFolderImportOutputMode] = useState<'plain' | 'encrypted'>('plain');
  const [folderImportCredentialMode, setFolderImportCredentialMode] = useState<DatasetCredentialMode>('password');
  const [folderImportPassword, setFolderImportPassword] = useState('');
  const [folderImportPasswordConfirm, setFolderImportPasswordConfirm] = useState('');
  const [folderImportKeyFile, setFolderImportKeyFile] = useState<File | null>(null);
  const [folderImportWorkerID, setFolderImportWorkerID] = useState('local');
  const [folderImportDatasetName, setFolderImportDatasetName] = useState('');
  const [isImportingFolders, setIsImportingFolders] = useState(false);
  const [folderImportStatus, setFolderImportStatus] = useState('');
  const [isHfImportModalOpen, setIsHfImportModalOpen] = useState(false);
  const [hfDatasetInput, setHfDatasetInput] = useState('');
  const [hfImportWorkerID, setHfImportWorkerID] = useState('local');
  const [hfPreview, setHfPreview] = useState<HfDatasetPreview | null>(null);
  const [hfConfig, setHfConfig] = useState('');
  const [hfSplit, setHfSplit] = useState('');
  const [hfImageColumn, setHfImageColumn] = useState('');
  const [hfCaptionMode, setHfCaptionMode] = useState<HfCaptionMode>('auto');
  const [hfCaptionColumn, setHfCaptionColumn] = useState('');
  const [hfOutputName, setHfOutputName] = useState('');
  const [hfMaxRows, setHfMaxRows] = useState('');
  const [hfImportStatus, setHfImportStatus] = useState('');
  const [hfImportError, setHfImportError] = useState('');
  const [isLoadingHfPreview, setIsLoadingHfPreview] = useState(false);
  const [isImportingHfDataset, setIsImportingHfDataset] = useState(false);

  const datasetWatcherLive = useDatasetWatcherLiveRefresh({
    enabled: status === 'success',
    workerID: 'local',
    onRefresh: () => refreshDatasets({ background: true }),
  });
  const autoCaptionProgressByDataset = useMemo(
    () => aggregateAutoCaptionProgressByDataset(datasetWatcherLive.watchers, datasetWatcherLive.statuses),
    [datasetWatcherLive.statuses, datasetWatcherLive.watchers],
  );

  useEffect(() => {
    const rememberedView = readStoredDatasetView();
    if (rememberedView) setDatasetView(rememberedView);

    const handleStorage = (event: StorageEvent) => {
      if (event.key !== DATASET_VIEW_STORAGE_KEY) return;
      const nextView = parseDatasetExplorerView(event.newValue);
      if (nextView) setDatasetView(nextView);
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const updateDatasetView = useCallback((view: DatasetExplorerView) => {
    setDatasetView(view);
    writeStoredDatasetView(view);
  }, []);

  const isDatasetUnlocked = useCallback(
    (dataset: DatasetSummary) =>
      dataset.encrypted && (unlockedDatasetRefs.has(datasetRowKey(dataset)) || !!getRememberedDatasetKey(dataset)),
    [unlockedDatasetRefs],
  );

  const rememberUnlockedDataset = useCallback((dataset: DatasetSummary, rawKeyB64: string) => {
    rememberDatasetKey(dataset, rawKeyB64);
    setUnlockedDatasetRefs(previous => {
      const next = new Set(previous);
      next.add(datasetRowKey(dataset));
      return next;
    });
  }, []);

  // Transform datasets array into rows with objects
  const tableRows: DatasetExplorerRow[] = datasets.map(dataset => {
    const unlocked = isDatasetUnlocked(dataset);
    return {
      dataset,
      name: dataset.name,
      encrypted: dataset.encrypted,
      unlocked,
      source: dataset.source === 'remote' ? 'remote' : 'local',
      worker: dataset.worker_name || 'Local',
      captions: captionStatusSearchText(dataset, unlocked),
      ref: datasetRowKey(dataset),
      worker_id: datasetWorkerID(dataset),
    };
  });
  const filteredTableRows = useMemo(() => {
    const query = datasetFilter.trim().toLowerCase();
    if (!query) return tableRows;
    return tableRows.filter(row =>
      [row.name, row.source, row.worker, row.encrypted ? 'encrypted' : 'plain', row.captions]
        .filter(Boolean)
        .some(value => `${value}`.toLowerCase().includes(query)),
    );
  }, [datasetFilter, tableRows]);

  useEffect(() => {
    filteredTableRows
      .filter(row => !row.encrypted && !datasetPreviewRequestsRef.current.has(row.ref))
      .slice(0, 80)
      .forEach(row => {
        datasetPreviewRequestsRef.current.add(row.ref);
        apiClient
          .post('/api/datasets/listImages', { datasetName: row.name, worker_id: row.worker_id })
          .then(response => {
            const images = Array.isArray(response.data?.images) ? response.data.images : [];
            const firstImagePath = images
              .map((image: { img_path?: unknown }) => (typeof image.img_path === 'string' ? image.img_path : ''))
              .find((imagePath: string) => imagePath && isPreviewableImagePath(imagePath));
            setDatasetPreviewUrls(previous => ({
              ...previous,
              [row.ref]: firstImagePath ? getMediaUrl(firstImagePath) : null,
            }));
          })
          .catch(() => {
            setDatasetPreviewUrls(previous => ({ ...previous, [row.ref]: null }));
          });
      });
  }, [filteredTableRows]);

  const selectedDatasets = useMemo(
    () => tableRows.filter(row => selectedDatasetRefs.has(row.ref)).map(row => row.dataset),
    [selectedDatasetRefs, tableRows],
  );
  const selectedEncryptedDatasets = useMemo(
    () => selectedDatasets.filter(dataset => dataset.encrypted),
    [selectedDatasets],
  );
  const selectedWorkerIDs = useMemo(
    () => Array.from(new Set(selectedDatasets.map(datasetWorkerID))),
    [selectedDatasets],
  );
  const selectedWorkerID = selectedWorkerIDs.length === 1 ? selectedWorkerIDs[0] : null;
  const canCombineSelection = selectedDatasets.length >= 2 && selectedWorkerID !== null;
  const canBulkUnlockSelection = selectedEncryptedDatasets.length > 0;
  const bulkPasswordTargetCount = useMemo(
    () =>
      bulkUnlockTargets.filter(target => {
        const ref = datasetRowKey(target);
        return (
          bulkUnlockStatus[ref] !== 'unlocked' &&
          bulkUnlockManifests[ref]?.crypto.kdf.type === 'PBKDF2-SHA256'
        );
      }).length,
    [bulkUnlockManifests, bulkUnlockStatus, bulkUnlockTargets],
  );
  const bulkUnlockedCount = useMemo(
    () => bulkUnlockTargets.filter(target => bulkUnlockStatus[datasetRowKey(target)] === 'unlocked').length,
    [bulkUnlockStatus, bulkUnlockTargets],
  );
  const isBulkUnlockBusy = useMemo(
    () => isBulkUnlocking || Object.values(bulkUnlockStatus).some(status => status === 'unlocking'),
    [bulkUnlockStatus, isBulkUnlocking],
  );
  const combineEncryptedSources = useMemo(
    () => combineSources.filter(source => source.encrypted),
    [combineSources],
  );
  const combineWorkerID = combineSources.length > 0 ? datasetWorkerID(combineSources[0]) : 'local';
  const folderImportGroups = useMemo(() => {
    const groups = new Map<string, FolderImportEntry[]>();
    folderImportEntries.forEach(entry => {
      const current = groups.get(entry.rootName) || [];
      current.push(entry);
      groups.set(entry.rootName, current);
    });
    return Array.from(groups.entries()).map(([rootName, entries]) => ({ rootName, entries }));
  }, [folderImportEntries]);
  const folderImportFileCount = folderImportEntries.length;
  const folderImportWorkerOptions = useMemo(() => {
    const options = new Map<string, string>();
    options.set('local', 'Local');
    datasets.forEach(dataset => {
      const workerID = datasetWorkerID(dataset);
      if (workerID !== 'local') {
        options.set(workerID, dataset.worker_name || workerID);
      }
    });
    return Array.from(options.entries()).map(([id, name]) => ({ id, name }));
  }, [datasets]);

  const toggleDatasetSelection = (dataset: DatasetSummary) => {
    const ref = datasetRowKey(dataset);
    const workerID = datasetWorkerID(dataset);
    setSelectedDatasetRefs(previous => {
      const next = new Set(previous);
      if (next.has(ref)) {
        next.delete(ref);
        return next;
      }
      const selectedRows = tableRows.filter(row => next.has(row.ref));
      const existingWorkerID = selectedRows[0]?.worker_id;
      if (existingWorkerID && existingWorkerID !== workerID) {
        alert('Select datasets from one worker at a time.');
        return previous;
      }
      next.add(ref);
      return next;
    });
  };

  const loadBulkUnlockManifest = useCallback(async (dataset: DatasetSummary) => {
    const ref = datasetRowKey(dataset);
    setBulkUnlockStatus(previous => ({ ...previous, [ref]: 'loading' }));
    setBulkUnlockErrors(previous => ({ ...previous, [ref]: '' }));
    try {
      const res = await apiClient.post('/api/datasets/listImages', {
        datasetName: dataset.name,
        worker_id: datasetWorkerID(dataset),
      });
      if (res.data?.encrypted && res.data?.manifest) {
        setBulkUnlockManifests(previous => ({ ...previous, [ref]: res.data.manifest }));
        setBulkUnlockStatus(previous => ({ ...previous, [ref]: 'locked' }));
      } else {
        setBulkUnlockStatus(previous => ({ ...previous, [ref]: 'error' }));
        setBulkUnlockErrors(previous => ({ ...previous, [ref]: 'Encrypted manifest was not returned.' }));
      }
    } catch {
      setBulkUnlockStatus(previous => ({ ...previous, [ref]: 'error' }));
      setBulkUnlockErrors(previous => ({ ...previous, [ref]: 'Could not load encrypted manifest.' }));
    }
  }, []);

  const closeBulkUnlockModal = () => {
    if (isBulkUnlockBusy) return;
    setIsBulkUnlockModalOpen(false);
    setBulkUnlockTargets([]);
    setBulkUnlockManifests({});
    setBulkUnlockStatus({});
    setBulkUnlockErrors({});
    setBulkSharedPassword('');
    setBulkRowPasswords({});
    setBulkRowKeyFiles({});
  };

  const openBulkUnlockModal = () => {
    if (!canBulkUnlockSelection) {
      alert('Select at least one encrypted dataset.');
      return;
    }

    const targets = selectedEncryptedDatasets;
    const initialStatus: Record<string, BulkUnlockStatus> = {};
    const rememberedRefs = new Set<string>();
    targets.forEach(target => {
      const ref = datasetRowKey(target);
      if (getRememberedDatasetKey(target)) {
        initialStatus[ref] = 'unlocked';
        rememberedRefs.add(ref);
      } else {
        initialStatus[ref] = 'loading';
      }
    });

    if (rememberedRefs.size > 0) {
      setUnlockedDatasetRefs(previous => {
        const next = new Set(previous);
        rememberedRefs.forEach(ref => next.add(ref));
        return next;
      });
    }

    setBulkUnlockTargets(targets);
    setBulkUnlockManifests({});
    setBulkUnlockStatus(initialStatus);
    setBulkUnlockErrors({});
    setBulkSharedPassword('');
    setBulkRowPasswords({});
    setBulkRowKeyFiles({});
    setIsBulkUnlockModalOpen(true);
    targets.forEach(target => {
      if (!getRememberedDatasetKey(target)) void loadBulkUnlockManifest(target);
    });
  };

  const unlockBulkDataset = async (
    dataset: DatasetSummary,
    request: Parameters<typeof unlockEncryptedDatasetKey>[1],
    failureMessage = 'Could not unlock this dataset.',
  ) => {
    const ref = datasetRowKey(dataset);
    const manifest = bulkUnlockManifests[ref];
    if (!manifest) {
      setBulkUnlockStatus(previous => ({ ...previous, [ref]: 'error' }));
      setBulkUnlockErrors(previous => ({ ...previous, [ref]: 'Encrypted manifest is still loading.' }));
      return false;
    }

    setBulkUnlockStatus(previous => ({ ...previous, [ref]: 'unlocking' }));
    setBulkUnlockErrors(previous => ({ ...previous, [ref]: '' }));
    try {
      const unlocked = await unlockEncryptedDatasetKey(manifest, request);
      await decryptCatalog(manifest, unlocked.key);
      rememberUnlockedDataset(dataset, unlocked.rawKeyB64);
      setBulkUnlockStatus(previous => ({ ...previous, [ref]: 'unlocked' }));
      setBulkUnlockErrors(previous => ({ ...previous, [ref]: '' }));
      return true;
    } catch {
      setBulkUnlockStatus(previous => ({ ...previous, [ref]: 'error' }));
      setBulkUnlockErrors(previous => ({ ...previous, [ref]: failureMessage }));
      return false;
    }
  };

  const handleBulkSharedPasswordUnlock = async () => {
    if (!bulkSharedPassword || isBulkUnlocking) return;
    setIsBulkUnlocking(true);
    try {
      for (const target of bulkUnlockTargets) {
        const ref = datasetRowKey(target);
        const manifest = bulkUnlockManifests[ref];
        if (
          bulkUnlockStatus[ref] === 'unlocked' ||
          manifest?.crypto.kdf.type !== 'PBKDF2-SHA256'
        ) {
          continue;
        }
        await unlockBulkDataset(
          target,
          { provider: 'password', password: bulkSharedPassword },
          'Needs a different password.',
        );
      }
    } finally {
      setIsBulkUnlocking(false);
    }
  };

  const closeCombineModal = () => {
    if (isCombiningDatasets) return;
    setIsCombineModalOpen(false);
    setCombineSources([]);
    setCombineOutputName('');
    setCombinePassword('');
    setCombinePasswordConfirm('');
    setCombineKeyFile(null);
    setCombineSourceKeys({});
    setCombineSourceManifests({});
    setCombineSourcePasswords({});
    setCombineSourceKeyFiles({});
    setCombineSourceErrors({});
    setCombineSourceLoading({});
  };

  const openCombineModal = () => {
    if (!canCombineSelection) {
      alert('Select at least two datasets from the same worker.');
      return;
    }
    const sources = selectedDatasets;
    const encrypted = sources.some(source => source.encrypted);
    setCombineSources(sources);
    setCombineOutputName(`${sources[0].name}_combined`);
    setCombineOutputMode(encrypted ? 'encrypted' : 'plain');
    setCombineSourceKeys(
      Object.fromEntries(
        sources
          .filter(source => source.encrypted)
          .map(source => [datasetRowKey(source), getRememberedDatasetKey(source)])
          .filter((entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1].length > 0),
      ),
    );
    setIsCombineModalOpen(true);
  };

  const openRenameDatasetModal = (dataset: DatasetSummary) => {
    setRenameDataset(dataset);
    setRenameDatasetName(dataset.name);
    setRenameDatasetError('');
  };

  const closeRenameDatasetModal = () => {
    if (isRenamingDataset) return;
    setRenameDataset(null);
    setRenameDatasetName('');
    setRenameDatasetError('');
  };

  const handleRenameDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!renameDataset || isRenamingDataset) return;

    const normalizedName = cleanClientDatasetName(renameDatasetName);
    if (!normalizedName) {
      setRenameDatasetError('Dataset name is required.');
      return;
    }

    try {
      setIsRenamingDataset(true);
      setRenameDatasetError('');
      const workerID = datasetWorkerID(renameDataset);
      const res = await apiClient.post('/api/datasets/rename', {
        name: renameDataset.name,
        newName: renameDatasetName,
        worker_id: workerID,
      });
      const renamedName = res.data?.name || normalizedName;
      const rememberedKey = getRememberedDatasetKey(renameDataset);
      if (rememberedKey) {
        rememberUnlockedDataset(
          {
            ...renameDataset,
            name: renamedName,
            worker_id: workerID,
            ref:
              workerID === 'local'
                ? `aitk-dataset://local/${encodeURIComponent(renamedName)}`
                : makeRemoteDatasetRef(workerID, renamedName),
          },
          rememberedKey,
        );
      }
      setSelectedDatasetRefs(previous => {
        const next = new Set(previous);
        next.delete(datasetRowKey(renameDataset));
        return next;
      });
      refreshDatasets();
      setRenameDataset(null);
      setRenameDatasetName('');
      setRenameDatasetError('');
    } catch (error: any) {
      setRenameDatasetError(error?.response?.data?.error || 'Failed to rename dataset.');
    } finally {
      setIsRenamingDataset(false);
    }
  };

  const handleDeleteDataset = (datasetName: string, workerID = 'local') => {
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/datasets/delete', { name: datasetName, worker_id: workerID })
          .then(() => {
            console.log('Dataset deleted:', datasetName);
            refreshDatasets();
          })
          .catch(error => {
            console.error('Error deleting dataset:', error);
          });
      },
    });
  };

  const handleImportDataset = async (dataset: any) => {
    if (!dataset?.worker_id || dataset.worker_id === 'local') return;
    const ref = dataset.ref || `${dataset.worker_id}:${dataset.name}`;
    setImportingRef(ref);
    try {
      const res = await apiClient.post('/api/datasets/import-remote', {
        worker_id: dataset.worker_id,
        datasetName: dataset.name,
      });
      refreshDatasets();
      const importedName = res.data?.dataset?.name;
      const importedPath = res.data?.path;
      const remembered =
        getRememberedEncryptedDatasetKey(ref) ||
        getRememberedEncryptedDatasetKey(remoteDatasetRememberKey(dataset.worker_id, dataset.name)) ||
        getRememberedEncryptedDatasetKey(dataset.name);
      if (remembered && importedName && importedPath) {
        rememberUnlockedDataset(
          {
            name: importedName,
            encrypted: true,
            worker_id: 'local',
            ref: `aitk-dataset://local/${encodeURIComponent(importedName)}`,
            path: importedPath,
          },
          remembered,
        );
      }
      if (importedName) router.push(`/datasets/${encodeURIComponent(importedName)}`);
    } catch (error: any) {
      alert(error?.response?.data?.error || 'Failed to import remote dataset.');
    } finally {
      setImportingRef(null);
    }
  };

  const closeHfImportModal = () => {
    if (isLoadingHfPreview || isImportingHfDataset) return;
    setIsHfImportModalOpen(false);
    setHfDatasetInput('');
    setHfImportWorkerID('local');
    setHfPreview(null);
    setHfConfig('');
    setHfSplit('');
    setHfImageColumn('');
    setHfCaptionMode('auto');
    setHfCaptionColumn('');
    setHfOutputName('');
    setHfMaxRows('');
    setHfImportStatus('');
    setHfImportError('');
  };

  const hfRequestPayload = (action: 'preview' | 'import') => {
    const maxRowsValue = hfMaxRows.trim() ? Number(hfMaxRows) : undefined;
    return {
      action,
      worker_id: hfImportWorkerID,
      dataset: hfDatasetInput,
      config: hfConfig || undefined,
      split: hfSplit || undefined,
      imageColumn: hfImageColumn || undefined,
      captionMode: hfCaptionMode,
      captionColumn: hfCaptionMode === 'column' ? hfCaptionColumn || undefined : undefined,
      outputName: action === 'import' ? hfOutputName || undefined : undefined,
      maxRows: Number.isFinite(maxRowsValue) && Number(maxRowsValue) > 0 ? Math.floor(Number(maxRowsValue)) : undefined,
    };
  };

  const handleLoadHfPreview = async () => {
    if (!hfDatasetInput.trim() || isLoadingHfPreview) return;
    try {
      setIsLoadingHfPreview(true);
      setHfImportError('');
      setHfImportStatus('Loading preview...');
      const res = await apiClient.post('/api/datasets/import-huggingface', hfRequestPayload('preview'), { timeout: 0 });
      const preview = res.data as HfDatasetPreview;
      setHfPreview(preview);
      setHfConfig(preview.selectedConfig || '');
      setHfSplit(preview.selectedSplit || '');
      setHfImageColumn(preview.suggestedImageColumn || preview.imageColumns[0] || '');
      setHfCaptionColumn(preview.suggestedCaptionColumn || preview.textColumns[0] || '');
      if (!hfOutputName) {
        setHfOutputName(hfOutputNameFromDataset(preview.datasetID, preview.selectedSplit));
      }
      const countText = preview.rowCount != null ? `${preview.rowCount} row${preview.rowCount === 1 ? '' : 's'}` : 'Preview ready';
      setHfImportStatus(countText);
    } catch (error: any) {
      setHfPreview(null);
      setHfImportError(error?.response?.data?.error || error?.message || 'Failed to load Hugging Face dataset preview.');
      setHfImportStatus('');
    } finally {
      setIsLoadingHfPreview(false);
    }
  };

  const handleImportHfDataset = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!hfDatasetInput.trim() || isImportingHfDataset) return;
    try {
      setIsImportingHfDataset(true);
      setHfImportError('');
      setHfImportStatus('Importing dataset...');
      const res = await apiClient.post('/api/datasets/import-huggingface', hfRequestPayload('import'), { timeout: 0 });
      const importedName = res.data?.dataset?.name;
      const workerID = hfImportWorkerID;
      const imported = res.data?.imported;
      setHfImportStatus(
        imported
          ? `Imported ${imported.imagesWritten} image${imported.imagesWritten === 1 ? '' : 's'}.`
          : 'Imported dataset.',
      );
      refreshDatasets();
      setIsHfImportModalOpen(false);
      if (importedName) {
        router.push(
          workerID === 'local'
            ? `/datasets/${encodeURIComponent(importedName)}`
            : `/datasets/${encodeURIComponent(importedName)}?worker_id=${encodeURIComponent(workerID)}`,
        );
      }
    } catch (error: any) {
      setHfImportError(error?.response?.data?.error || error?.message || 'Failed to import Hugging Face dataset.');
      setHfImportStatus('');
    } finally {
      setIsImportingHfDataset(false);
    }
  };

  useEffect(() => {
    if (!isCombineModalOpen || combineEncryptedSources.length === 0) return;
    combineEncryptedSources.forEach(source => {
      const ref = datasetRowKey(source);
      if (combineSourceManifests[ref] || combineSourceLoading[ref] || combineSourceErrors[ref]) return;
      setCombineSourceLoading(previous => ({ ...previous, [ref]: true }));
      apiClient
        .post('/api/datasets/listImages', {
          datasetName: source.name,
          worker_id: datasetWorkerID(source),
        })
        .then(res => {
          if (res.data?.encrypted && res.data?.manifest) {
            setCombineSourceManifests(previous => ({ ...previous, [ref]: res.data.manifest }));
          } else {
            setCombineSourceErrors(previous => ({ ...previous, [ref]: 'Encrypted manifest was not returned.' }));
          }
        })
        .catch(() => {
          setCombineSourceErrors(previous => ({ ...previous, [ref]: 'Could not load encrypted manifest.' }));
        })
        .finally(() => {
          setCombineSourceLoading(previous => ({ ...previous, [ref]: false }));
        });
    });
  }, [combineEncryptedSources, combineSourceErrors, combineSourceLoading, combineSourceManifests, isCombineModalOpen]);

  const unlockCombineSource = async (source: DatasetSummary) => {
    const ref = datasetRowKey(source);
    const manifest = combineSourceManifests[ref];
    if (!manifest) {
      setCombineSourceErrors(previous => ({ ...previous, [ref]: 'Encrypted manifest is still loading.' }));
      return;
    }
    try {
      let unlocked;
      if (manifest.crypto.kdf.type === 'PBKDF2-SHA256') {
        unlocked = await unlockEncryptedDatasetKey(manifest, {
          provider: 'password',
          password: combineSourcePasswords[ref] || '',
        });
      } else if (manifest.crypto.kdf.type === 'KEYFILE-SHA256') {
        if (!combineSourceKeyFiles[ref]) {
          setCombineSourceErrors(previous => ({ ...previous, [ref]: 'Select the key file for this dataset.' }));
          return;
        }
        unlocked = await unlockEncryptedDatasetKey(manifest, {
          provider: 'keyFile',
          file: combineSourceKeyFiles[ref] as File,
        });
      } else if (manifest.crypto.kdf.type === 'WEBAUTHN-PRF') {
        unlocked = await unlockEncryptedDatasetKey(manifest, { provider: 'webauthnPrf' });
      } else {
        setCombineSourceErrors(previous => ({ ...previous, [ref]: 'Select the key file for this dataset.' }));
        return;
      }
      const rawKeyB64 = unlocked.rawKeyB64;
      setCombineSourceKeys(previous => ({ ...previous, [ref]: rawKeyB64 }));
      setCombineSourceErrors(previous => ({ ...previous, [ref]: '' }));
      rememberUnlockedDataset(source, rawKeyB64);
    } catch {
      setCombineSourceErrors(previous => ({ ...previous, [ref]: 'Could not unlock this dataset.' }));
    }
  };

  const handleCombineDatasets = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isCombiningDatasets) return;
    if (combineSources.length < 2) return;

    const missingKey = combineEncryptedSources.find(source => !combineSourceKeys[datasetRowKey(source)]);
    if (missingKey) {
      alert(`Unlock "${missingKey.name}" before combining.`);
      return;
    }

    try {
      setIsCombiningDatasets(true);
      let outputEncryptedManifest = null;
      let outputKeyB64: string | null = null;
      if (combineOutputMode === 'encrypted') {
        if (combineCredentialMode === 'password') {
          if (!combinePassword || combinePassword !== combinePasswordConfirm) {
            alert('Password and confirmation must match.');
            return;
          }
          const result = await createEmptyEncryptedManifest('password', combinePassword);
          outputEncryptedManifest = result.manifest;
          outputKeyB64 = result.rawKeyB64;
        } else {
          if (combineCredentialMode === 'yubiKey') {
            const result = await createEmptyEncryptedManifest('yubiKey');
            outputEncryptedManifest = result.manifest;
            outputKeyB64 = result.rawKeyB64;
          } else {
            if (!combineKeyFile) {
              alert('Select a key file.');
              return;
            }
            const result = await createEmptyEncryptedManifest('keyFile', combineKeyFile);
            outputEncryptedManifest = result.manifest;
            outputKeyB64 = result.rawKeyB64;
          }
        }
      }

      const res = await apiClient.post('/api/datasets/combine', {
        worker_id: combineWorkerID,
        sourceDatasets: combineSources.map(source => source.name),
        outputName: combineOutputName,
        outputEncrypted: combineOutputMode === 'encrypted',
        encryptedDatasetKeys: combineEncryptedSources.map(source => ({
          datasetName: source.name,
          keyB64: combineSourceKeys[datasetRowKey(source)],
        })),
        outputEncryptedManifest,
        outputKeyB64,
      });

      const combined = res.data?.dataset as DatasetSummary | undefined;
      if (outputKeyB64 && combined?.name) {
        rememberUnlockedDataset(
          {
            ...combined,
            worker_id: combineWorkerID,
            ref:
              combineWorkerID === 'local'
                ? combined.ref || `aitk-dataset://local/${encodeURIComponent(combined.name)}`
                : makeRemoteDatasetRef(combineWorkerID, combined.name),
          },
          outputKeyB64,
        );
      }

      refreshDatasets();
      setSelectedDatasetRefs(new Set());
      setIsCombineModalOpen(false);
      if (combined?.name) {
        router.push(
          combineWorkerID === 'local'
            ? `/datasets/${encodeURIComponent(combined.name)}`
            : `/datasets/${encodeURIComponent(combined.name)}?worker_id=${encodeURIComponent(combineWorkerID)}`,
        );
      }
    } catch (error: any) {
      alert(error?.response?.data?.error || 'Failed to combine datasets.');
    } finally {
      setIsCombiningDatasets(false);
    }
  };

  const addFolderImportFiles = (fileList: FileList | null) => {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;
    const entries = files
      .map(file => {
        const relativePath = relativePathForFile(file);
        return {
          id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
          file,
          relativePath,
          rootName: folderImportRootName(relativePath),
        };
      })
      .filter(entry => {
        const parts = entry.relativePath.split('/').filter(Boolean);
        if (parts.some(part => part.startsWith('.'))) return false;
        return FOLDER_IMPORT_SUPPORTED_EXTENSIONS.has(folderImportExtension(entry.relativePath));
      });

    if (entries.length === 0) {
      alert('No supported media or caption files were found in that folder.');
      return;
    }

    setFolderImportEntries(previous => [...previous, ...entries]);
    setFolderImportStatus('');
    if (!folderImportDatasetName && folderImportMode === 'combined') {
      setFolderImportDatasetName(`${entries[0].rootName}_combined`);
    }
  };

  const closeFolderImportModal = () => {
    if (isImportingFolders) return;
    setIsFolderImportModalOpen(false);
    setFolderImportEntries([]);
    setFolderImportDatasetName('');
    setFolderImportStatus('');
    setFolderImportMode('separate');
    setFolderImportOutputMode('plain');
    setFolderImportCredentialMode('password');
    setFolderImportPassword('');
    setFolderImportPasswordConfirm('');
    setFolderImportKeyFile(null);
    setFolderImportWorkerID('local');
  };

  const uploadFolderImportBatch = async (
    workerID: string,
    datasetName: string,
    entries: FolderImportEntry[],
    relativePaths: string[],
  ) => {
    const formData = new FormData();
    formData.append('datasetName', datasetName);
    if (workerID !== 'local') formData.append('worker_id', workerID);
    formData.append('failIfDatasetExists', '1');
    formData.append('preserveRelativePaths', '1');
    formData.append('relativePaths', JSON.stringify(relativePaths));
    const sourceFolderPath = workerID === 'local' ? sourceFolderPathForEntries(entries) : '';
    if (sourceFolderPath) formData.append('sourceFolderPath', sourceFolderPath);
    entries.forEach(entry => {
      formData.append('files', entry.file);
    });
    await apiClient.post('/api/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 0,
    });
  };

  const createFolderImportEncryption = async () => {
    if (folderImportCredentialMode === 'password') {
      if (!folderImportPassword || folderImportPassword !== folderImportPasswordConfirm) {
        throw new Error('Password and confirmation must match.');
      }
      return createEmptyEncryptedManifest('password', folderImportPassword);
    }
    if (folderImportCredentialMode === 'yubiKey') {
      return createEmptyEncryptedManifest('yubiKey');
    }
    if (!folderImportKeyFile) {
      throw new Error('Select a key file.');
    }
    return createEmptyEncryptedManifest('keyFile', folderImportKeyFile);
  };

  const rememberFolderImportOutputKey = (workerID: string, datasetName: string, rawKeyB64: string) => {
    rememberUnlockedDataset(
      {
        name: datasetName,
        encrypted: true,
        worker_id: workerID,
        ref:
          workerID === 'local'
            ? `${workerID}:${datasetName}`
            : makeRemoteDatasetRef(workerID, datasetName),
      },
      rawKeyB64,
    );
  };

  const buildEncryptedFolderImportPayload = async (
    entries: FolderImportEntry[],
    relativePaths: string[],
    manifest: EncryptedDatasetManifest,
    key: CryptoKey,
  ) => {
    const captionFiles = new Map<string, File>();
    entries.forEach((entry, index) => {
      const relativePath = relativePaths[index] || entry.relativePath || entry.file.name;
      if (isFolderImportCaptionSidecarPath(relativePath)) {
        captionFiles.set(folderImportCaptionKey(relativePath), entry.file);
      }
    });

    const allocateCatalogName = createFlattenedFileNameAllocator();
    const rootCaption = await readRootCaptionFile(
      entries.map(entry => entry.file),
      relativePaths,
    );
    const catalog: EncryptedDatasetCatalog = {
      version: 1,
      items: [],
      ...(rootCaption !== null ? { rootCaption } : {}),
    };
    const encryptedObjects: Array<{ objectPath: string; blob: Blob }> = [];

    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      const mediaKind = getMediaKind(entry.file);
      if (!mediaKind) continue;
      const relativePath = relativePaths[index] || entry.relativePath || entry.file.name;
      const captionFile = captionFiles.get(folderImportCaptionKey(relativePath));
      const catalogName = allocateCatalogName(relativePath);
      const { item, encryptedObjects: itemObjects } = await buildEncryptedDatasetItem(
        entry.file,
        key,
        captionFile ? await captionFile.text() : null,
        catalogName,
      );
      catalog.items.push(item);
      encryptedObjects.push(...itemObjects);
    }

    if (catalog.items.length === 0) {
      throw new Error('No supported media files were found in the selected folders.');
    }

    const { manifest: encryptedManifest } = await encryptCatalog(catalog, key, manifest);
    return { manifest: encryptedManifest, encryptedObjects, itemCount: catalog.items.length };
  };

  const uploadEncryptedFolderImportBatch = async (
    workerID: string,
    datasetName: string,
    entries: FolderImportEntry[],
    relativePaths: string[],
  ) => {
    const encryption = await createFolderImportEncryption();
    const createResult = await apiClient
      .post('/api/datasets/create', {
        worker_id: workerID,
        name: datasetName,
        encrypted: true,
        encryptedManifest: encryption.manifest,
      })
      .then(res => res.data);
    const createdName = createResult?.name || datasetName;
    const encryptedPayload = await buildEncryptedFolderImportPayload(
      entries,
      relativePaths,
      encryption.manifest,
      encryption.key,
    );

    const formData = new FormData();
    formData.append('datasetName', createdName);
    if (workerID !== 'local') formData.append('worker_id', workerID);
    formData.append('encrypted', '1');
    formData.append('manifest', JSON.stringify(encryptedPayload.manifest));
    const sourceFolderPath = workerID === 'local' ? sourceFolderPathForEntries(entries) : '';
    if (sourceFolderPath) formData.append('sourceFolderPath', sourceFolderPath);
    formData.append(
      'objectPaths',
      JSON.stringify(encryptedPayload.encryptedObjects.map(encryptedObject => encryptedObject.objectPath)),
    );
    encryptedPayload.encryptedObjects.forEach(encryptedObject => {
      formData.append('files', encryptedObject.blob, encryptedObject.objectPath.split('/').pop() || 'object.bin');
    });
    await apiClient.post('/api/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 0,
    });
    rememberFolderImportOutputKey(workerID, createdName, encryption.rawKeyB64);
    return { datasetName: createdName, itemCount: encryptedPayload.itemCount };
  };

  const handleImportFolders = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isImportingFolders) return;
    if (folderImportEntries.length === 0) {
      alert('Choose at least one folder.');
      return;
    }

    try {
      setIsImportingFolders(true);
      const targetDatasetNames = new Set(
        datasets
          .filter(dataset => datasetWorkerID(dataset) === folderImportWorkerID)
          .map(dataset => dataset.name.toLowerCase()),
      );

      if (folderImportMode === 'separate') {
        let importedCount = 0;
        for (const group of folderImportGroups) {
          const datasetName = nextClientDatasetName(group.rootName, targetDatasetNames);
          setFolderImportStatus(`Importing ${datasetName}...`);
          const relativePaths = group.entries.map(entry => stripFolderImportRoot(entry.relativePath, entry.file.name));
          if (folderImportOutputMode === 'encrypted') {
            await uploadEncryptedFolderImportBatch(folderImportWorkerID, datasetName, group.entries, relativePaths);
          } else {
            await uploadFolderImportBatch(folderImportWorkerID, datasetName, group.entries, relativePaths);
          }
          importedCount += 1;
        }
        setFolderImportStatus(`Imported ${importedCount} datasets.`);
      } else {
        const datasetName = cleanClientDatasetName(folderImportDatasetName);
        if (!datasetName) {
          alert('Output dataset name is required.');
          return;
        }
        if (targetDatasetNames.has(datasetName.toLowerCase())) {
          alert('A dataset with that name already exists.');
          return;
        }
        setFolderImportStatus(`Importing ${datasetName}...`);
        const relativePaths = folderImportEntries.map(entry => entry.relativePath);
        const importResult =
          folderImportOutputMode === 'encrypted'
            ? await uploadEncryptedFolderImportBatch(
                folderImportWorkerID,
                datasetName,
                folderImportEntries,
                relativePaths,
              )
            : await uploadFolderImportBatch(folderImportWorkerID, datasetName, folderImportEntries, relativePaths).then(
                () => ({ datasetName }),
              );
        const importedDatasetName = importResult.datasetName;
        setFolderImportStatus(`Imported ${importedDatasetName}.`);
        router.push(
          folderImportWorkerID === 'local'
            ? `/datasets/${encodeURIComponent(importedDatasetName)}`
            : `/datasets/${encodeURIComponent(importedDatasetName)}?worker_id=${encodeURIComponent(folderImportWorkerID)}`,
        );
      }

      refreshDatasets();
      setIsFolderImportModalOpen(false);
      setFolderImportEntries([]);
      setFolderImportDatasetName('');
      setFolderImportStatus('');
      setFolderImportPassword('');
      setFolderImportPasswordConfirm('');
      setFolderImportKeyFile(null);
    } catch (error: any) {
      alert(error?.response?.data?.error || error?.message || 'Failed to import folders.');
    } finally {
      setIsImportingFolders(false);
    }
  };

  const handleCreateDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isCreatingDataset) return;
    try {
      setIsCreatingDataset(true);
      let encryptedManifest = null;
      let rawKeyB64: string | null = null;
      if (newDatasetMode === 'encrypted') {
        if (credentialMode === 'password') {
          if (!datasetPassword || datasetPassword !== datasetPasswordConfirm) {
            alert('Password and confirmation must match.');
            return;
          }
          const result = await createEmptyEncryptedManifest('password', datasetPassword);
          encryptedManifest = result.manifest;
          rawKeyB64 = result.rawKeyB64;
        } else {
          if (credentialMode === 'yubiKey') {
            const result = await createEmptyEncryptedManifest('yubiKey');
            encryptedManifest = result.manifest;
            rawKeyB64 = result.rawKeyB64;
          } else {
            if (!datasetKeyFile) {
              alert('Select a key file.');
              return;
            }
            const result = await createEmptyEncryptedManifest('keyFile', datasetKeyFile);
            encryptedManifest = result.manifest;
            rawKeyB64 = result.rawKeyB64;
          }
        }
      }

      const data = await apiClient
        .post('/api/datasets/create', {
          name: newDatasetName,
          encrypted: newDatasetMode === 'encrypted',
          encryptedManifest,
        })
        .then(res => res.data);
      console.log('New dataset created:', data);
      if (rawKeyB64 && data.name) {
        rememberUnlockedDataset(
          {
            name: data.name,
            encrypted: true,
            worker_id: 'local',
            ref: `aitk-dataset://local/${encodeURIComponent(data.name)}`,
          },
          rawKeyB64,
        );
      }
      refreshDatasets();
      setNewDatasetName('');
      setDatasetPassword('');
      setDatasetPasswordConfirm('');
      setDatasetKeyFile(null);
      setIsNewDatasetModalOpen(false);
      if (data.name) router.push(`/datasets/${data.name}`);
    } catch (error) {
      console.error('Error creating new dataset:', error);
    } finally {
      setIsCreatingDataset(false);
    }
  };

  const openNewDatasetModal = () => {
    setIsNewDatasetModalOpen(true);
  };

  const datasetHref = (row: DatasetExplorerRow) =>
    row.source === 'remote'
      ? `/datasets/${encodeURIComponent(row.name)}?worker_id=${encodeURIComponent(row.worker_id)}`
      : `/datasets/${encodeURIComponent(row.name)}`;

  const datasetMediaLabel = (dataset: DatasetSummary, unlocked = false) => {
    if (dataset.encrypted) return unlocked ? 'Unlocked' : 'Locked';
    if (typeof dataset.itemCount !== 'number') return 'Not scanned';
    if (dataset.itemCount === 0) return 'No media';
    return `${dataset.itemCount} media item${dataset.itemCount === 1 ? '' : 's'}`;
  };

  const datasetCaptionLabel = (dataset: DatasetSummary, unlocked = false) => {
    if (dataset.encrypted) return unlocked ? 'Captions unlocked' : 'Captions locked';
    if (typeof dataset.itemCount !== 'number' || typeof dataset.missingCaptionCount !== 'number') return 'Not scanned';
    if (dataset.itemCount === 0) return 'No captions';
    if (dataset.missingCaptionCount > 0) {
      return `${dataset.missingCaptionCount} missing caption${dataset.missingCaptionCount === 1 ? '' : 's'}`;
    }
    return 'Captions complete';
  };

  const isRowSelected = (row: DatasetExplorerRow) => selectedDatasetRefs.has(row.ref);
  const isRowSelectionDisabled = (row: DatasetExplorerRow) =>
    !isRowSelected(row) && selectedWorkerID !== null && selectedWorkerID !== row.worker_id;

  const renderSelectionCheckbox = (row: DatasetExplorerRow) => (
    <input
      type="checkbox"
      aria-label={`Select ${row.name}`}
      checked={isRowSelected(row)}
      disabled={isRowSelectionDisabled(row)}
      onChange={() => toggleDatasetSelection(row.dataset)}
      className="h-4 w-4 rounded border-gray-600 bg-gray-900 text-cyan-500 disabled:cursor-not-allowed disabled:opacity-40"
    />
  );

  const renderDatasetActions = (row: DatasetExplorerRow, compact = false) => (
    <div className="flex justify-end gap-1">
      {row.source === 'remote' && (
        <button
          type="button"
          className={`inline-flex items-center justify-center rounded-sm text-gray-300 transition-colors hover:bg-blue-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-50 ${
            compact ? 'h-7 w-7' : 'h-8 w-8'
          }`}
          disabled={importingRef === row.ref}
          onClick={() => handleImportDataset(row.dataset)}
          title="Import to Local"
          aria-label={`Import ${row.name} to Local`}
        >
          {importingRef === row.ref ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Download className="h-4 w-4" />
          )}
        </button>
      )}
      {row.source === 'local' && !row.encrypted && (
        <DatasetWatchFoldersButton
          datasetName={row.name}
          workerID={row.worker_id}
          defaultSourcePath={row.dataset.importSourcePath}
          label={`Watch folders for ${row.name}`}
          icon="eye"
          iconOnly
          className={`inline-flex items-center justify-center rounded-sm text-gray-300 transition-colors hover:bg-cyan-700 hover:text-white ${
            compact ? 'h-7 w-7' : 'h-8 w-8'
          }`}
          onRefresh={() => refreshDatasets({ background: true })}
        />
      )}
      <button
        type="button"
        className={`inline-flex items-center justify-center rounded-sm text-gray-300 transition-colors hover:bg-cyan-700 hover:text-white ${
          compact ? 'h-7 w-7' : 'h-8 w-8'
        }`}
        onClick={() => openRenameDatasetModal(row.dataset)}
        title="Rename dataset"
        aria-label={`Rename ${row.name}`}
      >
        <Pencil className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={`inline-flex items-center justify-center rounded-sm text-gray-300 transition-colors hover:bg-red-600 hover:text-white ${
          compact ? 'h-7 w-7' : 'h-8 w-8'
        }`}
        onClick={() => handleDeleteDataset(row.name, row.worker_id)}
        title="Delete dataset"
        aria-label={`Delete ${row.name}`}
      >
        <Trash2 className="h-4 w-4" />
      </button>
    </div>
  );

  const renderSourceBadge = (row: DatasetExplorerRow) => (
    <span
      className={`inline-flex max-w-full items-center rounded-sm border px-2 py-0.5 text-xs font-medium ${
        row.source === 'remote'
          ? 'border-sky-700/70 bg-sky-950/35 text-sky-200'
          : 'border-gray-700 bg-gray-900/80 text-gray-300'
      }`}
      title={row.source === 'remote' ? row.worker : 'Local'}
    >
      <span className="truncate">{row.source === 'remote' ? row.worker : 'Local'}</span>
    </span>
  );

  const renderTypeBadge = (row: DatasetExplorerRow) => (
    <span
      className={`inline-flex items-center rounded-sm border px-2 py-0.5 text-xs font-medium ${
        row.encrypted
          ? row.unlocked
            ? 'border-emerald-700/70 bg-emerald-950/35 text-emerald-200'
            : 'border-cyan-700/70 bg-cyan-950/35 text-cyan-200'
          : 'border-gray-700 bg-gray-900/80 text-gray-300'
      }`}
    >
      {row.encrypted ? (row.unlocked ? 'Unlocked' : 'Encrypted') : 'Plain'}
    </span>
  );

  const renderBrowserState = () => {
    if (status === 'loading') {
      return (
        <div className="flex min-h-64 items-center justify-center border border-gray-800 bg-gray-950/40">
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <Loader2 className="h-4 w-4 animate-spin text-cyan-300" />
            Loading datasets
          </div>
        </div>
      );
    }

    if (status === 'error') {
      return (
        <PageNotice
          tone="danger"
          title="Datasets could not be loaded"
          action={
            <button type="button" onClick={() => refreshDatasets()} className="operator-button py-1 text-xs">
              Retry
            </button>
          }
        />
      );
    }

    if (filteredTableRows.length === 0) {
      return (
        <PageNotice
          tone="neutral"
          title={datasetFilter ? 'No datasets match the filter' : 'No datasets found'}
          action={
            <button type="button" onClick={() => refreshDatasets()} className="operator-button py-1 text-xs">
              Refresh
            </button>
          }
        >
          {datasetFilter ? 'Try a different dataset, worker, caption, or type filter.' : 'Create a dataset or import folders to prepare training data.'}
        </PageNotice>
      );
    }

    return null;
  };

  const renderDetailsView = () => {
    const stateContent = renderBrowserState();
    if (stateContent) return stateContent;

    return (
      <div className="overflow-hidden border border-gray-800 bg-gray-950/40">
        <div className="overflow-x-auto">
          <div className="min-w-[920px]">
            <div className="grid grid-cols-[2.75rem_minmax(17rem,1.7fr)_8.5rem_minmax(12rem,0.85fr)_10rem_8rem_7.5rem] border-b border-gray-800 bg-gray-900/85 text-xs uppercase text-gray-500">
              <div className="px-3 py-2" />
              <div className="px-3 py-2 font-medium">Name</div>
              <div className="px-3 py-2 font-medium">Items</div>
              <div className="px-3 py-2 font-medium">Captions</div>
              <div className="px-3 py-2 font-medium">Source</div>
              <div className="px-3 py-2 font-medium">Type</div>
              <div className="px-3 py-2 text-right font-medium">Actions</div>
            </div>
            {filteredTableRows.map((row, index) => {
              const selected = isRowSelected(row);
              const rowClass = selected
                ? 'bg-cyan-950/35 ring-1 ring-inset ring-cyan-700/45'
                : index % 2 === 0
                  ? 'bg-gray-950/20'
                  : 'bg-gray-900/35';
              return (
                <div
                  key={row.ref}
                  className={`grid grid-cols-[2.75rem_minmax(17rem,1.7fr)_8.5rem_minmax(12rem,0.85fr)_10rem_8rem_7.5rem] items-center border-b border-gray-800 text-sm text-gray-300 last:border-b-0 hover:bg-gray-800/70 ${rowClass}`}
                >
                  <div className="flex items-center justify-center px-3 py-2">{renderSelectionCheckbox(row)}</div>
                  <div className="min-w-0 px-3 py-2">
                    <Link href={datasetHref(row)} className="flex min-w-0 items-center gap-3 text-gray-100 hover:text-cyan-100">
                      <DatasetFolderIcon
                        size="sm"
                        encrypted={row.encrypted}
                        unlocked={row.unlocked}
                        remote={row.source === 'remote'}
                        previewSrc={datasetPreviewUrls[row.ref]}
                      />
                      <span className="min-w-0 truncate font-medium">{row.name}</span>
                    </Link>
                    {row.dataset.path && <div className="mt-0.5 truncate text-xs text-gray-600">{row.dataset.path}</div>}
                  </div>
                  <div className="px-3 py-2 text-xs text-gray-400">{datasetMediaLabel(row.dataset, row.unlocked)}</div>
                  <div className="px-3 py-2">
                    <div className="flex min-w-0 flex-col gap-1.5">
                      <CaptionStatusBadge dataset={row.dataset} unlocked={row.unlocked} />
                      <DatasetWatcherProgressBadge progress={autoCaptionProgressByDataset[row.name]} />
                    </div>
                  </div>
                  <div className="min-w-0 px-3 py-2">{renderSourceBadge(row)}</div>
                  <div className="px-3 py-2">{renderTypeBadge(row)}</div>
                  <div className="px-3 py-2">{renderDatasetActions(row)}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  const renderIconView = () => {
    const stateContent = renderBrowserState();
    if (stateContent) return stateContent;

    return (
      <div className="grid grid-cols-[repeat(auto-fill,minmax(11.5rem,1fr))] gap-3">
        {filteredTableRows.map(row => {
          const selected = isRowSelected(row);
          return (
            <div
              key={row.ref}
              className={`relative min-h-56 rounded-sm border p-3 transition-colors ${
                selected
                  ? 'border-cyan-700 bg-cyan-950/30 shadow-sm shadow-cyan-950/50'
                  : 'border-gray-800 bg-gray-950/45 hover:border-gray-700 hover:bg-gray-900/55'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                {renderSelectionCheckbox(row)}
                {renderDatasetActions(row, true)}
              </div>
              <Link href={datasetHref(row)} className="mt-2 flex min-w-0 flex-col items-center gap-2 text-center">
                <DatasetFolderIcon
                  size="lg"
                  encrypted={row.encrypted}
                  unlocked={row.unlocked}
                  remote={row.source === 'remote'}
                  previewSrc={datasetPreviewUrls[row.ref]}
                />
                <span className="line-clamp-2 min-h-10 max-w-full break-words text-sm font-medium text-gray-100 hover:text-cyan-100">
                  {row.name}
                </span>
              </Link>
              <div className="mt-2 flex min-w-0 items-center justify-center gap-1.5">
                {renderSourceBadge(row)}
                {renderTypeBadge(row)}
              </div>
              <div className="mt-2 text-center text-xs text-gray-500">{datasetMediaLabel(row.dataset, row.unlocked)}</div>
              <div className="mt-1 truncate text-center text-xs text-gray-500">
                {datasetCaptionLabel(row.dataset, row.unlocked)}
              </div>
              <DatasetWatcherProgressBadge progress={autoCaptionProgressByDataset[row.name]} className="mx-auto mt-2" />
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <>
      <TopBar>
        <div className="flex shrink-0 items-center gap-2">
          <Database className="h-4 w-4 text-cyan-300" />
          <h1 className="text-base font-semibold">Datasets</h1>
        </div>
        <div className="flex-1"></div>
        {selectedDatasets.length > 0 && (
          <div className="hidden text-sm text-gray-400 sm:block">
            {selectedDatasets.length} selected
            {selectedWorkerID ? ` on ${selectedDatasets[0]?.worker_name || 'Local'}` : ''}
          </div>
        )}
        <div className="flex items-center gap-2">
          <Button
            className="operator-button shrink-0 py-1"
            disabled={!canBulkUnlockSelection}
            onClick={() => openBulkUnlockModal()}
            title="Unlock selected encrypted datasets"
            aria-label="Unlock selected encrypted datasets"
          >
            <KeyRound className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Unlock</span>
          </Button>
          <Button
            className="operator-button shrink-0 py-1"
            onClick={() => setIsHfImportModalOpen(true)}
            title="Import Hugging Face dataset"
            aria-label="Import Hugging Face dataset"
          >
            <CloudDownload className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Import HF</span>
          </Button>
          <Button
            className="operator-button shrink-0 py-1"
            onClick={() => setIsFolderImportModalOpen(true)}
            title="Import folders"
            aria-label="Import folders"
          >
            <FolderPlus className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Import Folders</span>
          </Button>
          <Button
            className="operator-button shrink-0 py-1"
            disabled={!canCombineSelection}
            onClick={() => openCombineModal()}
            title="Combine selected datasets"
            aria-label="Combine selected datasets"
          >
            <Layers className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Combine</span>
          </Button>
          <Button
            className="operator-button shrink-0 py-1"
            onClick={() => openNewDatasetModal()}
            title="New dataset"
            aria-label="New dataset"
          >
            <Plus className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">New Dataset</span>
          </Button>
        </div>
      </TopBar>

      <MainContent>
        <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-xs text-gray-500">
            {filteredTableRows.length} of {tableRows.length} datasets shown
            {selectedDatasets.length > 0 ? `, ${selectedDatasets.length} selected` : ''}
          </div>
          <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:items-center">
            <div className="inline-flex h-8 w-full rounded-sm border border-gray-800 bg-gray-950 p-0.5 sm:w-auto">
              <button
                type="button"
                onClick={() => updateDatasetView('details')}
                className={`inline-flex flex-1 items-center justify-center gap-1.5 rounded-sm px-2 text-xs font-medium transition-colors sm:flex-none ${
                  datasetView === 'details' ? 'bg-gray-800 text-gray-100' : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'
                }`}
                title="Details view"
                aria-pressed={datasetView === 'details'}
              >
                <List className="h-3.5 w-3.5" />
                Details
              </button>
              <button
                type="button"
                onClick={() => updateDatasetView('icons')}
                className={`inline-flex flex-1 items-center justify-center gap-1.5 rounded-sm px-2 text-xs font-medium transition-colors sm:flex-none ${
                  datasetView === 'icons' ? 'bg-gray-800 text-gray-100' : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'
                }`}
                title="Large icons view"
                aria-pressed={datasetView === 'icons'}
              >
                <Grid2X2 className="h-3.5 w-3.5" />
                Icons
              </button>
            </div>
            <label className="relative block w-full sm:w-80">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
              <input
                value={datasetFilter}
                onChange={event => setDatasetFilter(event.target.value)}
                placeholder="Filter datasets, workers, type"
                className="h-8 w-full rounded-sm border border-gray-800 bg-gray-950 pl-8 pr-3 text-sm text-gray-100 placeholder:text-gray-500 focus:border-cyan-700 focus:outline-none"
              />
            </label>
          </div>
        </div>
        {errors.length > 0 && (
          <div className="mb-3 rounded-md border border-yellow-700 bg-yellow-950/40 px-3 py-2 text-sm text-yellow-200">
            Some remote datasets could not be loaded:{' '}
            {errors.map(error => `${error.worker_name}: ${error.error}`).join('; ')}
          </div>
        )}
        {datasetView === 'details' ? renderDetailsView() : renderIconView()}
      </MainContent>

      <Modal
        isOpen={isNewDatasetModalOpen}
        onClose={() => setIsNewDatasetModalOpen(false)}
        title="New Dataset"
        size="md"
      >
        <div className="space-y-4 text-gray-200">
          <form onSubmit={handleCreateDataset}>
            <div className="mt-4">
              <TextInput label="Dataset Name" value={newDatasetName} onChange={value => setNewDatasetName(value)} />
            </div>
            <div className="mt-4 grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setNewDatasetMode('plain')}
                className={`rounded-md border px-3 py-2 text-left ${
                  newDatasetMode === 'plain'
                    ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                    : 'border-gray-700 bg-gray-900 text-gray-300'
                }`}
              >
                Plain
              </button>
              <button
                type="button"
                onClick={() => setNewDatasetMode('encrypted')}
                className={`rounded-md border px-3 py-2 text-left ${
                  newDatasetMode === 'encrypted'
                    ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                    : 'border-gray-700 bg-gray-900 text-gray-300'
                }`}
              >
                Encrypted
              </button>
            </div>

            {newDatasetMode === 'encrypted' && (
              <div className="mt-4 space-y-3 rounded-md border border-gray-700 bg-gray-900 p-3">
                <div className="grid grid-cols-3 gap-2">
                  <button
                    type="button"
                    onClick={() => setCredentialMode('password')}
                    className={`rounded-md px-3 py-2 text-sm ${
                      credentialMode === 'password' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300'
                    }`}
                  >
                    Password
                  </button>
                  <button
                    type="button"
                    onClick={() => setCredentialMode('keyFile')}
                    className={`rounded-md px-3 py-2 text-sm ${
                      credentialMode === 'keyFile' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300'
                    }`}
                  >
                    Key File
                  </button>
                  <button
                    type="button"
                    onClick={() => setCredentialMode('yubiKey')}
                    className={`rounded-md px-3 py-2 text-sm ${
                      credentialMode === 'yubiKey' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300'
                    }`}
                  >
                    YubiKey
                  </button>
                </div>
                {credentialMode === 'password' ? (
                  <div className="space-y-3">
                    <input
                      type="password"
                      value={datasetPassword}
                      onChange={e => setDatasetPassword(e.target.value)}
                      placeholder="Password"
                      className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                    />
                    <input
                      type="password"
                      value={datasetPasswordConfirm}
                      onChange={e => setDatasetPasswordConfirm(e.target.value)}
                      placeholder="Confirm password"
                      className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                    />
                  </div>
                ) : credentialMode === 'keyFile' ? (
                  <input
                    type="file"
                    onChange={e => setDatasetKeyFile(e.target.files?.[0] || null)}
                    className="block w-full text-sm text-gray-300 file:mr-3 file:rounded-md file:border-0 file:bg-gray-700 file:px-3 file:py-2 file:text-gray-100"
                  />
                ) : (
                  <div className="rounded-md border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-300">
                    YubiKey / USB Security Key
                  </div>
                )}
              </div>
            )}

            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
                onClick={() => setIsNewDatasetModalOpen(false)}
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isCreatingDataset}
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {isCreatingDataset ? 'Creating...' : 'Confirm'}
              </button>
            </div>
          </form>
        </div>
      </Modal>

      <Modal
        isOpen={isBulkUnlockModalOpen}
        onClose={closeBulkUnlockModal}
        title="Unlock Encrypted Datasets"
        size="lg"
        closeOnOverlayClick={!isBulkUnlockBusy}
      >
        <div className="space-y-4 text-gray-200">
          <div className="rounded-md border border-gray-700 bg-gray-900 p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-gray-100">
                  {bulkUnlockedCount} of {bulkUnlockTargets.length} unlocked
                </div>
                <div className="mt-0.5 text-xs text-gray-400">
                  {bulkUnlockTargets.length} encrypted dataset{bulkUnlockTargets.length === 1 ? '' : 's'} selected
                </div>
              </div>
              <span className="text-xs text-gray-400">
                {selectedWorkerID ? bulkUnlockTargets[0]?.worker_name || 'Local' : 'Mixed workers'}
              </span>
            </div>
          </div>

          {bulkPasswordTargetCount > 0 && (
            <div className="rounded-md border border-gray-700 bg-gray-900 p-3">
              <label className="mb-2 block text-sm font-medium text-gray-200">Shared Password</label>
              <div className="flex flex-col gap-2 sm:flex-row">
                <input
                  type="password"
                  value={bulkSharedPassword}
                  onChange={event => setBulkSharedPassword(event.target.value)}
                  placeholder="Dataset password"
                  className="min-w-0 flex-1 rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                />
                <button
                  type="button"
                  onClick={handleBulkSharedPasswordUnlock}
                  disabled={!bulkSharedPassword || isBulkUnlockBusy}
                  className="rounded-md bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isBulkUnlocking ? 'Unlocking...' : `Unlock ${bulkPasswordTargetCount}`}
                </button>
              </div>
            </div>
          )}

          <div className="max-h-[26rem] space-y-3 overflow-y-auto pr-1">
            {bulkUnlockTargets.map(target => {
              const ref = datasetRowKey(target);
              const manifest = bulkUnlockManifests[ref];
              const status = bulkUnlockStatus[ref] || 'loading';
              const error = bulkUnlockErrors[ref];
              const isUnlocked = status === 'unlocked';
              const isBusy = status === 'loading' || status === 'unlocking';
              const rowPassword = bulkRowPasswords[ref] || '';
              const rowKeyFile = bulkRowKeyFiles[ref] || null;

              return (
                <div key={ref} className="rounded-md border border-gray-800 bg-gray-950 p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium text-gray-100">{target.name}</div>
                      <div className="mt-0.5 text-xs text-gray-500">
                        {target.worker_name || (datasetWorkerID(target) === 'local' ? 'Local' : datasetWorkerID(target))}
                      </div>
                    </div>
                    <div
                      className={
                        isUnlocked
                          ? 'text-sm text-green-300'
                          : isBusy
                            ? 'text-sm text-blue-300'
                            : error
                              ? 'text-sm text-red-300'
                              : 'text-sm text-gray-400'
                      }
                    >
                      {isUnlocked
                        ? 'Unlocked'
                        : status === 'loading'
                          ? 'Loading...'
                          : status === 'unlocking'
                            ? 'Unlocking...'
                            : error
                              ? 'Needs attention'
                              : 'Locked'}
                    </div>
                  </div>

                  {!isUnlocked && !isBusy && manifest?.crypto.kdf.type === 'PBKDF2-SHA256' && (
                    <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                      <input
                        type="password"
                        value={rowPassword}
                        onChange={event =>
                          setBulkRowPasswords(previous => ({ ...previous, [ref]: event.target.value }))
                        }
                        placeholder="Different password"
                        className="min-w-0 flex-1 rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-gray-100"
                      />
                      <button
                        type="button"
                        onClick={() =>
                          unlockBulkDataset(
                            target,
                            { provider: 'password', password: rowPassword },
                            'Could not decrypt with this password.',
                          )
                        }
                        disabled={!rowPassword || isBulkUnlockBusy}
                        className="rounded-md bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        Unlock
                      </button>
                    </div>
                  )}

                  {!isUnlocked && !isBusy && manifest?.crypto.kdf.type === 'KEYFILE-SHA256' && (
                    <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                      <input
                        type="file"
                        onChange={event =>
                          setBulkRowKeyFiles(previous => ({ ...previous, [ref]: event.target.files?.[0] || null }))
                        }
                        className="min-w-0 flex-1 text-sm text-gray-300 file:mr-3 file:rounded-md file:border-0 file:bg-gray-700 file:px-3 file:py-2 file:text-gray-100"
                      />
                      <button
                        type="button"
                        onClick={() =>
                          rowKeyFile &&
                          unlockBulkDataset(
                            target,
                            { provider: 'keyFile', file: rowKeyFile },
                            'Could not decrypt with this key file.',
                          )
                        }
                        disabled={!rowKeyFile || isBulkUnlockBusy}
                        className="rounded-md bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        Unlock
                      </button>
                    </div>
                  )}

                  {!isUnlocked && !isBusy && manifest?.crypto.kdf.type === 'WEBAUTHN-PRF' && (
                    <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center">
                      <div className="min-w-0 flex-1 rounded-md border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-300">
                        YubiKey / USB Security Key
                      </div>
                      <button
                        type="button"
                        onClick={() =>
                          unlockBulkDataset(
                            target,
                            { provider: 'webauthnPrf' },
                            'Could not unlock with this YubiKey.',
                          )
                        }
                        disabled={isBulkUnlockBusy}
                        className="rounded-md bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        Unlock
                      </button>
                    </div>
                  )}

                  {!isUnlocked && !isBusy && !manifest && (
                    <div className="mt-3">
                      <button
                        type="button"
                        onClick={() => void loadBulkUnlockManifest(target)}
                        className="rounded-md bg-gray-700 px-3 py-2 text-sm text-gray-100 hover:bg-gray-600"
                      >
                        Retry
                      </button>
                    </div>
                  )}

                  {error && <div className="mt-2 text-sm text-red-400">{error}</div>}
                </div>
              );
            })}
          </div>

          <div className="flex justify-end">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 disabled:opacity-50"
              onClick={closeBulkUnlockModal}
              disabled={isBulkUnlockBusy}
            >
              Close
            </button>
          </div>
        </div>
      </Modal>

      <Modal
        isOpen={!!renameDataset}
        onClose={closeRenameDatasetModal}
        title="Rename Dataset"
        size="md"
        closeOnOverlayClick={!isRenamingDataset}
      >
        <form onSubmit={handleRenameDataset} className="space-y-4 text-gray-200">
          <div className="rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-300">
            {renameDataset
              ? renameDataset.worker_name ||
                (datasetWorkerID(renameDataset) === 'local' ? 'Local' : datasetWorkerID(renameDataset))
              : 'Local'}
          </div>
          <TextInput label="Dataset Name" value={renameDatasetName} onChange={setRenameDatasetName} />
          {renameDatasetError && <div className="text-sm text-red-400">{renameDatasetError}</div>}
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 disabled:opacity-50"
              onClick={closeRenameDatasetModal}
              disabled={isRenamingDataset}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isRenamingDataset}
              className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isRenamingDataset ? 'Renaming...' : 'Rename'}
            </button>
          </div>
        </form>
      </Modal>

      <Modal
        isOpen={isHfImportModalOpen}
        onClose={closeHfImportModal}
        title="Import Hugging Face Dataset"
        size="lg"
        closeOnOverlayClick={!isLoadingHfPreview && !isImportingHfDataset}
      >
        <form onSubmit={handleImportHfDataset} className="space-y-4 text-gray-200">
          <TextInput
            label="Dataset URL or ID"
            value={hfDatasetInput}
            onChange={value => {
              setHfDatasetInput(value);
              setHfPreview(null);
              setHfImportStatus('');
              setHfImportError('');
            }}
          />

          {folderImportWorkerOptions.length > 1 && (
            <div>
              <label className="mb-1 block text-sm text-gray-300">Import To</label>
              <select
                value={hfImportWorkerID}
                onChange={event => {
                  setHfImportWorkerID(event.target.value);
                  setHfPreview(null);
                  setHfImportStatus('');
                  setHfImportError('');
                }}
                className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
              >
                {folderImportWorkerOptions.map(worker => (
                  <option key={worker.id} value={worker.id}>
                    {worker.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm text-gray-300">Config</label>
              {hfPreview?.configs?.length ? (
                <select
                  value={hfConfig}
                  onChange={event => setHfConfig(event.target.value)}
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                >
                  {hfPreview.configs.map(config => (
                    <option key={config} value={config}>
                      {config}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  value={hfConfig}
                  onChange={event => setHfConfig(event.target.value)}
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                />
              )}
            </div>
            <div>
              <label className="mb-1 block text-sm text-gray-300">Split</label>
              {hfPreview?.splits?.length ? (
                <select
                  value={hfSplit}
                  onChange={event => setHfSplit(event.target.value)}
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                >
                  {hfPreview.splits.map(split => (
                    <option key={split} value={split}>
                      {split}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  value={hfSplit}
                  onChange={event => setHfSplit(event.target.value)}
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                />
              )}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm text-gray-300">Image Column</label>
              {hfPreview?.imageColumns?.length ? (
                <select
                  value={hfImageColumn}
                  onChange={event => setHfImageColumn(event.target.value)}
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                >
                  {hfPreview.imageColumns.map(column => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  value={hfImageColumn}
                  onChange={event => setHfImageColumn(event.target.value)}
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                />
              )}
            </div>
            <div>
              <label className="mb-1 block text-sm text-gray-300">Captions</label>
              <select
                value={hfCaptionMode}
                onChange={event => setHfCaptionMode(event.target.value as HfCaptionMode)}
                className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
              >
                <option value="auto">Auto</option>
                <option value="none">None</option>
                <option value="column">Column</option>
              </select>
            </div>
          </div>

          {hfCaptionMode === 'column' && (
            <div>
              <label className="mb-1 block text-sm text-gray-300">Caption Column</label>
              {hfPreview?.textColumns?.length ? (
                <select
                  value={hfCaptionColumn}
                  onChange={event => setHfCaptionColumn(event.target.value)}
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                >
                  {hfPreview.textColumns.map(column => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  value={hfCaptionColumn}
                  onChange={event => setHfCaptionColumn(event.target.value)}
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                />
              )}
            </div>
          )}

          <div className="grid gap-3 sm:grid-cols-2">
            <TextInput label="Output Dataset Name" value={hfOutputName} onChange={setHfOutputName} />
            <div>
              <label className="mb-1 block text-sm text-gray-300">Max Rows</label>
              <input
                type="number"
                min="1"
                value={hfMaxRows}
                onChange={event => setHfMaxRows(event.target.value)}
                className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
              />
            </div>
          </div>

          {hfPreview && (
            <div className="rounded-md border border-gray-700 bg-gray-900 p-3 text-sm text-gray-300">
              <div className="flex flex-wrap gap-x-4 gap-y-1">
                <span>{hfPreview.datasetID}</span>
                <span>{hfPreview.selectedConfig}</span>
                <span>{hfPreview.selectedSplit}</span>
                {hfPreview.rowCount != null && <span>{hfPreview.rowCount} rows</span>}
              </div>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <div className="min-w-0 truncate text-gray-400">
                  Images: {hfPreview.imageColumns.length ? hfPreview.imageColumns.join(', ') : 'none'}
                </div>
                <div className="min-w-0 truncate text-gray-400">
                  Text: {hfPreview.textColumns.length ? hfPreview.textColumns.join(', ') : 'none'}
                </div>
              </div>
            </div>
          )}

          {hfImportError && <div className="text-sm text-red-400">{hfImportError}</div>}
          {hfImportStatus && !hfImportError && <div className="text-sm text-blue-300">{hfImportStatus}</div>}

          <div className="flex flex-wrap justify-end gap-3">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 disabled:opacity-50"
              onClick={closeHfImportModal}
              disabled={isLoadingHfPreview || isImportingHfDataset}
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={isLoadingHfPreview || isImportingHfDataset || !hfDatasetInput.trim()}
              className="inline-flex items-center gap-2 rounded-md bg-slate-600 px-4 py-2 text-white hover:bg-slate-500 disabled:cursor-not-allowed disabled:opacity-50"
              onClick={handleLoadHfPreview}
            >
              {isLoadingHfPreview && <Loader2 className="h-4 w-4 animate-spin" />}
              Preview
            </button>
            <button
              type="submit"
              disabled={isLoadingHfPreview || isImportingHfDataset || !hfDatasetInput.trim()}
              className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isImportingHfDataset && <Loader2 className="h-4 w-4 animate-spin" />}
              Import
            </button>
          </div>
        </form>
      </Modal>

      <Modal
        isOpen={isFolderImportModalOpen}
        onClose={closeFolderImportModal}
        title="Import Folders"
        size="lg"
        closeOnOverlayClick={!isImportingFolders}
      >
        <form onSubmit={handleImportFolders} className="space-y-4 text-gray-200">
          <input
            ref={setFolderImportInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={event => {
              addFolderImportFiles(event.currentTarget.files);
              event.currentTarget.value = '';
            }}
          />

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setFolderImportMode('separate')}
              className={`rounded-md border px-3 py-2 text-left ${
                folderImportMode === 'separate'
                  ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                  : 'border-gray-700 bg-gray-900 text-gray-300'
              }`}
            >
              Separate Datasets
            </button>
            <button
              type="button"
              onClick={() => {
                setFolderImportMode('combined');
                if (!folderImportDatasetName && folderImportGroups[0]) {
                  setFolderImportDatasetName(`${folderImportGroups[0].rootName}_combined`);
                }
              }}
              className={`rounded-md border px-3 py-2 text-left ${
                folderImportMode === 'combined'
                  ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                  : 'border-gray-700 bg-gray-900 text-gray-300'
              }`}
            >
              One Dataset
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setFolderImportOutputMode('plain')}
              className={`rounded-md border px-3 py-2 text-left ${
                folderImportOutputMode === 'plain'
                  ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                  : 'border-gray-700 bg-gray-900 text-gray-300'
              }`}
            >
              Plain
            </button>
            <button
              type="button"
              onClick={() => setFolderImportOutputMode('encrypted')}
              className={`rounded-md border px-3 py-2 text-left ${
                folderImportOutputMode === 'encrypted'
                  ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                  : 'border-gray-700 bg-gray-900 text-gray-300'
              }`}
            >
              Encrypted
            </button>
          </div>

          {folderImportWorkerOptions.length > 1 && (
            <div>
              <label className="mb-1 block text-sm text-gray-300">Import To</label>
              <select
                value={folderImportWorkerID}
                onChange={e => setFolderImportWorkerID(e.target.value)}
                className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
              >
                {folderImportWorkerOptions.map(worker => (
                  <option key={worker.id} value={worker.id}>
                    {worker.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {folderImportMode === 'combined' && (
            <TextInput
              label="Output Dataset Name"
              value={folderImportDatasetName}
              onChange={setFolderImportDatasetName}
            />
          )}

          {folderImportOutputMode === 'encrypted' && (
            <div className="space-y-3 rounded-md border border-gray-700 bg-gray-900 p-3">
              <div className="grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => setFolderImportCredentialMode('password')}
                  className={`rounded-md px-3 py-2 text-sm ${
                    folderImportCredentialMode === 'password'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-300'
                  }`}
                >
                  Password
                </button>
                <button
                  type="button"
                  onClick={() => setFolderImportCredentialMode('keyFile')}
                  className={`rounded-md px-3 py-2 text-sm ${
                    folderImportCredentialMode === 'keyFile'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-300'
                  }`}
                >
                  Key File
                </button>
                <button
                  type="button"
                  onClick={() => setFolderImportCredentialMode('yubiKey')}
                  className={`rounded-md px-3 py-2 text-sm ${
                    folderImportCredentialMode === 'yubiKey'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-300'
                  }`}
                >
                  YubiKey
                </button>
              </div>
              {folderImportCredentialMode === 'password' ? (
                <div className="space-y-3">
                  <input
                    type="password"
                    value={folderImportPassword}
                    onChange={e => setFolderImportPassword(e.target.value)}
                    placeholder="Password"
                    className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                  />
                  <input
                    type="password"
                    value={folderImportPasswordConfirm}
                    onChange={e => setFolderImportPasswordConfirm(e.target.value)}
                    placeholder="Confirm password"
                    className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                  />
                </div>
              ) : folderImportCredentialMode === 'keyFile' ? (
                <input
                  type="file"
                  onChange={e => setFolderImportKeyFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm text-gray-300 file:mr-3 file:rounded-md file:border-0 file:bg-gray-700 file:px-3 file:py-2 file:text-gray-100"
                />
              ) : (
                <div className="rounded-md border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-300">
                  YubiKey / USB Security Key
                </div>
              )}
            </div>
          )}

          <div className="rounded-md border border-gray-700 bg-gray-900 p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-gray-200">
                  {folderImportGroups.length} folder{folderImportGroups.length === 1 ? '' : 's'} selected
                </div>
                <div className="text-sm text-gray-400">
                  {folderImportFileCount} file{folderImportFileCount === 1 ? '' : 's'} ready
                </div>
              </div>
              <button
                type="button"
                onClick={() => folderImportInputRef.current?.click()}
                className="rounded-md bg-slate-600 px-3 py-2 text-sm text-white hover:bg-slate-500"
              >
                Add Folder
              </button>
            </div>

            {folderImportGroups.length > 0 && (
              <div className="mt-3 max-h-48 overflow-y-auto rounded-md border border-gray-800 bg-gray-950">
                {folderImportGroups.map(group => (
                  <div
                    key={group.rootName}
                    className="flex items-center justify-between gap-3 border-b border-gray-800 px-3 py-2 last:border-b-0"
                  >
                    <span className="min-w-0 truncate text-sm">{group.rootName}</span>
                    <span className="text-xs text-gray-400">
                      {group.entries.length} file{group.entries.length === 1 ? '' : 's'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {folderImportMode === 'separate' && folderImportGroups.length > 0 && (
            <div className="rounded-md border border-gray-700 bg-gray-900 p-3 text-sm text-gray-300">
              {folderImportGroups.map(group => group.rootName).join(', ')}
            </div>
          )}

          {folderImportStatus && <div className="text-sm text-blue-300">{folderImportStatus}</div>}

          <div className="flex justify-end space-x-3">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600"
              onClick={closeFolderImportModal}
              disabled={isImportingFolders}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isImportingFolders || folderImportEntries.length === 0}
              className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isImportingFolders ? 'Importing...' : 'Import'}
            </button>
          </div>
        </form>
      </Modal>

      <Modal
        isOpen={isCombineModalOpen}
        onClose={closeCombineModal}
        title="Combine Datasets"
        size="lg"
        closeOnOverlayClick={!isCombiningDatasets}
      >
        <form onSubmit={handleCombineDatasets} className="space-y-4 text-gray-200">
          <div className="rounded-md border border-gray-700 bg-gray-900 p-3">
            <div className="text-sm text-gray-400">
              Worker: {combineSources[0]?.worker_name || (combineWorkerID === 'local' ? 'Local' : combineWorkerID)}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {combineSources.map(source => (
                <span
                  key={datasetRowKey(source)}
                  className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1 text-sm text-gray-200"
                >
                  {source.name}
                  {source.encrypted ? <span className="ml-1 text-blue-300">(encrypted)</span> : null}
                </span>
              ))}
            </div>
          </div>

          <TextInput label="Output Dataset Name" value={combineOutputName} onChange={setCombineOutputName} />

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setCombineOutputMode('plain')}
              className={`rounded-md border px-3 py-2 text-left ${
                combineOutputMode === 'plain'
                  ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                  : 'border-gray-700 bg-gray-900 text-gray-300'
              }`}
            >
              Plain
            </button>
            <button
              type="button"
              onClick={() => setCombineOutputMode('encrypted')}
              className={`rounded-md border px-3 py-2 text-left ${
                combineOutputMode === 'encrypted'
                  ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                  : 'border-gray-700 bg-gray-900 text-gray-300'
              }`}
            >
              Encrypted
            </button>
          </div>

          {combineEncryptedSources.length > 0 && combineOutputMode === 'plain' && (
            <div className="rounded-md border border-yellow-700 bg-yellow-950/40 px-3 py-2 text-sm text-yellow-200">
              Encrypted source files will be decrypted into a plain output dataset.
            </div>
          )}

          {combineEncryptedSources.length > 0 && (
            <div className="space-y-3 rounded-md border border-gray-700 bg-gray-900 p-3">
              <div className="text-sm font-semibold text-gray-200">Encrypted Sources</div>
              {combineEncryptedSources.map(source => {
                const ref = datasetRowKey(source);
                const manifest = combineSourceManifests[ref];
                const unlocked = !!combineSourceKeys[ref];
                const loading = !!combineSourceLoading[ref];
                const error = combineSourceErrors[ref];
                return (
                  <div key={ref} className="rounded-md border border-gray-800 bg-gray-950 p-3">
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0 truncate text-sm">{source.name}</div>
                      <div className={unlocked ? 'text-sm text-green-300' : 'text-sm text-gray-400'}>
                        {unlocked ? 'Unlocked' : loading ? 'Loading...' : 'Locked'}
                      </div>
                    </div>
                    {!unlocked && manifest && (
                      <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                        {manifest.crypto.kdf.type === 'PBKDF2-SHA256' ? (
                          <input
                            type="password"
                            value={combineSourcePasswords[ref] || ''}
                            onChange={e =>
                              setCombineSourcePasswords(previous => ({ ...previous, [ref]: e.target.value }))
                            }
                            placeholder="Dataset password"
                            className="min-w-0 flex-1 rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-gray-100"
                          />
                        ) : manifest.crypto.kdf.type === 'KEYFILE-SHA256' ? (
                          <input
                            type="file"
                            onChange={e =>
                              setCombineSourceKeyFiles(previous => ({
                                ...previous,
                                [ref]: e.target.files?.[0] || null,
                              }))
                            }
                            className="min-w-0 flex-1 text-sm text-gray-300 file:mr-3 file:rounded-md file:border-0 file:bg-gray-700 file:px-3 file:py-2 file:text-gray-100"
                          />
                        ) : (
                          <div className="min-w-0 flex-1 rounded-md border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-300">
                            YubiKey / USB Security Key
                          </div>
                        )}
                        <button
                          type="button"
                          onClick={() => unlockCombineSource(source)}
                          className="rounded-md bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700"
                        >
                          Unlock
                        </button>
                      </div>
                    )}
                    {error && <div className="mt-2 text-sm text-red-400">{error}</div>}
                  </div>
                );
              })}
            </div>
          )}

          {combineOutputMode === 'encrypted' && (
            <div className="space-y-3 rounded-md border border-gray-700 bg-gray-900 p-3">
              <div className="grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => setCombineCredentialMode('password')}
                  className={`rounded-md px-3 py-2 text-sm ${
                    combineCredentialMode === 'password' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300'
                  }`}
                >
                  Password
                </button>
                <button
                  type="button"
                  onClick={() => setCombineCredentialMode('keyFile')}
                  className={`rounded-md px-3 py-2 text-sm ${
                    combineCredentialMode === 'keyFile' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300'
                  }`}
                >
                  Key File
                </button>
                <button
                  type="button"
                  onClick={() => setCombineCredentialMode('yubiKey')}
                  className={`rounded-md px-3 py-2 text-sm ${
                    combineCredentialMode === 'yubiKey' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300'
                  }`}
                >
                  YubiKey
                </button>
              </div>
              {combineCredentialMode === 'password' ? (
                <div className="space-y-3">
                  <input
                    type="password"
                    value={combinePassword}
                    onChange={e => setCombinePassword(e.target.value)}
                    placeholder="Output password"
                    className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                  />
                  <input
                    type="password"
                    value={combinePasswordConfirm}
                    onChange={e => setCombinePasswordConfirm(e.target.value)}
                    placeholder="Confirm output password"
                    className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                  />
                </div>
              ) : combineCredentialMode === 'keyFile' ? (
                <input
                  type="file"
                  onChange={e => setCombineKeyFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm text-gray-300 file:mr-3 file:rounded-md file:border-0 file:bg-gray-700 file:px-3 file:py-2 file:text-gray-100"
                />
              ) : (
                <div className="rounded-md border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-300">
                  YubiKey / USB Security Key
                </div>
              )}
            </div>
          )}

          <div className="flex justify-end space-x-3">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600"
              onClick={closeCombineModal}
              disabled={isCombiningDatasets}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isCombiningDatasets}
              className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isCombiningDatasets ? 'Combining...' : 'Combine'}
            </button>
          </div>
        </form>
      </Modal>
    </>
  );
}
