'use client';
import { useSyncExternalStore } from 'react';
import { getTrainingReturnPath } from '@/hooks/useTrainingDraft';
import { reportWorkflowError } from '@/components/WorkflowFeedback';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from '@/navigation/Link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import useDatasetWatcherLiveRefresh from '@/hooks/useDatasetWatcherLiveRefresh';
import { Button } from '@headlessui/react';
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  Download,
  Grid2X2,
  FolderPlus,
  HelpCircle,
  Layers,
  List,
  Loader2,
  LockKeyhole,
  MinusCircle,
  Pencil,
  Plus,
  RefreshCw,
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
import { useRouter, useSearchParams } from '@/navigation';
import { aggregateAutoCaptionProgressByDataset } from '@/utils/datasetWatcherStatus';
import {
  folderImportExtension,
  folderImportRootName,
  FOLDER_IMPORT_SUPPORTED_EXTENSIONS,
  stripFolderImportRoot,
} from '@/utils/folderImport';
import { isImage } from '@/utils/basic';
import { getDisplayPath, getMediaUrl } from '@/utils/media';
import type { DatasetSummary } from '@/types';
import { planDatasetUploads } from '@/utils/datasetUploadPlan';
import { uploadDatasetFile } from '@/utils/streamedUploads';
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
  const cookie = document.cookie.split('; ').find(value => value.startsWith(`${DATASET_VIEW_STORAGE_KEY}=`));
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
function subscribeDatasetView(listener: () => void) {
  window.addEventListener('storage', listener);
  return () => window.removeEventListener('storage', listener);
}
function datasetRowKey(dataset: DatasetSummary) {
  return dataset.ref || `global:${dataset.worker_id || 'local'}:${dataset.name}`;
}
function datasetWorkerID(dataset: DatasetSummary) {
  return dataset.worker_id || 'local';
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
        className="inline-flex w-fit items-center gap-1.5 rounded-md border border-emerald-500/35 bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-800 dark:text-emerald-100"
        title="This encrypted dataset is unlocked in the current browser session"
      >
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-300" />
        Unlocked
      </span>
    );
  }
  if (dataset.encrypted) {
    return (
      <span
        className="inline-flex w-fit items-center gap-1.5 rounded-md border border-gray-700 bg-gray-900 px-2 py-0.5 text-xs font-medium text-gray-300"
        title="Caption counts are encrypted until the dataset is unlocked"
      >
        <LockKeyhole className="h-3.5 w-3.5 text-gray-400" />
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
            ? 'inline-flex w-fit items-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs font-medium text-amber-800 dark:text-amber-100'
            : 'inline-flex w-fit items-center gap-1.5 rounded-md border border-emerald-500/35 bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-800 dark:text-emerald-100'
        }
      >
        {hasMissingCaptions ? (
          <AlertTriangle className="h-3.5 w-3.5 text-amber-700 dark:text-amber-300" />
        ) : (
          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-300" />
        )}
        <span>{hasMissingCaptions ? `${missingCount} missing` : 'Complete'}</span>
        <span
          className={
            hasMissingCaptions ? 'text-amber-700 dark:text-amber-200/70' : 'text-emerald-700 dark:text-emerald-200/70'
          }
        >
          of {itemCount}
        </span>
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
  features: Array<{
    name: string;
    kind: string;
  }>;
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
  return (
    (
      file as File & {
        webkitRelativePath?: string;
      }
    ).webkitRelativePath || file.name
  ).replace(/\\/g, '/');
}
function fileSystemPathForFile(file: File) {
  const value = (
    file as File & {
      path?: unknown;
    }
  ).path;
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
  const searchParams = useSearchParams();
  const returnToTraining = searchParams.get('returnTo') === 'training';
  const importAction = searchParams.get('action');
  const [importChooserOpen, setImportChooserOpen] = useState(importAction === 'import');
  const [previousAction, setPreviousAction] = useState(importAction);
  if (previousAction !== importAction) {
    setPreviousAction(importAction);
    if (importAction === 'import') setImportChooserOpen(true);
  }
  const folderImportInputRef = useRef<HTMLInputElement | null>(null);
  const setFolderImportInputRef = useCallback((node: HTMLInputElement | null) => {
    folderImportInputRef.current = node;
    if (!node) return;
    node.setAttribute('webkitdirectory', '');
    node.setAttribute('directory', '');
  }, []);
  const { datasets, errors, status, refreshDatasets } = useDatasetList({
    includeRemote: true,
  });
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [datasetFilter, setDatasetFilter] = useState('');
  const storedView = useSyncExternalStore(
    subscribeDatasetView,
    () => readStoredDatasetView() ?? 'details',
    () => 'details',
  );
  const [selectedView, setDatasetView] = useState<DatasetExplorerView | null>(null);
  const datasetView = selectedView ?? storedView;
  const [datasetPreviewUrls, setDatasetPreviewUrls] = useState<Record<string, string | null>>({});
  const datasetPreviewRequestsRef = useRef<Set<string>>(new Set());
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);
  const [renameDataset, setRenameDataset] = useState<DatasetSummary | null>(null);
  const [renameDatasetName, setRenameDatasetName] = useState('');
  const [isRenamingDataset, setIsRenamingDataset] = useState(false);
  const [renameDatasetError, setRenameDatasetError] = useState('');
  const [importingRef, setImportingRef] = useState<string | null>(null);
  const [selectedDatasetRefs, setSelectedDatasetRefs] = useState<Set<string>>(() => new Set());
  const [isCombineModalOpen, setIsCombineModalOpen] = useState(false);
  const [combineSources, setCombineSources] = useState<DatasetSummary[]>([]);
  const [combineOutputName, setCombineOutputName] = useState('');
  const [isCombiningDatasets, setIsCombiningDatasets] = useState(false);
  const [isFolderImportModalOpen, setIsFolderImportModalOpen] = useState(false);
  const [folderImportEntries, setFolderImportEntries] = useState<FolderImportEntry[]>([]);
  const [folderImportMode, setFolderImportMode] = useState<'separate' | 'combined'>('separate');
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
  // Transform datasets array into rows with objects
  const tableRows: DatasetExplorerRow[] = datasets.map(dataset => {
    const unlocked = false;
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
          .post('/api/datasets/listImages', {
            datasetName: row.name,
            worker_id: row.worker_id,
          })
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
  const selectedWorkerIDs = useMemo(
    () => Array.from(new Set(selectedDatasets.map(datasetWorkerID))),
    [selectedDatasets],
  );
  const selectedWorkerID = selectedWorkerIDs.length === 1 ? selectedWorkerIDs[0] : null;
  const canCombineSelection =
    selectedDatasets.length >= 2 && selectedWorkerID !== null && selectedDatasets.every(dataset => !dataset.encrypted);
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
        reportWorkflowError('Select datasets from one worker at a time.');
        return previous;
      }
      next.add(ref);
      return next;
    });
  };
  const closeCombineModal = () => {
    if (isCombiningDatasets) return;
    setIsCombineModalOpen(false);
    setCombineSources([]);
    setCombineOutputName('');
  };
  const openCombineModal = () => {
    if (!canCombineSelection) {
      reportWorkflowError('Select at least two datasets from the same worker.');
      return;
    }
    const sources = selectedDatasets;
    if (sources.some(source => source.encrypted)) {
      reportWorkflowError('Decrypt legacy datasets before combining.');
      return;
    }
    const encrypted = sources.some(source => source.encrypted);
    setCombineSources(sources);
    setCombineOutputName(`${sources[0].name}_combined`);
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
  const handleDeleteDataset = (dataset: DatasetSummary) => {
    const datasetName = dataset.name;
    const workerID = datasetWorkerID(dataset);
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/datasets/delete', {
            name: datasetName,
            worker_id: workerID,
          })
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
      if (importedName) {
        router.push(`/datasets/${encodeURIComponent(importedName)}`);
      }
    } catch (error: any) {
      reportWorkflowError(error?.response?.data?.error || 'Failed to import remote dataset.');
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
      const countText =
        preview.rowCount != null ? `${preview.rowCount} row${preview.rowCount === 1 ? '' : 's'}` : 'Preview ready';
      setHfImportStatus(countText);
    } catch (error: any) {
      setHfPreview(null);
      setHfImportError(
        error?.response?.data?.error || error?.message || 'Failed to load Hugging Face dataset preview.',
      );
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
  const handleCombineDatasets = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isCombiningDatasets) return;
    if (combineSources.length < 2) return;
    try {
      setIsCombiningDatasets(true);

      const res = await apiClient.post('/api/datasets/combine', {
        worker_id: combineWorkerID,

        sourceDatasets: combineSources.map(source => source.name),
        outputName: combineOutputName,
      });
      const combined = res.data?.dataset as DatasetSummary | undefined;
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
      reportWorkflowError(error?.response?.data?.error || 'Failed to combine datasets.');
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
      reportWorkflowError('No supported media or caption files were found in that folder.');
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
    setFolderImportWorkerID('local');
  };
  const uploadFolderImportBatch = async (
    workerID: string,
    datasetName: string,
    entries: FolderImportEntry[],
    relativePaths: string[],
  ) => {
    const sourceFolderPath = workerID === 'local' ? sourceFolderPathForEntries(entries) : '';
    const groups = planDatasetUploads(
      entries.map((entry, index) => relativePaths[index] || entry.relativePath || entry.file.name),
      () => crypto.randomUUID(),
    );
    let first = true;
    for (const group of groups)
      for (const index of group.indexes) {
        const entry = entries[index];
        await uploadDatasetFile(entry.file, {
          uploadGroup: group.id,
          datasetName,
          workerID,
          relativePath: relativePaths[index] || entry.relativePath || entry.file.name,
          sourceFolderPath,
          failIfDatasetExists: first,
        });
        first = false;
      }
  };
  const handleImportFolders = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isImportingFolders) return;
    if (folderImportEntries.length === 0) {
      reportWorkflowError('Choose at least one folder.');
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
          {
            await uploadFolderImportBatch(folderImportWorkerID, datasetName, group.entries, relativePaths);
          }
          importedCount += 1;
        }
        setFolderImportStatus(`Imported ${importedCount} datasets.`);
      } else {
        const datasetName = cleanClientDatasetName(folderImportDatasetName);
        if (!datasetName) {
          reportWorkflowError('Output dataset name is required.');
          return;
        }
        if (targetDatasetNames.has(datasetName.toLowerCase())) {
          reportWorkflowError('A dataset with that name already exists.');
          return;
        }
        setFolderImportStatus(`Importing ${datasetName}...`);
        const relativePaths = folderImportEntries.map(entry => entry.relativePath);
        const importResult = await uploadFolderImportBatch(
          folderImportWorkerID,
          datasetName,
          folderImportEntries,
          relativePaths,
        ).then(() => ({ datasetName }));
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
    } catch (error: any) {
      reportWorkflowError(error?.response?.data?.error || error?.message || 'Failed to import folders.');
    } finally {
      setIsImportingFolders(false);
    }
  };
  const handleCreateDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isCreatingDataset) return;
    try {
      setIsCreatingDataset(true);
      const data = await apiClient
        .post('/api/datasets/create', {
          name: newDatasetName,
        })
        .then(res => res.data);
      console.log('New dataset created:', data);
      refreshDatasets();
      setNewDatasetName('');
      setIsNewDatasetModalOpen(false);
      if (data.name) {
        router.push(`/datasets/${encodeURIComponent(data.name)}`);
      }
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
      className="h-4 w-4 rounded border-gray-600 bg-gray-900 text-brand-500 disabled:cursor-not-allowed disabled:opacity-40"
    />
  );
  const renderDatasetActions = (row: DatasetExplorerRow, compact = false) => (
    <div className="flex min-w-0 flex-wrap justify-end gap-1">
      {row.source === 'remote' && !row.encrypted && (
        <button
          type="button"
          className={`inline-flex items-center justify-center rounded-sm text-gray-300 transition-colors hover:bg-brand-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-50 ${compact ? 'h-7 w-7' : 'h-8 w-8'}`}
          disabled={importingRef === row.ref}
          onClick={() => handleImportDataset(row.dataset)}
          title="Import to Local"
          aria-label={`Import ${row.name} to Local`}
        >
          {importingRef === row.ref ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
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
          className={`inline-flex items-center justify-center rounded-sm text-gray-300 transition-colors hover:bg-brand-700 hover:text-white ${compact ? 'h-7 w-7' : 'h-8 w-8'}`}
          onRefresh={() => refreshDatasets({ background: true })}
        />
      )}
      <Link
        className="operator-button max-w-full whitespace-normal text-center text-xs"
        href={
          row.encrypted
            ? `/datasets/${encodeURIComponent(row.name)}?worker_id=${encodeURIComponent(row.worker_id)}`
            : `/jobs/new?dataset=${encodeURIComponent(row.dataset.ref || row.dataset.path || row.name)}`
        }
      >
        {row.encrypted ? 'Decrypt to plain dataset' : 'Train'}
      </Link>
      <button
        type="button"
        className={`inline-flex items-center justify-center rounded-sm text-gray-300 transition-colors hover:bg-brand-700 hover:text-white disabled:cursor-not-allowed disabled:opacity-35 ${compact ? 'h-7 w-7' : 'h-8 w-8'}`}
        onClick={() => openRenameDatasetModal(row.dataset)}
        title="Rename dataset"
        aria-label={`Rename ${row.name}`}
      >
        <Pencil className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={`inline-flex items-center justify-center rounded-sm text-gray-300 transition-colors hover:bg-red-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-35 ${compact ? 'h-7 w-7' : 'h-8 w-8'}`}
        onClick={() => handleDeleteDataset(row.dataset)}
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
            ? 'border-emerald-700/70 bg-emerald-500/10 text-emerald-800 dark:bg-emerald-950/35 dark:text-emerald-200'
            : 'border-brand-700/70 bg-brand-950/35 text-brand-200'
          : 'border-gray-700 bg-gray-900/80 text-gray-300'
      }`}
    >
      {row.encrypted ? 'Legacy encrypted' : 'Plain'}
    </span>
  );
  const renderBrowserState = () => {
    if (status === 'loading') {
      return (
        <div className="flex min-h-64 items-center justify-center border border-gray-800 bg-gray-950/40">
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <Loader2 className="h-4 w-4 animate-spin text-brand-300" />
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
        <div className="border-y border-gray-800 px-4 py-8">
          <h2 className="text-sm font-semibold text-gray-100">
            {datasetFilter ? 'No matching datasets' : tableRows.length > 0 ? 'No datasets in this view' : 'No datasets'}
          </h2>
          <p className="mt-2 text-sm text-gray-400">
            {datasetFilter
              ? 'Try a different dataset, worker, caption, or type filter.'
              : 'Import media or create an empty dataset from the toolbar.'}
          </p>
          {datasetFilter && (
            <button type="button" className="operator-button mt-5" onClick={() => setDatasetFilter('')}>
              Clear filter
            </button>
          )}
        </div>
      );
    }
    return null;
  };
  const renderDetailsView = () => {
    const stateContent = renderBrowserState();
    if (stateContent) return stateContent;
    return (
      <div className="overflow-hidden border-y border-gray-800">
        <div className="overflow-x-auto">
          <div className="min-w-[1120px]">
            <div className="grid grid-cols-[2.75rem_minmax(16rem,1.5fr)_8rem_minmax(11rem,0.8fr)_9rem_7rem_16rem] border-b border-gray-800 bg-gray-900 text-xs text-gray-400">
              <div className="px-3 py-2" />
              <div className="px-3 py-2 font-medium">Name</div>
              <div className="px-3 py-2 font-medium">Items</div>
              <div className="px-3 py-2 font-medium">Captions</div>
              <div className="px-3 py-2 font-medium">Source</div>
              <div className="px-3 py-2 font-medium">Type</div>
              <div className="px-3 py-2 text-right font-medium">Actions</div>
            </div>
            {filteredTableRows.map(row => {
              const selected = isRowSelected(row);
              const rowClass = selected ? 'bg-brand-950/35 ring-1 ring-inset ring-brand-700/45' : '';
              return (
                <div
                  key={row.ref}
                  className={`grid grid-cols-[2.75rem_minmax(16rem,1.5fr)_8rem_minmax(11rem,0.8fr)_9rem_7rem_16rem] items-center border-b border-gray-800 text-sm text-gray-300 last:border-b-0 hover:bg-gray-800/70 ${rowClass}`}
                >
                  <div className="flex items-center justify-center px-3 py-2">{renderSelectionCheckbox(row)}</div>
                  <div className="min-w-0 px-3 py-2">
                    <Link
                      href={datasetHref(row)}
                      className="flex min-w-0 items-center gap-3 text-gray-100 hover:text-brand-100"
                    >
                      <DatasetFolderIcon
                        size="sm"
                        encrypted={row.encrypted}
                        unlocked={row.unlocked}
                        remote={row.source === 'remote'}
                        previewSrc={datasetPreviewUrls[row.ref]}
                      />
                      <span className="min-w-0 truncate font-medium">{row.name}</span>
                    </Link>
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
                  ? 'border-brand-700 bg-brand-950/30 shadow-sm shadow-brand-950/50'
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
                <span className="line-clamp-2 min-h-10 max-w-full break-words text-sm font-medium text-gray-100 hover:text-brand-100">
                  {row.name}
                </span>
              </Link>
              <div className="mt-2 flex min-w-0 items-center justify-center gap-1.5">
                {renderSourceBadge(row)}
                {renderTypeBadge(row)}
              </div>
              <div className="mt-2 text-center text-xs text-gray-500">
                {datasetMediaLabel(row.dataset, row.unlocked)}
              </div>
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
          <Database className="h-4 w-4 text-brand-300" />
          <h1 className="text-base font-semibold">Datasets</h1>
          <span className="text-xs tabular-nums text-gray-500">{tableRows.length}</span>
        </div>
        <div className="flex-1"></div>
        {selectedDatasets.length > 0 && (
          <div className="hidden text-sm text-gray-400 sm:block">
            {selectedDatasets.length} selected
            {selectedWorkerID ? ` on ${selectedDatasets[0]?.worker_name || 'Local'}` : ''}
          </div>
        )}
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="operator-button"
            onClick={() => refreshDatasets()}
            aria-label="Refresh datasets"
            title="Refresh datasets"
          >
            <RefreshCw size={15} aria-hidden="true" />
          </button>
          <Button
            className="operator-button shrink-0"
            aria-label="Import data"
            onClick={() => setImportChooserOpen(true)}
          >
            <FolderPlus size={16} aria-hidden="true" />
            <span className="hidden sm:inline">Import data</span>
          </Button>
          {selectedDatasets.length > 0 && (
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
          )}
          <Button
            className="operator-button-primary shrink-0"
            onClick={() => openNewDatasetModal()}
            title="Create dataset"
            aria-label="Create dataset"
          >
            <Plus className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Create dataset</span>
          </Button>
        </div>
      </TopBar>

      <MainContent>
        {returnToTraining && (
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3 border border-gray-700 px-3 py-2 text-sm">
            <p>Training draft open in this tab.</p>
            <Link href={getTrainingReturnPath()} className="studio-text-link">
              Return to training
            </Link>
          </div>
        )}
        <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
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
                  datasetView === 'details'
                    ? 'bg-gray-800 text-gray-100'
                    : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'
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
                  datasetView === 'icons'
                    ? 'bg-gray-800 text-gray-100'
                    : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'
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
                aria-label="Search datasets"
                placeholder="Search datasets, workers, type"
                className="h-8 w-full rounded border border-gray-700 bg-gray-950 pl-8 pr-3 text-sm text-gray-100 placeholder:text-gray-500 focus:border-brand-500 focus:outline-none"
              />
            </label>
          </div>
        </div>
        {errors.length > 0 && (
          <div className="mb-3 rounded-md border border-yellow-700 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-900 dark:bg-yellow-950/40 dark:text-yellow-200">
            Some datasets could not be loaded: {errors.map(error => `${error.worker_name}: ${error.error}`).join('; ')}
          </div>
        )}
        {datasetView === 'details' ? renderDetailsView() : renderIconView()}
      </MainContent>

      <Modal isOpen={importChooserOpen} onClose={() => setImportChooserOpen(false)} title="Import data" size="md">
        <div className="space-y-3">
          <button
            type="button"
            className="operator-button w-full justify-start p-4"
            onClick={() => {
              setImportChooserOpen(false);
              setIsFolderImportModalOpen(true);
            }}
          >
            Import folders from this computer
          </button>
          <button
            type="button"
            className="operator-button w-full justify-start p-4"
            onClick={() => {
              setImportChooserOpen(false);
              setIsHfImportModalOpen(true);
            }}
          >
            Import from Hugging Face
          </button>
          <button
            type="button"
            className="operator-button w-full justify-start p-4"
            onClick={() => {
              setImportChooserOpen(false);
              setIsNewDatasetModalOpen(true);
            }}
          >
            Create an empty dataset
          </button>
        </div>
      </Modal>
      <Modal
        isOpen={isNewDatasetModalOpen}
        onClose={() => setIsNewDatasetModalOpen(false)}
        title="Create dataset"
        size="md"
      >
        <div className="space-y-4 text-gray-200">
          <form onSubmit={handleCreateDataset}>
            <div className="mt-4">
              <TextInput label="Dataset Name" value={newDatasetName} onChange={value => setNewDatasetName(value)} />
            </div>

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
                className="rounded-md bg-brand-500 px-4 py-2 text-[var(--brand-ink)] hover:bg-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-500"
              >
                {isCreatingDataset ? 'Creating...' : 'Create dataset'}
              </button>
            </div>
          </form>
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
              className="rounded-md bg-brand-500 px-4 py-2 text-[var(--brand-ink)] hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-50"
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
          {hfImportStatus && !hfImportError && <div className="text-sm text-brand-300">{hfImportStatus}</div>}

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
              className="inline-flex items-center gap-2 rounded-md bg-brand-500 px-4 py-2 text-[var(--brand-ink)] hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-50"
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
                  ? 'border-brand-500 bg-brand-500/10 text-gray-100'
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
                  ? 'border-brand-500 bg-brand-500/10 text-gray-100'
                  : 'border-gray-700 bg-gray-900 text-gray-300'
              }`}
            >
              One Dataset
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

          {folderImportStatus && <div className="text-sm text-brand-300">{folderImportStatus}</div>}

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
              className="rounded-md bg-brand-500 px-4 py-2 text-[var(--brand-ink)] hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-50"
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
                  {source.encrypted ? <span className="ml-1 text-brand-300">(encrypted)</span> : null}
                </span>
              ))}
            </div>
          </div>

          <TextInput label="Output Dataset Name" value={combineOutputName} onChange={setCombineOutputName} />

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
              className="rounded-md bg-brand-500 px-4 py-2 text-[var(--brand-ink)] hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isCombiningDatasets ? 'Combining...' : 'Combine'}
            </button>
          </div>
        </form>
      </Modal>
    </>
  );
}
