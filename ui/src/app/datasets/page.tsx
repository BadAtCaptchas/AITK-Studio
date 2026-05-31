'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { Download, Search, Database, FolderPlus, Layers, Pencil, Plus } from 'lucide-react';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';
import {
  buildEncryptedDatasetItem,
  createEmptyEncryptedManifest,
  encryptCatalog,
  getMediaKind,
  getRememberedEncryptedDatasetKey,
  rememberEncryptedDatasetKey,
  unlockEncryptedDatasetKey,
  type DatasetCredentialMode,
} from '@/utils/encryptedDatasets';
import {
  createFlattenedFileNameAllocator,
  folderImportCaptionKey,
  folderImportExtension,
  folderImportRootName,
  stripFolderImportRoot,
} from '@/utils/folderImport';
import { makeRemoteDatasetRef, remoteDatasetRememberKey } from '@/utils/remoteDatasetRefs';
import type { DatasetSummary, EncryptedDatasetCatalog, EncryptedDatasetManifest } from '@/types';

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

type FolderImportEntry = {
  id: string;
  file: File;
  relativePath: string;
  rootName: string;
};

const FOLDER_IMPORT_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.gif',
  '.bmp',
  '.mp4',
  '.avi',
  '.mov',
  '.mkv',
  '.wmv',
  '.m4v',
  '.flv',
  '.webm',
  '.mp3',
  '.wav',
  '.flac',
  '.ogg',
  '.m4a',
  '.aac',
  '.txt',
]);

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

  // Transform datasets array into rows with objects
  const tableRows = datasets.map(dataset => ({
    dataset,
    name: dataset.name,
    encrypted: dataset.encrypted,
    source: dataset.source || 'local',
    worker: dataset.worker_name || 'Local',
    ref: datasetRowKey(dataset),
    worker_id: datasetWorkerID(dataset),
  }));
  const filteredTableRows = useMemo(() => {
    const query = datasetFilter.trim().toLowerCase();
    if (!query) return tableRows;
    return tableRows.filter(row =>
      [row.name, row.source, row.worker, row.encrypted ? 'encrypted' : 'plain']
        .filter(Boolean)
        .some(value => `${value}`.toLowerCase().includes(query)),
    );
  }, [datasetFilter, tableRows]);

  const selectedDatasets = useMemo(
    () => tableRows.filter(row => selectedDatasetRefs.has(row.ref)).map(row => row.dataset),
    [selectedDatasetRefs, tableRows],
  );
  const selectedWorkerIDs = useMemo(
    () => Array.from(new Set(selectedDatasets.map(datasetWorkerID))),
    [selectedDatasets],
  );
  const selectedWorkerID = selectedWorkerIDs.length === 1 ? selectedWorkerIDs[0] : null;
  const canCombineSelection = selectedDatasets.length >= 2 && selectedWorkerID !== null;
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

  const columns: TableColumn[] = [
    {
      title: '',
      key: 'select',
      className: 'w-10',
      render: row => {
        const selected = selectedDatasetRefs.has(row.ref);
        const disabled = !selected && selectedWorkerID !== null && selectedWorkerID !== row.worker_id;
        return (
          <input
            type="checkbox"
            aria-label={`Select ${row.name}`}
            checked={selected}
            disabled={disabled}
            onChange={() => toggleDatasetSelection(row.dataset)}
            className="h-4 w-4 rounded border-gray-600 bg-gray-900 text-blue-600 disabled:opacity-40"
          />
        );
      },
    },
    {
      title: 'Dataset Name',
      key: 'name',
      render: row => (
        <Link
          href={
            row.source === 'remote'
              ? `/datasets/${encodeURIComponent(row.name)}?worker_id=${encodeURIComponent(row.worker_id)}`
              : `/datasets/${encodeURIComponent(row.name)}`
          }
          className="text-gray-200 hover:text-gray-100"
        >
          {row.name}
        </Link>
      ),
    },
    {
      title: 'Source',
      key: 'source',
      className: 'w-40',
      render: row => (
        <span className={row.source === 'remote' ? 'text-blue-300' : 'text-gray-400'}>
          {row.source === 'remote' ? row.worker : 'Local'}
        </span>
      ),
    },
    {
      title: 'Type',
      key: 'encrypted',
      className: 'w-32',
      render: row => (
        <span className={row.encrypted ? 'text-blue-300' : 'text-gray-400'}>
          {row.encrypted ? 'Encrypted' : 'Plain'}
        </span>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-36 text-right',
      render: row => (
        <div className="flex justify-end gap-1">
          {row.source === 'remote' && (
            <button
              className="text-gray-200 hover:bg-blue-600 p-2 rounded-full transition-colors disabled:opacity-50"
              disabled={importingRef === row.ref}
              onClick={() => handleImportDataset(row.dataset)}
              title="Import to Local"
            >
              <Download className="h-4 w-4" />
            </button>
          )}
          <button
            className="text-gray-200 hover:bg-cyan-700 p-2 rounded-full transition-colors"
            onClick={() => openRenameDatasetModal(row.dataset)}
            title="Rename dataset"
            aria-label={`Rename ${row.name}`}
          >
            <Pencil className="h-4 w-4" />
          </button>
          <button
            className="text-gray-200 hover:bg-red-600 p-2 rounded-full transition-colors"
            onClick={() => handleDeleteDataset(row.name, row.worker_id)}
            title="Delete dataset"
            aria-label={`Delete ${row.name}`}
          >
            <FaRegTrashAlt />
          </button>
        </div>
      ),
    },
  ];

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
        rememberEncryptedDatasetKey(renamedName, rememberedKey);
        if (workerID !== 'local') {
          rememberEncryptedDatasetKey(makeRemoteDatasetRef(workerID, renamedName), rememberedKey);
          rememberEncryptedDatasetKey(remoteDatasetRememberKey(workerID, renamedName), rememberedKey);
        }
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
        rememberEncryptedDatasetKey(importedName, remembered);
        rememberEncryptedDatasetKey(importedPath, remembered);
      }
      if (importedName) router.push(`/datasets/${encodeURIComponent(importedName)}`);
    } catch (error: any) {
      alert(error?.response?.data?.error || 'Failed to import remote dataset.');
    } finally {
      setImportingRef(null);
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
      rememberDatasetKey(source, rawKeyB64);
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
        rememberDatasetKey(
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
        return FOLDER_IMPORT_EXTENSIONS.has(folderImportExtension(entry.relativePath));
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
    rememberDatasetKey(
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
      if (/\.txt$/i.test(relativePath)) {
        captionFiles.set(folderImportCaptionKey(relativePath), entry.file);
      }
    });

    const allocateCatalogName = createFlattenedFileNameAllocator();
    const catalog: EncryptedDatasetCatalog = { version: 1, items: [] };
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
        rememberEncryptedDatasetKey(data.name, rawKeyB64);
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
          <label className="relative block w-full sm:w-80">
            <Search className="pointer-events-none absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
            <input
              value={datasetFilter}
              onChange={event => setDatasetFilter(event.target.value)}
              placeholder="Filter datasets, workers, type"
              className="h-8 w-full border border-gray-800 bg-gray-950 pl-8 pr-3 text-sm text-gray-100 placeholder:text-gray-500 focus:border-cyan-700 focus:outline-none"
            />
          </label>
        </div>
        {errors.length > 0 && (
          <div className="mb-3 rounded-md border border-yellow-700 bg-yellow-950/40 px-3 py-2 text-sm text-yellow-200">
            Some remote datasets could not be loaded:{' '}
            {errors.map(error => `${error.worker_name}: ${error.error}`).join('; ')}
          </div>
        )}
        <UniversalTable
          columns={columns}
          rows={filteredTableRows}
          isLoading={status === 'loading'}
          onRefresh={refreshDatasets}
          emptyTitle={datasetFilter ? 'No datasets match the filter' : 'No datasets found'}
          emptyDescription="Create a dataset or import folders to prepare training data."
          errorMessage={status === 'error' ? 'Datasets could not be loaded.' : null}
        />
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
