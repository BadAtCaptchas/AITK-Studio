'use client';

import { useEffect, useMemo, useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { Download } from 'lucide-react';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';
import {
  createEmptyEncryptedManifest,
  deriveKeyFileKey,
  derivePasswordKey,
  exportRawAesKey,
  getRememberedEncryptedDatasetKey,
  rememberEncryptedDatasetKey,
} from '@/utils/encryptedDatasets';
import { makeRemoteDatasetRef, remoteDatasetRememberKey } from '@/utils/remoteDatasetRefs';
import type { DatasetSummary, EncryptedDatasetManifest } from '@/types';

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

export default function Datasets() {
  const router = useRouter();
  const { datasets, errors, status, refreshDatasets } = useDatasetList({ includeRemote: true });
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [newDatasetMode, setNewDatasetMode] = useState<'plain' | 'encrypted'>('plain');
  const [credentialMode, setCredentialMode] = useState<'password' | 'keyFile'>('password');
  const [datasetPassword, setDatasetPassword] = useState('');
  const [datasetPasswordConfirm, setDatasetPasswordConfirm] = useState('');
  const [datasetKeyFile, setDatasetKeyFile] = useState<File | null>(null);
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);
  const [importingRef, setImportingRef] = useState<string | null>(null);
  const [selectedDatasetRefs, setSelectedDatasetRefs] = useState<Set<string>>(() => new Set());
  const [isCombineModalOpen, setIsCombineModalOpen] = useState(false);
  const [combineSources, setCombineSources] = useState<DatasetSummary[]>([]);
  const [combineOutputName, setCombineOutputName] = useState('');
  const [combineOutputMode, setCombineOutputMode] = useState<'plain' | 'encrypted'>('plain');
  const [combineCredentialMode, setCombineCredentialMode] = useState<'password' | 'keyFile'>('password');
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
      className: 'w-28 text-right',
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
            className="text-gray-200 hover:bg-red-600 p-2 rounded-full transition-colors"
            onClick={() => handleDeleteDataset(row.name, row.worker_id)}
            title="Delete dataset"
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
      const key =
        manifest.crypto.kdf.type === 'PBKDF2-SHA256'
          ? await derivePasswordKey(combineSourcePasswords[ref] || '', manifest)
          : combineSourceKeyFiles[ref]
            ? await deriveKeyFileKey(combineSourceKeyFiles[ref] as File)
            : null;
      if (!key) {
        setCombineSourceErrors(previous => ({ ...previous, [ref]: 'Select the key file for this dataset.' }));
        return;
      }
      const rawKeyB64 = await exportRawAesKey(key);
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
          if (!combineKeyFile) {
            alert('Select a key file.');
            return;
          }
          const result = await createEmptyEncryptedManifest('keyFile', combineKeyFile);
          outputEncryptedManifest = result.manifest;
          outputKeyB64 = result.rawKeyB64;
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
          if (!datasetKeyFile) {
            alert('Select a key file.');
            return;
          }
          const result = await createEmptyEncryptedManifest('keyFile', datasetKeyFile);
          encryptedManifest = result.manifest;
          rawKeyB64 = result.rawKeyB64;
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
        <div>
          <h1 className="text-lg">Datasets</h1>
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
            className="text-white bg-slate-600 px-3 py-1 rounded-md hover:bg-slate-500 transition-colors disabled:cursor-not-allowed disabled:opacity-40"
            disabled={!canCombineSelection}
            onClick={() => openCombineModal()}
          >
            Combine
          </Button>
          <Button
            className="text-white bg-slate-600 px-3 py-1 rounded-md hover:bg-slate-500 transition-colors"
            onClick={() => openNewDatasetModal()}
          >
            New Dataset
          </Button>
        </div>
      </TopBar>

      <MainContent>
        {errors.length > 0 && (
          <div className="mb-3 rounded-md border border-yellow-700 bg-yellow-950/40 px-3 py-2 text-sm text-yellow-200">
            Some remote datasets could not be loaded:{' '}
            {errors.map(error => `${error.worker_name}: ${error.error}`).join('; ')}
          </div>
        )}
        <UniversalTable
          columns={columns}
          rows={tableRows}
          isLoading={status === 'loading'}
          onRefresh={refreshDatasets}
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
                <div className="grid grid-cols-2 gap-2">
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
                ) : (
                  <input
                    type="file"
                    onChange={e => setDatasetKeyFile(e.target.files?.[0] || null)}
                    className="block w-full text-sm text-gray-300 file:mr-3 file:rounded-md file:border-0 file:bg-gray-700 file:px-3 file:py-2 file:text-gray-100"
                  />
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
                        ) : (
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
              <div className="grid grid-cols-2 gap-2">
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
              ) : (
                <input
                  type="file"
                  onChange={e => setCombineKeyFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm text-gray-300 file:mr-3 file:rounded-md file:border-0 file:bg-gray-700 file:px-3 file:py-2 file:text-gray-100"
                />
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
