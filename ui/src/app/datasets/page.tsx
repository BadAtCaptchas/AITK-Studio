'use client';

import { useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';
import {
  createEmptyEncryptedManifest,
  rememberEncryptedDatasetKey,
} from '@/utils/encryptedDatasets';

export default function Datasets() {
  const router = useRouter();
  const { datasets, status, refreshDatasets } = useDatasetList();
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [newDatasetMode, setNewDatasetMode] = useState<'plain' | 'encrypted'>('plain');
  const [credentialMode, setCredentialMode] = useState<'password' | 'keyFile'>('password');
  const [datasetPassword, setDatasetPassword] = useState('');
  const [datasetPasswordConfirm, setDatasetPasswordConfirm] = useState('');
  const [datasetKeyFile, setDatasetKeyFile] = useState<File | null>(null);
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);

  // Transform datasets array into rows with objects
  const tableRows = datasets.map(dataset => ({
    name: dataset.name,
    encrypted: dataset.encrypted,
    actions: dataset.name, // Pass full dataset name for actions
  }));

  const columns: TableColumn[] = [
    {
      title: 'Dataset Name',
      key: 'name',
      render: row => (
        <Link href={`/datasets/${row.name}`} className="text-gray-200 hover:text-gray-100">
          {row.name}
        </Link>
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
      className: 'w-20 text-right',
      render: row => (
        <button
          className="text-gray-200 hover:bg-red-600 p-2 rounded-full transition-colors"
          onClick={() => handleDeleteDataset(row.name)}
        >
          <FaRegTrashAlt />
        </button>
      ),
    },
  ];

  const handleDeleteDataset = (datasetName: string) => {
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/datasets/delete', { name: datasetName })
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
        <div>
          <Button
            className="text-white bg-slate-600 px-3 py-1 rounded-md hover:bg-slate-500 transition-colors"
            onClick={() => openNewDatasetModal()}
          >
            New Dataset
          </Button>
        </div>
      </TopBar>

      <MainContent>
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
    </>
  );
}
