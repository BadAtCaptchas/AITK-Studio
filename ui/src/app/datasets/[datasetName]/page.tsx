'use client';

import { useEffect, useState, use, useMemo, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { FaChevronLeft } from 'react-icons/fa';
import { VirtuosoGrid } from 'react-virtuoso';
import { Pencil } from 'lucide-react';
import DatasetImageCard from '@/components/DatasetImageCard';
import DatasetImageViewer from '@/components/DatasetImageViewer';
import EncryptedDatasetItemCard from '@/components/EncryptedDatasetItemCard';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal, useOpenImagesModalOnDrag } from '@/components/AddImagesModal';
import { Modal } from '@/components/Modal';
import { TextInput } from '@/components/formInputs';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import useSettings from '@/hooks/useSettings';
import { pathJoin } from '@/utils/basic';
import AutoCaptionButton from '@/components/AutoCaptionButton';
import { PageNotice } from '@/components/OperatorPrimitives';
import type { EncryptedDatasetCatalog, EncryptedDatasetItem, EncryptedDatasetManifest } from '@/types';
import {
  arrayBufferToBase64,
  decryptCatalog,
  encryptCatalog,
  exportRawAesKey,
  getRememberedEncryptedDatasetKey,
  importRawAesKey,
  rememberEncryptedDatasetKey,
  unlockEncryptedDatasetKey,
} from '@/utils/encryptedDatasets';
import { makeRemoteDatasetRef, remoteDatasetRememberKey } from '@/utils/remoteDatasetRefs';

export default function DatasetPage({ params }: { params: Promise<{ datasetName: string }> }) {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const [isAutoCaptioning, setIsAutoCaptioning] = useState(false);
  const usableParams = use(params);
  const datasetName = usableParams.datasetName;
  const router = useRouter();
  const searchParams = useSearchParams();
  const workerID = searchParams.get('worker_id') || 'local';
  const isRemoteDataset = workerID !== 'local';
  const datasetRef = useMemo(
    () => (isRemoteDataset ? makeRemoteDatasetRef(workerID, datasetName) : null),
    [datasetName, isRemoteDataset, workerID],
  );
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const { settings } = useSettings();
  const [encryptedManifest, setEncryptedManifest] = useState<EncryptedDatasetManifest | null>(null);
  const [encryptedCatalog, setEncryptedCatalog] = useState<EncryptedDatasetCatalog | null>(null);
  const [encryptedKey, setEncryptedKey] = useState<CryptoKey | null>(null);
  const [encryptedRawKeyB64, setEncryptedRawKeyB64] = useState<string | null>(null);
  const [unlockPassword, setUnlockPassword] = useState('');
  const [unlockKeyFile, setUnlockKeyFile] = useState<File | null>(null);
  const [unlockError, setUnlockError] = useState<string | null>(null);
  const [selectedImgPath, setSelectedImgPath] = useState<string | null>(null);
  const [captionRefreshKeys, setCaptionRefreshKeys] = useState<Record<string, number>>({});
  const [scrollParent, setScrollParent] = useState<HTMLDivElement | null>(null);
  const [isRenameModalOpen, setIsRenameModalOpen] = useState(false);
  const [renameDatasetName, setRenameDatasetName] = useState(datasetName);
  const [isRenamingDataset, setIsRenamingDataset] = useState(false);
  const [renameDatasetError, setRenameDatasetError] = useState('');
  const scrollParentCallback = useCallback((el: HTMLDivElement | null) => setScrollParent(el), []);

  const refreshImageList = (dbName: string) => {
    setStatus('loading');
    apiClient
      .post('/api/datasets/listImages', { datasetName: dbName, worker_id: workerID })
      .then((res: any) => {
        const data = res.data;
        if (data.encrypted) {
          setEncryptedManifest(data.manifest);
          setImgList([]);
          setSelectedImgPath(null);
          setStatus('success');
          return;
        }

        setEncryptedManifest(null);
        setEncryptedCatalog(null);
        setEncryptedKey(null);
        setEncryptedRawKeyB64(null);
        setImgList(data.images);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching images:', error);
        setStatus('error');
      });
  };

  const encryptedUploadOptions = useMemo(() => {
    if (!encryptedManifest || !encryptedCatalog || !encryptedKey) return undefined;
    return {
      encrypted: {
        manifest: encryptedManifest,
        catalog: encryptedCatalog,
        cryptoKey: encryptedKey,
        onUpdate: (manifest: EncryptedDatasetManifest, catalog: EncryptedDatasetCatalog) => {
          setEncryptedManifest(manifest);
          setEncryptedCatalog(catalog);
        },
      },
    };
  }, [encryptedCatalog, encryptedKey, encryptedManifest]);

  useOpenImagesModalOnDrag(datasetName, () => refreshImageList(datasetName), {
    ...encryptedUploadOptions,
    workerID,
  });

  const unlockEncryptedDataset = async (key: CryptoKey, manifest: EncryptedDatasetManifest) => {
    const catalog = await decryptCatalog(manifest, key);
    const rawKeyB64 = await exportRawAesKey(key);
    setEncryptedKey(key);
    setEncryptedCatalog(catalog);
    setEncryptedRawKeyB64(rawKeyB64);
    rememberEncryptedDatasetKey(datasetName, rawKeyB64);
    if (datasetRef) {
      rememberEncryptedDatasetKey(datasetRef, rawKeyB64);
      rememberEncryptedDatasetKey(remoteDatasetRememberKey(workerID, datasetName), rawKeyB64);
    }
    if (settings?.DATASETS_FOLDER) {
      rememberEncryptedDatasetKey(pathJoin(settings.DATASETS_FOLDER, datasetName), rawKeyB64);
    }
  };

  const handleUnlock = async () => {
    if (!encryptedManifest) return;
    setUnlockError(null);
    try {
      const key =
        encryptedManifest.crypto.kdf.type === 'PBKDF2-SHA256'
          ? (await unlockEncryptedDatasetKey(encryptedManifest, {
              provider: 'password',
              password: unlockPassword,
            })).key
          : encryptedManifest.crypto.kdf.type === 'KEYFILE-SHA256' && unlockKeyFile
            ? (await unlockEncryptedDatasetKey(encryptedManifest, {
                provider: 'keyFile',
                file: unlockKeyFile,
              })).key
            : encryptedManifest.crypto.kdf.type === 'WEBAUTHN-PRF'
              ? (await unlockEncryptedDatasetKey(encryptedManifest, { provider: 'webauthnPrf' })).key
              : null;
      if (!key) {
        setUnlockError('Select the key file for this dataset.');
        return;
      }
      await unlockEncryptedDataset(key, encryptedManifest);
      setUnlockPassword('');
      setUnlockKeyFile(null);
    } catch {
      setUnlockError('Could not decrypt this dataset with the provided secret.');
    }
  };

  const imgPaths = useMemo(() => imgList.map(img => img.img_path), [imgList]);

  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
    }
  }, [datasetName, workerID]);

  useEffect(() => {
    if (!encryptedManifest || encryptedKey || encryptedCatalog) return;
    const remembered =
      (datasetRef ? getRememberedEncryptedDatasetKey(datasetRef) : null) ||
      getRememberedEncryptedDatasetKey(remoteDatasetRememberKey(workerID, datasetName)) ||
      getRememberedEncryptedDatasetKey(datasetName) ||
      (settings?.DATASETS_FOLDER
        ? getRememberedEncryptedDatasetKey(pathJoin(settings.DATASETS_FOLDER, datasetName))
        : null);
    if (!remembered) return;
    importRawAesKey(remembered)
      .then(key => unlockEncryptedDataset(key, encryptedManifest))
      .catch(() => undefined);
  }, [datasetName, datasetRef, encryptedCatalog, encryptedKey, encryptedManifest, settings?.DATASETS_FOLDER, workerID]);

  useEffect(() => {
    if (!isRenameModalOpen) {
      setRenameDatasetName(datasetName);
    }
  }, [datasetName, isRenameModalOpen]);

  const openRenameModal = () => {
    setRenameDatasetName(datasetName);
    setRenameDatasetError('');
    setIsRenameModalOpen(true);
  };

  const closeRenameModal = () => {
    if (isRenamingDataset) return;
    setIsRenameModalOpen(false);
    setRenameDatasetError('');
  };

  const rememberRenamedEncryptedKey = (renamedName: string) => {
    if (!encryptedRawKeyB64) return;
    rememberEncryptedDatasetKey(renamedName, encryptedRawKeyB64);
    if (isRemoteDataset) {
      rememberEncryptedDatasetKey(makeRemoteDatasetRef(workerID, renamedName), encryptedRawKeyB64);
      rememberEncryptedDatasetKey(remoteDatasetRememberKey(workerID, renamedName), encryptedRawKeyB64);
    }
    if (settings?.DATASETS_FOLDER) {
      rememberEncryptedDatasetKey(pathJoin(settings.DATASETS_FOLDER, renamedName), encryptedRawKeyB64);
    }
  };

  const handleRenameDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isRenamingDataset) return;

    try {
      setIsRenamingDataset(true);
      setRenameDatasetError('');
      const res = await apiClient.post('/api/datasets/rename', {
        name: datasetName,
        newName: renameDatasetName,
        worker_id: workerID,
      });
      const renamedName = res.data?.name || renameDatasetName.trim();
      rememberRenamedEncryptedKey(renamedName);
      setIsRenameModalOpen(false);
      router.replace(
        isRemoteDataset
          ? `/datasets/${encodeURIComponent(renamedName)}?worker_id=${encodeURIComponent(workerID)}`
          : `/datasets/${encodeURIComponent(renamedName)}`,
      );
    } catch (error: any) {
      setRenameDatasetError(error?.response?.data?.error || 'Failed to rename dataset.');
    } finally {
      setIsRenamingDataset(false);
    }
  };

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Images';
      subtitle = 'Please wait while we fetch your dataset images...';
      showIt = true;
      bgColor = 'bg-gray-800/50';
      textColor = 'text-gray-100';
      iconColor = 'text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Images';
      subtitle = 'There was a problem fetching the images. Please try refreshing the page.';
      showIt = true;
      bgColor = 'bg-red-600/20';
      textColor = 'text-red-100';
      iconColor = 'text-red-400';
    }
    if (status == 'success' && !encryptedManifest && imgList.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Images Found';
      subtitle = 'This dataset is empty. Click "Add Images" to get started.';
      showIt = true;
      bgColor = 'bg-gray-800/50';
      textColor = 'text-gray-100';
      iconColor = 'text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div className={`mx-auto mt-10 max-w-xl border border-dashed border-gray-700 px-4 py-6 ${bgColor} ${textColor}`}>
        <div className="flex items-start gap-3">
          <div className={`${iconColor} mt-0.5`}>{icon}</div>
          <div>
            <h3 className="text-sm font-semibold">{text}</h3>
            <p className="mt-1 text-sm opacity-75">{subtitle}</p>
          </div>
        </div>
      </div>
    );
  }, [status, imgList.length, encryptedManifest]);

  const saveEncryptedCaption = async (
    item: EncryptedDatasetItem,
    captionPath: string,
    encryptedCaptionJson: string,
  ) => {
    if (!encryptedManifest || !encryptedCatalog || !encryptedKey) return;
    const updatedItems = encryptedCatalog.items.map(existing =>
      existing.id === item.id
        ? { ...existing, captionObjectPath: captionPath, updatedAt: new Date().toISOString() }
        : existing,
    );
    const nextCatalog: EncryptedDatasetCatalog = { ...encryptedCatalog, items: updatedItems };
    const { manifest: nextManifest } = await encryptCatalog(nextCatalog, encryptedKey, encryptedManifest);
    await apiClient.post('/api/datasets/encrypted/update', {
      datasetName,
      worker_id: workerID,
      manifest: nextManifest,
      objects: [
        {
          objectPath: captionPath,
          dataBase64: arrayBufferToBase64(new TextEncoder().encode(encryptedCaptionJson)),
        },
      ],
    });
    setEncryptedManifest(nextManifest);
    setEncryptedCatalog(nextCatalog);
  };

  const deleteEncryptedItem = async (item: EncryptedDatasetItem) => {
    if (!encryptedManifest || !encryptedCatalog || !encryptedKey) return;
    const nextCatalog: EncryptedDatasetCatalog = {
      ...encryptedCatalog,
      items: encryptedCatalog.items.filter(existing => existing.id !== item.id),
    };
    const { manifest: nextManifest } = await encryptCatalog(nextCatalog, encryptedKey, encryptedManifest);
    await apiClient.post('/api/datasets/encrypted/update', {
      datasetName,
      worker_id: workerID,
      manifest: nextManifest,
      deleteObjects: [item.objectPath, item.captionObjectPath].filter(Boolean),
    });
    setEncryptedManifest(nextManifest);
    setEncryptedCatalog(nextCatalog);
  };

  return (
    <>
      <TopBar>
        <div className="flex-shrink-0">
          <Button className="operator-icon-button" onClick={() => history.back()} title="Back">
            <FaChevronLeft />
          </Button>
        </div>
        <div className="min-w-0 flex-shrink">
          <h1 className="truncate text-base font-semibold">
            <span className="hidden sm:inline">Dataset: </span>
            {datasetName}
            {isRemoteDataset ? <span className="ml-2 text-xs text-blue-300">Remote</span> : null}
          </h1>
        </div>
        <div className="flex-1"></div>
        <div className="flex-shrink-0 flex items-center gap-1 sm:gap-2">
          <Button
            className="operator-button whitespace-nowrap py-1 text-sm"
            onClick={openRenameModal}
            title="Rename dataset"
            aria-label="Rename dataset"
          >
            <Pencil className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Rename</span>
          </Button>
          {!isRemoteDataset && (
            <AutoCaptionButton
              datasetPath={`${pathJoin(settings.DATASETS_FOLDER, datasetName)}`}
              setIsAutoCaptioning={setIsAutoCaptioning}
              encryptedDatasetKeyB64={encryptedRawKeyB64 || undefined}
            />
          )}
          <Button
            className="operator-button whitespace-nowrap py-1 text-sm"
            disabled={!!encryptedManifest && !encryptedCatalog}
            onClick={() =>
              openImagesModal(datasetName, () => refreshImageList(datasetName), {
                ...encryptedUploadOptions,
                workerID,
              })
            }
          >
            <span className="sm:hidden">+ Add</span>
            <span className="hidden sm:inline">Add Images</span>
          </Button>
        </div>
      </TopBar>
      <MainContent ref={scrollParentCallback}>
        {encryptedManifest && !encryptedCatalog && (
          <div className="mx-auto mt-10 max-w-md border border-gray-700 bg-gray-900 p-5 text-gray-200">
            <h2 className="text-base font-semibold">Encrypted Dataset Locked</h2>
            <p className="mt-2 text-sm text-gray-400">
              Unlock in this browser to preview, edit captions, upload, or start encrypted training.
            </p>
            <div className="mt-4 space-y-3">
              {encryptedManifest.crypto.kdf.type === 'PBKDF2-SHA256' ? (
                <input
                  type="password"
                  value={unlockPassword}
                  onChange={e => setUnlockPassword(e.target.value)}
                  placeholder="Dataset password"
                  className="w-full border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                />
              ) : encryptedManifest.crypto.kdf.type === 'KEYFILE-SHA256' ? (
                <input
                  type="file"
                  onChange={e => setUnlockKeyFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm text-gray-300 file:mr-3 file:border-0 file:bg-gray-700 file:px-3 file:py-2 file:text-gray-100"
                />
              ) : (
                <div className="border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-300">
                  YubiKey / USB Security Key
                </div>
              )}
              {unlockError && <div className="text-sm text-red-400">{unlockError}</div>}
              <Button className="operator-button w-full border-cyan-800 bg-cyan-950/60 text-cyan-100" onClick={handleUnlock}>
                Unlock
              </Button>
            </div>
          </div>
        )}
        {PageInfoContent}
        {status === 'success' && encryptedCatalog && encryptedKey && encryptedCatalog.items.length === 0 && (
          <PageNotice tone="neutral" title="No images found" className="mx-auto mt-10 max-w-xl">
            This encrypted dataset is empty. Add images to make it available for training.
          </PageNotice>
        )}
        {status === 'success' && imgList.length > 0 && scrollParent && (
          <VirtuosoGrid
            totalCount={imgList.length}
            customScrollParent={scrollParent}
            overscan={400}
            listClassName="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"
            itemContent={index => {
              const img = imgList[index];
              if (!img) return null;
              return (
                <DatasetImageCard
                  alt="image"
                  isAutoCaptioning={isAutoCaptioning}
                  imageUrl={img.img_path}
                  onDelete={() => refreshImageList(datasetName)}
                  onImageClick={() => setSelectedImgPath(img.img_path)}
                  captionRefreshKey={captionRefreshKeys[img.img_path] || 0}
                  observerRoot={scrollParent}
                />
              );
            }}
            computeItemKey={index => imgList[index]?.img_path ?? index}
          />
        )}
        {status === 'success' && encryptedCatalog && encryptedKey && encryptedCatalog.items.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {encryptedCatalog.items.map(item => (
              <EncryptedDatasetItemCard
                key={item.id}
                datasetName={datasetName}
                workerID={workerID}
                item={item}
                cryptoKey={encryptedKey}
                isAutoCaptioning={isAutoCaptioning}
                onSaveCaption={saveEncryptedCaption}
                onDelete={deleteEncryptedItem}
              />
            ))}
          </div>
        )}
      </MainContent>
      <Modal
        isOpen={isRenameModalOpen}
        onClose={closeRenameModal}
        title="Rename Dataset"
        size="md"
        closeOnOverlayClick={!isRenamingDataset}
      >
        <form onSubmit={handleRenameDataset} className="space-y-4 text-gray-200">
          <TextInput label="Dataset Name" value={renameDatasetName} onChange={setRenameDatasetName} />
          {renameDatasetError && <div className="text-sm text-red-400">{renameDatasetError}</div>}
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 disabled:opacity-50"
              onClick={closeRenameModal}
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
      <AddImagesModal />
      <DatasetImageViewer
        imgPath={selectedImgPath}
        imageList={imgPaths}
        onChange={setSelectedImgPath}
        refreshImages={() => refreshImageList(datasetName)}
        onCaptionSaved={path => setCaptionRefreshKeys(prev => ({ ...prev, [path]: (prev[path] || 0) + 1 }))}
      />
    </>
  );
}
