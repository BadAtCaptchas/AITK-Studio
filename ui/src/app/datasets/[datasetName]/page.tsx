'use client';

import { useEffect, useState, use, useMemo } from 'react';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { FaChevronLeft } from 'react-icons/fa';
import DatasetImageCard from '@/components/DatasetImageCard';
import EncryptedDatasetItemCard from '@/components/EncryptedDatasetItemCard';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal, useOpenImagesModalOnDrag } from '@/components/AddImagesModal';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import useSettings from '@/hooks/useSettings';
import { pathJoin } from '@/utils/basic';
import AutoCaptionButton from '@/components/AutoCaptionButton';
import type { EncryptedDatasetCatalog, EncryptedDatasetItem, EncryptedDatasetManifest } from '@/types';
import {
  arrayBufferToBase64,
  decryptCatalog,
  deriveKeyFileKey,
  derivePasswordKey,
  encryptCatalog,
  exportRawAesKey,
  getRememberedEncryptedDatasetKey,
  importRawAesKey,
  rememberEncryptedDatasetKey,
} from '@/utils/encryptedDatasets';

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const [isAutoCaptioning, setIsAutoCaptioning] = useState(false);
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const { settings, isSettingsLoaded } = useSettings();
  const [encryptedManifest, setEncryptedManifest] = useState<EncryptedDatasetManifest | null>(null);
  const [encryptedCatalog, setEncryptedCatalog] = useState<EncryptedDatasetCatalog | null>(null);
  const [encryptedKey, setEncryptedKey] = useState<CryptoKey | null>(null);
  const [encryptedRawKeyB64, setEncryptedRawKeyB64] = useState<string | null>(null);
  const [unlockPassword, setUnlockPassword] = useState('');
  const [unlockKeyFile, setUnlockKeyFile] = useState<File | null>(null);
  const [unlockError, setUnlockError] = useState<string | null>(null);

  const refreshImageList = (dbName: string) => {
    setStatus('loading');
    console.log('Fetching images for dataset:', dbName);
    apiClient
      .post('/api/datasets/listImages', { datasetName: dbName })
      .then((res: any) => {
        const data = res.data;
        if (data.encrypted) {
          setEncryptedManifest(data.manifest);
          setImgList([]);
          setStatus('success');
          return;
        }
        setEncryptedManifest(null);
        setEncryptedCatalog(null);
        setEncryptedKey(null);
        setEncryptedRawKeyB64(null);
        console.log('Images:', data.images);
        // sort
        data.images.sort((a: { img_path: string }, b: { img_path: string }) => a.img_path.localeCompare(b.img_path));
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

  useOpenImagesModalOnDrag(datasetName, () => refreshImageList(datasetName), encryptedUploadOptions);

  const unlockEncryptedDataset = async (key: CryptoKey, manifest: EncryptedDatasetManifest) => {
    const catalog = await decryptCatalog(manifest, key);
    const rawKeyB64 = await exportRawAesKey(key);
    setEncryptedKey(key);
    setEncryptedCatalog(catalog);
    setEncryptedRawKeyB64(rawKeyB64);
    rememberEncryptedDatasetKey(datasetName, rawKeyB64);
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
          ? await derivePasswordKey(unlockPassword, encryptedManifest)
          : unlockKeyFile
            ? await deriveKeyFileKey(unlockKeyFile)
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

  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
    }
  }, [datasetName]);

  useEffect(() => {
    if (!encryptedManifest || encryptedKey || encryptedCatalog) return;
    const remembered =
      getRememberedEncryptedDatasetKey(datasetName) ||
      (settings?.DATASETS_FOLDER
        ? getRememberedEncryptedDatasetKey(pathJoin(settings.DATASETS_FOLDER, datasetName))
        : null);
    if (!remembered) return;
    importRawAesKey(remembered)
      .then(key => unlockEncryptedDataset(key, encryptedManifest))
      .catch(() => undefined);
  }, [datasetName, encryptedCatalog, encryptedKey, encryptedManifest, settings?.DATASETS_FOLDER]);

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
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
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
      manifest: nextManifest,
      deleteObjects: [item.objectPath, item.captionObjectPath].filter(Boolean),
    });
    setEncryptedManifest(nextManifest);
    setEncryptedCatalog(nextCatalog);
  };

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">Dataset: {datasetName}</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <AutoCaptionButton
            datasetPath={`${pathJoin(settings.DATASETS_FOLDER, datasetName)}`}
            setIsAutoCaptioning={setIsAutoCaptioning}
            encryptedDatasetKeyB64={encryptedRawKeyB64 || undefined}
          />
          <Button
            className="text-white bg-slate-600 px-3 py-1 rounded-md disabled:opacity-50"
            disabled={!!encryptedManifest && !encryptedCatalog}
            onClick={() => openImagesModal(datasetName, () => refreshImageList(datasetName), encryptedUploadOptions)}
          >
            Add Images
          </Button>
        </div>
      </TopBar>
      <MainContent>
        {encryptedManifest && !encryptedCatalog && (
          <div className="mx-auto mt-10 max-w-md rounded-md border border-gray-700 bg-gray-900 p-5 text-gray-200">
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
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-gray-100"
                />
              ) : (
                <input
                  type="file"
                  onChange={e => setUnlockKeyFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm text-gray-300 file:mr-3 file:rounded-md file:border-0 file:bg-gray-700 file:px-3 file:py-2 file:text-gray-100"
                />
              )}
              {unlockError && <div className="text-sm text-red-400">{unlockError}</div>}
              <Button className="w-full rounded-md bg-blue-600 px-3 py-2 text-white" onClick={handleUnlock}>
                Unlock
              </Button>
            </div>
          </div>
        )}
        {PageInfoContent}
        {status === 'success' && encryptedCatalog && encryptedKey && encryptedCatalog.items.length === 0 && (
          <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-800/50 text-gray-100 mx-auto max-w-md text-center">
            <div className="text-gray-400 mb-4">
              <LuImageOff className="w-8 h-8" />
            </div>
            <h3 className="text-lg font-semibold mb-2">No Images Found</h3>
            <p className="text-sm opacity-75 leading-relaxed">This encrypted dataset is empty. Click "Add Images" to get started.</p>
          </div>
        )}
        {status === 'success' && imgList.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {imgList.map(img => (
              <DatasetImageCard
                key={img.img_path}
                alt="image"
                isAutoCaptioning={isAutoCaptioning}
                imageUrl={img.img_path}
                onDelete={() => refreshImageList(datasetName)}
              />
            ))}
          </div>
        )}
        {status === 'success' && encryptedCatalog && encryptedKey && encryptedCatalog.items.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {encryptedCatalog.items.map(item => (
              <EncryptedDatasetItemCard
                key={item.id}
                datasetName={datasetName}
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
      <AddImagesModal />
    </>
  );
}
