'use client';

import { useEffect, useState, useMemo, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { ArrowLeft, Ban, ImageOff, Loader2, Pencil, Trash2 } from 'lucide-react';
import DatasetImageStudio, {
  type BulkCaptionActionRequest,
  type BulkCaptionActionResult,
  type DatasetStudioItem,
  type DeleteImagesResult,
} from '@/components/DatasetImageStudio';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal, useOpenImagesModalOnDrag } from '@/components/AddImagesModal';
import DatasetFolderIcon from '@/components/DatasetFolderIcon';
import { Modal } from '@/components/Modal';
import { openConfirm } from '@/components/ConfirmModal';
import { TextInput } from '@/components/formInputs';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import useSettings from '@/hooks/useSettings';
import { isImage, pathJoin } from '@/utils/basic';
import { getDisplayPath, getMediaUrl } from '@/utils/media';
import AutoCaptionButton from '@/components/AutoCaptionButton';
import DatasetWatchFoldersButton from '@/components/DatasetWatchFoldersButton';
import DatasetWatcherProgressBadge from '@/components/DatasetWatcherProgressBadge';
import { PageNotice } from '@/components/OperatorPrimitives';
import { openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import useDatasetWatcherLiveRefresh from '@/hooks/useDatasetWatcherLiveRefresh';
import { aggregateAutoCaptionProgress } from '@/utils/datasetWatcherStatus';
import type { DatasetSummary, EncryptedDatasetCatalog, EncryptedDatasetItem, EncryptedDatasetManifest } from '@/types';
import {
  arrayBufferToBase64,
  captionObjectPath,
  decryptCatalog,
  encryptCaptionObject,
  encryptCatalog,
  exportRawAesKey,
  getRememberedEncryptedDatasetKey,
  importRawAesKey,
  rememberEncryptedDatasetKey,
  unlockEncryptedDatasetKey,
} from '@/utils/encryptedDatasets';
import { buildEncryptedObjectRequestBody } from '@/utils/encryptedObjectMediaCache';
import { makeRemoteDatasetRef, remoteDatasetRememberKey } from '@/utils/remoteDatasetRefs';
import { parseCaptionKeywordQuery, removeCaptionKeywords } from '@/utils/captionKeywordSearch';

function isPreviewableImagePath(value: string) {
  return isImage(value) || isImage(getDisplayPath(value));
}

type DatasetEditorPageProps = {
  datasetName: string;
  projectID?: string | null;
  datasetRoot?: string | null;
  returnHref?: string | null;
  projectName?: string | null;
};

export default function DatasetEditorPage({
  datasetName,
  projectID = null,
  datasetRoot = null,
  returnHref = null,
  projectName = null,
}: DatasetEditorPageProps) {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const [isAutoCaptioning, setIsAutoCaptioning] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();
  const workerID = projectID ? 'local' : searchParams.get('worker_id') || 'local';
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
  const [isRenameModalOpen, setIsRenameModalOpen] = useState(false);
  const [renameDatasetName, setRenameDatasetName] = useState(datasetName);
  const [isRenamingDataset, setIsRenamingDataset] = useState(false);
  const [renameDatasetError, setRenameDatasetError] = useState('');
  const [isDeletingDataset, setIsDeletingDataset] = useState(false);
  const [datasetActionError, setDatasetActionError] = useState('');
  const [defaultWatchSourcePath, setDefaultWatchSourcePath] = useState('');
  const projectPayload = useMemo(() => (projectID ? { project_id: projectID } : {}), [projectID]);
  const effectiveDatasetRoot = datasetRoot || settings?.DATASETS_FOLDER || '';
  const datasetPath = useMemo(
    () => (effectiveDatasetRoot ? pathJoin(effectiveDatasetRoot, datasetName) : datasetName),
    [datasetName, effectiveDatasetRoot],
  );
  const canUseWatchFolders = !isRemoteDataset && !encryptedManifest;

  const refreshImageList = (dbName: string, options: { background?: boolean } = {}) => {
    if (!options.background) setStatus('loading');
    apiClient
      .post('/api/datasets/listImages', { datasetName: dbName, worker_id: workerID, ...projectPayload })
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
        setImgList(data.images);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching images:', error);
        if (!options.background) setStatus('error');
      });
  };

  const datasetWatcherLive = useDatasetWatcherLiveRefresh({
    enabled: status === 'success' && canUseWatchFolders,
    datasetName,
    projectID,
    workerID,
    onRefresh: () => refreshImageList(datasetName, { background: true }),
  });
  const hasActiveDatasetWatchers = datasetWatcherLive.hasActiveWatchers;
  const autoCaptionProgress = useMemo(
    () => aggregateAutoCaptionProgress(datasetWatcherLive.watchers, datasetWatcherLive.statuses, datasetName),
    [datasetName, datasetWatcherLive.statuses, datasetWatcherLive.watchers],
  );

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
    projectID,
  });

  const unlockEncryptedDataset = async (key: CryptoKey, manifest: EncryptedDatasetManifest) => {
    const catalog = await decryptCatalog(manifest, key);
    const rawKeyB64 = await exportRawAesKey(key);
    setEncryptedKey(key);
    setEncryptedCatalog(catalog);
    setEncryptedRawKeyB64(rawKeyB64);
    rememberEncryptedDatasetKey(datasetName, rawKeyB64);
    if (projectID) {
      rememberEncryptedDatasetKey(`project:${projectID}:${datasetName}`, rawKeyB64);
    }
    if (datasetRef) {
      rememberEncryptedDatasetKey(datasetRef, rawKeyB64);
      rememberEncryptedDatasetKey(remoteDatasetRememberKey(workerID, datasetName), rawKeyB64);
    }
    if (effectiveDatasetRoot) {
      rememberEncryptedDatasetKey(pathJoin(effectiveDatasetRoot, datasetName), rawKeyB64);
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

  const datasetHeaderPreviewUrl = useMemo(() => {
    const firstImagePath = imgList.map(image => image.img_path).find(path => isPreviewableImagePath(path));
    return firstImagePath ? getMediaUrl(firstImagePath) : null;
  }, [imgList]);

  const plainStudioItems = useMemo<DatasetStudioItem[]>(
    () => imgList.map(img => ({ kind: 'plain' as const, path: img.img_path })),
    [imgList],
  );

  const encryptedStudioItems = useMemo<DatasetStudioItem[]>(
    () => (encryptedCatalog?.items || []).map(item => ({ kind: 'encrypted' as const, item })),
    [encryptedCatalog?.items],
  );

  const openJsonConversion = useCallback(() => {
    if (isRemoteDataset) return;
    openCaptionDatasetModal(datasetPath, () => refreshImageList(datasetName), {
      encryptedDatasetKeyB64: encryptedRawKeyB64 || undefined,
      projectID,
      datasetName,
      rootCaption: encryptedCatalog ? encryptedCatalog.rootCaption ?? null : undefined,
      preset: 'ideogram_json',
    });
  }, [datasetName, datasetPath, encryptedCatalog, encryptedRawKeyB64, isRemoteDataset, projectID]);

  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
    }
  }, [datasetName, projectPayload, workerID]);

  useEffect(() => {
    if (!datasetName || isRemoteDataset) {
      setDefaultWatchSourcePath('');
      return;
    }

    let cancelled = false;
    apiClient
      .get('/api/datasets/list', {
        params: {
          worker_id: workerID,
          ...projectPayload,
        },
      })
      .then(res => {
        if (cancelled) return;
        const rawDatasets = Array.isArray(res.data)
          ? res.data
          : Array.isArray(res.data?.datasets)
            ? res.data.datasets
            : [];
        const dataset = rawDatasets.find((item: DatasetSummary) => item?.name === datasetName);
        setDefaultWatchSourcePath(typeof dataset?.importSourcePath === 'string' ? dataset.importSourcePath : '');
      })
      .catch(() => {
        if (!cancelled) setDefaultWatchSourcePath('');
      });

    return () => {
      cancelled = true;
    };
  }, [datasetName, isRemoteDataset, projectPayload, workerID]);

  useEffect(() => {
    if (!encryptedManifest || encryptedKey || encryptedCatalog) return;
    const remembered =
      (datasetRef ? getRememberedEncryptedDatasetKey(datasetRef) : null) ||
      (projectID ? getRememberedEncryptedDatasetKey(`project:${projectID}:${datasetName}`) : null) ||
      getRememberedEncryptedDatasetKey(remoteDatasetRememberKey(workerID, datasetName)) ||
      getRememberedEncryptedDatasetKey(datasetName) ||
      (effectiveDatasetRoot
        ? getRememberedEncryptedDatasetKey(pathJoin(effectiveDatasetRoot, datasetName))
        : null);
    if (!remembered) return;
    importRawAesKey(remembered)
      .then(key => unlockEncryptedDataset(key, encryptedManifest))
      .catch(() => undefined);
  }, [datasetName, datasetRef, effectiveDatasetRoot, encryptedCatalog, encryptedKey, encryptedManifest, projectID, workerID]);

  useEffect(() => {
    if (!encryptedManifest || !encryptedKey) return;
    let cancelled = false;
    decryptCatalog(encryptedManifest, encryptedKey)
      .then(catalog => {
        if (!cancelled) setEncryptedCatalog(catalog);
      })
      .catch(error => {
        console.warn('Could not refresh encrypted dataset catalog:', error);
      });
    return () => {
      cancelled = true;
    };
  }, [encryptedKey, encryptedManifest]);

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
    if (projectID) {
      rememberEncryptedDatasetKey(`project:${projectID}:${renamedName}`, encryptedRawKeyB64);
    }
    if (effectiveDatasetRoot) {
      rememberEncryptedDatasetKey(pathJoin(effectiveDatasetRoot, renamedName), encryptedRawKeyB64);
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
        ...projectPayload,
      });
      const renamedName = res.data?.name || renameDatasetName.trim();
      rememberRenamedEncryptedKey(renamedName);
      setIsRenameModalOpen(false);
      if (projectID) {
        router.replace(`/projects/${encodeURIComponent(projectID)}/datasets/${encodeURIComponent(renamedName)}`);
      } else {
        router.replace(
          isRemoteDataset
            ? `/datasets/${encodeURIComponent(renamedName)}?worker_id=${encodeURIComponent(workerID)}`
            : `/datasets/${encodeURIComponent(renamedName)}`,
        );
      }
    } catch (error: any) {
      setRenameDatasetError(error?.response?.data?.error || 'Failed to rename dataset.');
    } finally {
      setIsRenamingDataset(false);
    }
  };

  const handleDeleteDataset = () => {
    if (isDeletingDataset) return;
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        setIsDeletingDataset(true);
        setDatasetActionError('');
        try {
          await apiClient.post('/api/datasets/delete', {
            name: datasetName,
            worker_id: workerID,
            ...projectPayload,
          });
          router.replace(returnHref || '/datasets');
        } catch (error: any) {
          setDatasetActionError(error?.response?.data?.error || error?.message || 'Failed to delete dataset.');
        } finally {
          setIsDeletingDataset(false);
        }
      },
    });
  };

  const PageInfoContent = (() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (status == 'loading') {
      icon = <Loader2 className="h-8 w-8 animate-spin" />;
      text = 'Loading Images';
      subtitle = 'Please wait while we fetch your dataset images...';
      showIt = true;
      bgColor = 'bg-gray-800/50';
      textColor = 'text-gray-100';
      iconColor = 'text-gray-400';
    }
    if (status == 'error') {
      icon = <Ban className="h-8 w-8" />;
      text = 'Error Loading Images';
      subtitle = 'There was a problem fetching the images. Please try refreshing the page.';
      showIt = true;
      bgColor = 'bg-red-600/20';
      textColor = 'text-red-100';
      iconColor = 'text-red-400';
    }
    if (status == 'success' && !encryptedManifest && imgList.length === 0) {
      icon = <ImageOff className="h-8 w-8" />;
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
            {status === 'success' && !encryptedManifest && imgList.length === 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                <Button
                  className="operator-button whitespace-nowrap border-blue-800 bg-blue-950/70 text-blue-100"
                  onClick={() =>
                    openImagesModal(datasetName, () => refreshImageList(datasetName), {
                      ...encryptedUploadOptions,
                      workerID,
                      projectID,
                    })
                  }
                >
                  Add Images
                </Button>
                {canUseWatchFolders && (
                  <DatasetWatchFoldersButton
                    datasetName={datasetName}
                    projectID={projectID}
                    workerID={workerID}
                    defaultSourcePath={defaultWatchSourcePath}
                    className="operator-button whitespace-nowrap border-cyan-800 bg-cyan-950/70 text-cyan-100"
                    onRefresh={() => refreshImageList(datasetName, { background: true })}
                  />
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  })();

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
      ...projectPayload,
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

  const handleDeleteImages = async (targetItems: DatasetStudioItem[]): Promise<DeleteImagesResult> => {
    const uniqueItems = Array.from(
      new Map(
        targetItems.map(item => [item.kind === 'plain' ? item.path : item.item.id, item] as const),
      ).values(),
    );
    const plainPaths = uniqueItems.flatMap(item => (item.kind === 'plain' ? [item.path] : []));
    const encryptedItems = uniqueItems.flatMap(item => (item.kind === 'encrypted' ? [item.item] : []));
    const removedKeys: string[] = [];
    let deleted = 0;
    let skipped = 0;
    let failed = 0;

    if (plainPaths.length > 0) {
      const response = await apiClient.post('/api/img/delete-bulk', { imgPaths: plainPaths, ...projectPayload });
      const data = response.data || {};
      const removedPaths = Array.isArray(data.removedPaths)
        ? data.removedPaths.filter((value: unknown): value is string => typeof value === 'string')
        : [];
      deleted += Number(data.deleted || 0);
      skipped += Number(data.skipped || 0);
      failed += Number(data.failed || 0);
      removedKeys.push(...removedPaths);
      if (removedPaths.length > 0) {
        const removedPathSet = new Set(removedPaths);
        setImgList(previous => previous.filter(image => !removedPathSet.has(image.img_path)));
      }
    }

    if (encryptedItems.length > 0) {
      if (!encryptedManifest || !encryptedCatalog || !encryptedKey) {
        throw new Error('Unlock the encrypted dataset first.');
      }
      const encryptedIDs = new Set(encryptedItems.map(item => item.id));
      const nextCatalog: EncryptedDatasetCatalog = {
        ...encryptedCatalog,
        items: encryptedCatalog.items.filter(item => !encryptedIDs.has(item.id)),
      };
      const { manifest: nextManifest } = await encryptCatalog(nextCatalog, encryptedKey, encryptedManifest);
      await apiClient.post('/api/datasets/encrypted/update', {
        datasetName,
        worker_id: workerID,
        ...projectPayload,
        manifest: nextManifest,
        deleteObjects: encryptedItems.flatMap(item =>
          [item.objectPath, item.captionObjectPath].filter((value): value is string => Boolean(value)),
        ),
      });
      setEncryptedManifest(nextManifest);
      setEncryptedCatalog(nextCatalog);
      deleted += encryptedItems.length;
      removedKeys.push(...encryptedItems.map(item => item.id));
    }

    return {
      requested: uniqueItems.length,
      deleted,
      skipped,
      failed,
      removedKeys,
    };
  };

  const encryptedObjectUpdate = async (objectPath: string) => {
    const response = await apiClient.post(
      '/api/datasets/encrypted/object',
      buildEncryptedObjectRequestBody({ datasetName, workerID, projectID, objectPath }),
      { responseType: 'blob' },
    );
    const bytes = await (response.data as Blob).arrayBuffer();
    return { objectPath, dataBase64: arrayBufferToBase64(bytes) };
  };

  const handleBulkEncryptedCaptionAction = async (
    request: BulkCaptionActionRequest,
  ): Promise<BulkCaptionActionResult> => {
    if (!encryptedManifest || !encryptedCatalog || !encryptedKey) {
      throw new Error('Unlock the encrypted dataset first.');
    }

    const matches = request.matches.flatMap(match =>
      match.item.kind === 'encrypted' ? [{ ...match, encryptedItem: match.item.item }] : [],
    );
    const matchedItems = Array.from(
      new Map<string, EncryptedDatasetItem>(
        matches.map(match => [match.encryptedItem.id, match.encryptedItem] as const),
      ).values(),
    );
    const matchedIDs = new Set(matchedItems.map(item => item.id));
    const now = new Date().toISOString();

    if (request.action === 'delete') {
      const nextCatalog: EncryptedDatasetCatalog = {
        ...encryptedCatalog,
        items: encryptedCatalog.items.filter(item => !matchedIDs.has(item.id)),
      };
      const { manifest: nextManifest } = await encryptCatalog(nextCatalog, encryptedKey, encryptedManifest);
      await apiClient.post('/api/datasets/encrypted/update', {
        datasetName,
        worker_id: workerID,
        ...projectPayload,
        manifest: nextManifest,
        deleteObjects: matchedItems.flatMap(item => [item.objectPath, item.captionObjectPath].filter(Boolean)),
      });
      setEncryptedManifest(nextManifest);
      setEncryptedCatalog(nextCatalog);
      return {
        action: request.action,
        found: matches.length,
        affected: matchedItems.length,
        deleted: matchedItems.length,
        removedKeys: matchedItems.map(item => item.id),
      };
    }

    if (request.action === 'move') {
      const destinationName = request.destinationName?.trim();
      if (!destinationName) throw new Error('Destination dataset name is required.');

      const { manifest: emptyManifest } = await encryptCatalog({ version: 1, items: [] }, encryptedKey, encryptedManifest);
      const createResponse = await apiClient.post('/api/datasets/create', {
        name: destinationName,
        worker_id: workerID,
        ...projectPayload,
        encrypted: true,
        encryptedManifest: emptyManifest,
      });
      const createdName = createResponse.data?.name || destinationName;

      const objects = [];
      for (const item of matchedItems) {
        objects.push(await encryptedObjectUpdate(item.objectPath));
        if (item.captionObjectPath) objects.push(await encryptedObjectUpdate(item.captionObjectPath));
      }

      const targetCatalog: EncryptedDatasetCatalog = {
        version: 1,
        items: matchedItems.map(item => ({ ...item, updatedAt: now })),
      };
      const { manifest: targetManifest } = await encryptCatalog(targetCatalog, encryptedKey, emptyManifest);
      await apiClient.post('/api/datasets/encrypted/update', {
        datasetName: createdName,
        worker_id: workerID,
        ...projectPayload,
        manifest: targetManifest,
        objects,
      });

      const nextCatalog: EncryptedDatasetCatalog = {
        ...encryptedCatalog,
        items: encryptedCatalog.items.filter(item => !matchedIDs.has(item.id)),
      };
      const { manifest: nextManifest } = await encryptCatalog(nextCatalog, encryptedKey, encryptedManifest);
      await apiClient.post('/api/datasets/encrypted/update', {
        datasetName,
        worker_id: workerID,
        ...projectPayload,
        manifest: nextManifest,
        deleteObjects: matchedItems.flatMap(item => [item.objectPath, item.captionObjectPath].filter(Boolean)),
      });
      setEncryptedManifest(nextManifest);
      setEncryptedCatalog(nextCatalog);
      if (encryptedRawKeyB64) {
        rememberEncryptedDatasetKey(createdName, encryptedRawKeyB64);
        if (projectID) {
          rememberEncryptedDatasetKey(`project:${projectID}:${createdName}`, encryptedRawKeyB64);
        }
        if (effectiveDatasetRoot) {
          rememberEncryptedDatasetKey(pathJoin(effectiveDatasetRoot, createdName), encryptedRawKeyB64);
        }
        if (isRemoteDataset) {
          rememberEncryptedDatasetKey(makeRemoteDatasetRef(workerID, createdName), encryptedRawKeyB64);
          rememberEncryptedDatasetKey(remoteDatasetRememberKey(workerID, createdName), encryptedRawKeyB64);
        }
      }

      return {
        action: request.action,
        found: matches.length,
        affected: matchedItems.length,
        moved: matchedItems.length,
        destinationName: createdName,
        removedKeys: matchedItems.map(item => item.id),
      };
    }

    const updatedCaptions: Record<string, string> = {};
    const updatedItems = new Map<string, EncryptedDatasetItem>();
    const objects: Array<{ objectPath: string; dataBase64: string }> = [];
    let removedWords = 0;
    const terms = parseCaptionKeywordQuery(request.query);

    for (const match of matches) {
      const result = removeCaptionKeywords(match.caption, terms, request.matchMode);
      if (!result.changed) continue;
      const item = match.encryptedItem;
      const targetCaptionPath = item.captionObjectPath || captionObjectPath();
      const encryptedCaption = await encryptCaptionObject(encryptedKey, targetCaptionPath, result.caption);
      objects.push({
        objectPath: targetCaptionPath,
        dataBase64: arrayBufferToBase64(new TextEncoder().encode(JSON.stringify(encryptedCaption))),
      });
      updatedItems.set(item.id, { ...item, captionObjectPath: targetCaptionPath, updatedAt: now });
      updatedCaptions[item.id] = result.caption;
      removedWords += result.removedCount;
    }

    if (updatedItems.size === 0) {
      return { action: request.action, found: matches.length, affected: 0, updated: 0, removedWords: 0 };
    }

    const nextCatalog: EncryptedDatasetCatalog = {
      ...encryptedCatalog,
      items: encryptedCatalog.items.map(item => updatedItems.get(item.id) || item),
    };
    const { manifest: nextManifest } = await encryptCatalog(nextCatalog, encryptedKey, encryptedManifest);
    await apiClient.post('/api/datasets/encrypted/update', {
      datasetName,
      worker_id: workerID,
      ...projectPayload,
      manifest: nextManifest,
      objects,
    });
    setEncryptedManifest(nextManifest);
    setEncryptedCatalog(nextCatalog);

    return {
      action: request.action,
      found: matches.length,
      affected: updatedItems.size,
      updated: updatedItems.size,
      removedWords,
      updatedCaptions,
    };
  };

  return (
    <>
      <TopBar className="h-14 bg-[#070b10]">
        <div className="flex-shrink-0">
          <Button
            className="operator-icon-button"
            onClick={() => {
              if (returnHref) {
                router.push(returnHref);
              } else {
                history.back();
              }
            }}
            title="Back"
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </div>
        <DatasetFolderIcon
          size="sm"
          encrypted={!!encryptedManifest}
          unlocked={!!encryptedCatalog}
          remote={isRemoteDataset}
          previewSrc={datasetHeaderPreviewUrl}
          className="hidden sm:block"
        />
        <div className="min-w-0 flex-shrink text-sm">
          <h1 className="truncate font-semibold text-gray-100">
            <span className="hidden text-gray-400 sm:inline">
              {projectName ? `${projectName} / Datasets / ` : 'AI Toolkit / Datasets / '}
            </span>
            <span>{datasetName}</span>
            <span className="hidden text-gray-500 sm:inline"> / Edit Dataset</span>
            {projectID ? <span className="ml-2 text-xs text-cyan-300">Project</span> : null}
            {isRemoteDataset ? <span className="ml-2 text-xs text-blue-300">Remote</span> : null}
          </h1>
        </div>
        <div className="flex-1"></div>
        <div className="flex-shrink-0 flex items-center gap-1 sm:gap-2">
          <Button
            className="operator-button whitespace-nowrap py-1 text-sm"
            onClick={openRenameModal}
            disabled={isDeletingDataset}
            title="Rename dataset"
            aria-label="Rename dataset"
          >
            <Pencil className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Rename</span>
          </Button>
          {!isRemoteDataset && !encryptedManifest && (
            <>
              <DatasetWatchFoldersButton
                datasetName={datasetName}
                projectID={projectID}
                workerID={workerID}
                defaultSourcePath={defaultWatchSourcePath}
                label="Watch Folders"
                onRefresh={() => refreshImageList(datasetName, { background: true })}
              />
              <DatasetWatcherProgressBadge progress={autoCaptionProgress} className="hidden sm:inline-flex" />
            </>
          )}
          {!isRemoteDataset && (
            <AutoCaptionButton
              datasetPath={datasetPath}
              datasetName={datasetName}
              projectID={projectID}
              setIsAutoCaptioning={setIsAutoCaptioning}
              encryptedDatasetKeyB64={encryptedRawKeyB64 || undefined}
              rootCaption={encryptedCatalog ? encryptedCatalog.rootCaption ?? null : undefined}
            />
          )}
          <Button
            className="operator-button whitespace-nowrap py-1 text-sm"
            disabled={isDeletingDataset || (!!encryptedManifest && !encryptedCatalog)}
            onClick={() =>
              openImagesModal(datasetName, () => refreshImageList(datasetName), {
                ...encryptedUploadOptions,
                workerID,
                projectID,
              })
            }
          >
            <span className="sm:hidden">+ Add</span>
            <span className="hidden sm:inline">Add Images</span>
          </Button>
          <Button
            className="operator-button whitespace-nowrap border-red-900/70 bg-red-950/60 py-1 text-sm text-red-100 hover:bg-red-900"
            onClick={handleDeleteDataset}
            disabled={isDeletingDataset}
            title="Delete dataset"
            aria-label="Delete dataset"
          >
            {isDeletingDataset ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
            <span className="hidden sm:inline">Delete</span>
          </Button>
        </div>
      </TopBar>
      <MainContent className="!top-14 !h-[calc(100%-3.5rem)] overflow-hidden !px-0 !pt-0 sm:!px-0">
        {datasetActionError && (
          <PageNotice tone="danger" title="Dataset action failed" className="mx-auto mt-4 max-w-xl">
            {datasetActionError}
          </PageNotice>
        )}
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
        {status === 'success' && plainStudioItems.length > 0 && (
          <DatasetImageStudio
            datasetName={datasetName}
            workerID={workerID}
            projectID={projectID}
            datasetPath={!isRemoteDataset ? datasetPath : null}
            items={plainStudioItems}
            isAutoCaptioning={isAutoCaptioning}
            liveCaptionRefresh={hasActiveDatasetWatchers}
            onRefresh={() => refreshImageList(datasetName)}
            onAddImages={() =>
              openImagesModal(datasetName, () => refreshImageList(datasetName), {
                ...encryptedUploadOptions,
                workerID,
                projectID,
              })
            }
            onConvertDatasetToJson={!isRemoteDataset ? openJsonConversion : undefined}
            onDeleteImages={handleDeleteImages}
          />
        )}
        {status === 'success' && encryptedCatalog && encryptedKey && encryptedStudioItems.length > 0 && (
          <DatasetImageStudio
            datasetName={datasetName}
            workerID={workerID}
            projectID={projectID}
            datasetPath={!isRemoteDataset ? datasetPath : null}
            items={encryptedStudioItems}
            isAutoCaptioning={isAutoCaptioning}
            encryptedKey={encryptedKey}
            encryptedRawKeyB64={encryptedRawKeyB64}
            rootCaption={encryptedCatalog.rootCaption ?? null}
            onRefresh={() => refreshImageList(datasetName)}
            onAddImages={() =>
              openImagesModal(datasetName, () => refreshImageList(datasetName), {
                ...encryptedUploadOptions,
                workerID,
                projectID,
              })
            }
            onConvertDatasetToJson={!isRemoteDataset ? openJsonConversion : undefined}
            onDeleteImages={handleDeleteImages}
            onBulkEncryptedCaptionAction={handleBulkEncryptedCaptionAction}
            onSaveEncryptedCaption={saveEncryptedCaption}
          />
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
    </>
  );
}
