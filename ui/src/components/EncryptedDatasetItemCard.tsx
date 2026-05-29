import React, { KeyboardEvent, useCallback, useEffect, useRef, useState } from 'react';
import { FaEye, FaEyeSlash, FaTrashAlt } from 'react-icons/fa';
import classNames from 'classnames';
import type { EncryptedDatasetItem } from '@/types';
import { apiClient } from '@/utils/api';
import {
  decryptEncryptedObjectBlob,
  encryptCaptionObject,
  captionObjectPath,
  randomId,
} from '@/utils/encryptedDatasets';
import { openConfirm } from './ConfirmModal';
import AudioPlayer from './AudioPlayer';

type EncryptedDatasetItemCardProps = {
  datasetName: string;
  workerID?: string;
  item: EncryptedDatasetItem;
  cryptoKey: CryptoKey;
  isAutoCaptioning: boolean;
  onSaveCaption: (item: EncryptedDatasetItem, captionObjectPath: string, encryptedCaptionJson: string) => Promise<void>;
  onDelete: (item: EncryptedDatasetItem) => Promise<void>;
};

const EncryptedDatasetItemCard: React.FC<EncryptedDatasetItemCardProps> = ({
  datasetName,
  workerID = 'local',
  item,
  cryptoKey,
  isAutoCaptioning,
  onSaveCaption,
  onDelete,
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(true);
  const [inViewport, setInViewport] = useState(false);
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  const [caption, setCaption] = useState('');
  const [savedCaption, setSavedCaption] = useState('');
  const [captionLoaded, setCaptionLoaded] = useState(false);
  const [loading, setLoading] = useState(false);

  const fetchEncryptedObject = useCallback(
    async (objectPath: string) => {
      const res = await apiClient.post(
        '/api/datasets/encrypted/object',
        { datasetName, objectPath, worker_id: workerID },
        { responseType: 'blob' },
      );
      return res.data as Blob;
    },
    [datasetName, workerID],
  );

  const loadMedia = useCallback(async () => {
    if (mediaUrl || loading) return;
    setLoading(true);
    try {
      const blob = await fetchEncryptedObject(item.objectPath);
      const decrypted = await decryptEncryptedObjectBlob(cryptoKey, item.objectPath, blob);
      const url = URL.createObjectURL(new Blob([decrypted], { type: item.mimeType || 'application/octet-stream' }));
      setMediaUrl(url);
    } finally {
      setLoading(false);
    }
  }, [cryptoKey, fetchEncryptedObject, item.mimeType, item.objectPath, loading, mediaUrl]);

  const loadCaption = useCallback(async () => {
    if (captionLoaded) return;
    if (!item.captionObjectPath) {
      setCaption('');
      setSavedCaption('');
      setCaptionLoaded(true);
      return;
    }
    try {
      const blob = await fetchEncryptedObject(item.captionObjectPath);
      const decrypted = await decryptEncryptedObjectBlob(cryptoKey, item.captionObjectPath, blob);
      const text = new TextDecoder().decode(decrypted);
      setCaption(text);
      setSavedCaption(text);
    } catch (error) {
      console.error('Error loading encrypted caption:', error);
    } finally {
      setCaptionLoaded(true);
    }
  }, [captionLoaded, cryptoKey, fetchEncryptedObject, item.captionObjectPath]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        setInViewport(entries[0].isIntersecting);
      },
      { threshold: 0.1 },
    );
    if (cardRef.current) observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!inViewport || !isVisible) return;
    void loadMedia();
    void loadCaption();
  }, [inViewport, isVisible, loadCaption, loadMedia]);

  useEffect(() => {
    return () => {
      if (mediaUrl) URL.revokeObjectURL(mediaUrl);
    };
  }, [mediaUrl]);

  const saveCaption = async () => {
    const trimmedCaption = caption.trim();
    if (trimmedCaption === savedCaption) return;
    const targetCaptionPath = item.captionObjectPath || captionObjectPath(randomId());
    const encryptedCaption = await encryptCaptionObject(cryptoKey, targetCaptionPath, trimmedCaption);
    await onSaveCaption(item, targetCaptionPath, JSON.stringify(encryptedCaption));
    setSavedCaption(trimmedCaption);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void saveCaption();
    }
  };

  const captionCurrent = caption.trim() === savedCaption;

  return (
    <div className="flex flex-col">
      <div ref={cardRef} className="relative w-full" style={{ paddingBottom: '100%' }}>
        <div className="absolute inset-0 rounded-t-lg bg-gray-900 shadow-md">
          {inViewport && isVisible && mediaUrl && item.mediaKind === 'image' && (
            <img src={mediaUrl} alt={item.name} className="h-full w-full object-contain" />
          )}
          {inViewport && isVisible && mediaUrl && item.mediaKind === 'video' && (
            <video src={mediaUrl} className="h-full w-full object-contain" controls loop />
          )}
          {inViewport && isVisible && mediaUrl && item.mediaKind === 'audio' && (
            <AudioPlayer src={mediaUrl} title={item.name} />
          )}
          {(!isVisible || !mediaUrl) && (
            <div className="absolute inset-0 flex items-center justify-center rounded-t-lg bg-gray-800 text-sm text-gray-400">
              {isVisible ? 'Decrypting...' : ''}
            </div>
          )}
          <div className="absolute top-1 right-1 z-10 flex gap-2">
            <button className="rounded-full bg-gray-800 p-2" onClick={() => setIsVisible(value => !value)}>
              {isVisible ? <FaEyeSlash /> : <FaEye />}
            </button>
            <button
              className="rounded-full bg-gray-800 p-2"
              onClick={() => {
                openConfirm({
                  title: 'Delete encrypted item',
                  message: `Delete "${item.name}" from this encrypted dataset?`,
                  type: 'warning',
                  confirmText: 'Delete',
                  onConfirm: () => void onDelete(item),
                });
              }}
            >
              <FaTrashAlt />
            </button>
          </div>
        </div>
      </div>
      <div
        className={classNames('h-[75px] w-full rounded-b-lg bg-gray-800 p-2 text-sm text-white', {
          'border-2 border-blue-500': !captionCurrent,
          'border-2 border-transparent': captionCurrent,
        })}
      >
        {captionLoaded ? (
          <form
            onSubmit={e => {
              e.preventDefault();
              void saveCaption();
            }}
            onBlur={() => void saveCaption()}
          >
            <textarea
              className={classNames('w-full resize-none bg-transparent outline-none focus:outline-none focus:ring-0', {
                'cursor-not-allowed opacity-50': isAutoCaptioning,
              })}
              value={caption}
              rows={3}
              readOnly={isAutoCaptioning}
              onChange={e => setCaption(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          </form>
        ) : (
          <div className="flex h-full items-center justify-center text-gray-400">Loading caption...</div>
        )}
      </div>
    </div>
  );
};

export default EncryptedDatasetItemCard;
