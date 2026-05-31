import React from 'react';
import useFilesList from '@/hooks/useFilesList';
import { Loader2, AlertCircle, Download, Box, Brain, Trash2, SlidersHorizontal } from 'lucide-react';
import { openMergeLoRAsModal } from './MergeLoRAsModal';
import { getDisplayPath, getDownloadUrl, parseRemoteAssetRef } from '@/utils/media';
import { openConfirm } from './ConfirmModal';
import { apiClient } from '@/utils/api';

const getFilename = (filePath: string) => getDisplayPath(filePath).split(/[\\/]/).pop() || '';
const getFoldername = (filePath: string) => filePath.replace(/[\\/][^\\/]*$/, '');

export default function FilesWidget({ jobID, jobName }: { jobID: string; jobName?: string }) {
  const { files, status, refreshFiles } = useFilesList(jobID, 5000);
  const localFiles = files.filter(file => !parseRemoteAssetRef(file.path));

  const isOptimizerFile = (filePath: string) => getFilename(filePath) === 'optimizer.pt';
  const checkpointFiles = files.filter(file => !isOptimizerFile(file.path));
  const localCheckpointFiles = localFiles.filter(file => !isOptimizerFile(file.path));
  const optimizerFile = files.find(file => isOptimizerFile(file.path));

  const cleanSize = (size: number) => {
    if (size < 1024) {
      return `${size} B`;
    } else if (size < 1024 * 1024) {
      return `${(size / 1024).toFixed(1)} KB`;
    } else if (size < 1024 * 1024 * 1024) {
      return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    } else {
      return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    }
  };

  const handleDeleteFile = (filePath: string) => {
    const fileName = getFilename(filePath);
    openConfirm({
      title: 'Delete Checkpoint',
      message: `Are you sure you want to delete "${fileName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        try {
          await apiClient.post('/api/files/delete', { filePath });
          refreshFiles();
        } catch (error) {
          console.error('Error deleting checkpoint:', error);
          alert('Failed to delete checkpoint. Please try again.');
        }
      },
    });
  };

  return (
    <div className="col-span-2 overflow-hidden border border-gray-800 bg-gray-900/60">
      <div className="flex items-center justify-between border-b border-gray-800 bg-gray-900 px-3 py-2">
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          <h2 className="font-semibold text-gray-100">Checkpoints</h2>
          <span className="border border-gray-700 bg-gray-950 px-2 py-0.5 text-xs text-gray-300">
            {checkpointFiles.length}
          </span>
        </div>
        {localCheckpointFiles.length > 0 && (
          <button
            type="button"
            className="border border-purple-800 bg-purple-950/40 px-3 py-1 text-xs font-medium uppercase text-purple-200 hover:bg-purple-900/50"
            onClick={() => {
              const firstPath = localCheckpointFiles[0].path;
              openMergeLoRAsModal(
                getFoldername(firstPath),
                `${jobName || 'job'}_merged`,
                localCheckpointFiles.map(file => ({
                  path: file.path,
                  label: getFilename(file.path).replace(/\.safetensors$/i, ''),
                })),
                refreshFiles,
              );
            }}
          >
            Merge
          </button>
        )}
      </div>

      <div className="p-2">
        {status === 'loading' && (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="w-5 h-5 text-gray-400 animate-spin" />
          </div>
        )}

        {status === 'error' && (
          <div className="flex items-center justify-center py-4 text-rose-400 space-x-2">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">Error loading checkpoints</span>
          </div>
        )}

        {['success', 'refreshing'].includes(status) && (
          <div className="space-y-1">
            {checkpointFiles.map((file, index) => {
              const fileName = getFilename(file.path);
              const nameWithoutExt = fileName.replace('.safetensors', '');
              const isRemote = !!parseRemoteAssetRef(file.path);
              const downloadUrl = getDownloadUrl(file.path);
              return (
                <div
                  key={index}
                  className="group flex items-center justify-between px-2 py-1.5 transition-all duration-200 hover:bg-gray-800"
                >
                  <a target="_blank" href={downloadUrl} className="flex items-center space-x-2 min-w-0 flex-1">
                    <Box className="w-4 h-4 text-purple-600 dark:text-purple-400 flex-shrink-0" />
                    <div className="flex flex-col min-w-0">
                      <div className="flex text-sm text-gray-200">
                        <span className="overflow-hidden text-ellipsis direction-rtl whitespace-nowrap">
                          {nameWithoutExt}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">.safetensors</span>
                    </div>
                  </a>
                  <div className="flex items-center space-x-3 flex-shrink-0">
                    <span className="text-xs text-gray-400">{cleanSize(file.size)}</span>
                    <a
                      target="_blank"
                      href={downloadUrl}
                      className="bg-purple-500 bg-opacity-0 p-1 transition-all group-hover:bg-opacity-10"
                      title="Download checkpoint"
                    >
                      <Download className="w-3 h-3 text-purple-600 dark:text-purple-400" />
                    </a>
                    {!isRemote && (
                      <button
                        type="button"
                        onClick={() => handleDeleteFile(file.path)}
                        className="bg-red-500 bg-opacity-0 p-1 transition-all group-hover:bg-opacity-10 hover:!bg-opacity-30"
                        title="Delete checkpoint"
                      >
                        <Trash2 className="w-3 h-3 text-red-500" />
                      </button>
                    )}
                  </div>
                </div>
              );
            })}

            {optimizerFile && (
              <div className="group mt-1 flex items-center justify-between border-t border-gray-800 px-2 py-1.5 pt-2 transition-all duration-200 hover:bg-gray-800">
                <a
                  target="_blank"
                  href={getDownloadUrl(optimizerFile.path)}
                  className="flex items-center space-x-2 min-w-0 flex-1"
                >
                  <SlidersHorizontal className="w-4 h-4 text-amber-500 flex-shrink-0" />
                  <div className="flex flex-col min-w-0">
                    <div className="flex text-sm text-amber-200">
                      <span className="overflow-hidden text-ellipsis direction-rtl whitespace-nowrap">optimizer</span>
                    </div>
                    <span className="text-xs text-amber-600/70">.pt - optimizer state</span>
                  </div>
                </a>
                <div className="flex items-center space-x-3 flex-shrink-0">
                  <span className="text-xs text-gray-400">{cleanSize(optimizerFile.size)}</span>
                  <a
                    target="_blank"
                    href={getDownloadUrl(optimizerFile.path)}
                    className="bg-amber-500 bg-opacity-0 p-1 transition-all group-hover:bg-opacity-10"
                  >
                    <Download className="w-3 h-3 text-amber-500" />
                  </a>
                  {!parseRemoteAssetRef(optimizerFile.path) && (
                    <button
                      type="button"
                      onClick={() => handleDeleteFile(optimizerFile.path)}
                      className="bg-red-500 bg-opacity-0 p-1 transition-all group-hover:bg-opacity-10 hover:!bg-opacity-30"
                      title="Delete optimizer state"
                    >
                      <Trash2 className="w-3 h-3 text-red-500" />
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {['success', 'refreshing'].includes(status) && files.length === 0 && (
          <div className="text-center py-4 text-gray-400 text-sm">No checkpoints available</div>
        )}
      </div>
    </div>
  );
}
