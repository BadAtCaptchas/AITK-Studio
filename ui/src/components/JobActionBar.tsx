import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { Eye, Trash2, Pen, Play, Pause, Cog, X, Download, Loader2, CheckCircle2, CloudDownload, Save, RefreshCcw } from 'lucide-react';
import {
  Button,
  Dialog,
  DialogBackdrop,
  DialogPanel,
  DialogTitle,
  Menu,
  MenuButton,
  MenuItem,
  MenuItems,
} from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import type { Job } from '@/types';
import {
  startJob,
  restartJobFromScratch,
  stopJob,
  deleteJob,
  getAvaliableJobActions,
  markJobAsStopped,
  startTrainingJobExport,
  getTrainingJobExportProgress,
  cancelTrainingJobExport,
  downloadServerFile,
  downloadJobModelReferences,
  saveJobNow,
  retryRemoteCaptionResult,
  type TrainingJobCheckpointExportMode,
  type TrainingJobExportProgress,
} from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { openCaptionDatasetModal } from '@/components/CaptionDatasetModal';

interface JobActionBarProps {
  job: Job;
  onRefresh?: () => void;
  afterDelete?: () => void;
  hideView?: boolean;
  className?: string;
  autoStartQueue?: boolean;
}

type ExportMode = 'state' | 'datasets';
type ExportStatus = {
  mode: ExportMode;
  phase: 'exporting' | 'ready' | 'failed' | 'canceled';
  progress: TrainingJobExportProgress | null;
};
type ExportDialogState = { includeDatasets: boolean } | null;
type ModelDownloadStatus = {
  phase: 'downloading' | 'completed' | 'failed';
  handledCount: number;
  warnings: string[];
  error: string | null;
};

function sleep(ms: number) {
  return new Promise(resolve => window.setTimeout(resolve, ms));
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

function getExportPhase(progress: TrainingJobExportProgress): ExportStatus['phase'] {
  if (progress.status === 'completed') return 'ready';
  if (progress.status === 'failed') return 'failed';
  if (progress.status === 'canceled') return 'canceled';
  return 'exporting';
}

function getExportProgressDetail(progress: TrainingJobExportProgress | null) {
  if (!progress) return null;

  const files =
    progress.entriesTotal > 0 ? `${progress.entriesProcessed} / ${progress.entriesTotal} files` : null;
  const bytes =
    progress.bytesTotal > 0
      ? `${formatBytes(progress.bytesProcessed)} / ${formatBytes(progress.bytesTotal)}`
      : null;

  return [files, bytes].filter(Boolean).join(' · ');
}

function getExportStatusLabel(exportStatus: ExportStatus | null) {
  if (!exportStatus) return '';
  if (exportStatus.phase === 'canceled') return 'Export canceled';
  if (exportStatus.progress?.status === 'canceling') return 'Canceling export...';
  if (exportStatus.phase === 'failed') return 'Export failed';
  if (exportStatus.phase === 'ready') return 'Export ready';

  return (
    exportStatus.progress?.message ||
    (exportStatus.mode === 'datasets' ? 'Starting dataset export...' : 'Starting job export...')
  );
}

function getApiErrorMessage(error: unknown, fallback: string) {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { data?: { error?: string } } }).response;
    if (response?.data?.error) return response.data.error;
  }
  return error instanceof Error ? error.message : fallback;
}

function getModelDownloadStatusLabel(status: ModelDownloadStatus) {
  if (status.phase === 'downloading') return 'Downloading referenced models...';
  if (status.phase === 'failed') return status.error || 'Model download failed';
  if (status.handledCount > 0) {
    return `Downloaded ${status.handledCount} referenced model${status.handledCount === 1 ? '' : 's'}.`;
  }
  return status.warnings[0] || 'No downloadable model references were found.';
}

function getRemoteCaptionState(job: Job) {
  try {
    const state = JSON.parse(job.job_config)?.config?.remote_caption;
    return state && typeof state === 'object' ? state : null;
  } catch {
    return null;
  }
}

const actionButtonClass = 'operator-icon-button align-middle';
const dangerousActionButtonClass =
  'operator-icon-button align-middle hover:border-rose-800 hover:bg-rose-950/50 hover:text-rose-100';

export default function JobActionBar({
  job,
  onRefresh,
  afterDelete,
  className,
  hideView,
  autoStartQueue = false,
}: JobActionBarProps) {
  const { canStart, canStop, canDelete, canEdit, canRemoveFromQueue, canRestartFromScratch } =
    getAvaliableJobActions(job);
  const [exportStatus, setExportStatus] = useState<ExportStatus | null>(null);
  const [modelDownloadStatus, setModelDownloadStatus] = useState<ModelDownloadStatus | null>(null);
  const [captionResultSyncing, setCaptionResultSyncing] = useState(false);
  const [exportDialog, setExportDialog] = useState<ExportDialogState>(null);
  const [checkpointMode, setCheckpointMode] = useState<TrainingJobCheckpointExportMode>('latest');
  const exportStatusTimeout = useRef<number | null>(null);
  const modelDownloadStatusTimeout = useRef<number | null>(null);
  const exportInFlight = useRef(false);
  const modelDownloadInFlight = useRef(false);
  const captionResultInFlight = useRef(false);
  const activeExportID = useRef<string | null>(null);
  const cancelExportInFlight = useRef(false);
  const isMounted = useRef(true);
  const isExporting = exportStatus?.phase === 'exporting';
  const isDownloadingModels = modelDownloadStatus?.phase === 'downloading';
  const remoteCaptionState = getRemoteCaptionState(job);
  const remoteCaptionDownloadStatus = remoteCaptionState?.downloadStatus || null;
  const remoteCaptionLastError = typeof remoteCaptionState?.lastError === 'string' ? remoteCaptionState.lastError : null;
  const canRetryRemoteCaptionSync =
    job.job_type === 'caption' && job.worker_id !== 'local' && remoteCaptionDownloadStatus === 'failed';
  const canManualRemoteCaptionSync =
    job.job_type === 'caption' && job.worker_id !== 'local' && remoteCaptionDownloadStatus !== 'merged';
  const remoteCaptionActionBusy = captionResultSyncing || remoteCaptionDownloadStatus === 'downloading';

  useEffect(() => {
    return () => {
      isMounted.current = false;
      if (exportStatusTimeout.current !== null) {
        window.clearTimeout(exportStatusTimeout.current);
      }
      if (modelDownloadStatusTimeout.current !== null) {
        window.clearTimeout(modelDownloadStatusTimeout.current);
      }
      activeExportID.current = null;
    };
  }, []);

  if (!afterDelete) afterDelete = onRefresh;

  const clearExportStatusSoon = () => {
    if (exportStatusTimeout.current !== null) {
      window.clearTimeout(exportStatusTimeout.current);
    }
    exportStatusTimeout.current = window.setTimeout(() => {
      if (isMounted.current) {
        setExportStatus(null);
      }
      exportStatusTimeout.current = null;
    }, 2500);
  };

  const clearModelDownloadStatusSoon = () => {
    if (modelDownloadStatusTimeout.current !== null) {
      window.clearTimeout(modelDownloadStatusTimeout.current);
    }
    modelDownloadStatusTimeout.current = window.setTimeout(() => {
      if (isMounted.current) {
        setModelDownloadStatus(null);
      }
      modelDownloadStatusTimeout.current = null;
    }, 3500);
  };

  const updateExportStatus = (mode: ExportMode, progress: TrainingJobExportProgress) => {
    if (!isMounted.current) return;
    setExportStatus({ mode, phase: getExportPhase(progress), progress });
  };

  const waitForExport = async (mode: ExportMode, exportID: string) => {
    while (true) {
      await sleep(500);
      const progress = await getTrainingJobExportProgress(job.id, exportID);
      updateExportStatus(mode, progress);

      if (progress.status === 'completed') return progress;
      if (progress.status === 'canceled') return progress;
      if (progress.status === 'failed') {
        throw new Error(progress.error || 'Failed to export training job');
      }
    }
  };

  const handleCancelExport = async () => {
    const exportID = activeExportID.current || exportStatus?.progress?.exportID;
    if (!exportID || cancelExportInFlight.current) return;

    cancelExportInFlight.current = true;
    try {
      const progress = await cancelTrainingJobExport(job.id, exportID);
      updateExportStatus(exportStatus?.mode || (progress.includeDatasets ? 'datasets' : 'state'), progress);
    } catch (error) {
      console.error('Error canceling export:', error);
      alert('Failed to cancel export. Please try again.');
    } finally {
      cancelExportInFlight.current = false;
    }
  };

  const handleExport = async (includeDatasets: boolean, checkpointMode: TrainingJobCheckpointExportMode) => {
    const exportMode: ExportMode = includeDatasets ? 'datasets' : 'state';
    if (exportInFlight.current) return;

    exportInFlight.current = true;
    if (exportStatusTimeout.current !== null) {
      window.clearTimeout(exportStatusTimeout.current);
      exportStatusTimeout.current = null;
    }
    setExportStatus({ mode: exportMode, phase: 'exporting', progress: null });
    try {
      const started = await startTrainingJobExport(job.id, includeDatasets, checkpointMode);
      activeExportID.current = started.exportID;
      updateExportStatus(exportMode, started.progress);

      const progress = await waitForExport(exportMode, started.exportID);
      if (progress.status === 'canceled') {
        setExportStatus({ mode: exportMode, phase: 'canceled', progress });
        clearExportStatusSoon();
        return;
      }
      if (!progress.zipPath || !progress.fileName) {
        throw new Error('Export completed without a downloadable file.');
      }

      downloadServerFile(progress.zipPath, progress.fileName);
      setExportStatus({ mode: exportMode, phase: 'ready', progress });
      clearExportStatusSoon();
      if (progress.warnings?.length) {
        alert(`Export completed with warnings:\n\n${progress.warnings.join('\n')}`);
      }
    } catch (error) {
      console.error('Error exporting job:', error);
      alert('Failed to export job. Please try again.');
      setExportStatus({ mode: exportMode, phase: 'failed', progress: null });
      clearExportStatusSoon();
    } finally {
      exportInFlight.current = false;
      activeExportID.current = null;
    }
  };

  const openExportDialog = (includeDatasets: boolean) => {
    if (isExporting) return;
    setCheckpointMode('latest');
    setExportDialog({ includeDatasets });
  };

  const startDialogExport = () => {
    if (!exportDialog) return;
    const includeDatasets = exportDialog.includeDatasets;
    setExportDialog(null);
    void handleExport(includeDatasets, checkpointMode);
  };

  const handleDownloadModelReferences = async () => {
    if (job.job_type !== 'train' || modelDownloadInFlight.current) return;

    modelDownloadInFlight.current = true;
    if (modelDownloadStatusTimeout.current !== null) {
      window.clearTimeout(modelDownloadStatusTimeout.current);
      modelDownloadStatusTimeout.current = null;
    }
    setModelDownloadStatus({ phase: 'downloading', handledCount: 0, warnings: [], error: null });
    try {
      const result = await downloadJobModelReferences(job.id);
      setModelDownloadStatus({
        phase: 'completed',
        handledCount: result.handledValues.length,
        warnings: result.warnings || [],
        error: null,
      });
      onRefresh?.();
      clearModelDownloadStatusSoon();
    } catch (error) {
      console.error('Error downloading referenced models:', error);
      setModelDownloadStatus({
        phase: 'failed',
        handledCount: 0,
        warnings: [],
        error: getApiErrorMessage(error, 'Failed to download referenced models.'),
      });
      clearModelDownloadStatusSoon();
    } finally {
      modelDownloadInFlight.current = false;
    }
  };

  const handleSaveNextStep = async () => {
    try {
      await saveJobNow(job.id);
      onRefresh?.();
    } catch (error) {
      console.error('Error requesting checkpoint save:', error);
      alert(getApiErrorMessage(error, 'Failed to request a checkpoint save.'));
    }
  };

  const handleRestartFromScratch = () => {
    if (!canRestartFromScratch) return;

    openConfirm({
      title: 'Restart From Scratch',
      message: `Restart "${job.name}" from scratch? This will permanently delete its checkpoints, samples, logs, and training metrics. The job config and datasets will remain.`,
      type: 'danger',
      confirmText: 'Restart From Scratch',
      onConfirm: async () => {
        try {
          await restartJobFromScratch(job.id);
          onRefresh?.();
        } catch (error) {
          alert(getApiErrorMessage(error, 'Failed to restart job from scratch.'));
        }
      },
    });
  };

  const handleRetryRemoteCaptionResult = async () => {
    if (captionResultInFlight.current) return;
    captionResultInFlight.current = true;
    setCaptionResultSyncing(true);
    try {
      await retryRemoteCaptionResult(job.id);
      onRefresh?.();
    } catch (error) {
      console.error('Error syncing remote caption result:', error);
      alert(getApiErrorMessage(error, 'Failed to sync remote caption result.'));
      onRefresh?.();
    } finally {
      captionResultInFlight.current = false;
      setCaptionResultSyncing(false);
    }
  };

  const exportStatusLabel = getExportStatusLabel(exportStatus);
  const modelDownloadStatusLabel = modelDownloadStatus ? getModelDownloadStatusLabel(modelDownloadStatus) : '';
  const exportStatusDetail = getExportProgressDetail(exportStatus?.progress || null);
  const exportPercent = Math.round(exportStatus?.progress?.percent || (exportStatus?.phase === 'ready' ? 100 : 0));
  const canCancelExport =
    isExporting &&
    !!(activeExportID.current || exportStatus?.progress?.exportID) &&
    exportStatus?.progress?.status !== 'canceling' &&
    exportStatus?.progress?.cancelRequested !== true;

  return (
    <div className={`inline-flex items-center justify-end gap-1 ${className || ''}`}>
      {canStart && (
        <Button
          title="Start job"
          aria-label="Start job"
          onClick={async () => {
            if (!canStart) return;
            try {
              await startJob(job.id);
              // start the queue as well
              if (autoStartQueue) {
                await startQueue(job.gpu_ids, job.worker_id);
              }
              if (onRefresh) onRefresh();
            } catch (error) {
              alert(error instanceof Error ? error.message : 'Failed to start job.');
            }
          }}
          className={actionButtonClass}
        >
          <Play className="h-4 w-4" />
        </Button>
      )}
      {canRemoveFromQueue && (
        <Button
          title="Remove from queue"
          aria-label="Remove from queue"
          onClick={async () => {
            if (!canRemoveFromQueue) return;
            await markJobAsStopped(job.id);
            if (onRefresh) onRefresh();
          }}
          className={actionButtonClass}
        >
          <X className="h-4 w-4" />
        </Button>
      )}
      {canStop && (
        <Button
          title="Stop job"
          aria-label="Stop job"
          onClick={() => {
            if (!canStop) return;
            openConfirm({
              title: 'Stop Job',
              message: `Are you sure you want to stop the job "${job.name}"? You CAN resume later.`,
              type: 'info',
              confirmText: 'Stop',
              onConfirm: async () => {
                await stopJob(job.id);
                if (onRefresh) onRefresh();
              },
            });
          }}
          className={actionButtonClass}
        >
          <Pause className="h-4 w-4" />
        </Button>
      )}
      {!hideView && (
        <Link href={`/jobs/${job.id}`} className={actionButtonClass} title="View job" aria-label="View job">
          <Eye className="h-4 w-4" />
        </Link>
      )}
      {job.job_type === 'caption' && canEdit && (
        <div
          className={actionButtonClass}
          title="Edit captions"
          aria-label="Edit captions"
          onClick={() =>
            openCaptionDatasetModal(
              job.job_ref || '',
              () => {
                if (onRefresh) onRefresh();
              },
              { jobId: job.id },
            )
          }
        >
          <Pen className="h-4 w-4" />
        </div>
      )}
      {canRetryRemoteCaptionSync && (
        <Button
          title={remoteCaptionLastError ? `Retry caption sync: ${remoteCaptionLastError}` : 'Retry caption sync'}
          aria-label="Retry caption sync"
          disabled={captionResultSyncing}
          onClick={() => void handleRetryRemoteCaptionResult()}
          className={`${actionButtonClass} ${captionResultSyncing ? 'cursor-wait opacity-80' : 'text-rose-200 hover:border-rose-800 hover:bg-rose-950/50'}`}
        >
          {captionResultSyncing ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCcw className="h-4 w-4" />}
        </Button>
      )}
      {job.job_type === 'train' && canEdit && (
        <Link href={`/jobs/new?id=${job.id}`} className={actionButtonClass} title="Edit job" aria-label="Edit job">
          <Pen className="h-4 w-4" />
        </Link>
      )}
      <Button
        title={canDelete ? 'Delete job' : 'Stop the job before deleting'}
        aria-label="Delete job"
        disabled={!canDelete}
        onClick={() => {
          if (!canDelete) return;
          let message = `Are you sure you want to delete the job "${job.name}"? This will also permanently remove it from your disk.`;
          if (job.status === 'running') {
            message += ' WARNING: The job is currently running. You should stop it first if you can.';
          }
          openConfirm({
            title: 'Delete Job',
            message: message,
            type: 'warning',
            confirmText: 'Delete',
            onConfirm: async () => {
              if (job.status === 'running') {
                try {
                  await stopJob(job.id);
                } catch (e) {
                  console.error('Error stopping job before deleting:', e);
                }
              }
              await deleteJob(job.id);
              if (afterDelete) afterDelete();
            },
          });
        }}
        className={dangerousActionButtonClass}
      >
        <Trash2 className="h-4 w-4" />
      </Button>
      <div className="mx-1 h-5 border-r border-gray-700"></div>
      <Menu>
        <MenuButton
          title="More job actions"
          aria-label="More job actions"
          className={`${actionButtonClass} ${isExporting ? 'cursor-wait opacity-80' : ''}`}
        >
          {isExporting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Cog className="h-4 w-4" />}
        </MenuButton>
        <MenuItems anchor="bottom" className="z-50 mt-2 w-60 border border-gray-700 bg-gray-950 px-2 py-2">
          {job.job_type === 'train' && (
            <MenuItem>
              <Link
                href={`/jobs/new?cloneId=${job.id}`}
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded block"
              >
                Clone Job
              </Link>
            </MenuItem>
          )}
          {job.job_type === 'train' && canStop && (
            <MenuItem>
              <div
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
                onClick={() => void handleSaveNextStep()}
              >
                <Save className="w-4 h-4" />
                Save Next Step
              </div>
            </MenuItem>
          )}
          {job.job_type === 'train' && (
            <MenuItem>
              <div
                className={`px-4 py-1 rounded flex items-center gap-2 ${
                  isExporting ? 'cursor-wait opacity-60' : 'cursor-pointer hover:bg-gray-800'
                }`}
                aria-disabled={isExporting}
                onClick={() => openExportDialog(false)}
              >
                <Download className="w-4 h-4" />
                Export Job State
              </div>
            </MenuItem>
          )}
          {job.job_type === 'train' && (
            <MenuItem>
              <div
                className={`px-4 py-1 rounded flex items-center gap-2 ${
                  isExporting ? 'cursor-wait opacity-60' : 'cursor-pointer hover:bg-gray-800'
                }`}
                aria-disabled={isExporting}
                onClick={() => openExportDialog(true)}
              >
                <Download className="w-4 h-4" />
                Export With Datasets
              </div>
            </MenuItem>
          )}
          {job.job_type === 'train' && (
            <MenuItem>
              <div
                className={`px-4 py-1 rounded flex items-center gap-2 ${
                  isDownloadingModels ? 'cursor-wait opacity-60' : 'cursor-pointer hover:bg-gray-800'
                }`}
                aria-disabled={isDownloadingModels}
                onClick={() => void handleDownloadModelReferences()}
              >
                {isDownloadingModels ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <CloudDownload className="w-4 h-4" />
                )}
                Download Referenced Models
              </div>
            </MenuItem>
          )}
          {canManualRemoteCaptionSync && (
            <MenuItem>
              <div
                className={`px-4 py-1 rounded flex items-center gap-2 ${
                  remoteCaptionActionBusy ? 'cursor-wait opacity-60' : 'cursor-pointer hover:bg-gray-800'
                }`}
                aria-disabled={remoteCaptionActionBusy}
                onClick={() => {
                  if (!remoteCaptionActionBusy) void handleRetryRemoteCaptionResult();
                }}
              >
                {remoteCaptionActionBusy ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : remoteCaptionDownloadStatus === 'failed' ? (
                  <RefreshCcw className="w-4 h-4" />
                ) : (
                  <CloudDownload className="w-4 h-4" />
                )}
                {remoteCaptionDownloadStatus === 'failed' ? 'Retry Caption Sync' : 'Sync Caption Result'}
              </div>
            </MenuItem>
          )}
          {canRestartFromScratch && (
            <MenuItem>
              <div
                className="cursor-pointer rounded px-4 py-1 text-rose-200 hover:bg-rose-950/60 hover:text-rose-100 flex items-center gap-2"
                onClick={handleRestartFromScratch}
              >
                <RefreshCcw className="w-4 h-4" />
                Restart From Scratch
              </div>
            </MenuItem>
          )}
          <MenuItem>
            <div
              className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded"
              onClick={() => {
                let message = `Are you sure you want to mark this job as stopped? This will set the job status to 'stopped' if the status is hung. Only do this if you are 100% sure the job is stopped. This will NOT stop the job.`;
                openConfirm({
                  title: 'Mark Job as Stopped',
                  message: message,
                  type: 'warning',
                  confirmText: 'Mark as Stopped',
                  onConfirm: async () => {
                    await markJobAsStopped(job.id);
                    onRefresh && onRefresh();
                  },
                });
              }}
            >
              Mark as Stopped
            </div>
          </MenuItem>
        </MenuItems>
      </Menu>
      {exportDialog && (
        <Dialog open={exportDialog !== null} onClose={() => setExportDialog(null)} className="relative z-40">
          <DialogBackdrop
            transition
            className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-200 data-enter:ease-out data-leave:duration-150 data-leave:ease-in"
          />

          <div className="fixed inset-0 z-40 w-screen overflow-y-auto">
            <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
              <DialogPanel
                transition
                className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-200 data-enter:ease-out data-leave:duration-150 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-md data-closed:sm:translate-y-0 data-closed:sm:scale-95"
              >
                <div className="bg-gray-800 px-4 pt-5 pb-4 sm:p-6">
                  <DialogTitle as="h3" className="text-base font-semibold text-gray-100">
                    {exportDialog.includeDatasets ? 'Export With Datasets' : 'Export Job State'}
                  </DialogTitle>
                  <div className="mt-4 space-y-2">
                    <button
                      type="button"
                      onClick={() => setCheckpointMode('latest')}
                      aria-pressed={checkpointMode === 'latest'}
                      className={`w-full rounded-md border px-3 py-3 text-left transition-colors ${
                        checkpointMode === 'latest'
                          ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                          : 'border-gray-700 bg-gray-900 text-gray-200 hover:border-gray-500'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-medium">Latest checkpoint</span>
                        <span className="text-xs text-gray-400">Smaller</span>
                      </div>
                      <p className="mt-1 text-xs text-gray-400">
                        Exports the latest checkpoint and skips older checkpoint files.
                      </p>
                    </button>
                    <button
                      type="button"
                      onClick={() => setCheckpointMode('all')}
                      aria-pressed={checkpointMode === 'all'}
                      className={`w-full rounded-md border px-3 py-3 text-left transition-colors ${
                        checkpointMode === 'all'
                          ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                          : 'border-gray-700 bg-gray-900 text-gray-200 hover:border-gray-500'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-medium">All checkpoints</span>
                        <span className="text-xs text-gray-400">Large</span>
                      </div>
                      <p className="mt-1 text-xs text-gray-400">
                        Includes every checkpoint file found in the training folder.
                      </p>
                    </button>
                  </div>
                </div>
                <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
                  <button
                    type="button"
                    onClick={startDialogExport}
                    className="inline-flex w-full justify-center rounded-md bg-blue-700 px-3 py-2 text-sm font-semibold text-white shadow-xs hover:bg-blue-500 sm:ml-3 sm:w-auto"
                  >
                    Export
                  </button>
                  <button
                    type="button"
                    onClick={() => setExportDialog(null)}
                    className="mt-3 inline-flex w-full justify-center rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-900 sm:mt-0 sm:w-auto"
                  >
                    Cancel
                  </button>
                </div>
              </DialogPanel>
            </div>
          </div>
        </Dialog>
      )}
      {exportStatus && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-4 right-4 z-50 flex w-80 max-w-[calc(100vw-2rem)] items-start gap-2 rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 shadow-lg"
        >
          {exportStatus.phase === 'ready' ? (
            <CheckCircle2 className="mt-0.5 h-4 w-4 flex-none text-green-400" />
          ) : exportStatus.phase === 'failed' ? (
            <X className="mt-0.5 h-4 w-4 flex-none text-red-400" />
          ) : exportStatus.phase === 'canceled' ? (
            <X className="mt-0.5 h-4 w-4 flex-none text-gray-400" />
          ) : (
            <Loader2 className="mt-0.5 h-4 w-4 flex-none animate-spin text-blue-400" />
          )}
          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between gap-3">
              <span className="truncate">{exportStatusLabel}</span>
              <div className="flex flex-none items-center gap-2">
                <span className="text-xs text-gray-400">{exportPercent}%</span>
                {canCancelExport && (
                  <button
                    type="button"
                    onClick={() => void handleCancelExport()}
                    title="Cancel export"
                    aria-label="Cancel export"
                    className="rounded p-0.5 text-gray-400 hover:bg-gray-800 hover:text-gray-100"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </div>
            {exportStatusDetail && <div className="mt-0.5 truncate text-xs text-gray-400">{exportStatusDetail}</div>}
            {exportStatus.phase === 'exporting' && (
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-gray-700">
                <div
                  className="h-full rounded-full bg-blue-500 transition-all"
                  style={{ width: `${Math.max(2, exportPercent)}%` }}
                />
              </div>
            )}
          </div>
        </div>
      )}
      {modelDownloadStatus && (
        <div
          role="status"
          aria-live="polite"
          className={`fixed ${exportStatus ? 'bottom-24' : 'bottom-4'} right-4 z-50 flex w-80 max-w-[calc(100vw-2rem)] items-start gap-2 rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 shadow-lg`}
        >
          {modelDownloadStatus.phase === 'completed' ? (
            <CheckCircle2 className="mt-0.5 h-4 w-4 flex-none text-green-400" />
          ) : modelDownloadStatus.phase === 'failed' ? (
            <X className="mt-0.5 h-4 w-4 flex-none text-red-400" />
          ) : (
            <Loader2 className="mt-0.5 h-4 w-4 flex-none animate-spin text-blue-400" />
          )}
          <div className="min-w-0 flex-1">
            <div className="truncate">{modelDownloadStatusLabel}</div>
            {modelDownloadStatus.phase === 'completed' && modelDownloadStatus.warnings.length > 0 && (
              <div className="mt-0.5 truncate text-xs text-gray-400">{modelDownloadStatus.warnings[0]}</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
