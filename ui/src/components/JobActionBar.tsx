import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { Eye, Trash2, Pen, Play, Pause, Cog, X, Download, Loader2, CheckCircle2 } from 'lucide-react';
import { Button } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { Job } from '@prisma/client';
import {
  startJob,
  stopJob,
  deleteJob,
  getAvaliableJobActions,
  markJobAsStopped,
  exportTrainingJob,
  downloadServerFile,
} from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { redirect } from 'next/navigation';
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
type ExportStatus = { mode: ExportMode; phase: 'exporting' | 'ready' | 'failed' };

export default function JobActionBar({
  job,
  onRefresh,
  afterDelete,
  className,
  hideView,
  autoStartQueue = false,
}: JobActionBarProps) {
  const { canStart, canStop, canDelete, canEdit, canRemoveFromQueue } = getAvaliableJobActions(job);
  const [exportStatus, setExportStatus] = useState<ExportStatus | null>(null);
  const exportStatusTimeout = useRef<number | null>(null);
  const exportInFlight = useRef(false);
  const isExporting = exportStatus?.phase === 'exporting';

  useEffect(() => {
    return () => {
      if (exportStatusTimeout.current !== null) {
        window.clearTimeout(exportStatusTimeout.current);
      }
    };
  }, []);

  if (!afterDelete) afterDelete = onRefresh;

  const clearExportStatusSoon = () => {
    if (exportStatusTimeout.current !== null) {
      window.clearTimeout(exportStatusTimeout.current);
    }
    exportStatusTimeout.current = window.setTimeout(() => {
      setExportStatus(null);
      exportStatusTimeout.current = null;
    }, 2500);
  };

  const handleExport = async (includeDatasets: boolean) => {
    const exportMode: ExportMode = includeDatasets ? 'datasets' : 'state';
    if (exportInFlight.current) return;

    exportInFlight.current = true;
    if (exportStatusTimeout.current !== null) {
      window.clearTimeout(exportStatusTimeout.current);
      exportStatusTimeout.current = null;
    }
    setExportStatus({ mode: exportMode, phase: 'exporting' });
    try {
      const result = await exportTrainingJob(job.id, includeDatasets);
      downloadServerFile(result.zipPath, result.fileName);
      setExportStatus({ mode: exportMode, phase: 'ready' });
      clearExportStatusSoon();
      if (result.warnings?.length) {
        alert(`Export completed with warnings:\n\n${result.warnings.join('\n')}`);
      }
    } catch (error) {
      console.error('Error exporting job:', error);
      alert('Failed to export job. Please try again.');
      setExportStatus({ mode: exportMode, phase: 'failed' });
      clearExportStatusSoon();
    } finally {
      exportInFlight.current = false;
    }
  };

  const exportStatusLabel =
    exportStatus?.phase === 'failed'
      ? 'Export failed'
      : exportStatus?.phase === 'ready'
      ? 'Export ready'
      : exportStatus?.mode === 'datasets'
        ? 'Exporting with datasets...'
        : 'Exporting job state...';

  return (
    <div className={`${className}`}>
      {canStart && (
        <Button
          onClick={async () => {
            if (!canStart) return;
            await startJob(job.id);
            // start the queue as well
            if (autoStartQueue) {
              await startQueue(job.gpu_ids);
            }
            if (onRefresh) onRefresh();
          }}
          className={`ml-2 opacity-100`}
        >
          <Play />
        </Button>
      )}
      {canRemoveFromQueue && (
        <Button
          onClick={async () => {
            if (!canRemoveFromQueue) return;
            await markJobAsStopped(job.id);
            if (onRefresh) onRefresh();
          }}
          className={`ml-2 opacity-100`}
        >
          <X />
        </Button>
      )}
      {canStop && (
        <Button
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
          className={`ml-2 opacity-100`}
        >
          <Pause />
        </Button>
      )}
      {!hideView && (
        <Link href={`/jobs/${job.id}`} className="ml-2 text-gray-200 hover:text-gray-100 inline-block">
          <Eye />
        </Link>
      )}
      {job.job_type === 'caption' && canEdit && (
        <div
          className="ml-2 hover:text-gray-100 inline-block cursor-pointer"
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
          <Pen />
        </div>
      )}
      {job.job_type === 'train' && canEdit && (
        <Link href={`/jobs/new?id=${job.id}`} className="ml-2 hover:text-gray-100 inline-block">
          <Pen />
        </Link>
      )}
      <Button
        onClick={() => {
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
        className={`ml-2 opacity-100`}
      >
        <Trash2 />
      </Button>
      <div className="border-r border-1 border-gray-700 ml-2 inline"></div>
      <Menu>
        <MenuButton className={`ml-2 inline-flex items-center ${isExporting ? 'cursor-wait opacity-80' : ''}`}>
          {isExporting ? <Loader2 className="animate-spin" /> : <Cog />}
        </MenuButton>
        <MenuItems anchor="bottom" className="bg-gray-900 border border-gray-700 rounded shadow-lg w-56 px-2 py-2 mt-4">
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
          {job.job_type === 'train' && (
            <MenuItem>
              <div
                className={`px-4 py-1 rounded flex items-center gap-2 ${
                  isExporting ? 'cursor-wait opacity-60' : 'cursor-pointer hover:bg-gray-800'
                }`}
                aria-disabled={isExporting}
                onClickCapture={() => void handleExport(false)}
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
                onClickCapture={() => void handleExport(true)}
              >
                <Download className="w-4 h-4" />
                Export With Datasets
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
      {exportStatus && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-4 right-4 z-50 inline-flex items-center gap-2 rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 shadow-lg"
        >
          {exportStatus.phase === 'ready' ? (
            <CheckCircle2 className="h-4 w-4 text-green-400" />
          ) : exportStatus.phase === 'failed' ? (
            <X className="h-4 w-4 text-red-400" />
          ) : (
            <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
          )}
          {exportStatusLabel}
        </div>
      )}
    </div>
  );
}
