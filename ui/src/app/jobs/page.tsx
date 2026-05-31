'use client';

import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { Button } from '@headlessui/react';
import { AlertTriangle, ArrowRight, CheckCircle2, CloudDownload, FileArchive, ListOrdered, Loader2, Plus, Upload, X } from 'lucide-react';
import { SelectInput } from '@/components/formInputs';
import useGPUInfo from '@/hooks/useGPUInfo';
import { downloadJobModelReferences, importTrainingJob } from '@/utils/jobs';
import type { Job } from '@/types';

type ImportPhase = 'uploading' | 'processing' | 'completed' | 'failed';
type ModelDownloadPhase = 'downloading' | 'completed' | 'failed';

type ImportStatus = {
  phase: ImportPhase;
  fileName: string;
  fileSize: number;
  gpuIDs: string | null;
  loaded: number;
  total: number | null;
  uploadPercent: number | null;
  job: Job | null;
  warnings: string[];
  error: string | null;
};

type ModelDownloadStatus = {
  phase: ModelDownloadPhase;
  handledCount: number;
  warnings: string[];
  error: string | null;
};

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

function getApiErrorMessage(error: unknown, fallback: string) {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { data?: { error?: string } } }).response;
    if (response?.data?.error) return response.data.error;
  }
  return error instanceof Error ? error.message : fallback;
}

function getImportErrorMessage(error: unknown) {
  return getApiErrorMessage(error, 'Failed to import training job.');
}

function getImportProgressPercent(status: ImportStatus) {
  if (status.phase === 'completed') return 100;
  if (status.phase === 'processing') return 72;
  if (status.phase === 'failed') {
    if (status.uploadPercent !== null && status.uploadPercent >= 100) return 72;
    return status.uploadPercent ? Math.max(8, Math.round(status.uploadPercent * 0.65)) : 8;
  }
  if (status.uploadPercent === null) return 8;
  return Math.max(8, Math.round(status.uploadPercent * 0.65));
}

function getImportTitle(status: ImportStatus) {
  if (status.phase === 'uploading') return 'Uploading training archive';
  if (status.phase === 'processing') return 'Importing job files';
  if (status.phase === 'completed') return 'Training job imported';
  return 'Import failed';
}

function getImportDetail(status: ImportStatus) {
  if (status.phase === 'uploading') {
    const uploaded = formatBytes(status.loaded);
    const total = status.total ? formatBytes(status.total) : formatBytes(status.fileSize);
    return status.uploadPercent === null
      ? `${uploaded} uploaded`
      : `${status.uploadPercent}% uploaded (${uploaded} / ${total})`;
  }
  if (status.phase === 'processing') {
    return 'Validating the archive, restoring training files, and copying datasets.';
  }
  if (status.phase === 'completed') {
    return `${status.job?.name || 'Imported job'} is ready in the queue.`;
  }
  return status.error || 'Please check the archive and try again.';
}

function getStageClass(isActive: boolean, isComplete: boolean, isFailed = false) {
  if (isFailed) return 'border-red-500/50 bg-red-500/10 text-red-200';
  if (isComplete) return 'border-green-500/50 bg-green-500/10 text-green-200';
  if (isActive) return 'border-blue-500/60 bg-blue-500/10 text-blue-100';
  return 'border-gray-700 bg-gray-950 text-gray-400';
}

export default function Dashboard() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [importStatus, setImportStatus] = useState<ImportStatus | null>(null);
  const [modelDownloadStatus, setModelDownloadStatus] = useState<ModelDownloadStatus | null>(null);
  const [jobsTableKey, setJobsTableKey] = useState(0);
  const isImporting = importStatus?.phase === 'uploading' || importStatus?.phase === 'processing';
  const isModelDownloading = modelDownloadStatus?.phase === 'downloading';

  useEffect(() => {
    if (isGPUInfoLoaded && gpuIDs === null && gpuList.length > 0) {
      setGpuIDs(`${gpuList[0].index}`);
    }
  }, [gpuIDs, gpuList, isGPUInfoLoaded]);

  const handleFileSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file || isImporting) return;

    setModelDownloadStatus(null);
    setImportStatus({
      phase: 'uploading',
      fileName: file.name,
      fileSize: file.size,
      gpuIDs,
      loaded: 0,
      total: file.size || null,
      uploadPercent: 0,
      job: null,
      warnings: [],
      error: null,
    });
    try {
      const result = await importTrainingJob(file, gpuIDs, progress => {
        setImportStatus(current => {
          if (!current) return current;
          const uploadDone = progress.percent !== null && progress.percent >= 100;
          return {
            ...current,
            phase: uploadDone ? 'processing' : 'uploading',
            loaded: progress.loaded,
            total: progress.total,
            uploadPercent: progress.percent,
          };
        });
      });

      setImportStatus(current => ({
        phase: 'completed',
        fileName: current?.fileName || file.name,
        fileSize: current?.fileSize || file.size,
        gpuIDs,
        loaded: current?.total || file.size,
        total: current?.total || file.size || null,
        uploadPercent: 100,
        job: result.job,
        warnings: result.warnings || [],
        error: null,
      }));
      setJobsTableKey(key => key + 1);
    } catch (error) {
      console.error('Error importing training job:', error);
      setImportStatus(current => ({
        phase: 'failed',
        fileName: current?.fileName || file.name,
        fileSize: current?.fileSize || file.size,
        gpuIDs,
        loaded: current?.loaded || 0,
        total: current?.total || file.size || null,
        uploadPercent: current?.uploadPercent || null,
        job: null,
        warnings: [],
        error: getImportErrorMessage(error),
      }));
    }
  };

  const handleDownloadImportedModels = async () => {
    const job = importStatus?.job;
    if (!job || isModelDownloading) return;

    setModelDownloadStatus({ phase: 'downloading', handledCount: 0, warnings: [], error: null });
    try {
      const result = await downloadJobModelReferences(job.id);
      setModelDownloadStatus({
        phase: 'completed',
        handledCount: result.handledValues.length,
        warnings: result.warnings || [],
        error: null,
      });
      if (result.job) {
        setImportStatus(current => (current?.job?.id === result.job.id ? { ...current, job: result.job } : current));
      }
      setJobsTableKey(key => key + 1);
    } catch (error) {
      console.error('Error downloading referenced models:', error);
      setModelDownloadStatus({
        phase: 'failed',
        handledCount: 0,
        warnings: [],
        error: getApiErrorMessage(error, 'Failed to download referenced models.'),
      });
    }
  };

  const importProgressPercent = importStatus ? getImportProgressPercent(importStatus) : 0;
  const importTarget = importStatus?.gpuIDs ? `GPU #${importStatus.gpuIDs}` : 'Default GPU';
  const importUploadComplete = importStatus
    ? ['processing', 'completed'].includes(importStatus.phase) ||
      (importStatus.phase === 'failed' && importStatus.uploadPercent !== null && importStatus.uploadPercent >= 100)
    : false;
  const importProcessingComplete = importStatus?.phase === 'completed';
  const importUploadFailed =
    importStatus?.phase === 'failed' && (importStatus.uploadPercent === null || importStatus.uploadPercent < 100);
  const importRestoreFailed = importStatus?.phase === 'failed' && !importUploadFailed;

  return (
    <>
      <TopBar>
        <div className="flex shrink-0 items-center gap-2">
          <ListOrdered className="h-4 w-4 text-cyan-300" />
          <h1 className="text-base font-semibold">Queue</h1>
        </div>
        <div className="flex-1"></div>
        {gpuList.length > 0 && (
          <div className="mr-2">
            <SelectInput
              value={gpuIDs ?? ''}
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              disabled={isImporting}
            />
          </div>
        )}
        <div className="mr-2">
          <Button
            className="operator-button shrink-0 py-1"
            onClick={() => fileInputRef.current?.click()}
            disabled={isImporting}
            title={isImporting ? 'Importing training job' : 'Import training job'}
            aria-label={isImporting ? 'Importing training job' : 'Import training job'}
          >
            {isImporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            <span className="hidden sm:inline">{isImporting ? 'Importing...' : 'Import Training Job'}</span>
          </Button>
        </div>
        <div className="mr-2">
          <Link
            href="/jobs/secure-remote-captioning"
            className="operator-button shrink-0 py-1"
            title="Secure remote captioning"
            aria-label="Secure remote captioning"
          >
            <CloudDownload className="h-4 w-4" />
            <span className="hidden sm:inline">Secure Remote Captioning</span>
          </Link>
        </div>
        <div>
          <Link
            href="/jobs/new"
            className="operator-button shrink-0 border-emerald-800 bg-emerald-950/60 py-1 text-emerald-100 hover:bg-emerald-900"
            title="New training job"
            aria-label="New training job"
          >
            <Plus className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">New Training Job</span>
          </Link>
        </div>
      </TopBar>
      <input
        ref={fileInputRef}
        type="file"
        accept=".zip,.aitk"
        className="hidden"
        onChange={handleFileSelected}
      />
      <MainContent>
        {importStatus && (
          <section
            role="status"
            aria-live="polite"
            className="mb-4 overflow-hidden border border-gray-700 bg-gray-900 text-gray-100"
          >
            <div className="flex items-start gap-3 px-4 py-4">
              <div
                className={`mt-0.5 rounded-md p-2 ${
                  importStatus.phase === 'completed'
                    ? 'bg-green-500/10 text-green-300'
                    : importStatus.phase === 'failed'
                      ? 'bg-red-500/10 text-red-300'
                      : 'bg-blue-500/10 text-blue-300'
                }`}
              >
                {importStatus.phase === 'completed' ? (
                  <CheckCircle2 className="h-5 w-5" />
                ) : importStatus.phase === 'failed' ? (
                  <AlertTriangle className="h-5 w-5" />
                ) : (
                  <Loader2 className="h-5 w-5 animate-spin" />
                )}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h2 className="text-sm font-semibold text-gray-100">{getImportTitle(importStatus)}</h2>
                    <p className="mt-1 text-xs text-gray-400">{getImportDetail(importStatus)}</p>
                  </div>
                  {!isImporting && (
                    <button
                      type="button"
                      onClick={() => setImportStatus(null)}
                      title="Dismiss import status"
                      aria-label="Dismiss import status"
                    className="operator-icon-button h-7 w-7"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-gray-400">
                    <span className="inline-flex min-w-0 max-w-full items-center gap-1 border border-gray-700 bg-gray-950 px-2 py-1">
                    <FileArchive className="h-3.5 w-3.5 flex-none" />
                    <span className="truncate">{importStatus.fileName}</span>
                  </span>
                  <span className="border border-gray-700 bg-gray-950 px-2 py-1">
                    {formatBytes(importStatus.fileSize)}
                  </span>
                  <span className="border border-gray-700 bg-gray-950 px-2 py-1">{importTarget}</span>
                </div>

                <div className="mt-3 h-2 overflow-hidden rounded-full bg-gray-800">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      importStatus.phase === 'failed'
                        ? 'bg-red-500'
                        : importStatus.phase === 'completed'
                          ? 'bg-green-500'
                          : 'bg-blue-500'
                    }`}
                    style={{ width: `${importProgressPercent}%` }}
                  />
                </div>

                <div className="mt-3 grid grid-cols-1 gap-2 text-xs sm:grid-cols-3">
                  <div
                    className={`rounded border px-3 py-2 ${getStageClass(
                      importStatus.phase === 'uploading',
                      importUploadComplete,
                      importUploadFailed,
                    )}`}
                  >
                    <div className="font-medium">Upload</div>
                    <div className="mt-0.5 opacity-80">
                      {importStatus.uploadPercent === null ? 'Receiving file' : `${importStatus.uploadPercent}%`}
                    </div>
                  </div>
                  <div
                    className={`rounded border px-3 py-2 ${getStageClass(
                      importStatus.phase === 'processing',
                      importProcessingComplete,
                      importRestoreFailed,
                    )}`}
                  >
                    <div className="font-medium">Restore</div>
                    <div className="mt-0.5 opacity-80">
                      {importStatus.phase === 'processing'
                        ? 'In progress'
                        : importProcessingComplete
                          ? 'Complete'
                          : 'Waiting'}
                    </div>
                  </div>
                  <div
                    className={`rounded border px-3 py-2 ${getStageClass(
                      false,
                      importStatus.phase === 'completed',
                      importStatus.phase === 'failed',
                    )}`}
                  >
                    <div className="font-medium">{importStatus.phase === 'failed' ? 'Review' : 'Ready'}</div>
                    <div className="mt-0.5 opacity-80">
                      {importStatus.phase === 'completed'
                        ? 'Job created'
                        : importStatus.phase === 'failed'
                          ? 'Action needed'
                          : 'Waiting'}
                    </div>
                  </div>
                </div>

                {importStatus.warnings.length > 0 && (
                  <div className="mt-3 rounded border border-yellow-500/40 bg-yellow-500/10 px-3 py-2 text-xs text-yellow-100">
                    <div className="font-medium">Imported with warnings</div>
                    <ul className="mt-1 list-disc space-y-1 pl-4">
                      {importStatus.warnings.map((warning, index) => (
                        <li key={`${warning}-${index}`}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {!isImporting && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {importStatus.job && importStatus.phase === 'completed' && (
                      <button
                        type="button"
                        onClick={() => void handleDownloadImportedModels()}
                        disabled={isModelDownloading}
                        className="inline-flex items-center gap-2 rounded-md bg-green-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-green-600 disabled:cursor-wait disabled:opacity-70"
                      >
                        {isModelDownloading ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <CloudDownload className="h-4 w-4" />
                        )}
                        {isModelDownloading ? 'Downloading Models...' : 'Download Referenced Models'}
                      </button>
                    )}
                    {importStatus.job && (
                      <Link
                        href={`/jobs/${importStatus.job.id}`}
                        className="inline-flex items-center gap-2 rounded-md bg-blue-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-600"
                      >
                        View Job
                        <ArrowRight className="h-4 w-4" />
                      </Link>
                    )}
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="rounded-md bg-gray-800 px-3 py-1.5 text-sm font-medium text-gray-100 hover:bg-gray-700"
                    >
                      Import Another
                    </button>
                  </div>
                )}
                {modelDownloadStatus && (
                  <div
                    className={`mt-2 rounded border px-3 py-2 text-xs ${
                      modelDownloadStatus.phase === 'failed'
                        ? 'border-red-500/40 bg-red-500/10 text-red-100'
                        : modelDownloadStatus.phase === 'completed'
                          ? 'border-green-500/40 bg-green-500/10 text-green-100'
                          : 'border-blue-500/40 bg-blue-500/10 text-blue-100'
                    }`}
                  >
                    {modelDownloadStatus.phase === 'downloading'
                      ? 'Downloading referenced models...'
                      : modelDownloadStatus.phase === 'failed'
                        ? modelDownloadStatus.error
                        : modelDownloadStatus.handledCount > 0
                          ? `Downloaded ${modelDownloadStatus.handledCount} referenced model${modelDownloadStatus.handledCount === 1 ? '' : 's'}.`
                          : modelDownloadStatus.warnings[0] || 'No downloadable model references were found.'}
                    {modelDownloadStatus.phase === 'completed' && modelDownloadStatus.warnings.length > 0 && (
                      <ul className="mt-1 list-disc space-y-1 pl-4">
                        {modelDownloadStatus.warnings.map((warning, index) => (
                          <li key={`${warning}-${index}`}>{warning}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            </div>
          </section>
        )}
        <JobsTable key={jobsTableKey} />
      </MainContent>
    </>
  );
}
