import type { HFDownloadProgress as HFDownloadProgressType } from '@/types';
import classNames from 'classnames';
import { AlertCircle, CheckCircle2, CloudDownload } from 'lucide-react';

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${units[unitIndex]}`;
}

function getProgressDetail(progress: HFDownloadProgressType) {
  const bytes =
    progress.bytesTotal != null && progress.bytesTotal > 0
      ? `${formatBytes(progress.bytesDownloaded)} / ${formatBytes(progress.bytesTotal)}`
      : progress.bytesDownloaded > 0
        ? formatBytes(progress.bytesDownloaded)
        : null;
  const active = progress.activeCount > 1 ? `${progress.activeCount} files` : null;
  return [bytes, active].filter(Boolean).join(' · ');
}

function getPercent(progress: HFDownloadProgressType) {
  if (progress.status === 'completed') return 100;
  if (progress.percent == null || !Number.isFinite(progress.percent)) return null;
  return clamp(progress.percent, 0, 100);
}

function ProgressBar({ progress, compact = false }: { progress: HFDownloadProgressType; compact?: boolean }) {
  const percent = getPercent(progress);
  const isFailed = progress.status === 'failed';
  const isComplete = progress.status === 'completed';
  const fillColor = isFailed ? 'bg-rose-400' : isComplete ? 'bg-emerald-400' : 'bg-cyan-400';

  return (
    <div
      className={classNames(
        'relative overflow-hidden rounded-full bg-gray-800 ring-1 ring-white/10',
        compact ? 'h-1.5' : 'h-2',
      )}
    >
      {percent == null ? (
        <div className="absolute inset-y-0 w-1/2 animate-pulse rounded-full bg-gradient-to-r from-cyan-500/20 via-cyan-300 to-cyan-500/20" />
      ) : (
        <div className={classNames('h-full rounded-full transition-all duration-500', fillColor)} style={{ width: `${percent}%` }} />
      )}
    </div>
  );
}

export function HFDownloadProgressBand({ progress }: { progress: HFDownloadProgressType | null }) {
  if (!progress) return null;

  const percent = getPercent(progress);
  const detail = getProgressDetail(progress);
  const isFailed = progress.status === 'failed';
  const isComplete = progress.status === 'completed';
  const Icon = isFailed ? AlertCircle : isComplete ? CheckCircle2 : CloudDownload;

  return (
    <div
      role="status"
      className={classNames(
        'overflow-hidden rounded-lg border px-4 py-3 shadow-inner',
        isFailed
          ? 'border-rose-500/30 bg-rose-950/20'
          : isComplete
            ? 'border-emerald-500/30 bg-emerald-950/20'
            : 'border-cyan-500/30 bg-gradient-to-r from-gray-950 via-cyan-950/20 to-gray-950',
      )}
    >
      <div className="flex min-w-0 items-start gap-3">
        <div
          className={classNames(
            'mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md',
            isFailed ? 'bg-rose-500/15 text-rose-300' : isComplete ? 'bg-emerald-500/15 text-emerald-300' : 'bg-cyan-500/15 text-cyan-300',
          )}
        >
          <Icon className={classNames('h-4 w-4', progress.status === 'downloading' && 'animate-pulse')} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-3 text-sm">
            <div className="min-w-0">
              <p className="truncate font-medium text-gray-100">{progress.message}</p>
              {progress.fileName && <p className="truncate text-xs text-gray-400">{progress.fileName}</p>}
            </div>
            <div className="shrink-0 text-right">
              <p className={classNames('font-mono text-sm', isFailed ? 'text-rose-300' : isComplete ? 'text-emerald-300' : 'text-cyan-200')}>
                {percent == null ? 'Active' : `${Math.round(percent)}%`}
              </p>
              {detail && <p className="text-xs text-gray-500">{detail}</p>}
            </div>
          </div>
          <div className="mt-3">
            <ProgressBar progress={progress} />
          </div>
        </div>
      </div>
    </div>
  );
}

export function HFDownloadProgressInline({ progress, fallback }: { progress: HFDownloadProgressType | null | undefined; fallback: string }) {
  if (!progress) return <span>{fallback}</span>;

  const percent = getPercent(progress);
  const isFailed = progress.status === 'failed';
  const isComplete = progress.status === 'completed';

  return (
    <div className="min-w-[12rem] max-w-xs space-y-1">
      <div className="flex min-w-0 items-center justify-between gap-2">
        <span
          className={classNames(
            'truncate text-xs font-medium',
            isFailed ? 'text-rose-300' : isComplete ? 'text-emerald-300' : 'text-cyan-300',
          )}
        >
          {progress.message}
        </span>
        <span className="shrink-0 font-mono text-[11px] text-gray-400">{percent == null ? '' : `${Math.round(percent)}%`}</span>
      </div>
      <ProgressBar progress={progress} compact />
      {fallback && <div className="truncate text-[11px] text-gray-500">{fallback}</div>}
    </div>
  );
}
