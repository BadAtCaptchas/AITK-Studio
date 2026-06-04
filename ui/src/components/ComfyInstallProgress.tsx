import type { ComfyInstallProgress as ComfyInstallProgressType } from '@/types';
import classNames from 'classnames';
import { AlertCircle, CheckCircle2, Download, Loader2, ServerCog } from 'lucide-react';

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function getPercent(progress: ComfyInstallProgressType) {
  if (progress.status === 'completed') return 100;
  if (progress.percent == null || !Number.isFinite(progress.percent)) return null;
  return clamp(progress.percent, 0, 100);
}

function getStageLabel(progress: ComfyInstallProgressType) {
  if (progress.status === 'checking') return 'Checking install';
  if (progress.status === 'installing') return 'Installing';
  if (progress.status === 'launching') return 'Launching';
  if (progress.status === 'ready' || progress.status === 'completed') return 'Ready';
  if (progress.status === 'failed') return 'Install failed';
  return 'Managed ComfyUI';
}

function ProgressBar({ progress, compact = false }: { progress: ComfyInstallProgressType; compact?: boolean }) {
  const percent = getPercent(progress);
  const isFailed = progress.status === 'failed';
  const isComplete = progress.status === 'completed' || progress.status === 'ready';
  const fillColor = isFailed ? 'bg-rose-400' : isComplete ? 'bg-emerald-400' : 'bg-violet-400';

  return (
    <div
      className={classNames(
        'relative overflow-hidden rounded-full bg-gray-800 ring-1 ring-white/10',
        compact ? 'h-1.5' : 'h-2',
      )}
    >
      {percent == null ? (
        <div className="absolute inset-y-0 w-1/2 animate-pulse rounded-full bg-gradient-to-r from-violet-500/20 via-violet-300 to-violet-500/20" />
      ) : (
        <div className={classNames('h-full rounded-full transition-all duration-500', fillColor)} style={{ width: `${percent}%` }} />
      )}
    </div>
  );
}

function iconForProgress(progress: ComfyInstallProgressType) {
  if (progress.status === 'failed') return AlertCircle;
  if (progress.status === 'completed' || progress.status === 'ready') return CheckCircle2;
  if (progress.status === 'launching') return ServerCog;
  if (progress.status === 'installing') return Download;
  return Loader2;
}

export function ComfyInstallProgressBand({ progress }: { progress: ComfyInstallProgressType | null }) {
  if (!progress) return null;

  const percent = getPercent(progress);
  const isFailed = progress.status === 'failed';
  const isComplete = progress.status === 'completed' || progress.status === 'ready';
  const Icon = iconForProgress(progress);

  return (
    <div
      role="status"
      className={classNames(
        'overflow-hidden border px-4 py-3',
        isFailed
          ? 'border-rose-500/30 bg-rose-950/20'
          : isComplete
            ? 'border-emerald-500/30 bg-emerald-950/20'
            : 'border-violet-500/30 bg-gradient-to-r from-gray-950 via-violet-950/20 to-gray-950',
      )}
    >
      <div className="flex min-w-0 items-start gap-3">
        <div
          className={classNames(
            'mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md',
            isFailed ? 'bg-rose-500/15 text-rose-300' : isComplete ? 'bg-emerald-500/15 text-emerald-300' : 'bg-violet-500/15 text-violet-300',
          )}
        >
          <Icon className={classNames('h-4 w-4', !isFailed && !isComplete && 'animate-pulse')} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-3 text-sm">
            <div className="min-w-0">
              <p className="truncate font-medium text-gray-100">{progress.message}</p>
              <p className="truncate text-xs text-gray-400">
                {getStageLabel(progress)}
                {progress.root ? ` - ${progress.root}` : ''}
              </p>
              {progress.error && <p className="mt-1 truncate text-xs text-rose-300">{progress.error}</p>}
            </div>
            <div className="shrink-0 text-right">
              <p className={classNames('font-mono text-sm', isFailed ? 'text-rose-300' : isComplete ? 'text-emerald-300' : 'text-violet-200')}>
                {percent == null ? 'Active' : `${Math.round(percent)}%`}
              </p>
              <p className="text-xs text-gray-500">{progress.step.replaceAll('_', ' ')}</p>
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

export function ComfyInstallProgressInline({
  progress,
  fallback,
}: {
  progress: ComfyInstallProgressType | null | undefined;
  fallback: string;
}) {
  if (!progress) return <span>{fallback}</span>;

  const percent = getPercent(progress);
  const isFailed = progress.status === 'failed';
  const isComplete = progress.status === 'completed' || progress.status === 'ready';

  return (
    <div className="min-w-[12rem] max-w-xs space-y-1">
      <div className="flex min-w-0 items-center justify-between gap-2">
        <span
          className={classNames(
            'truncate text-xs font-medium',
            isFailed ? 'text-rose-300' : isComplete ? 'text-emerald-300' : 'text-violet-300',
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
