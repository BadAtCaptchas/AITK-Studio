import classNames from 'classnames';
import type React from 'react';
import type { LucideIcon } from 'lucide-react';
import { AlertTriangle, CheckCircle2, CircleDashed, Loader2, PauseCircle, PlayCircle, XCircle } from 'lucide-react';

type Tone = 'neutral' | 'info' | 'success' | 'warning' | 'danger';

const toneClasses: Record<Tone, string> = {
  neutral: 'border-gray-700 bg-gray-900/70 text-gray-300',
  info: 'border-cyan-700/70 bg-cyan-950/30 text-cyan-200',
  success: 'border-emerald-700/70 bg-emerald-950/30 text-emerald-200',
  warning: 'border-amber-700/70 bg-amber-950/30 text-amber-200',
  danger: 'border-rose-700/70 bg-rose-950/30 text-rose-200',
};

const jobStatusMeta: Record<string, { label: string; tone: Tone; icon: LucideIcon; active?: boolean }> = {
  queued: { label: 'Queued', tone: 'warning', icon: CircleDashed },
  running: { label: 'Running', tone: 'info', icon: Loader2, active: true },
  stopping: { label: 'Stopping', tone: 'warning', icon: Loader2, active: true },
  stopped: { label: 'Stopped', tone: 'neutral', icon: PauseCircle },
  completed: { label: 'Completed', tone: 'success', icon: CheckCircle2 },
  error: { label: 'Error', tone: 'danger', icon: XCircle },
  failed: { label: 'Failed', tone: 'danger', icon: XCircle },
};

export function getJobStatusMeta(status: string) {
  return jobStatusMeta[status?.toLowerCase()] || {
    label: status || 'Unknown',
    tone: 'neutral' as Tone,
    icon: CircleDashed,
  };
}

export function StatusBadge({ status, label, className }: { status: string; label?: string; className?: string }) {
  const meta = getJobStatusMeta(status);
  const Icon = meta.icon;

  return (
    <span
      className={classNames(
        'inline-flex max-w-full items-center gap-1.5 rounded-sm border px-2 py-0.5 text-xs font-medium',
        toneClasses[meta.tone],
        className,
      )}
    >
      <Icon className={classNames('h-3.5 w-3.5 flex-none', meta.active ? 'animate-spin' : '')} />
      <span className="truncate">{label || meta.label}</span>
    </span>
  );
}

export function QueueStateBadge({ running }: { running: boolean }) {
  return (
    <span
      className={classNames(
        'inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 text-xs font-medium',
        running ? toneClasses.success : toneClasses.danger,
      )}
    >
      {running ? <PlayCircle className="h-3.5 w-3.5" /> : <PauseCircle className="h-3.5 w-3.5" />}
      {running ? 'Queue running' : 'Queue stopped'}
    </span>
  );
}

export function ProgressBar({
  value,
  tone = 'info',
  className,
}: {
  value: number;
  tone?: Extract<Tone, 'info' | 'success' | 'warning' | 'danger'>;
  className?: string;
}) {
  const safeValue = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
  const fillClass = {
    info: 'bg-cyan-500',
    success: 'bg-emerald-500',
    warning: 'bg-amber-500',
    danger: 'bg-rose-500',
  }[tone];

  return (
    <div className={classNames('h-1.5 overflow-hidden rounded-sm bg-gray-800', className)}>
      <div className={classNames('h-full rounded-sm transition-all', fillClass)} style={{ width: `${safeValue}%` }} />
    </div>
  );
}

export function PageNotice({
  tone = 'neutral',
  title,
  children,
  className,
  action,
}: {
  tone?: Tone;
  title: string;
  children?: React.ReactNode;
  className?: string;
  action?: React.ReactNode;
}) {
  const Icon = tone === 'danger' || tone === 'warning' ? AlertTriangle : tone === 'success' ? CheckCircle2 : CircleDashed;

  return (
    <div className={classNames('border px-3 py-2 text-sm', toneClasses[tone], className)} role="status">
      <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-start">
        <Icon className="mt-0.5 h-4 w-4 flex-none" />
        <div className="min-w-0 flex-1">
          <div className="font-medium">{title}</div>
          {children ? <div className="mt-1 text-xs opacity-85">{children}</div> : null}
        </div>
        {action ? <div className="flex-none self-start sm:self-auto">{action}</div> : null}
      </div>
    </div>
  );
}
