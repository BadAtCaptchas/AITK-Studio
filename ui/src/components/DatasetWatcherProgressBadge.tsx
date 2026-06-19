'use client';

import { WandSparkles } from 'lucide-react';
import {
  autoCaptionProgressTitle,
  hasAutoCaptionProgress,
  type DatasetAutoCaptionProgress,
} from '@/utils/datasetWatcherStatus';

type Props = {
  progress?: DatasetAutoCaptionProgress | null;
  className?: string;
  compact?: boolean;
};

export default function DatasetWatcherProgressBadge({ progress, className = '', compact = false }: Props) {
  if (!hasAutoCaptionProgress(progress)) return null;

  const pending = progress.pending.toLocaleString();
  const label = compact ? `${pending} left` : `Auto-caption: ${pending} left`;

  return (
    <span
      className={`inline-flex w-fit max-w-full items-center gap-1.5 rounded-md border border-fuchsia-500/35 bg-fuchsia-500/10 px-2 py-0.5 text-xs font-medium text-fuchsia-100 ${className}`}
      title={autoCaptionProgressTitle(progress)}
    >
      <WandSparkles className="h-3.5 w-3.5 flex-none text-fuchsia-300" />
      <span className="truncate">{label}</span>
    </span>
  );
}
