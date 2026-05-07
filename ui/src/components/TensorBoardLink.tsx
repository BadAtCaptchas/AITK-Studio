'use client';

import classNames from 'classnames';
import { ExternalLink } from 'lucide-react';
import useTensorBoardStatus from '@/hooks/useTensorBoardStatus';

type TensorBoardLinkProps = {
  className?: string;
  compact?: boolean;
};

export default function TensorBoardLink({ className, compact = false }: TensorBoardLinkProps) {
  const { tensorBoardStatus } = useTensorBoardStatus();

  if (!tensorBoardStatus?.enabled || !tensorBoardStatus.running || !tensorBoardStatus.url) {
    return null;
  }

  const label = compact ? 'TensorBoard' : 'Open TensorBoard';
  const baseClass = classNames(
    'inline-flex items-center gap-1.5 border border-gray-700 bg-gray-800 text-gray-200 transition-colors',
    compact ? 'rounded-md px-2 py-1 text-xs' : 'rounded-lg px-3 py-1.5 text-sm',
    'hover:bg-gray-700',
    className,
  );

  return (
    <a href={tensorBoardStatus.url} target="_blank" rel="noreferrer" className={baseClass}>
      <ExternalLink className={compact ? 'h-3.5 w-3.5' : 'h-4 w-4'} />
      {label}
    </a>
  );
}
