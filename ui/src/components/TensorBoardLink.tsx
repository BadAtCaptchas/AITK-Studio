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
    compact ? 'rounded-sm px-2 py-1 text-xs' : 'rounded-sm px-2 py-1 text-sm sm:px-3 sm:py-1.5',
    'hover:bg-gray-700',
    className,
  );

  return (
    <a href={tensorBoardStatus.url} target="_blank" rel="noreferrer" className={baseClass}>
      <ExternalLink className={compact ? 'h-3.5 w-3.5' : 'h-4 w-4'} />
      <span className={compact ? '' : 'hidden sm:inline'}>{label}</span>
    </a>
  );
}
