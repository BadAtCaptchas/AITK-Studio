'use client';

import React from 'react';
import classNames from 'classnames';

export function ToolButton({
  active,
  disabled,
  label,
  icon,
  onClick,
}: {
  active?: boolean;
  disabled?: boolean;
  label: string;
  icon: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      title={label}
      onClick={onClick}
      className={classNames(
        'group inline-flex h-14 w-14 flex-shrink-0 flex-col items-center justify-center gap-1 rounded-md border text-[10px] transition-colors md:h-16 md:w-16 md:text-[11px]',
        {
          'border-blue-500 bg-blue-600/20 text-blue-100 shadow-[0_0_0_1px_rgba(59,130,246,0.25)]': active,
          'border-transparent text-gray-300 hover:border-gray-700 hover:bg-gray-800 hover:text-gray-100': !active,
          'cursor-not-allowed opacity-35 hover:border-transparent hover:bg-transparent': disabled,
        },
      )}
    >
      <span className="flex h-6 w-6 items-center justify-center">{icon}</span>
      <span>{label}</span>
    </button>
  );
}

export function SegmentedButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={classNames('h-8 min-w-24 border-r border-gray-800 px-3 text-sm last:border-r-0', {
        'bg-blue-600/30 text-blue-100': active,
        'text-gray-300 hover:bg-gray-800': !active,
      })}
    >
      {children}
    </button>
  );
}
