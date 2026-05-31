'use client';
import React from 'react';
import classNames from 'classnames';

interface Props {
  className?: string;
  children?: React.ReactNode;
}

export const TopBar: React.FC<Props> = ({ children, className }) => {
  return (
    <div
      className={classNames(
        'operator-scrollbar-none absolute top-0 left-0 z-10 flex h-12 w-full items-center gap-2 overflow-x-auto border-b border-gray-800 bg-gray-950/95 px-2',
        className,
      )}
    >
      {children ? children : null}
    </div>
  );
};

export const MainContent = React.forwardRef<HTMLDivElement, Props>(({ children, className }, ref) => {
  return (
    <div
      ref={ref}
      className={classNames('absolute top-0 left-0 h-full w-full overflow-auto px-3 pt-14 sm:px-4', className)}
    >
      {children ? children : null}
    </div>
  );
});
MainContent.displayName = 'MainContent';
