'use client';
import React from 'react';
import classNames from 'classnames';

interface Props {
  className?: string;
  style?: React.CSSProperties;
  children?: React.ReactNode;
}

export const TopBar: React.FC<Props> = ({ children, className }) => {
  return (
    <div
      className={classNames(
        'studio-page-topbar operator-scrollbar-none absolute top-0 left-0 z-10 flex w-full items-center gap-3 overflow-x-auto border-b border-gray-800 bg-gray-950/95',
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
      className={classNames('studio-page-content absolute top-0 left-0 h-full w-full overflow-auto', className)}
    >
      {children ? children : null}
    </div>
  );
});
MainContent.displayName = 'MainContent';
