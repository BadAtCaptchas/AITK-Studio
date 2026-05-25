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
        'absolute top-0 left-0 w-full h-12 bg-gray-900 shadow-sm z-10 flex items-center px-2',
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
      className={classNames('pt-14 px-2 sm:px-4 absolute top-0 left-0 w-full h-full overflow-auto', className)}
    >
      {children ? children : null}
    </div>
  );
});
MainContent.displayName = 'MainContent';
