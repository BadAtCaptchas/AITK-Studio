'use client';

import { useEffect } from 'react';
import type { ReactNode } from 'react';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { X } from 'lucide-react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  showCloseButton?: boolean;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  closeOnOverlayClick?: boolean;
}

const sizeClasses = {
  sm: 'max-w-md',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
  xl: 'max-w-4xl',
} as const;

export function Modal({
  isOpen,
  onClose,
  title,
  children,
  showCloseButton = true,
  size = 'md',
  closeOnOverlayClick = true,
}: ModalProps) {
  useEffect(() => {
    if (!isOpen || closeOnOverlayClick) return;
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleEscape, true);
    return () => document.removeEventListener('keydown', handleEscape, true);
  }, [closeOnOverlayClick, isOpen, onClose]);

  return (
    <Dialog
      open={isOpen}
      onClose={closeOnOverlayClick ? onClose : () => undefined}
      className="relative z-50"
    >
      <DialogBackdrop className="fixed inset-0 bg-gray-900/75 backdrop-blur-sm transition-opacity" />
      <div className="fixed inset-0 flex w-screen items-center justify-center overflow-y-auto p-4">
        <DialogPanel
          className={`relative mx-auto w-full ${sizeClasses[size]} rounded-xl border border-gray-800 bg-gray-950 text-gray-100 shadow-xl transition-all`}
        >
          {(title || showCloseButton) && (
            <div className="flex items-center justify-between rounded-t-xl border-b border-gray-800 px-6 py-5">
              {title ? (
                <DialogTitle className="text-xl font-semibold text-gray-100">{title}</DialogTitle>
              ) : null}
              {showCloseButton ? (
                <button
                  type="button"
                  className="ml-auto inline-flex h-9 w-9 items-center justify-center rounded-md text-gray-400 hover:bg-gray-700 hover:text-gray-200"
                  onClick={onClose}
                  aria-label="Close modal"
                >
                  <X className="h-5 w-5" aria-hidden="true" />
                </button>
              ) : null}
            </div>
          )}
          <div className="px-6 py-4">{children}</div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
