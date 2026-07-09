'use client';
import { useEffect, useRef, useState } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaExclamationTriangle, FaInfo, FaSpinner } from 'react-icons/fa';
import { TextInput } from './formInputs';
import { useFromNull } from '@/hooks/useFromNull';
import classNames from 'classnames';

export interface ConfirmState {
  title: string;
  message?: string;
  confirmText?: string;
  type?: 'danger' | 'warning' | 'info';
  inputTitle?: string;
  onConfirm?: (value?: string) => void | Promise<void>;
  onCancel?: () => void;
}

export const confirmstate = createGlobalState<ConfirmState | null>(null);

export const openConfirm = (confirmProps: ConfirmState) => {
  confirmstate.set(confirmProps);
};

export default function ConfirmModal() {
  const [confirm, setConfirm] = confirmstate.use();
  const [isOpen, setIsOpen] = useState(false);
  const [inputValue, setInputValue] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useFromNull(() => {
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus();
      }
    }, 100);
  }, [confirm]);

  useEffect(() => {
    if (confirm) {
      setIsOpen(true);
      setInputValue('');
      setIsSubmitting(false);
      setSubmitError(null);
    }
  }, [confirm]);

  useEffect(() => {
    if (isOpen) {
      return;
    }

    // Allow the leave transition to finish before discarding its content.
    const resetTimer = setTimeout(() => {
      setConfirm(null);
    }, 500);
    return () => clearTimeout(resetTimer);
  }, [isOpen, setConfirm]);

  const onCancel = () => {
    if (isSubmitting) {
      return;
    }
    if (confirm?.onCancel) {
      confirm.onCancel();
    }
    setIsOpen(false);
  };

  const onConfirm = async () => {
    if (isSubmitting) {
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);
    try {
      await confirm?.onConfirm?.(inputValue);
      setIsOpen(false);
    } catch (error: unknown) {
      setSubmitError(error instanceof Error ? error.message : 'The action could not be completed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  let Icon = FaExclamationTriangle;
  let color = confirm?.type || 'danger';

  // Use conditional rendering for icon
  if (color === 'info') {
    Icon = FaInfo;
  }

  // Color mapping for background colors
  const getBgColor = () => {
    switch (color) {
      case 'danger':
        return 'bg-red-500';
      case 'warning':
        return 'bg-yellow-500';
      case 'info':
        return 'bg-blue-500';
      default:
        return 'bg-red-500';
    }
  };

  // Color mapping for text colors
  const getTextColor = () => {
    switch (color) {
      case 'danger':
        return 'text-red-950';
      case 'warning':
        return 'text-yellow-950';
      case 'info':
        return 'text-blue-950';
      default:
        return 'text-red-950';
    }
  };

  // Color mapping for titles
  const getTitleColor = () => {
    switch (color) {
      case 'danger':
        return 'text-red-500';
      case 'warning':
        return 'text-yellow-500';
      case 'info':
        return 'text-blue-500';
      default:
        return 'text-red-500';
    }
  };

  // Button background color mapping
  const getButtonBgColor = () => {
    switch (color) {
      case 'danger':
        return 'bg-red-700 hover:bg-red-500';
      case 'warning':
        return 'bg-yellow-700 hover:bg-yellow-500';
      case 'info':
        return 'bg-blue-700 hover:bg-blue-500';
      default:
        return 'bg-red-700 hover:bg-red-500';
    }
  };

  return (
    <Dialog open={isOpen} onClose={onCancel} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />

      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
          <DialogPanel
            transition
            className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-lg data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="bg-gray-800 px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <div className="sm:flex sm:items-start">
                <div
                  className={`mx-auto flex size-12 shrink-0 items-center justify-center rounded-full ${getBgColor()} sm:mx-0 sm:size-10`}
                >
                  <Icon aria-hidden="true" className={`size-6 ${getTextColor()}`} />
                </div>
                <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left flex-1">
                  <DialogTitle as="h3" className={`text-base font-semibold ${getTitleColor()}`}>
                    {confirm?.title}
                  </DialogTitle>
                  <div className="mt-2">
                    <p className="text-sm text-gray-200">{confirm?.message}</p>
                    <div className={classNames('mt-4 w-full', { hidden: !confirm?.inputTitle })}>
                      <form onSubmit={(e) => {
                        e.preventDefault();
                        void onConfirm();
                      }}>
                        <TextInput
                          value={inputValue}
                          ref={inputRef}
                          onChange={setInputValue}
                          placeholder={confirm?.inputTitle}
                        />
                      </form>
                    </div>
                    {submitError ? (
                      <p role="alert" className="mt-3 text-sm text-red-300">
                        {submitError}
                      </p>
                    ) : null}
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
              <button
                type="button"
                onClick={() => void onConfirm()}
                disabled={isSubmitting}
                className={`inline-flex w-full items-center justify-center gap-2 rounded-md ${getButtonBgColor()} px-3 py-2 text-sm font-semibold text-white shadow-xs disabled:cursor-wait disabled:opacity-60 sm:ml-3 sm:w-auto`}
              >
                {isSubmitting ? <FaSpinner aria-hidden="true" className="animate-spin" /> : null}
                {isSubmitting ? 'Working…' : confirm?.confirmText || 'Confirm'}
              </button>
              <button
                type="button"
                data-autofocus
                onClick={onCancel}
                disabled={isSubmitting}
                className="mt-3 inline-flex w-full justify-center rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-60 sm:mt-0 sm:w-auto ring-0"
              >
                Cancel
              </button>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}
