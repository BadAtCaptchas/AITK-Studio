'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Loader2, Trash2 } from 'lucide-react';
import { Modal } from './Modal';
import { callScriptStream } from '@/utils/callScript';

export interface MergeLoRAFile {
  path: string;
  label: string;
}

export interface MergeLoRAsModalState {
  folderPath: string;
  outputName: string;
  availableLoRAs: MergeLoRAFile[];
  onComplete?: () => void;
}

interface SelectedLoRA {
  path: string;
  label: string;
  strength: number;
}

export const mergeLoRAsModalState = createGlobalState<MergeLoRAsModalState | null>(null);

export const openMergeLoRAsModal = (
  folderPath: string,
  outputName: string,
  availableLoRAs: MergeLoRAFile[],
  onComplete?: () => void,
) => {
  mergeLoRAsModalState.set({
    folderPath,
    outputName,
    availableLoRAs,
    onComplete,
  });
};

const stripSafetensorsExtension = (value: string) => value.replace(/\.safetensors$/i, '');

const cleanOutputName = (value: string) => stripSafetensorsExtension(value).replace(/[\\/]/g, '').trim();

const joinPath = (folder: string, name: string) => {
  const sep = folder.includes('\\') && !folder.includes('/') ? '\\' : '/';
  const trimmed = folder.replace(/[\\/]+$/, '');
  return `${trimmed}${sep}${cleanOutputName(name)}.safetensors`;
};

const rescaleStrengths = (items: SelectedLoRA[]) => {
  if (items.length === 0) return items;
  const strength = Math.round((1 / items.length) * 1000) / 1000;
  return items.map(item => ({ ...item, strength }));
};

const MergeLoRAsModal: React.FC = () => {
  const [modalInfo, setModalInfo] = mergeLoRAsModalState.use();
  const [selectedLoRAs, setSelectedLoRAs] = useState<SelectedLoRA[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [logOutput, setLogOutput] = useState('');
  const logRef = useRef<HTMLDivElement | null>(null);

  const isOpen = modalInfo !== null;
  const selectedPaths = useMemo(() => new Set(selectedLoRAs.map(item => item.path)), [selectedLoRAs]);
  const availableOptions = useMemo(
    () => (modalInfo?.availableLoRAs ?? []).filter(file => !selectedPaths.has(file.path)),
    [modalInfo?.availableLoRAs, selectedPaths],
  );

  useEffect(() => {
    if (!modalInfo) {
      setSelectedLoRAs([]);
      setIsRunning(false);
      setIsDone(false);
      setHasError(false);
      setLogOutput('');
    }
  }, [modalInfo]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logOutput]);

  const close = () => {
    if (isRunning) return;
    setModalInfo(null);
  };

  const addLoRA = (path: string) => {
    const match = modalInfo?.availableLoRAs.find(file => file.path === path);
    if (!match || selectedPaths.has(path)) return;
    setSelectedLoRAs(prev => rescaleStrengths([...prev, { ...match, strength: 1 }]));
  };

  const removeLoRA = (path: string) => {
    setSelectedLoRAs(prev => rescaleStrengths(prev.filter(item => item.path !== path)));
  };

  const updateStrength = (path: string, strength: number) => {
    setSelectedLoRAs(prev => prev.map(item => (item.path === path ? { ...item, strength } : item)));
  };

  const appendLog = (chunk: string) => setLogOutput(prev => prev + chunk);

  const submit = async () => {
    if (!modalInfo || isRunning || selectedLoRAs.length === 0) return;

    const outputName = cleanOutputName(modalInfo.outputName);
    if (!outputName) return;

    setIsRunning(true);
    setIsDone(false);
    setHasError(false);
    setLogOutput('');

    try {
      const finalEvent = await callScriptStream('merge_loras.py', {
        args: {
          loras: JSON.stringify(selectedLoRAs.map(({ path, strength }) => ({ path, strength }))),
          output: joinPath(modalInfo.folderPath, outputName),
          save_dtype: 'bfloat16',
          device: 'cpu',
        },
        onStdout: appendLog,
        onStderr: appendLog,
      });

      const ok = finalEvent?.type === 'exit' && finalEvent.ok === true;
      if (!ok) {
        setHasError(true);
        if (finalEvent?.type === 'error' && finalEvent.message) {
          appendLog(`\n${finalEvent.message}\n`);
        } else if (finalEvent?.type === 'exit') {
          appendLog(`\nScript exited with code ${finalEvent.exitCode}.\n`);
        }
      } else {
        modalInfo.onComplete?.();
      }
    } catch (error: any) {
      setHasError(true);
      appendLog(`\n${error?.message || 'Unknown error'}\n`);
    } finally {
      setIsRunning(false);
      setIsDone(true);
    }
  };

  const showLog = isRunning || isDone;
  const outputName = cleanOutputName(modalInfo?.outputName ?? '');

  return (
    <Modal isOpen={isOpen} onClose={close} title="Merge LoRAs" size="lg" showCloseButton={!isRunning} closeOnOverlayClick={!isRunning}>
      {showLog ? (
        <div>
          <div className="mb-2 flex items-center gap-2 text-sm">
            {isRunning && (
              <>
                <Loader2 className="h-4 w-4 animate-spin text-amber-400" />
                <span className="text-amber-400">Merging LoRAs</span>
              </>
            )}
            {isDone && hasError && <span className="text-rose-400">Merge failed</span>}
            {isDone && !hasError && <span className="text-emerald-400">Merge complete</span>}
          </div>
          <div
            ref={logRef}
            className="min-h-[360px] max-h-[60vh] overflow-y-auto rounded-md bg-black p-3 font-mono text-xs text-gray-100 whitespace-pre-wrap break-all"
          >
            {logOutput || (isRunning ? 'Starting...\n' : '')}
          </div>
          <div className="mt-4 flex justify-end">
            <button
              type="button"
              onClick={close}
              disabled={isRunning}
              className="rounded-md bg-gray-700 px-4 py-2 text-sm text-gray-100 hover:bg-gray-600 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Close
            </button>
          </div>
        </div>
      ) : (
        <form
          onSubmit={event => {
            event.preventDefault();
            void submit();
          }}
        >
          <label className="block text-xs mb-1 mt-2 text-gray-300">Output Filename</label>
          <div className="flex items-stretch rounded-sm border border-gray-700 bg-gray-950 focus-within:ring-2 focus-within:ring-gray-600">
            <input
              type="text"
              value={outputName}
              onChange={event => {
                if (!modalInfo) return;
                setModalInfo({ ...modalInfo, outputName: cleanOutputName(event.target.value) });
              }}
              className="min-w-0 flex-1 bg-transparent px-3 py-1 text-sm text-gray-100 placeholder:text-gray-500 focus:outline-none"
              required
            />
            <span className="flex items-center border-l border-gray-700 bg-gray-900/60 px-2 text-sm text-gray-400">.safetensors</span>
          </div>

          <label className="block text-xs mb-1 mt-4 text-gray-300">Add LoRA</label>
          <select
            value=""
            onChange={event => addLoRA(event.target.value)}
            className="w-full rounded-sm border border-gray-700 bg-gray-950 px-3 py-1 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-600"
          >
            <option value="" disabled>
              Select checkpoint
            </option>
            {availableOptions.map(file => (
              <option key={file.path} value={file.path}>
                {file.label}
              </option>
            ))}
          </select>

          {selectedLoRAs.length > 0 && (
            <div className="mt-4">
              <label className="block text-xs mb-1 text-gray-300">Selected LoRAs</label>
              <div className="max-h-48 space-y-1 overflow-y-auto rounded-md bg-gray-950 p-2">
                {selectedLoRAs.map(item => (
                  <div key={item.path} className="flex items-center gap-2 px-2 py-1">
                    <div className="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap text-xs text-gray-200" title={item.label}>
                      {item.label}
                    </div>
                    <input
                      type="number"
                      value={item.strength}
                      onChange={event => {
                        const value = Number(event.target.value);
                        if (Number.isFinite(value)) updateStrength(item.path, value);
                      }}
                      step="any"
                      className="w-20 flex-shrink-0 rounded-sm border border-gray-700 bg-gray-900 px-2 py-0.5 text-xs text-gray-100 focus:outline-none focus:ring-1 focus:ring-gray-600"
                    />
                    <button
                      type="button"
                      onClick={() => removeLoRA(item.path)}
                      className="flex-shrink-0 rounded p-1 text-gray-400 hover:bg-gray-800 hover:text-rose-400"
                      aria-label="Remove LoRA"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-6 flex justify-end gap-2">
            <button type="button" onClick={close} className="rounded-md px-4 py-2 text-sm text-gray-300 hover:text-gray-100">
              Cancel
            </button>
            <button
              type="submit"
              disabled={selectedLoRAs.length === 0 || !outputName}
              className="rounded-md bg-purple-600 px-4 py-2 text-sm text-white hover:bg-purple-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Merge
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
};

export default MergeLoRAsModal;
