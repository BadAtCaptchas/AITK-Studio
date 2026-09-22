'use client';

import { useEffect, useRef, useState } from 'react';
import { isAxiosError } from 'axios';
import { Layers, Loader2, Upload } from 'lucide-react';
import { Modal } from './Modal';
import { apiClient } from '@/utils/api';
import { parseLayeredDocumentsResponse, type LayeredImageDocument } from '@/domain/layeredImages';

const checkerboard = {
  backgroundColor: '#334155',
  backgroundImage: 'conic-gradient(#475569 25%, transparent 0 50%, #475569 0 75%, transparent 0)',
  backgroundSize: '20px 20px',
};
function errorMessage(error: unknown): string {
  if (isAxiosError<unknown>(error)) {
    const data = error.response?.data;
    if (data && typeof data === 'object' && 'error' in data && typeof data.error === 'string') return data.error;
  }
  return error instanceof Error ? error.message : 'Layered document operation failed';
}

export default function LayeredDocumentsButton({
  datasetName,
  workerID,
  onChanged,
}: {
  datasetName: string;
  workerID: string;
  onChanged: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [documents, setDocuments] = useState<LayeredImageDocument[]>([]);
  const [selectedID, setSelectedID] = useState('');
  const [draft, setDraft] = useState<LayeredImageDocument | null>(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const controller = useRef<AbortController | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const original = documents.find(document => document.manifest.id === selectedID);
  const dirty = Boolean(
    draft &&
      original &&
      (draft.captionText !== original.captionText ||
        JSON.stringify(draft.manifest.layers) !== JSON.stringify(original.manifest.layers)),
  );
  useEffect(() => () => controller.current?.abort(), []);
  const selectDocument = (document: LayeredImageDocument | undefined) => {
    setSelectedID(document?.manifest.id || '');
    setDraft(
      document
        ? {
            ...document,
            manifest: { ...document.manifest, layers: document.manifest.layers.map(layer => ({ ...layer })) },
          }
        : null,
    );
  };
  const load = async (preferredID = selectedID) => {
    const response = await apiClient.post<unknown>('/api/datasets/layered', { datasetName, worker_id: workerID });
    const next = parseLayeredDocumentsResponse(response.data);
    setDocuments(next);
    selectDocument(next.find(document => document.manifest.id === preferredID) || next[0]);
  };
  const openDocuments = async () => {
    setOpen(true);
    setError('');
    setNotice('');
    setBusy('Loading documents');
    try {
      await load();
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setBusy('');
    }
  };
  const close = () => {
    if (busy || (dirty && !window.confirm('Discard unsaved layer captions?'))) return;
    setOpen(false);
  };
  const importFiles = async (files: File[]) => {
    if (dirty && !window.confirm('Discard unsaved layer captions before importing?')) return;
    setError('');
    setNotice('');
    const warnings: string[] = [];
    let imported = 0;
    controller.current = new AbortController();
    try {
      for (const file of files) {
        if (!/\.(psd|ora)$/i.test(file.name)) throw new Error('Choose PSD or OpenRaster (.ora) documents');
        if (file.size > 512 * 1024 ** 2) throw new Error(`${file.name} exceeds the 512 MiB import limit`);
        setBusy(`Importing ${file.name} (${imported + 1}/${files.length})`);
        const response = await apiClient.post<unknown>('/api/datasets/import-layered', file, {
          signal: controller.current.signal,
          headers: {
            'Content-Type': 'application/octet-stream',
            'x-aitk-dataset-name': encodeURIComponent(datasetName),
            'x-aitk-worker-id': encodeURIComponent(workerID),
            'x-aitk-file-name': encodeURIComponent(file.name),
          },
        });
        const data = response.data;
        if (data && typeof data === 'object' && 'warnings' in data && Array.isArray(data.warnings))
          warnings.push(...data.warnings.filter((warning: unknown): warning is string => typeof warning === 'string'));
        imported++;
      }
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      try {
        await load();
      } catch (failure) {
        setError(errorMessage(failure));
      }
      if (imported) {
        onChanged();
        setNotice(
          `Imported ${imported} document${imported === 1 ? '' : 's'}.${warnings.length ? ` ${warnings.join(' ')}` : ''}`,
        );
      }
      controller.current = null;
      setBusy('');
      if (input.current) input.current.value = '';
    }
  };
  const save = async () => {
    if (!draft) return;
    setBusy('Saving captions');
    setError('');
    try {
      const response = await apiClient.post<unknown>('/api/datasets/layered', {
        datasetName,
        worker_id: workerID,
        action: 'save',
        id: draft.manifest.id,
        revision: draft.revision,
        caption: draft.captionText,
        layers: draft.manifest.layers.map(({ name, caption }) => ({ name, caption })),
      });
      const next = parseLayeredDocumentsResponse(response.data);
      setDocuments(next);
      selectDocument(next.find(document => document.manifest.id === selectedID));
      setNotice('Captions saved.');
      onChanged();
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setBusy('');
    }
  };
  return (
    <>
      <button type="button" className="operator-button h-8 gap-1.5 px-3 text-xs" onClick={openDocuments}>
        <Layers className="h-3.5 w-3.5" /> Layered documents
      </button>
      <Modal isOpen={open} onClose={close} title="Layered documents" size="xl" closeOnOverlayClick={false}>
        <div className="max-h-[78vh] space-y-4 overflow-auto p-5">
          <p className="text-sm text-gray-400">
            Import PSD or ORA documents. Each visible top-level layer or group becomes a full-canvas RGBA target. Layers
            are ordered from bottom to top. Deleting or moving the composite also includes its layers.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <input
              ref={input}
              type="file"
              multiple
              accept=".psd,.ora"
              className="hidden"
              aria-label="Import layered documents"
              onChange={event => {
                const files = Array.from(event.target.files || []);
                if (files.length) void importFiles(files);
              }}
            />
            <button
              type="button"
              className="operator-button gap-2"
              disabled={Boolean(busy)}
              onClick={() => input.current?.click()}
            >
              <Upload className="h-4 w-4" /> Import PSD / ORA
            </button>
            <button
              type="button"
              className="operator-button"
              disabled={Boolean(busy)}
              onClick={() => {
                if (!dirty || window.confirm('Discard unsaved layer captions and reload?')) void openDocuments();
              }}
            >
              Reload
            </button>
            {busy && (
              <span role="status" className="flex items-center gap-2 text-sm text-gray-300">
                <Loader2 className="h-4 w-4 animate-spin" /> {busy}
              </span>
            )}
            {busy.startsWith('Importing') && (
              <button type="button" className="operator-button" onClick={() => controller.current?.abort()}>
                Cancel import
              </button>
            )}
          </div>
          {error && (
            <p role="alert" className="rounded border border-red-900 bg-red-950/30 p-3 text-sm text-red-300">
              {error}
            </p>
          )}
          {notice && (
            <p role="status" className="text-sm text-brand-300">
              {notice}
            </p>
          )}
          {!documents.length && !busy && (
            <p className="text-sm text-gray-500">No layered documents in this dataset yet.</p>
          )}
          {documents.length > 0 && (
            <label className="block text-sm">
              Document
              <select
                disabled={Boolean(busy)}
                value={selectedID}
                className="mt-1 block w-full rounded border border-gray-700 bg-gray-900 p-2"
                onChange={event => {
                  if (!dirty || window.confirm('Discard unsaved layer captions?'))
                    selectDocument(documents.find(document => document.manifest.id === event.target.value));
                }}
              >
                {documents.map(document => (
                  <option key={document.manifest.id} value={document.manifest.id}>
                    {document.manifest.source?.filename || document.manifest.composite} ·{' '}
                    {document.manifest.layers.length} layers
                  </option>
                ))}
              </select>
            </label>
          )}
          {draft && (
            <>
              <div className="grid gap-4 sm:grid-cols-[180px_1fr]">
                <a href={draft.compositeUrl} target="_blank" rel="noreferrer" title="Open composite PNG">
                  <img
                    src={draft.compositeUrl}
                    alt="Composite"
                    className="h-44 w-full rounded object-contain"
                    style={checkerboard}
                  />
                </a>
                <label className="text-sm">
                  Composite caption{' '}
                  <span className="text-gray-500">
                    ({draft.manifest.width} × {draft.manifest.height})
                  </span>
                  <textarea
                    disabled={Boolean(busy)}
                    maxLength={65536}
                    rows={5}
                    value={draft.captionText}
                    onChange={event => setDraft({ ...draft, captionText: event.target.value })}
                    className="mt-2 w-full rounded border border-gray-700 bg-gray-900 p-2"
                  />
                </label>
              </div>
              <ol className="space-y-3">
                {draft.manifest.layers.map((layer, index) => (
                  <li
                    key={layer.path}
                    className="grid gap-3 rounded border border-gray-800 p-3 sm:grid-cols-[120px_1fr]"
                  >
                    <div>
                      <a href={draft.layerUrls[index]} download={`${index + 1}-${layer.name || 'layer'}.png`}>
                        <img
                          src={draft.layerUrls[index]}
                          alt={`Layer ${index + 1}: ${layer.name}`}
                          className="h-28 w-full rounded object-contain"
                          style={checkerboard}
                        />
                      </a>
                      <span className="mt-1 block text-xs text-gray-500">
                        {index + 1} ·{' '}
                        {index === 0 ? 'Bottom' : index === draft.manifest.layers.length - 1 ? 'Top' : 'Layer'} ·{' '}
                        <a href={draft.layerUrls[index]} download={`${index + 1}-layer.png`} className="text-brand-300">
                          PNG
                        </a>
                      </span>
                    </div>
                    <div className="space-y-2">
                      <label className="block text-xs text-gray-400">
                        Layer name
                        <input
                          disabled={Boolean(busy)}
                          maxLength={512}
                          value={layer.name}
                          onChange={event =>
                            setDraft({
                              ...draft,
                              manifest: {
                                ...draft.manifest,
                                layers: draft.manifest.layers.map((item, i) =>
                                  i === index ? { ...item, name: event.target.value } : item,
                                ),
                              },
                            })
                          }
                          className="mt-1 block w-full rounded border border-gray-700 bg-gray-900 p-2 text-sm text-gray-100"
                        />
                      </label>
                      <label className="block text-xs text-gray-400">
                        Layer caption
                        <textarea
                          disabled={Boolean(busy)}
                          maxLength={65536}
                          rows={3}
                          value={layer.caption}
                          onChange={event =>
                            setDraft({
                              ...draft,
                              manifest: {
                                ...draft.manifest,
                                layers: draft.manifest.layers.map((item, i) =>
                                  i === index ? { ...item, caption: event.target.value } : item,
                                ),
                              },
                            })
                          }
                          className="mt-1 block w-full rounded border border-gray-700 bg-gray-900 p-2 text-sm text-gray-100"
                        />
                      </label>
                    </div>
                  </li>
                ))}
              </ol>
              <button
                type="button"
                disabled={Boolean(busy) || !dirty}
                onClick={() => void save()}
                className="operator-button"
              >
                Save captions{dirty ? ' *' : ''}
              </button>
            </>
          )}
        </div>
      </Modal>
    </>
  );
}
