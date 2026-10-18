'use client';
import useStoredString from '@/hooks/useStoredString';
import useScopedState from '@/hooks/useScopedState';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { apiClient } from '@/utils/api';
import { unlockEncryptedDatasetKey } from '@/utils/encryptedDatasets';
import type { EncryptedDatasetManifest } from '@/types';
import type { RelinkPreview } from '@/server/datasetRelinking';

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
type Progress = {
  id: string;
  state: string;
  phase: string;
  canCancel: boolean;
  completed: number;
  total: number;
  error: string;
  result: Record<string, unknown> | null;
};
function progress(value: unknown): Progress {
  if (!record(value) || typeof value.id !== 'string' || typeof value.state !== 'string')
    throw new Error('Invalid migration response');
  const counts = record(value.progress) ? value.progress : {};
  return {
    id: value.id,
    state: value.state,
    phase: typeof value.phase === 'string' ? value.phase : '',
    canCancel: value.canCancel === true,
    completed: typeof counts.completed === 'number' ? counts.completed : 0,
    total: typeof counts.total === 'number' ? counts.total : 0,
    error: typeof value.error === 'string' ? value.error : '',
    result: record(value.result) ? value.result : null,
  };
}
export default function DatasetDecryption({
  datasetName,
  workerID,
  manifest,
}: {
  datasetName: string;
  workerID: string;
  manifest: EncryptedDatasetManifest;
}) {
  const storageKey = `dataset-decrypt:${workerID}:${datasetName}`;
  const [savedDestination] = useStoredString(storageKey + ':destination', `${datasetName}_decrypted`);
  const [editedDestination, setOutputName] = useScopedState<string | null>(storageKey, null);
  const outputName = editedDestination ?? savedDestination;
  const [password, setPassword] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [operation, setOperation] = useState<Progress | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [jobs, setJobs] = useState<RelinkPreview[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [outcomes, setOutcomes] = useState<string[]>([]);
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      const id = window.localStorage.getItem(storageKey);
      if (!id) return;
      try {
        const response = await apiClient.get(
          `/api/operations/${encodeURIComponent(id)}?worker_id=${encodeURIComponent(workerID)}`,
        );
        const next = progress(response.data as unknown);
        if (disposed) return;
        setOperation(next);
        if (!['completed', 'failed', 'canceled', 'needs-keys'].includes(next.state)) timer = setTimeout(poll, 1500);
      } catch (cause) {
        if (!disposed) setError(cause instanceof Error ? cause.message : 'Unable to read migration progress');
      }
    };
    void poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  }, [storageKey, workerID, operation?.id, busy]);
  const running = operation && !['completed', 'failed', 'canceled', 'needs-keys'].includes(operation.state);
  async function start() {
    setBusy(true);
    setError('');
    try {
      const type = manifest.crypto.kdf.type;
      if (type === 'KEYFILE-SHA256' && !file) throw new Error('Choose the original key file');
      const credential =
        type === 'PBKDF2-SHA256'
          ? { provider: 'password' as const, password }
          : type === 'KEYFILE-SHA256' && file
            ? { provider: 'keyFile' as const, file }
            : type === 'WEBAUTHN-PRF'
              ? { provider: 'webauthnPrf' as const }
              : null;
      if (!credential) throw new Error('This credential provider is not supported');
      const unlocked = await unlockEncryptedDatasetKey(manifest, credential);
      const response = await apiClient.post('/api/datasets/decrypt', {
        datasetName,
        outputName,
        worker_id: workerID,
        keyB64: unlocked.rawKeyB64,
      });
      const next = progress(response.data as unknown);
      window.localStorage.setItem(storageKey, next.id);
      window.localStorage.setItem(storageKey + ':destination', outputName);
      setOperation(next);
      setPassword('');
      setFile(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to decrypt dataset');
    } finally {
      setBusy(false);
    }
  }
  async function relink(apply = false) {
    if (!operation) return;
    setBusy(true);
    setError('');
    try {
      const response = await apiClient.post('/api/datasets/relink', {
        operationID: operation.id,
        worker_id: workerID,
        action: apply ? 'apply' : 'preview',
        jobs: apply
          ? jobs
              .filter(job => selected.includes(job.id))
              .map(job => ({ id: job.id, revision: job.revision, remoteRevision: job.remoteRevision }))
          : undefined,
      });
      const value: unknown = response.data;
      if (!record(value)) throw new Error('Invalid relinking response');
      if (apply && Array.isArray(value.outcomes))
        setOutcomes(
          value.outcomes.map((result: unknown) =>
            record(result)
              ? `${String(result.id)}: ${result.success === true ? 'Relinked' : String(result.error)}`
              : 'Invalid job outcome',
          ),
        );
      else if (Array.isArray(value.jobs))
        setJobs(
          value.jobs.filter(
            (job: unknown): job is RelinkPreview =>
              record(job) &&
              typeof job.id === 'string' &&
              typeof job.name === 'string' &&
              typeof job.revision === 'string' &&
              typeof job.status === 'string' &&
              Array.isArray(job.replacements) &&
              job.replacements.every(
                (replacement: unknown) =>
                  record(replacement) &&
                  typeof replacement.field === 'string' &&
                  typeof replacement.from === 'string' &&
                  typeof replacement.to === 'string',
              ),
          ),
        );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to relink jobs');
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="mx-auto max-w-3xl space-y-4 rounded-lg border border-gray-700 bg-gray-900 p-6">
      <h1 className="text-xl font-semibold">Decrypt to plain dataset</h1>
      <p className="text-sm text-gray-300">
        Encrypted datasets have been retired. Create a plain copy to edit, caption, or train. The encrypted original
        remains intact.
      </p>
      {!running && operation?.state !== 'completed' && (
        <>
          <label className="block">
            Destination name
            <input
              className="mt-1 block w-full rounded border border-gray-600 bg-gray-950 p-2"
              value={outputName}
              onChange={event => setOutputName(event.target.value)}
              disabled={busy}
            />
          </label>
          {manifest.crypto.kdf.type === 'PBKDF2-SHA256' && (
            <label className="block">
              Password
              <input
                type="password"
                autoComplete="off"
                className="mt-1 block w-full rounded border border-gray-600 bg-gray-950 p-2"
                value={password}
                onChange={event => setPassword(event.target.value)}
              />
            </label>
          )}
          {manifest.crypto.kdf.type === 'KEYFILE-SHA256' && (
            <label className="block">
              Original key file
              <input type="file" className="mt-1 block" onChange={event => setFile(event.target.files?.[0] || null)} />
            </label>
          )}
          {manifest.crypto.kdf.type === 'WEBAUTHN-PRF' && <p>Use the security key registered with this dataset.</p>}
          {operation?.state === 'needs-keys' && <p>Unlock again to resume after a restart.</p>}
          <button className="operator-button" disabled={busy || !outputName.trim()} onClick={() => void start()}>
            {busy ? 'Unlocking…' : 'Decrypt to plain dataset'}
          </button>
        </>
      )}
      {operation && (
        <div aria-live="polite">
          <p>
            {operation.state} · {operation.completed} / {operation.total} items
          </p>
          {operation.error && <p className="text-red-300">{operation.error}</p>}
          {operation.canCancel && (
            <button
              className="operator-button mt-2"
              onClick={() => {
                void apiClient
                  .delete(`/api/operations/${operation.id}?worker_id=${encodeURIComponent(workerID)}`)
                  .catch(cause => setError(String(cause)));
              }}
            >
              Cancel migration
            </button>
          )}
        </div>
      )}
      {operation?.state === 'completed' && (
        <>
          <p>
            Created {String(operation.result?.datasetName)}: {String(operation.result?.itemCount)} items and{' '}
            {String(operation.result?.captionCount)} captions.
          </p>
          {Array.isArray(operation.result?.renamed) &&
            operation.result.renamed.map(
              (item: unknown, index: number) =>
                record(item) && (
                  <p key={index} className="text-sm">
                    Renamed {String(item.original)} → {String(item.name)}
                  </p>
                ),
            )}
          <Link
            className="operator-button inline-block"
            href={`/datasets/${encodeURIComponent(String(operation.result?.datasetName))}?worker_id=${encodeURIComponent(workerID)}`}
          >
            Open plain dataset
          </Link>
          <button className="operator-button ml-2" disabled={busy} onClick={() => void relink()}>
            Preview job relinking
          </button>
          <p className="text-sm text-gray-400">
            Select idle jobs to update. Queued jobs will become stopped. Training history and checkpoints are preserved.
          </p>
          {jobs.map(job => (
            <div key={job.id} className="rounded border border-gray-700 p-3">
              <label>
                <input
                  type="checkbox"
                  disabled={!['queued', 'stopped', 'error', 'completed'].includes(job.status) || busy}
                  checked={selected.includes(job.id)}
                  onChange={event =>
                    setSelected(current =>
                      event.target.checked ? [...current, job.id] : current.filter(id => id !== job.id),
                    )
                  }
                />{' '}
                {job.name} · {job.status}
              </label>
              {job.replacements.map(change => (
                <p key={change.field} className="break-all text-xs text-gray-400">
                  {change.field}: {change.from} → {change.to}
                </p>
              ))}
            </div>
          ))}
          {jobs.length > 0 && (
            <button className="operator-button" disabled={busy || !selected.length} onClick={() => void relink(true)}>
              Relink selected jobs
            </button>
          )}
          {outcomes.map((outcome, index) => (
            <p key={index}>{outcome}</p>
          ))}
        </>
      )}
      {error && (
        <p role="alert" className="text-red-300">
          {error}
        </p>
      )}
    </section>
  );
}
