'use client';
import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowRight, Box, Search, Loader2 } from 'lucide-react';
import { apiClient } from '@/utils/api';

type LibraryModel = {
  id: string;
  label: string;
  filename: string;
  updatedAt: string;
  sizeBytes: number;
  jobId?: string;
  triggerWords: string[];
};
function readModels(value: unknown): LibraryModel[] {
  if (!value || typeof value !== 'object' || !('loras' in value) || !Array.isArray(value.loras))
    throw new Error('Invalid model library response.');
  return value.loras.flatMap((item: unknown) => {
    if (!item || typeof item !== 'object') return [];
    const record = item as Record<string, unknown>;
    if (typeof record.id !== 'string' || typeof record.label !== 'string' || typeof record.filename !== 'string')
      return [];
    return [
      {
        id: record.id,
        label: record.label,
        filename: record.filename,
        updatedAt: typeof record.updatedAt === 'string' ? record.updatedAt : '',
        sizeBytes: typeof record.sizeBytes === 'number' ? record.sizeBytes : 0,
        jobId: typeof record.jobId === 'string' ? record.jobId : undefined,
        triggerWords: Array.isArray(record.triggerWords)
          ? record.triggerWords.filter((word): word is string => typeof word === 'string')
          : [],
      },
    ];
  });
}

export default function ModelsPage() {
  const [models, setModels] = useState<LibraryModel[]>([]);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [query, setQuery] = useState('');
  const refresh = useCallback(async () => {
    setStatus('loading');
    try {
      const response = await apiClient.get('/api/generate/loras');
      setModels(readModels(response.data));
      setStatus('success');
    } catch {
      setStatus('error');
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  const filtered = models.filter(model =>
    `${model.label} ${model.triggerWords.join(' ')}`.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <div className="journey-page">
      <header className="journey-heading">
        <div>
          <h1 className="journey-title">Models</h1>
          <p className="mt-2 text-gray-400">Your training checkpoints and uploaded LoRAs, ready to explore.</p>
        </div>
        <Link className="operator-button" href="/generate">
          Upload in Generate <ArrowRight size={16} />
        </Link>
      </header>
      <label className="my-8 flex max-w-lg items-center gap-3 rounded-lg border border-gray-700 px-4 py-3">
        <Search size={18} aria-hidden="true" />
        <span className="sr-only">Search models</span>
        <input
          className="min-w-0 flex-1 bg-transparent outline-none"
          value={query}
          onChange={event => setQuery(event.target.value)}
          placeholder="Search models or trigger words"
        />
      </label>
      {status === 'loading' ? (
        <p role="status" className="flex items-center gap-3">
          <Loader2 className="animate-spin" /> Loading models…
        </p>
      ) : status === 'error' ? (
        <div role="alert" className="journey-card p-6">
          <p>Couldn’t load the model library.</p>
          <button className="studio-primary mt-4" onClick={() => void refresh()}>
            Retry
          </button>
        </div>
      ) : models.length === 0 ? (
        <div className="journey-card p-8">
          <Box className="mb-5 text-gray-400" size={42} />
          <h2 className="text-2xl font-semibold">Your models will appear here</h2>
          <p className="mt-3 max-w-xl text-gray-400">
            Train a LoRA or upload an existing one in Generate. Checkpoints saved by local training runs appear
            automatically.
          </p>
          <Link className="studio-primary mt-6" href="/jobs/new">
            Set up training <ArrowRight size={16} />
          </Link>
        </div>
      ) : filtered.length === 0 ? (
        <p role="status">No models match this search.</p>
      ) : (
        <ul className="grid gap-5 lg:grid-cols-2">
          {filtered.map(model => (
            <li key={model.id} className="journey-card min-w-0 p-6">
              <Box className="mb-5 text-[var(--studio-accent)]" aria-hidden="true" />
              <h2 className="break-words text-lg font-semibold">{model.label}</h2>
              <p className="mt-2 text-sm text-gray-400">
                {(model.sizeBytes / 1024 / 1024).toFixed(1)} MB
                {model.updatedAt && ` · ${new Date(model.updatedAt).toLocaleDateString()}`}
              </p>
              <p className="mt-4 text-sm text-gray-300">
                {model.triggerWords.length
                  ? `Trigger words: ${model.triggerWords.join(', ')}`
                  : 'No trigger words recorded'}
              </p>
              <div className="mt-6 flex flex-wrap items-center gap-5">
                <Link className="studio-primary" href={`/generate?model_ref=${encodeURIComponent(model.id)}`}>
                  Try this model <ArrowRight size={16} />
                </Link>
                {model.jobId && (
                  <Link className="studio-text-link text-sm" href={`/jobs/${encodeURIComponent(model.jobId)}`}>
                    View training run
                  </Link>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
