'use client';
import { useCallback, useEffect, useMemo, useState } from 'react';
import Link from '@/navigation/Link';
import { ArrowDown, ArrowUp, ArrowUpDown, Box, Search, Loader2, RefreshCw, Upload } from 'lucide-react';
import { apiClient } from '@/utils/api';
import { TopBar, MainContent } from '@/components/layout';

type LibraryModel = {
  id: string;
  label: string;
  filename: string;
  updatedAt: string;
  sizeBytes: number;
  jobId?: string;
  triggerWords: string[];
};
type ModelSort = 'name' | 'size' | 'updated';

function modelTimestamp(value: string): number {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function modelSize(bytes: number): string {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}
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
  const [sort, setSort] = useState<ModelSort>('updated');
  const [direction, setDirection] = useState<'asc' | 'desc'>('desc');
  const loadModels = useCallback(
    () =>
      apiClient
        .get('/api/generate/loras')
        .then(response => {
          setModels(readModels(response.data));
          setStatus('success');
        })
        .catch(() => setStatus('error')),
    [],
  );
  const refresh = () => {
    setStatus('loading');
    return loadModels();
  };
  useEffect(() => {
    void loadModels();
  }, [loadModels]);
  const filtered = useMemo(() => {
    const search = query.trim().toLowerCase();
    return models
      .filter(model =>
        `${model.label} ${model.filename} ${model.triggerWords.join(' ')}`.toLowerCase().includes(search),
      )
      .sort((a, b) => {
        const compared =
          sort === 'name'
            ? a.label.localeCompare(b.label, undefined, { numeric: true })
            : sort === 'size'
              ? a.sizeBytes - b.sizeBytes
              : modelTimestamp(a.updatedAt) - modelTimestamp(b.updatedAt);
        return (direction === 'asc' ? compared : -compared) || a.label.localeCompare(b.label);
      });
  }, [direction, models, query, sort]);
  const changeSort = (nextSort: ModelSort) => {
    if (nextSort === sort) setDirection(current => (current === 'asc' ? 'desc' : 'asc'));
    else {
      setSort(nextSort);
      setDirection(nextSort === 'name' ? 'asc' : 'desc');
    }
  };
  const sortHeader = (key: ModelSort, label: string) => (
    <th
      scope="col"
      className="px-4 py-2 text-left font-medium"
      aria-sort={sort === key ? (direction === 'asc' ? 'ascending' : 'descending') : 'none'}
    >
      <button
        type="button"
        onClick={() => changeSort(key)}
        className="inline-flex items-center gap-2 hover:text-gray-100"
      >
        {label}
        {sort !== key ? (
          <ArrowUpDown size={12} aria-hidden="true" />
        ) : direction === 'asc' ? (
          <ArrowUp size={12} aria-hidden="true" />
        ) : (
          <ArrowDown size={12} aria-hidden="true" />
        )}
      </button>
    </th>
  );
  return (
    <>
      <TopBar>
        <Box size={16} className="text-gray-400" aria-hidden="true" />
        <h1 className="text-base font-semibold">Models</h1>
        <span className="text-xs tabular-nums text-gray-500">{models.length}</span>
        <div className="flex-1" />
        <button
          type="button"
          className="operator-button"
          onClick={() => void refresh()}
          disabled={status === 'loading'}
          aria-label="Refresh models"
        >
          <RefreshCw size={15} className={status === 'loading' ? 'animate-spin' : ''} aria-hidden="true" />
          <span className="hidden sm:inline">Refresh</span>
        </button>
        <Link className="operator-button" href="/generate">
          <Upload size={15} aria-hidden="true" /> Upload LoRA
        </Link>
      </TopBar>
      <MainContent>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs text-gray-400">
            {filtered.length} of {models.length} models
          </span>
          <label className="flex h-8 w-full items-center gap-2 rounded border border-gray-700 bg-gray-950 px-3 sm:w-80">
            <Search size={15} className="text-gray-500" aria-hidden="true" />
            <span className="sr-only">Search models</span>
            <input
              className="min-w-0 flex-1 bg-transparent text-sm outline-none"
              value={query}
              onChange={event => setQuery(event.target.value)}
              placeholder="Filter models or trigger words"
            />
          </label>
        </div>
        {status === 'loading' ? (
          <p role="status" className="flex items-center gap-2 border-y border-gray-800 px-4 py-6 text-sm text-gray-400">
            <Loader2 size={16} className="animate-spin" /> Loading models…
          </p>
        ) : status === 'error' ? (
          <div role="alert" className="flex items-center justify-between gap-3 border border-red-800 p-4 text-sm">
            <p>Couldn’t load the model library.</p>
            <button className="operator-button" onClick={() => void refresh()}>
              Retry
            </button>
          </div>
        ) : models.length === 0 ? (
          <div className="border-y border-gray-800 px-4 py-8">
            <h2 className="text-sm font-semibold">No models</h2>
            <p className="mt-2 text-sm text-gray-400">Local checkpoints and uploaded LoRAs appear here.</p>
            <Link className="operator-button mt-4" href="/jobs/new">
              New training
            </Link>
          </div>
        ) : filtered.length === 0 ? (
          <p role="status" className="border-y border-gray-800 p-4 text-sm text-gray-400">
            No models match this filter.
          </p>
        ) : (
          <div className="overflow-x-auto border-y border-gray-800">
            <table className="w-full min-w-[760px] border-collapse text-sm">
              <thead className="border-b border-gray-800 bg-gray-900 text-xs text-gray-400">
                <tr>
                  {sortHeader('name', 'Name')}
                  {sortHeader('size', 'Size')}
                  {sortHeader('updated', 'Updated')}
                  <th scope="col" className="px-4 py-2 text-left font-medium">
                    Trigger words
                  </th>
                  <th scope="col" className="px-4 py-2 text-right font-medium">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(model => (
                  <tr key={model.id} className="border-b border-gray-800 last:border-b-0 hover:bg-gray-900/60">
                    <th scope="row" className="max-w-[320px] px-4 py-2.5 text-left font-medium">
                      <Link
                        className="block truncate hover:text-brand-300"
                        title={model.filename}
                        href={`/generate?model_ref=${encodeURIComponent(model.id)}`}
                      >
                        {model.label}
                      </Link>
                    </th>
                    <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-gray-400">
                      {modelSize(model.sizeBytes)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-gray-400">
                      {modelTimestamp(model.updatedAt) ? (
                        <time dateTime={model.updatedAt} title={new Date(model.updatedAt).toLocaleString()}>
                          {new Date(model.updatedAt).toLocaleDateString()}
                        </time>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="max-w-[240px] px-4 py-2.5 text-gray-400">
                      <span className="block truncate" title={model.triggerWords.join(', ')}>
                        {model.triggerWords.join(', ') || '—'}
                      </span>
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex justify-end gap-3 whitespace-nowrap">
                        {model.jobId && (
                          <Link
                            className="text-gray-400 hover:text-gray-100"
                            href={`/jobs/${encodeURIComponent(model.jobId)}`}
                          >
                            Training run
                          </Link>
                        )}
                        <Link
                          className="text-brand-300 hover:text-brand-200"
                          href={`/generate?model_ref=${encodeURIComponent(model.id)}`}
                        >
                          Generate
                        </Link>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </MainContent>
    </>
  );
}
