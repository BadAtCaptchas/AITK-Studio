'use client';

import { Boxes, RefreshCw } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import useModels from '@/hooks/useModels';

function formatBytes(bytes: number | null) {
  if (!bytes) return '-';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

export default function ModelsPage() {
  const { artifacts, status, reindex } = useModels(10000);

  return (
    <>
      <TopBar>
        <div className="flex items-center gap-3">
          <Boxes className="h-5 w-5 text-blue-400" />
          <h1 className="text-base text-white">Models</h1>
          <span className="text-xs text-gray-500">Artifact registry</span>
        </div>
        <button
          type="button"
          onClick={() => void reindex()}
          className="ml-auto inline-flex items-center gap-2 rounded border border-white/10 px-3 py-1.5 text-xs text-gray-300 hover:bg-white/5 hover:text-white"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${status === 'refreshing' ? 'animate-spin' : ''}`} />
          Reindex
        </button>
      </TopBar>
      <MainContent>
        <div className="overflow-hidden border border-white/10 bg-black">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-white/10 text-[11px] uppercase tracking-wide text-gray-500">
              <tr>
                <th className="px-3 py-2">Name</th>
                <th className="px-3 py-2">Kind</th>
                <th className="px-3 py-2">Source</th>
                <th className="px-3 py-2">Step</th>
                <th className="px-3 py-2">Size</th>
                <th className="px-3 py-2">Modified</th>
                <th className="px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {artifacts.map(artifact => (
                <tr key={artifact.id} className="border-b border-white/5 hover:bg-white/[0.03]">
                  <td className="max-w-sm truncate px-3 py-2 text-white" title={artifact.path}>{artifact.name}</td>
                  <td className="px-3 py-2 text-gray-300">{artifact.kind.replaceAll('_', ' ')}</td>
                  <td className="px-3 py-2 text-gray-400">{artifact.source}</td>
                  <td className="px-3 py-2 font-mono text-gray-300">{artifact.step?.toLocaleString() ?? '-'}</td>
                  <td className="px-3 py-2 text-gray-300">{formatBytes(artifact.size)}</td>
                  <td className="px-3 py-2 text-gray-400">{artifact.modified_at ? new Date(artifact.modified_at).toLocaleString() : '-'}</td>
                  <td className="px-3 py-2">
                    <span className={artifact.exists ? 'text-green-400' : 'text-yellow-400'}>{artifact.exists ? 'available' : 'missing'}</span>
                  </td>
                </tr>
              ))}
              {artifacts.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-3 py-10 text-center text-gray-500">
                    No model artifacts indexed yet. Run reindex or wait for the worker.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </MainContent>
    </>
  );
}
