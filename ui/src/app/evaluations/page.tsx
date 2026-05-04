'use client';

import { useMemo, useState } from 'react';
import { ShieldCheck, Play } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import useEvaluations from '@/hooks/useEvaluations';
import useJobsList from '@/hooks/useJobsList';

function metricSummary(metrics: string) {
  try {
    const parsed = JSON.parse(metrics || '{}');
    return Object.entries(parsed)
      .filter(([, value]) => typeof value === 'number')
      .map(([key, value]) => `${key}: ${(value as number).toFixed(4)}`)
      .join('  ');
  } catch {
    return '';
  }
}

export default function EvaluationsPage() {
  const { runs, createRun } = useEvaluations(5000);
  const { jobs } = useJobsList({ reloadInterval: 5000, job_type: 'train' });
  const [selectedJobs, setSelectedJobs] = useState<string[]>([]);
  const [referencePath, setReferencePath] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const selectedNames = useMemo(
    () => jobs.filter(job => selectedJobs.includes(job.id)).map(job => job.name).join(', '),
    [jobs, selectedJobs],
  );

  const submit = async () => {
    if (selectedJobs.length === 0 || isSubmitting) return;
    setIsSubmitting(true);
    try {
      await createRun({
        name: selectedNames ? `Evaluation: ${selectedNames}` : undefined,
        jobIds: selectedJobs,
        referencePath: referencePath.trim() || undefined,
      });
      setSelectedJobs([]);
      setReferencePath('');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <TopBar>
        <div className="flex items-center gap-3">
          <ShieldCheck className="h-5 w-5 text-blue-400" />
          <h1 className="text-base text-white">Evaluations</h1>
          <span className="text-xs text-gray-500">Automated CLIP / LPIPS / FID runs</span>
        </div>
      </TopBar>
      <MainContent>
        <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
          <section className="border border-white/10 bg-black p-4">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-white">New Evaluation</h2>
            <div className="mt-4 space-y-2">
              {jobs.map(job => (
                <label key={job.id} className="flex items-center gap-2 rounded border border-white/5 px-3 py-2 text-sm text-gray-300 hover:bg-white/5">
                  <input
                    type="checkbox"
                    checked={selectedJobs.includes(job.id)}
                    onChange={event => {
                      setSelectedJobs(prev => event.target.checked ? [...prev, job.id] : prev.filter(id => id !== job.id));
                    }}
                    className="accent-blue-500"
                  />
                  <span className="truncate">{job.name}</span>
                  <span className="ml-auto text-xs text-gray-500">{job.status}</span>
                </label>
              ))}
              {jobs.length === 0 && <div className="text-sm text-gray-500">No training jobs found.</div>}
            </div>
            <label className="mt-4 block text-xs uppercase tracking-wide text-gray-500">Reference dataset/path</label>
            <input
              value={referencePath}
              onChange={event => setReferencePath(event.target.value)}
              placeholder="Optional absolute path or dataset folder"
              className="mt-2 w-full border border-white/10 bg-black px-3 py-2 text-sm text-white outline-none focus:border-blue-500"
            />
            <button
              type="button"
              disabled={selectedJobs.length === 0 || isSubmitting}
              onClick={() => void submit()}
              className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded bg-blue-600 px-3 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              Queue Evaluation
            </button>
          </section>

          <section className="overflow-hidden border border-white/10 bg-black">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-white/10 text-[11px] uppercase tracking-wide text-gray-500">
                <tr>
                  <th className="px-3 py-2">Run</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2">Created</th>
                  <th className="px-3 py-2">Metrics</th>
                  <th className="px-3 py-2">Error</th>
                </tr>
              </thead>
              <tbody>
                {runs.map(run => (
                  <tr key={run.id} className="border-b border-white/5 hover:bg-white/[0.03]">
                    <td className="px-3 py-2 text-white">{run.name}</td>
                    <td className="px-3 py-2">
                      <span className={run.status === 'failed' ? 'text-red-400' : run.status === 'completed' ? 'text-green-400' : 'text-yellow-400'}>{run.status}</span>
                    </td>
                    <td className="px-3 py-2 text-gray-400">{new Date(run.created_at).toLocaleString()}</td>
                    <td className="max-w-md truncate px-3 py-2 text-gray-300">{metricSummary(run.metrics)}</td>
                    <td className="max-w-xs truncate px-3 py-2 text-red-300">{run.error}</td>
                  </tr>
                ))}
                {runs.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-3 py-10 text-center text-gray-500">
                      No evaluations yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>
        </div>
      </MainContent>
    </>
  );
}
