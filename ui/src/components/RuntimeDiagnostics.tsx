'use client';
import { useState } from 'react';
import { apiClient } from '@/utils/api';
type Report = {
  checkedAt: string;
  ready: boolean;
  checks: Array<{ name: string; ok: boolean; detail: string; action?: string | null }>;
};
export default function RuntimeDiagnostics() {
  const [report, setReport] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState('');
  const check = async () => {
    setBusy(true);
    setError('');
    try {
      const response = await apiClient.get<Report>('/api/diagnostics', { timeout: 100_000 });
      setReport(response.data);
    } catch {
      setError('Diagnostics could not finish. Check worker startup logs.');
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="space-y-3 rounded border border-gray-800 bg-gray-950 p-4">
      <div className="flex items-center justify-between gap-4">
        <h2 className="font-semibold">Environment readiness</h2>
        <button className="operator-button" disabled={busy} onClick={check}>
          {busy ? 'Checking environment…' : 'Check environment'}
        </button>
      </div>
      <p className="text-sm text-gray-400">
        Checks Python packages, device support, storage access, and the queue worker.
      </p>
      {error && (
        <p role="alert" className="text-red-300">
          {error}
        </p>
      )}
      {report && (
        <>
          <p role="status" className={report.ready ? 'text-green-300' : 'text-amber-300'}>
            {report.ready ? 'Ready' : 'Action required'} · Checked {new Date(report.checkedAt).toLocaleTimeString()}
          </p>
          <ul className="space-y-2 text-sm">
            {report.checks.map(check => (
              <li key={check.name}>
                <strong className={check.ok ? 'text-green-300' : 'text-amber-300'}>{check.name}</strong>
                <p className="break-words text-gray-300">{check.detail}</p>
                {check.action && <p className="text-gray-400">{check.action}</p>}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
