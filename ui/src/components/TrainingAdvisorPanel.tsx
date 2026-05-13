'use client';

import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle2, Info, Loader2 } from 'lucide-react';
import classNames from 'classnames';
import type { AdvisorFinding, AdvisorResult, AdvisorSeverity, Job, JobConfig } from '@/types';
import { apiClient } from '@/utils/api';

type AdvisorStatus = 'idle' | 'loading' | 'success' | 'error' | 'refreshing';
type Variant = 'card' | 'inline';

const severityLabel: Record<AdvisorSeverity, string> = {
  critical: 'Critical',
  warning: 'Warning',
  info: 'Note',
};

const severityClasses: Record<AdvisorSeverity, string> = {
  critical: 'border-rose-700 bg-rose-950/35 text-rose-100',
  warning: 'border-amber-700 bg-amber-950/35 text-amber-100',
  info: 'border-sky-800 bg-sky-950/30 text-sky-100',
};

const severityBadgeClasses: Record<AdvisorSeverity, string> = {
  critical: 'bg-rose-500/15 text-rose-200',
  warning: 'bg-amber-500/15 text-amber-200',
  info: 'bg-sky-500/15 text-sky-200',
};

function severityIcon(severity: AdvisorSeverity) {
  if (severity === 'critical' || severity === 'warning') return <AlertTriangle className="h-4 w-4" />;
  return <Info className="h-4 w-4" />;
}

function isTerminalStatus(status?: string) {
  return status === 'completed' || status === 'error' || status === 'stopped';
}

function FindingCard({ finding }: { finding: AdvisorFinding }) {
  return (
    <div className={classNames('rounded-sm border px-3 py-2', severityClasses[finding.severity])}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1 text-xs font-medium uppercase">
          {severityIcon(finding.severity)}
          {severityLabel[finding.severity]}
        </span>
        <span className="rounded-sm bg-black/20 px-1.5 py-0.5 text-xs uppercase text-gray-300">{finding.category}</span>
        <span className="rounded-sm bg-black/20 px-1.5 py-0.5 text-xs uppercase text-gray-300">{finding.stage}</span>
      </div>
      <div className="mt-2 font-medium text-gray-100">{finding.title}</div>
      <div className="mt-1 text-sm text-gray-300">{finding.message}</div>
      <div className="mt-2 text-sm text-gray-200">{finding.recommendation}</div>
      {!!finding.evidence?.length && (
        <div className="mt-2 text-xs text-gray-400">{finding.evidence.join(' | ')}</div>
      )}
      {!!finding.relatedConfigPaths?.length && (
        <div className="mt-2 flex flex-wrap gap-1">
          {finding.relatedConfigPaths.map(configPath => (
            <span key={configPath} className="rounded-sm bg-black/20 px-1.5 py-0.5 text-xs text-gray-300">
              {configPath}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function AdvisorResultPanel({
  result,
  status,
  error,
  variant = 'card',
}: {
  result: AdvisorResult | null;
  status: AdvisorStatus;
  error: string | null;
  variant?: Variant;
}) {
  const groupedFindings = useMemo(() => {
    const grouped: Record<AdvisorSeverity, AdvisorFinding[]> = {
      critical: [],
      warning: [],
      info: [],
    };
    for (const finding of result?.findings ?? []) {
      grouped[finding.severity].push(finding);
    }
    return grouped;
  }, [result]);

  const loading = status === 'loading' || status === 'refreshing';
  const hasFindings = !!result?.findings.length;
  const stats = result?.datasetStats;

  return (
    <section
      className={classNames(
        'rounded-lg border border-gray-800',
        variant === 'card' ? 'bg-gray-900 p-4' : 'bg-gray-950/50 p-3',
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold uppercase text-gray-400">Training Advisor</h2>
          <p className="mt-1 text-sm text-gray-300">
            {result?.summary.text ?? (loading ? 'Analyzing training setup...' : 'Warn-only quality checks')}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {loading && <Loader2 className="h-4 w-4 animate-spin text-gray-400" />}
          {result && (
            <>
              <span className={classNames('rounded-sm px-2 py-1', severityBadgeClasses.critical)}>
                {result.summary.critical} critical
              </span>
              <span className={classNames('rounded-sm px-2 py-1', severityBadgeClasses.warning)}>
                {result.summary.warning} warning
              </span>
              <span className={classNames('rounded-sm px-2 py-1', severityBadgeClasses.info)}>
                {result.summary.info} notes
              </span>
            </>
          )}
        </div>
      </div>

      {error && <div className="mt-3 rounded-sm border border-rose-800 bg-rose-950/30 px-3 py-2 text-sm text-rose-200">{error}</div>}

      {stats && (
        <div className="mt-3 flex flex-wrap gap-2 text-xs text-gray-400">
          <span>{stats.mediaFiles.toLocaleString()} media scanned</span>
          <span>{stats.missingCaptions.toLocaleString()} missing captions</span>
          <span>{stats.emptyCaptions.toLocaleString()} empty captions</span>
          {stats.truncated && <span>scan capped</span>}
        </div>
      )}

      {!loading && result && !hasFindings && (
        <div className="mt-3 flex items-center gap-2 rounded-sm border border-emerald-800 bg-emerald-950/25 px-3 py-2 text-sm text-emerald-100">
          <CheckCircle2 className="h-4 w-4" />
          No training quality issues found.
        </div>
      )}

      {hasFindings && (
        <div className="mt-3 space-y-3">
          {(['critical', 'warning', 'info'] as AdvisorSeverity[]).map(severity => {
            const findings = groupedFindings[severity];
            if (!findings.length) return null;
            return (
              <div key={severity} className="space-y-2">
                <div className="text-xs font-medium uppercase text-gray-500">{severityLabel[severity]}</div>
                {findings.map(finding => (
                  <FindingCard key={finding.id} finding={finding} />
                ))}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

export function TrainingAdvisorPanel({ jobConfig, gpuIDs }: { jobConfig: JobConfig; gpuIDs: string | null }) {
  const [result, setResult] = useState<AdvisorResult | null>(null);
  const [status, setStatus] = useState<AdvisorStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const payload = useMemo(() => JSON.stringify({ job_config: jobConfig, gpu_ids: gpuIDs }), [jobConfig, gpuIDs]);

  useEffect(() => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => {
      setStatus(current => (current === 'success' ? 'refreshing' : 'loading'));
      setError(null);
      apiClient
        .post('/api/training-advisor/preflight', JSON.parse(payload), { signal: controller.signal })
        .then(res => {
          setResult(res.data as AdvisorResult);
          setStatus('success');
        })
        .catch(error => {
          if (controller.signal.aborted || error?.code === 'ERR_CANCELED') return;
          console.error('Error loading training advisor:', error);
          setError('Failed to run training advisor.');
          setStatus('error');
        });
    }, 700);

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [payload]);

  return <AdvisorResultPanel result={result} status={status} error={error} />;
}

export function JobAdvisorPanel({ job }: { job: Job }) {
  const [result, setResult] = useState<AdvisorResult | null>(null);
  const [status, setStatus] = useState<AdvisorStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    let controller: AbortController | null = null;

    const loadAdvisor = () => {
      controller?.abort();
      controller = new AbortController();
      setStatus(current => (current === 'success' ? 'refreshing' : 'loading'));
      setError(null);
      apiClient
        .get(`/api/jobs/${job.id}/advisor`, { signal: controller.signal })
        .then(res => {
          if (stopped) return;
          setResult(res.data as AdvisorResult);
          setStatus('success');
        })
        .catch(error => {
          if (stopped || controller?.signal.aborted || error?.code === 'ERR_CANCELED') return;
          console.error('Error loading job advisor:', error);
          setError('Failed to load training advisor.');
          setStatus('error');
        });
    };

    loadAdvisor();
    const interval = isTerminalStatus(job.status) ? null : window.setInterval(loadAdvisor, 10000);

    return () => {
      stopped = true;
      if (interval !== null) window.clearInterval(interval);
      controller?.abort();
    };
  }, [job.id, job.status]);

  return <AdvisorResultPanel result={result} status={status} error={error} variant="inline" />;
}
