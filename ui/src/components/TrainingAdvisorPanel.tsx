'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, Info, Loader2, RefreshCw } from 'lucide-react';
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

function isEditableElement(element: Element | null) {
  if (!(element instanceof HTMLElement)) return false;
  if (element.isContentEditable) return true;
  return element.matches('input, textarea, select, [role="textbox"], [contenteditable="true"]');
}

function severityIcon(severity: AdvisorSeverity) {
  if (severity === 'critical' || severity === 'warning') return <AlertTriangle className="h-4 w-4" />;
  return <Info className="h-4 w-4" />;
}

function isLiveStatus(status?: string) {
  return status === 'queued' || status === 'running' || status === 'stopping';
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
  onRefresh,
}: {
  result: AdvisorResult | null;
  status: AdvisorStatus;
  error: string | null;
  variant?: Variant;
  onRefresh?: () => void;
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
  const initialLoading = status === 'loading' && !result;
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
            {result?.summary.text ?? (initialLoading ? 'Analyzing training setup...' : 'Warn-only quality checks')}
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
          {onRefresh && (
            <button
              type="button"
              onClick={onRefresh}
              disabled={loading}
              className="rounded-sm border border-gray-700 p-1 text-gray-300 hover:bg-gray-800 hover:text-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
              title="Run training advisor checks"
              aria-label="Run training advisor checks"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
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

      {!initialLoading && result && !hasFindings && (
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
  const mountedRef = useRef(true);
  const requestIdRef = useRef(0);
  const lastCompletedPayloadRef = useRef<string | null>(null);
  const payload = useMemo(() => JSON.stringify({ job_config: jobConfig, gpu_ids: gpuIDs }), [jobConfig, gpuIDs]);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const runPreflight = useCallback(
    (payloadSnapshot = payload, options: { force?: boolean } = {}) => {
      if (!options.force && lastCompletedPayloadRef.current === payloadSnapshot) return;

      const requestId = requestIdRef.current + 1;
      requestIdRef.current = requestId;

      setStatus(current => (current === 'success' ? 'refreshing' : 'loading'));
      setError(null);
      apiClient
        .post('/api/training-advisor/preflight', JSON.parse(payloadSnapshot))
        .then(res => {
          if (!mountedRef.current || requestId !== requestIdRef.current) return;
          lastCompletedPayloadRef.current = payloadSnapshot;
          setResult(res.data as AdvisorResult);
          setStatus('success');
        })
        .catch(error => {
          if (!mountedRef.current || requestId !== requestIdRef.current || error?.code === 'ERR_CANCELED') return;
          console.error('Error loading training advisor:', error);
          setError('Failed to run training advisor.');
          setStatus('error');
        });
    },
    [payload],
  );

  useEffect(() => {
    let focusOutHandler: (() => void) | null = null;
    let focusOutDelay: number | null = null;
    let didRun = false;

    const timeout = window.setTimeout(() => {
      if (isEditableElement(document.activeElement)) {
        focusOutHandler = () => {
          if (focusOutDelay !== null) window.clearTimeout(focusOutDelay);
          focusOutDelay = window.setTimeout(() => {
            if (!didRun && !isEditableElement(document.activeElement)) {
              didRun = true;
              if (focusOutHandler) document.removeEventListener('focusout', focusOutHandler);
              runPreflight(payload);
            }
          }, 250);
        };
        document.addEventListener('focusout', focusOutHandler);
        return;
      }

      runPreflight(payload);
    }, 700);

    return () => {
      window.clearTimeout(timeout);
      if (focusOutDelay !== null) window.clearTimeout(focusOutDelay);
      if (focusOutHandler) document.removeEventListener('focusout', focusOutHandler);
    };
  }, [payload, runPreflight]);

  return (
    <AdvisorResultPanel
      result={result}
      status={status}
      error={error}
      onRefresh={() => runPreflight(payload, { force: true })}
    />
  );
}

export function JobAdvisorPanel({ job }: { job: Job }) {
  const [result, setResult] = useState<AdvisorResult | null>(null);
  const [status, setStatus] = useState<AdvisorStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const isLive = isLiveStatus(job.status);

  useEffect(() => {
    if (isLive) {
      setResult(null);
      setStatus('idle');
      setError(null);
      return;
    }

    let stopped = false;
    const controller = new AbortController();
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
        if (stopped || controller.signal.aborted || error?.code === 'ERR_CANCELED') return;
        console.error('Error loading job advisor:', error);
        setError('Failed to load training advisor.');
        setStatus('error');
      });

    return () => {
      stopped = true;
      controller.abort();
    };
  }, [job.id, isLive]);

  if (isLive) return null;
  return <AdvisorResultPanel result={result} status={status} error={error} variant="inline" />;
}
