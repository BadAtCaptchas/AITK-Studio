import type { Job } from '@/types';
import useGPUInfo from '@/hooks/useGPUInfo';
import useCPUInfo from '@/hooks/useCPUInfo';
import GPUWidget from '@/components/GPUWidget';
import CPUWidget from '@/components/CPUWidget';
import FilesWidget from '@/components/FilesWidget';
import TensorBoardLink from '@/components/TensorBoardLink';
import { JobAdvisorPanel } from '@/components/TrainingAdvisorPanel';
import { getJobConfig, getTotalSteps } from '@/utils/jobs';
import {
  buildTrainingPhaseSummary,
  hasTrainingPhases,
  type TrainingPhaseSummary,
} from '@/utils/trainingPhaseSummary';
import { Cpu, HardDrive, Info, Gauge, Layers } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import useJobLog from '@/hooks/useJobLog';
import useJobMetrics from '@/hooks/useJobMetrics';
import useJobDownloadProgress from '@/hooks/useJobDownloadProgress';
import { HFDownloadProgressBand } from '@/components/HFDownloadProgress';
import { PageNotice, ProgressBar, StatusBadge } from '@/components/OperatorPrimitives';

interface JobOverviewProps {
  job: Job;
}

function isLiveJob(job: Job) {
  return job.status === 'queued' || job.status === 'running' || job.status === 'stopping';
}

function formatPhaseValue(value: number) {
  return value.toLocaleString();
}

function usefulPhaseReason(reason: string | null) {
  if (!reason || reason === 'initial' || reason === 'resume') return null;
  return reason.replace(/_/g, ' ');
}

function phaseDetailText(summary: TrainingPhaseSummary) {
  if (summary.phaseStep !== null && summary.phaseSteps !== null) {
    return `Phase step ${formatPhaseValue(summary.phaseStep)} of ${formatPhaseValue(summary.phaseSteps)}`;
  }
  if (summary.phaseStep !== null) {
    return `Phase step ${formatPhaseValue(summary.phaseStep)}`;
  }
  if (summary.telemetryPending) {
    return 'Open-ended auto learn phase; waiting for telemetry';
  }
  if (summary.isAutoTrain) {
    return 'Open-ended auto learn phase';
  }
  return 'Phase step unavailable';
}

function phaseStatusText(summary: TrainingPhaseSummary) {
  const reason = usefulPhaseReason(summary.reason);
  if (reason) return `Last transition: ${reason}`;
  if (summary.telemetryPending) return 'Telemetry pending';
  return summary.source === 'metrics' ? 'Live telemetry' : 'Estimated from config';
}

function TrainingPhaseOverview({ summary }: { summary: TrainingPhaseSummary }) {
  return (
    <div className="space-y-2 border border-gray-800 bg-gray-950/50 px-3 py-2">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <Layers className="w-5 h-5 text-cyan-400 shrink-0" />
          <div className="min-w-0">
            <p className="text-xs text-gray-400">Training phase</p>
            <p className="text-sm font-medium text-gray-100 truncate">
              Phase {summary.index + 1} of {summary.count}: {summary.name}
            </p>
          </div>
        </div>
        <span className="rounded-full border border-gray-700 bg-gray-900 px-2.5 py-1 text-xs text-gray-300">
          {summary.isAutoTrain ? 'Auto learn' : 'Phased'}
        </span>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="text-gray-300">{phaseDetailText(summary)}</span>
        <span className="text-gray-500">{phaseStatusText(summary)}</span>
      </div>

      {summary.progress !== null && (
        <ProgressBar value={summary.progress} />
      )}
    </div>
  );
}

export default function JobOverview({ job }: JobOverviewProps) {
  const gpuIds = useMemo(() => {
    if (job.gpu_ids === 'mps') {
      return [0]; // For MPS, we can just return a single GPU ID since it's virtualized
    }
    return job.gpu_ids.split(',').map(id => parseInt(id));
  }, [job.gpu_ids]);
  const isStopping = job.stop && job.status === 'running';
  const shouldPoll = isLiveJob(job) || isStopping;
  const logPollInterval = shouldPoll ? 5000 : null;
  const downloadPollInterval = shouldPoll ? 2000 : null;
  const systemPollInterval = shouldPoll ? 10000 : null;
  const { log, status: statusLog } = useJobLog(job.id, logPollInterval);
  const { progress: hfDownloadProgress } = useJobDownloadProgress(
    job.id,
    job.hf_download_progress || null,
    downloadPollInterval,
  );
  const logRef = useRef<HTMLDivElement>(null);
  // Track whether we should auto-scroll to bottom
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo(gpuIds, systemPollInterval, job.worker_id);
  const { cpuInfo, isCPUInfoLoaded } = useCPUInfo(systemPollInterval, job.worker_id);
  const totalSteps = getTotalSteps(job);
  const progress = totalSteps && totalSteps > 0 ? (job.step / totalSteps) * 100 : null;
  const jobType = job?.job_type || 'unknown';
  const trainConfig = useMemo(() => {
    if (jobType !== 'train') return null;
    try {
      return getJobConfig(job).config.process[0]?.train ?? null;
    } catch {
      return null;
    }
  }, [job.job_config, jobType]);
  const hasPhaseOverview = jobType === 'train' && hasTrainingPhases(trainConfig);
  const phaseMetricsOptions = useMemo(
    () => ({
      keys: ['phase/*'],
      maxPoints: 100,
      enabled: hasPhaseOverview,
    }),
    [hasPhaseOverview],
  );
  const { latest: phaseMetricLatest } = useJobMetrics(job.id, job.status, phaseMetricsOptions);
  const phaseSummary = useMemo(
    () =>
      buildTrainingPhaseSummary(trainConfig, job.step, {
        index: phaseMetricLatest['phase/index'],
        name: phaseMetricLatest['phase/name'],
        step: phaseMetricLatest['phase/step'],
        reason: phaseMetricLatest['phase/reason'],
      }),
    [job.step, phaseMetricLatest, trainConfig],
  );

  const logLines: string[] = useMemo(() => {
    // split at line breaks on \n or \r\n but not \r
    let splits: string[] = log.split(/\n|\r\n/);

    splits = splits.map(line => {
      return line.split(/\r/).pop();
    }) as string[];

    // only return last 100 lines max
    const maxLines = 1000;
    if (splits.length > maxLines) {
      splits = splits.slice(splits.length - maxLines);
    }

    return splits;
  }, [log]);

  // Handle scroll events to determine if user has scrolled away from bottom
  const handleScroll = () => {
    if (logRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logRef.current;
      // Consider "at bottom" if within 10 pixels of the bottom
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 10;
      setIsScrolledToBottom(isAtBottom);
    }
  };

  // Auto-scroll to bottom only if we were already at the bottom
  useEffect(() => {
    if (logRef.current && isScrolledToBottom) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log, isScrolledToBottom]);

  let status = job.status;
  if (isStopping) {
    status = 'stopping';
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {/* Job Information Panel */}
      <div className="col-span-2 flex flex-col overflow-hidden border border-gray-800 bg-gray-900/60">
        <div className="flex items-center justify-between gap-3 border-b border-gray-800 bg-gray-900 px-3 py-2">
          <h2 className="min-w-0 truncate text-gray-100">
            <Info className="w-5 h-5 mr-2 -mt-1 text-amber-600 dark:text-amber-400 inline-block" /> {job.info}
          </h2>
          <div className="flex flex-none items-center gap-2">
            {jobType === 'train' && <TensorBoardLink compact />}
            <StatusBadge status={status} />
          </div>
        </div>

        <div className="flex flex-grow flex-col space-y-4 p-3">
          <HFDownloadProgressBand progress={hfDownloadProgress} />
          {job.remote_error && (
            <PageNotice tone="warning" title="Remote worker reported a problem">
              {job.remote_error}
            </PageNotice>
          )}
          {jobType === 'train' && <JobAdvisorPanel job={job} />}

          {/* Progress Bar */}
          {job.job_type === 'train' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Progress</span>
                <span className="text-gray-200">
                  {totalSteps ? `Step ${job.step} of ${totalSteps}` : `Step ${job.step}`}
                </span>
              </div>
              {progress !== null && (
                <ProgressBar value={progress} />
              )}
            </div>
          )}
          {phaseSummary && <TrainingPhaseOverview summary={phaseSummary} />}

          {/* Job Info Grid */}
          <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
            <div className="flex items-center space-x-4">
              <HardDrive className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <div>
                <p className="text-xs text-gray-400">Job Name</p>
                <p className="text-sm font-medium text-gray-200">{job.name}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <Cpu className="w-5 h-5 text-cyan-500 dark:text-cyan-400" />
              <div>
                <p className="text-xs text-gray-400">Assigned GPUs</p>
                <p className="text-sm font-medium text-gray-200">
                  {job.worker_id === 'local' ? 'Local' : 'Remote'} GPUs: {job.gpu_ids}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <Gauge className="w-5 h-5 text-green-600 dark:text-green-400" />
              <div>
                <p className="text-xs text-gray-400">Speed</p>
                <p className="text-sm font-medium text-gray-200">{job.speed_string == '' ? '?' : job.speed_string}</p>
              </div>
            </div>
          </div>

          {/* Log - Now using flex-grow to fill remaining space */}
          <div className="relative min-h-60 flex-grow border border-gray-800 bg-gray-950">
            <div
              ref={logRef}
              className="absolute inset-0 overflow-y-auto p-3 text-xs text-gray-300"
              onScroll={handleScroll}
            >
              {statusLog === 'loading' && 'Loading log...'}
              {statusLog === 'error' && 'Error loading log'}
              {['success', 'refreshing'].includes(statusLog) && (
                <div>
                  {logLines.length > 0 ? logLines.map((line, index) => <pre key={index}>{line}</pre>) : 'No log output yet.'}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* GPU Widget Panel */}
      <div className="col-span-1">
        <div>{isCPUInfoLoaded && cpuInfo && <CPUWidget cpu={cpuInfo} />}</div>
        <div className="mt-4">{isGPUInfoLoaded && gpuList.length > 0 && <GPUWidget gpu={gpuList[0]} />}</div>
        {jobType === 'train' && (
          <div className="mt-4">
            <FilesWidget jobID={job.id} jobName={job.name} />
          </div>
        )}
      </div>
    </div>
  );
}
