import fs from 'fs';
import path from 'path';
import { getTrainingFolder } from './settings';
import { db, type LossPoint } from './db';
import type { Job, JobConfig } from '../types';

export type CheckpointSummary = {
  name: string;
  path: string;
  step: number | null;
  size: number;
  modified_at: string;
  is_latest: boolean;
};

function safeJsonParse<T>(value: string, fallback: T): T {
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function formatDuration(seconds: number | null) {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return null;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function parseStep(fileName: string) {
  const match = fileName.match(/(?:_|step-?)(\d{1,9})(?:\D|$)/i) || fileName.match(/(\d{4,9})(?:\D|$)/);
  return match ? Number(match[1]) : null;
}

function listCheckpoints(jobFolder: string): CheckpointSummary[] {
  if (!fs.existsSync(jobFolder)) return [];
  const checkpoints = fs
    .readdirSync(jobFolder)
    .filter(file => file.toLowerCase().endsWith('.safetensors'))
    .map(file => {
      const filePath = path.join(jobFolder, file);
      const stat = fs.statSync(filePath);
      return {
        name: file.replace(/\.safetensors$/i, ''),
        path: filePath,
        step: parseStep(file),
        size: stat.size,
        modified_at: stat.mtime.toISOString(),
        is_latest: false,
      };
    })
    .sort((a, b) => {
      const stepDelta = (b.step ?? 0) - (a.step ?? 0);
      return stepDelta || new Date(b.modified_at).getTime() - new Date(a.modified_at).getTime();
    });
  if (checkpoints[0]) checkpoints[0].is_latest = true;
  return checkpoints;
}

function readLog(jobFolder: string) {
  const logPath = path.join(jobFolder, 'log.txt');
  if (!fs.existsSync(logPath)) return '';
  try {
    return fs.readFileSync(logPath, 'utf-8').replace(/\x1B\[A/g, '');
  } catch {
    return 'Error reading log file';
  }
}

function splitLog(log: string) {
  return log
    .split(/\n|\r\n/)
    .map(line => line.split(/\r/).pop() || '')
    .filter((line, index, lines) => line.length > 0 || index === lines.length - 1)
    .slice(-1500);
}

function summarizeConfig(job: Job) {
  const parsed = safeJsonParse<JobConfig | null>(job.job_config, null);
  const processConfig = parsed?.config?.process?.[0] as any;
  const train = processConfig?.train ?? {};
  const model = processConfig?.model ?? {};
  const save = processConfig?.save ?? {};
  const sample = processConfig?.sample ?? {};
  const dataset = processConfig?.datasets?.[0] ?? {};

  return {
    model: model.name_or_path ?? 'Unknown',
    model_arch: model.arch ?? null,
    quantization: model.quantize ? model.qtype || 'enabled' : 'disabled',
    precision: train.dtype ?? save.dtype ?? 'Unknown',
    dataset: dataset.folder_path ? path.basename(dataset.folder_path) : 'Unknown',
    train_val_split: 'n/a',
    total_samples: null,
    dataloader_workers: processConfig?.dataloader_workers ?? null,
    max_steps: train.steps ?? 0,
    save_steps: save.save_every ?? null,
    log_steps: processConfig?.performance_log_every ?? processConfig?.logging?.log_every ?? null,
    seed: sample.seed ?? null,
    optimizer: train.optimizer ?? 'Unknown',
    scheduler: train.noise_scheduler ?? 'Unknown',
    batch_size: train.batch_size ?? null,
    gradient_accumulation: train.gradient_accumulation ?? null,
    sequence_length: train.sequence_length ?? null,
  };
}

function getJobTotalSteps(job: Job) {
  const parsed = safeJsonParse<JobConfig | null>(job.job_config, null);
  return parsed?.config?.process?.[0]?.train?.steps ?? 0;
}

async function loadLossSeries(job: Job, jobFolder: string) {
  const logPath = path.join(jobFolder, 'loss_log.db');
  const first = await db.metrics.getLossLog(job.id, logPath, { key: 'loss', limit: 1, sinceStep: null, stride: 1 });
  const keys = first.keys.filter(key => /loss/i.test(key));
  const wantedKeys = keys.length ? keys : ['loss'];
  const results = await Promise.all(
    wantedKeys.map(key => db.metrics.getLossLog(job.id, logPath, { key, limit: 5000, sinceStep: null, stride: 1 })),
  );
  return results.map(result => ({ key: result.key, points: result.points }));
}

function deriveTiming(job: Job, lossPoints: LossPoint[], totalSteps: number) {
  const sorted = lossPoints.filter(point => point.wall_time).sort((a, b) => a.step - b.step);
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  const fallbackStart = new Date(job.created_at).getTime() / 1000;
  const fallbackEnd = new Date(job.updated_at).getTime() / 1000;
  const elapsed = first && last ? last.wall_time - first.wall_time : Math.max(0, fallbackEnd - fallbackStart);
  const completedSteps = first && last ? Math.max(1, last.step - first.step) : Math.max(1, job.step || 1);
  const secondsPerStep = elapsed > 0 ? elapsed / completedSteps : null;
  const remainingSteps = Math.max(0, totalSteps - job.step);
  return {
    runtime_seconds: elapsed,
    runtime: formatDuration(elapsed),
    eta_seconds: secondsPerStep == null ? null : remainingSteps * secondsPerStep,
    eta: formatDuration(secondsPerStep == null ? null : remainingSteps * secondsPerStep),
  };
}

export async function getJobDashboard(jobID: string) {
  const job = await db.jobs.findById(jobID);
  if (!job) return null;

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const totalSteps = getJobTotalSteps(job);
  const progress = totalSteps > 0 ? Math.min(100, (job.step / totalSteps) * 100) : 0;
  const checkpoints = listCheckpoints(jobFolder);
  const log = readLog(jobFolder);
  const lossSeries = await loadLossSeries(job, jobFolder);
  const primaryLoss = lossSeries.find(series => series.key === 'loss')?.points ?? lossSeries[0]?.points ?? [];
  const timing = deriveTiming(job, primaryLoss, totalSteps);

  return {
    job,
    progress,
    totalSteps,
    timing,
    log,
    logLines: splitLog(log),
    lossSeries,
    checkpoints,
    configSummary: summarizeConfig(job),
  };
}
