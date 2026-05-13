import fs from 'fs';
import path from 'path';
import type {
  AdvisorCategory,
  AdvisorDatasetStats,
  AdvisorFinding,
  AdvisorResult,
  AdvisorSeverity,
  AdvisorStage,
  DatasetConfig,
  Job,
  JobConfig,
  ProcessConfig,
  TrainingPhaseConfig,
} from '../types';

export type AdvisorMetricPoint = {
  step: number;
  wall_time?: number;
  value: number | null;
  value_text?: string | null;
};

export type AdvisorMetricSeries = {
  key: string;
  totalCount?: number;
  firstStep?: number | null;
  lastStep?: number | null;
  latest?: AdvisorMetricPoint | null;
  points: AdvisorMetricPoint[];
};

export type AdvisorMetricsInput = {
  keys?: string[];
  series: Record<string, AdvisorMetricSeries>;
};

export type AnalyzeTrainingAdvisorOptions = {
  gpuIds?: string | null;
  job?: Pick<Job, 'id' | 'step' | 'status' | 'speed_string'> | null;
  metrics?: AdvisorMetricsInput | null;
  scanDatasets?: boolean;
  scanFileLimit?: number;
};

type DatasetScanSummary = {
  index: number;
  path: string;
  captionExt: string;
  mediaFiles: number;
  captionFiles: number;
  missingCaptions: number;
  emptyCaptions: number;
  captionExtensionMismatches: number;
  truncated: boolean;
  inaccessible: boolean;
  placeholder: boolean;
};

const DEFAULT_SCAN_FILE_LIMIT = 300;
const MEDIA_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.bmp',
  '.gif',
  '.mp4',
  '.avi',
  '.mov',
  '.mkv',
  '.wmv',
  '.m4v',
  '.flv',
  '.mp3',
  '.wav',
  '.flac',
  '.ogg',
]);
const CAPTION_EXTENSIONS = ['.txt', '.caption', '.md'];
const PLACEHOLDER_PATHS = new Set(['/path/to/images/folder', 'path/to/images/folder']);

function addFinding(
  findings: AdvisorFinding[],
  severity: AdvisorSeverity,
  stage: AdvisorStage,
  category: AdvisorCategory,
  id: string,
  title: string,
  message: string,
  recommendation: string,
  evidence?: string[],
  relatedConfigPaths?: string[],
) {
  findings.push({
    id,
    severity,
    stage,
    category,
    title,
    message,
    recommendation,
    evidence,
    relatedConfigPaths,
  });
}

function getProcess(jobConfig: JobConfig): ProcessConfig | null {
  return jobConfig?.config?.process?.[0] ?? null;
}

function isPlaceholderPath(value: string | null | undefined) {
  if (!value) return true;
  const normalized = value.trim().replace(/\\/g, '/').replace(/\/+$/, '');
  return PLACEHOLDER_PATHS.has(normalized);
}

function safeStat(filePath: string) {
  try {
    return fs.statSync(filePath);
  } catch {
    return null;
  }
}

function normalizeCaptionExt(ext: unknown) {
  const raw = typeof ext === 'string' && ext.trim() ? ext.trim() : 'txt';
  return raw.startsWith('.') ? raw.toLowerCase() : `.${raw.toLowerCase()}`;
}

function walkMediaFiles(root: string, limit: number) {
  const mediaFiles: string[] = [];
  const stack = [root];
  let truncated = false;

  while (stack.length > 0) {
    const current = stack.pop() as string;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        if (entry.name !== '_controls') stack.push(fullPath);
        continue;
      }
      if (!entry.isFile()) continue;
      if (!MEDIA_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) continue;
      mediaFiles.push(fullPath);
      if (mediaFiles.length >= limit) {
        truncated = true;
        return { mediaFiles, truncated };
      }
    }
  }

  return { mediaFiles, truncated };
}

function scanDataset(dataset: DatasetConfig, index: number, limit: number): DatasetScanSummary {
  const folderPath = dataset.folder_path || '';
  const captionExt = normalizeCaptionExt(dataset.caption_ext);
  const summary: DatasetScanSummary = {
    index,
    path: folderPath,
    captionExt,
    mediaFiles: 0,
    captionFiles: 0,
    missingCaptions: 0,
    emptyCaptions: 0,
    captionExtensionMismatches: 0,
    truncated: false,
    inaccessible: false,
    placeholder: isPlaceholderPath(folderPath),
  };

  if (summary.placeholder) return summary;

  const stat = safeStat(folderPath);
  if (!stat?.isDirectory()) {
    summary.inaccessible = true;
    return summary;
  }

  const scan = walkMediaFiles(folderPath, limit);
  summary.truncated = scan.truncated;
  summary.mediaFiles = scan.mediaFiles.length;

  for (const mediaPath of scan.mediaFiles) {
    const parsed = path.parse(mediaPath);
    const expectedCaptionPath = path.join(parsed.dir, `${parsed.name}${captionExt}`);
    const captionStat = safeStat(expectedCaptionPath);

    if (captionStat?.isFile()) {
      summary.captionFiles += 1;
      if (captionStat.size === 0) summary.emptyCaptions += 1;
      continue;
    }

    summary.missingCaptions += 1;
    for (const altExt of CAPTION_EXTENSIONS) {
      if (altExt === captionExt) continue;
      if (safeStat(path.join(parsed.dir, `${parsed.name}${altExt}`))?.isFile()) {
        summary.captionExtensionMismatches += 1;
        break;
      }
    }
  }

  return summary;
}

function summarizeDatasets(scans: DatasetScanSummary[]): AdvisorDatasetStats {
  return scans.reduce<AdvisorDatasetStats>(
    (acc, scan) => {
      acc.datasetCount += 1;
      acc.scannedFiles += scan.mediaFiles;
      acc.mediaFiles += scan.mediaFiles;
      acc.captionFiles += scan.captionFiles;
      acc.missingCaptions += scan.missingCaptions;
      acc.emptyCaptions += scan.emptyCaptions;
      acc.captionExtensionMismatches += scan.captionExtensionMismatches;
      acc.inaccessibleDatasets += scan.inaccessible ? 1 : 0;
      acc.placeholderDatasets += scan.placeholder ? 1 : 0;
      acc.truncated = acc.truncated || scan.truncated;
      return acc;
    },
    {
      datasetCount: 0,
      scannedFiles: 0,
      mediaFiles: 0,
      captionFiles: 0,
      missingCaptions: 0,
      emptyCaptions: 0,
      captionExtensionMismatches: 0,
      inaccessibleDatasets: 0,
      placeholderDatasets: 0,
      truncated: false,
    },
  );
}

function analyzeDatasets(findings: AdvisorFinding[], processConfig: ProcessConfig, scanFileLimit: number) {
  const datasets = processConfig.datasets ?? [];
  const scans = datasets.map((dataset, index) => scanDataset(dataset, index, scanFileLimit));

  if (datasets.length === 0) {
    addFinding(
      findings,
      'critical',
      'preflight',
      'dataset',
      'dataset.none',
      'No dataset configured',
      'This job has no dataset entries, so training cannot learn from any media.',
      'Add at least one dataset folder before starting the job.',
      undefined,
      ['config.process[0].datasets'],
    );
  }

  for (const scan of scans) {
    const prefix = `config.process[0].datasets[${scan.index}]`;
    if (scan.placeholder) {
      addFinding(
        findings,
        'critical',
        'preflight',
        'dataset',
        `dataset.${scan.index}.placeholder`,
        'Dataset path is still a placeholder',
        `Dataset ${scan.index + 1} points to "${scan.path || '(empty)'}".`,
        'Choose a real dataset folder before starting the job.',
        [scan.path || '(empty)'],
        [`${prefix}.folder_path`],
      );
      continue;
    }
    if (scan.inaccessible) {
      addFinding(
        findings,
        'critical',
        'preflight',
        'dataset',
        `dataset.${scan.index}.inaccessible`,
        'Dataset folder is not accessible',
        `Dataset ${scan.index + 1} could not be read as a folder.`,
        'Verify the folder exists and that the UI process can read it.',
        [scan.path],
        [`${prefix}.folder_path`],
      );
      continue;
    }
    if (scan.mediaFiles === 0) {
      addFinding(
        findings,
        'critical',
        'preflight',
        'dataset',
        `dataset.${scan.index}.empty`,
        'Dataset has no trainable media',
        `No supported image, video, or audio files were found in dataset ${scan.index + 1}.`,
        'Add supported media files or select a different dataset folder.',
        [scan.path],
        [`${prefix}.folder_path`],
      );
      continue;
    }

    const missingCaptionRatio = scan.mediaFiles > 0 ? scan.missingCaptions / scan.mediaFiles : 0;
    if (missingCaptionRatio >= 0.8) {
      addFinding(
        findings,
        'warning',
        'preflight',
        'dataset',
        `dataset.${scan.index}.captions.mostly_missing`,
        'Most dataset captions are missing',
        `${scan.missingCaptions} of ${scan.mediaFiles} scanned media files do not have ${scan.captionExt} captions.`,
        'Caption the dataset or set a deliberate default caption before training.',
        [`${Math.round(missingCaptionRatio * 100)}% missing captions`],
        [`${prefix}.caption_ext`, `${prefix}.default_caption`],
      );
    } else if (missingCaptionRatio >= 0.2) {
      addFinding(
        findings,
        'info',
        'preflight',
        'dataset',
        `dataset.${scan.index}.captions.some_missing`,
        'Some dataset captions are missing',
        `${scan.missingCaptions} of ${scan.mediaFiles} scanned media files do not have ${scan.captionExt} captions.`,
        'Review missing captions if prompt fidelity matters for this training run.',
        undefined,
        [`${prefix}.caption_ext`, `${prefix}.default_caption`],
      );
    }

    if (scan.emptyCaptions > 0) {
      addFinding(
        findings,
        'warning',
        'preflight',
        'dataset',
        `dataset.${scan.index}.captions.empty`,
        'Empty caption files found',
        `${scan.emptyCaptions} scanned caption files are empty.`,
        'Fill or remove empty caption files so they do not behave like blank prompts by accident.',
        undefined,
        [`${prefix}.caption_ext`],
      );
    }

    if (scan.captionExtensionMismatches > 0) {
      addFinding(
        findings,
        'warning',
        'preflight',
        'dataset',
        `dataset.${scan.index}.caption_ext_mismatch`,
        'Caption extension may be wrong',
        `${scan.captionExtensionMismatches} files had captions with a different common extension than ${scan.captionExt}.`,
        'Set the dataset caption extension to match the caption files you intend to use.',
        undefined,
        [`${prefix}.caption_ext`],
      );
    }

    if (scan.truncated) {
      addFinding(
        findings,
        'info',
        'preflight',
        'dataset',
        `dataset.${scan.index}.scan_truncated`,
        'Dataset scan was capped',
        `Only the first ${scanFileLimit} media files were scanned to keep the advisor responsive.`,
        'Use this as a representative check; large datasets may still contain issues beyond the scan cap.',
      );
    }
  }

  return summarizeDatasets(scans);
}

function configuredControlPaths(dataset: DatasetConfig) {
  return [
    dataset.control_path,
    dataset.control_path_1,
    dataset.control_path_2,
    dataset.control_path_3,
  ].filter((value): value is string => typeof value === 'string' && value.trim().length > 0);
}

function analyzeControlPaths(findings: AdvisorFinding[], processConfig: ProcessConfig) {
  for (const [index, dataset] of (processConfig.datasets ?? []).entries()) {
    const controls = dataset.controls ?? [];
    const controlPaths = configuredControlPaths(dataset);
    const prefix = `config.process[0].datasets[${index}]`;

    if (controls.length > 0 && controlPaths.length === 0) {
      addFinding(
        findings,
        'warning',
        'preflight',
        'dataset',
        `dataset.${index}.controls.missing_path`,
        'Control dataset is missing control paths',
        `Dataset ${index + 1} has controls configured but no readable control folder path.`,
        'Set the matching control path for each configured control input.',
        controls,
        [`${prefix}.controls`, `${prefix}.control_path`],
      );
    }

    for (const [controlIndex, controlPath] of controlPaths.entries()) {
      if (!safeStat(controlPath)?.isDirectory()) {
        addFinding(
          findings,
          'warning',
          'preflight',
          'dataset',
          `dataset.${index}.controls.${controlIndex}.inaccessible`,
          'Control folder is not accessible',
          `Control path "${controlPath}" could not be read as a folder.`,
          'Verify the control folder exists and is available to the worker.',
          [controlPath],
          [`${prefix}.control_path`],
        );
      }
    }
  }
}

function getEffectiveBatch(processConfig: ProcessConfig) {
  const batch = Number(processConfig.train?.batch_size ?? 1);
  const accumulation = Number(processConfig.train?.gradient_accumulation ?? 1);
  return Math.max(1, batch) * Math.max(1, accumulation);
}

function analyzeConfig(findings: AdvisorFinding[], processConfig: ProcessConfig, gpuIds?: string | null) {
  const train = processConfig.train;
  const sample = processConfig.sample;
  const save = processConfig.save;
  const logging = processConfig.logging;
  const steps = Math.max(0, Number(train?.steps ?? 0));

  if (gpuIds != null && String(gpuIds).trim().length === 0) {
    addFinding(
      findings,
      'critical',
      'preflight',
      'config',
      'config.gpu_ids.empty',
      'No GPU assignment selected',
      'The job does not have a GPU assignment.',
      'Select a GPU or MPS target before starting.',
    );
  }

  const effectiveBatch = getEffectiveBatch(processConfig);
  if (effectiveBatch >= 8) {
    addFinding(
      findings,
      effectiveBatch >= 16 ? 'critical' : 'warning',
      'preflight',
      'performance',
      'train.effective_batch.high',
      'Effective batch size is high',
      `The effective batch size is ${effectiveBatch} (${train.batch_size} batch x ${train.gradient_accumulation} accumulation).`,
      'Use a smaller effective batch unless you intentionally tuned LR and VRAM for it.',
      undefined,
      ['config.process[0].train.batch_size', 'config.process[0].train.gradient_accumulation'],
    );
  }

  const lr = Number(train?.lr ?? 0);
  const arch = String(processConfig.model?.arch ?? '');
  const sensitiveArch = /hidream|qwen|zimage|flux2|wan|ltx/i.test(arch);
  const warnLr = sensitiveArch ? 1e-4 : 3e-4;
  const criticalLr = sensitiveArch ? 3e-4 : 1e-3;
  if (lr > criticalLr) {
    addFinding(
      findings,
      'critical',
      'preflight',
      'config',
      'train.lr.critical',
      'Learning rate looks too aggressive',
      `The configured LR is ${lr}, which is high for ${arch || 'this model'}.`,
      'Lower LR before starting unless this is a deliberate short stress test.',
      undefined,
      ['config.process[0].train.lr'],
    );
  } else if (lr > warnLr) {
    addFinding(
      findings,
      'warning',
      'preflight',
      'config',
      'train.lr.warning',
      'Learning rate may be aggressive',
      `The configured LR is ${lr}, which can destabilize LoRA quality for ${arch || 'this model'}.`,
      'Consider lowering LR or using training phases with a short warmup/finetune stage.',
      undefined,
      ['config.process[0].train.lr'],
    );
  }

  if (train?.disable_sampling) {
    addFinding(
      findings,
      'warning',
      'preflight',
      'sampling',
      'sample.disabled',
      'Sampling is disabled',
      'The run will not produce visual/audio checkpoints for quality inspection.',
      'Enable sampling unless this is a smoke test or resumed cleanup run.',
      undefined,
      ['config.process[0].train.disable_sampling'],
    );
  } else if (!sample?.samples?.length) {
    addFinding(
      findings,
      'warning',
      'preflight',
      'sampling',
      'sample.none',
      'No sample prompts configured',
      'The run has no sample prompts, making quality drift harder to catch.',
      'Add at least one stable validation prompt before starting.',
      undefined,
      ['config.process[0].sample.samples'],
    );
  }

  const sampleEvery = Number(sample?.sample_every ?? 0);
  if (!train?.disable_sampling && steps > 0 && sampleEvery > 0 && sampleEvery > Math.max(500, steps / 3)) {
    addFinding(
      findings,
      'info',
      'preflight',
      'sampling',
      'sample.cadence.sparse',
      'Sample cadence is sparse',
      `Samples are scheduled every ${sampleEvery} steps across a ${steps} step run.`,
      'Sample more often early in a new recipe so quality regressions are visible sooner.',
      undefined,
      ['config.process[0].sample.sample_every'],
    );
  }

  const saveEvery = Number(save?.save_every ?? 0);
  if (steps > 0 && saveEvery > steps) {
    addFinding(
      findings,
      'warning',
      'preflight',
      'sampling',
      'save.after_run',
      'Checkpoint cadence exceeds total steps',
      `Checkpoints are scheduled every ${saveEvery} steps, but the run has ${steps} steps.`,
      'Lower save_every or rely on the final save only if that is intentional.',
      undefined,
      ['config.process[0].save.save_every'],
    );
  }

  const monitorEvery = logging?.monitor_every ?? logging?.log_every;
  if (monitorEvery == null || Number(monitorEvery) > 50) {
    addFinding(
      findings,
      'info',
      'preflight',
      'metrics',
      'logging.monitor.sparse',
      'Training diagnostics are sparse',
      `Monitor metrics are recorded every ${monitorEvery ?? 'unknown'} steps.`,
      'Use a lower monitor interval while tuning a new model or dataset.',
      undefined,
      ['config.process[0].logging.monitor_every', 'config.process[0].logging.log_every'],
    );
  }
}

function analyzePhases(findings: AdvisorFinding[], processConfig: ProcessConfig) {
  const train = processConfig.train;
  const phases = train?.phases ?? [];
  if (!phases.length) return;

  const trainSteps = Math.max(0, Number(train.steps ?? 0));
  const phaseSteps = phases.reduce((sum, phase) => sum + Math.max(1, Number(phase.steps) || 1), 0);
  if (trainSteps !== phaseSteps) {
    addFinding(
      findings,
      'critical',
      'preflight',
      'phases',
      'phases.steps.mismatch',
      'Phase steps do not match total steps',
      `Total train steps are ${trainSteps}, but phases add up to ${phaseSteps}.`,
      'Synchronize train.steps with the sum of all phase steps before starting.',
      undefined,
      ['config.process[0].train.steps', 'config.process[0].train.phases'],
    );
  }

  for (const [index, phase] of phases.entries()) {
    const autoAdvance = phase.auto_advance;
    if (!autoAdvance) continue;
    const phaseStepCount = Math.max(1, Number(phase.steps) || 1);
    const minSteps = Number(autoAdvance.min_steps ?? Math.max(200, Number(autoAdvance.window ?? 100) * 2));
    if (minSteps >= phaseStepCount) {
      addFinding(
        findings,
        'info',
        'preflight',
        'phases',
        `phases.${index}.auto_advance.min_steps`,
        'Auto-advance may never trigger',
        `Phase ${index + 1} has ${phaseStepCount} steps, but auto-advance waits at least ${minSteps} steps.`,
        'Lower auto-advance min steps or extend the phase if plateau detection should matter.',
        undefined,
        [`config.process[0].train.phases[${index}].auto_advance.min_steps`],
      );
    }
  }
}

function numericPoints(metrics: AdvisorMetricsInput | null | undefined, key: string) {
  const payload = metrics?.series?.[key];
  return (payload?.points ?? []).filter(point => point.value != null && Number.isFinite(point.value)) as Array<
    AdvisorMetricPoint & { value: number }
  >;
}

function latestNumeric(metrics: AdvisorMetricsInput | null | undefined, key: string) {
  const payload = metrics?.series?.[key];
  if (payload?.latest?.value != null && Number.isFinite(payload.latest.value)) return payload.latest.value;
  const points = numericPoints(metrics, key);
  return points.length ? points[points.length - 1].value : null;
}

function average(values: number[]) {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function getPrimaryLossPoints(metrics: AdvisorMetricsInput | null | undefined) {
  const candidates = ['loss/loss', 'loss', 'loss/mse', 'loss/total'];
  for (const key of candidates) {
    const points = numericPoints(metrics, key);
    if (points.length >= 10) return { key, points };
  }
  const dynamicKey = Object.keys(metrics?.series ?? {}).find(key => key.startsWith('loss/'));
  if (!dynamicKey) return { key: 'loss/loss', points: [] };
  return { key: dynamicKey, points: numericPoints(metrics, dynamicKey) };
}

function analyzeLossDynamics(findings: AdvisorFinding[], processConfig: ProcessConfig, metrics?: AdvisorMetricsInput | null) {
  const { key, points } = getPrimaryLossPoints(metrics);
  if (points.length < 40) return;

  const recent = points.slice(-120).map(point => point.value);
  const split = Math.floor(recent.length / 2);
  const earlierAvg = average(recent.slice(0, split));
  const laterAvg = average(recent.slice(split));
  if (earlierAvg == null || laterAvg == null || earlierAvg <= 0) return;

  const improvementPct = ((earlierAvg - laterAvg) / Math.abs(earlierAvg)) * 100;
  if (improvementPct < 1 && points.length >= 100) {
    const hasAutoAdvance = (processConfig.train.phases ?? []).some((phase: TrainingPhaseConfig) => !!phase.auto_advance);
    addFinding(
      findings,
      'info',
      'live',
      hasAutoAdvance ? 'stability' : 'phases',
      hasAutoAdvance ? 'live.loss.plateau' : 'live.loss.plateau.no_auto_advance',
      hasAutoAdvance ? 'Loss looks plateaued' : 'Loss plateau could drive a phase change',
      `${key} improved by ${improvementPct.toFixed(1)}% across the recent window.`,
      hasAutoAdvance
        ? 'Inspect recent samples before spending many more steps in this phase.'
        : 'Consider adding phase auto-advance when using staged training recipes.',
      [`${key}: ${earlierAvg.toFixed(4)} -> ${laterAvg.toFixed(4)}`],
    );
  }

  const recentMedian = median(recent);
  const latest = recent[recent.length - 1];
  if (recentMedian != null && latest > recentMedian * 2.5) {
    addFinding(
      findings,
      'warning',
      'live',
      'stability',
      'live.loss.spike',
      'Recent loss spike detected',
      `Latest ${key} is ${latest.toFixed(4)}, above the recent median ${recentMedian.toFixed(4)}.`,
      'Check recent samples and consider lowering LR if spikes persist.',
      undefined,
      ['config.process[0].train.lr'],
    );
  }
}

function analyzeLiveMetrics(
  findings: AdvisorFinding[],
  processConfig: ProcessConfig,
  options: AnalyzeTrainingAdvisorOptions,
) {
  const metrics = options.metrics;
  if (!metrics) return;

  analyzeLossDynamics(findings, processConfig, metrics);

  const oomPoints = numericPoints(metrics, 'train/oom_skipped');
  const oomCount = oomPoints.filter(point => point.value > 0).length;
  if (oomCount > 0) {
    addFinding(
      findings,
      'critical',
      'live',
      'performance',
      'live.oom_skips',
      'OOM skips recorded',
      `${oomCount} logged steps reported an OOM skip.`,
      'Reduce batch size, resolution, frame count, or enable stronger memory-saving options before continuing.',
      undefined,
      ['config.process[0].train.batch_size', 'config.process[0].datasets[0].resolution'],
    );
  }

  const memoryPct = latestNumeric(metrics, 'train/gpu_mem_used_pct');
  if (memoryPct != null && memoryPct >= 90) {
    addFinding(
      findings,
      memoryPct >= 95 ? 'critical' : 'warning',
      'live',
      'performance',
      'live.gpu_memory.high',
      'GPU memory pressure is high',
      `Latest GPU memory use is ${memoryPct.toFixed(1)}%.`,
      'Expect OOM risk during sampling/checkpointing; reduce batch/resolution or enable offloading if this persists.',
      undefined,
      ['config.process[0].train.batch_size', 'config.process[0].model.layer_offloading'],
    );
  }

  const freeGb = latestNumeric(metrics, 'train/gpu_mem_free_gb');
  if (freeGb != null && freeGb < 1) {
    addFinding(
      findings,
      'warning',
      'live',
      'performance',
      'live.gpu_memory.low_free',
      'GPU free memory is very low',
      `Latest free GPU memory is ${freeGb.toFixed(2)} GB.`,
      'Leave more VRAM headroom if samples or checkpoints fail around scheduled events.',
      undefined,
      ['config.process[0].train.batch_size'],
    );
  }

  const stepsPerSec = numericPoints(metrics, 'train/steps_per_sec');
  if (stepsPerSec.length >= 40) {
    const recent = average(stepsPerSec.slice(-20).map(point => point.value));
    const previous = average(stepsPerSec.slice(-40, -20).map(point => point.value));
    if (recent != null && previous != null && previous > 0 && recent < previous * 0.5) {
      addFinding(
        findings,
        'warning',
        'live',
        'performance',
        'live.throughput.drop',
        'Training throughput dropped',
        `Recent throughput is ${recent.toFixed(3)} steps/sec versus ${previous.toFixed(3)} earlier.`,
        'Check whether sampling, checkpointing, disk cache, or memory pressure is slowing the run.',
      );
    }
  }

  const gradNorm = latestNumeric(metrics, 'train/grad_norm');
  const gradLimit = latestNumeric(metrics, 'train/grad_norm_limit');
  if (gradNorm != null && gradLimit != null && gradLimit > 0 && gradNorm > gradLimit * 2) {
    addFinding(
      findings,
      'warning',
      'live',
      'stability',
      'live.grad_norm.limit',
      'Gradient norm is above clipping limit',
      `Latest grad norm is ${gradNorm.toFixed(3)} with a configured limit of ${gradLimit.toFixed(3)}.`,
      'If this repeats with poor samples, lower LR or increase warmup/staging.',
      undefined,
      ['config.process[0].train.lr', 'config.process[0].train.max_grad_norm'],
    );
  } else if (gradNorm != null && gradNorm > 100) {
    addFinding(
      findings,
      'warning',
      'live',
      'stability',
      'live.grad_norm.high',
      'Gradient norm is unusually high',
      `Latest grad norm is ${gradNorm.toFixed(3)}.`,
      'Watch for loss spikes and consider lowering LR if samples degrade.',
      undefined,
      ['config.process[0].train.lr'],
    );
  }

  const currentStep = Number(options.job?.step ?? 0);
  const sampleEvery = Number(processConfig.sample?.sample_every ?? 0);
  const saveEvery = Number(processConfig.save?.save_every ?? 0);
  const sampleEvents = metrics.series['event/sample']?.points ?? [];
  const checkpointEvents = metrics.series['event/checkpoint']?.points ?? [];

  if (!processConfig.train?.disable_sampling && sampleEvery > 0 && currentStep > sampleEvery && sampleEvents.length === 0) {
    addFinding(
      findings,
      'warning',
      'live',
      'sampling',
      'live.sample.events_missing',
      'No sample events have been logged',
      `The job is at step ${currentStep}, but no sample event is present.`,
      'Confirm sample prompts are valid and sampling is not failing silently.',
      undefined,
      ['config.process[0].sample.samples', 'config.process[0].sample.sample_every'],
    );
  }

  if (saveEvery > 0 && currentStep > saveEvery && checkpointEvents.length === 0) {
    addFinding(
      findings,
      'warning',
      'live',
      'sampling',
      'live.checkpoint.events_missing',
      'No checkpoint events have been logged',
      `The job is at step ${currentStep}, but no checkpoint event is present.`,
      'Confirm save cadence and output folder permissions before relying on this run.',
      undefined,
      ['config.process[0].save.save_every'],
    );
  }
}

function rankFinding(finding: AdvisorFinding) {
  const severityRank: Record<AdvisorSeverity, number> = { critical: 0, warning: 1, info: 2 };
  const stageRank: Record<AdvisorStage, number> = { live: 0, preflight: 1 };
  return severityRank[finding.severity] * 10 + stageRank[finding.stage];
}

function buildResult(findings: AdvisorFinding[], datasetStats?: AdvisorDatasetStats): AdvisorResult {
  const sorted = [...findings].sort((a, b) => rankFinding(a) - rankFinding(b) || a.title.localeCompare(b.title));
  const critical = sorted.filter(finding => finding.severity === 'critical').length;
  const warning = sorted.filter(finding => finding.severity === 'warning').length;
  const info = sorted.filter(finding => finding.severity === 'info').length;
  const text =
    critical > 0
      ? `${critical} critical training risk${critical === 1 ? '' : 's'} found`
      : warning > 0
        ? `${warning} warning${warning === 1 ? '' : 's'} found`
        : info > 0
          ? `${info} note${info === 1 ? '' : 's'} found`
          : 'No training quality issues found';

  return {
    summary: { critical, warning, info, text },
    findings: sorted,
    scannedAt: new Date().toISOString(),
    datasetStats,
  };
}

export function analyzeTrainingAdvisor(
  jobConfig: JobConfig,
  options: AnalyzeTrainingAdvisorOptions = {},
): AdvisorResult {
  const findings: AdvisorFinding[] = [];
  const processConfig = getProcess(jobConfig);
  if (!processConfig) {
    addFinding(
      findings,
      'critical',
      'preflight',
      'config',
      'config.process.missing',
      'Training process is missing',
      'The job config does not contain config.process[0].',
      'Use a valid training config before starting.',
      undefined,
      ['config.process[0]'],
    );
    return buildResult(findings);
  }

  const scanFileLimit = Math.max(1, Number(options.scanFileLimit ?? DEFAULT_SCAN_FILE_LIMIT));
  const datasetStats = options.scanDatasets === false ? undefined : analyzeDatasets(findings, processConfig, scanFileLimit);
  analyzeControlPaths(findings, processConfig);
  analyzeConfig(findings, processConfig, options.gpuIds);
  analyzePhases(findings, processConfig);
  analyzeLiveMetrics(findings, processConfig, options);

  return buildResult(findings, datasetStats);
}
