import type { AdvisorResult, DatasetConfig, JobConfig } from '../types';
import { isRecord } from './commandInput';
import { analyzeTrainingAdvisor, scanAdvisorDataset, type DatasetScanSummary } from './trainingAdvisor';
import { fetchWorkerHealth, getRemoteWorker, isLocalWorker, remoteJson } from './remoteClient';
import { parseRemoteDatasetRef } from '../utils/remoteDatasetRefs';
import path from 'path';
import { resolveDatasetScope } from './datasetScope';
import { resolveDatasetDirectoryInsideRoot } from './remoteCaptionSecurity';

export async function readWorkerDatasetCheck(value: unknown, resolveRoot = async () => (await resolveDatasetScope()).datasetsRoot): Promise<DatasetScanSummary> {
  if (!isRecord(value) || typeof value.dataset_name !== 'string' || !value.dataset_name.trim() || value.dataset_name.length > 255 ||
      value.dataset_name === '.' || value.dataset_name === '..' || /[\\/:\0]/.test(value.dataset_name) ||
      (value.caption_ext !== undefined && (typeof value.caption_ext !== 'string' || !/^\.?[a-zA-Z0-9_-]{1,20}$/.test(value.caption_ext))))
    throw new Error('Invalid dataset check');
  const root = await resolveRoot();
  const folderPath = await resolveDatasetDirectoryInsideRoot(path.join(root, value.dataset_name), root);
  const scan = scanAdvisorDataset({ folder_path: folderPath, caption_ext: typeof value.caption_ext === 'string' ? value.caption_ext : 'txt' }, 0);
  return { ...scan, path: value.dataset_name };
}

export function isAdvisorJobConfig(value: unknown): value is JobConfig {
  if (!isRecord(value) || !isRecord(value.config) || !Array.isArray(value.config.process) || !value.config.process.length || value.config.process.length > 32) return false;
  const fields = (object: Record<string, unknown>, keys: string[], kind: 'string' | 'number' | 'boolean') =>
    keys.every(key => object[key] == null || (typeof object[key] === kind && (kind !== 'number' || Number.isFinite(object[key]))));
  return value.config.process.every(process => {
    if (!isRecord(process) || !['model', 'train', 'save', 'logging', 'sample', 'network'].every(key => process[key] === undefined || isRecord(process[key]))) return false;
    const train = isRecord(process.train) ? process.train : {}, model = isRecord(process.model) ? process.model : {};
    const sample = isRecord(process.sample) ? process.sample : {}, logging = isRecord(process.logging) ? process.logging : {};
    const save = isRecord(process.save) ? process.save : {}, network = isRecord(process.network) ? process.network : {};
    if (!fields(model, ['arch', 'name_or_path'], 'string') || !fields(model, ['use_flux_cfg'], 'boolean') ||
        (model.model_kwargs !== undefined && !isRecord(model.model_kwargs)) || !fields(network, ['type'], 'string') ||
        !fields(train, ['optimizer'], 'string') || !fields(train, ['steps', 'batch_size', 'gradient_accumulation', 'lr'], 'number') ||
        !fields(train, ['train_text_encoder', 'unload_text_encoder', 'cache_text_embeddings', 'bypass_guidance_embedding', 'sega_distill',
          'diff_output_preservation', 'blank_prompt_preservation', 'do_prior_divergence', 'inverted_mask_prior', 'do_differential_guidance', 'do_guidance_loss', 'disable_sampling'], 'boolean') ||
        !fields(sample, ['sample_every'], 'number') || !fields(save, ['save_every'], 'number') || !fields(logging, ['monitor_every', 'log_every'], 'number') ||
        (sample.samples != null && (!Array.isArray(sample.samples) || sample.samples.length > 1000 || sample.samples.some(item => !isRecord(item) || !fields(item, ['prompt'], 'string'))))) return false;
    return process.datasets === undefined || (Array.isArray(process.datasets) && process.datasets.length <= 100 && process.datasets.every(dataset =>
      isRecord(dataset) && fields(dataset, ['folder_path', 'caption_ext', 'control_path', 'control_path_1', 'control_path_2', 'control_path_3'], 'string') &&
      fields(dataset, ['cache_text_embeddings', 'encrypted'], 'boolean') &&
      (dataset.controls == null || (Array.isArray(dataset.controls) && dataset.controls.every(item => typeof item === 'string')))));
  });
}

function parseDatasetScan(value: unknown, dataset: DatasetConfig, index: number): DatasetScanSummary {
  if (!isRecord(value)) throw new Error('Worker returned an invalid dataset check');
  const counts = ['mediaFiles', 'captionFiles', 'missingCaptions', 'emptyCaptions', 'captionExtensionMismatches'] as const;
  if (counts.some(key => typeof value[key] !== 'number' || !Number.isSafeInteger(value[key]) || Number(value[key]) < 0) ||
      ['truncated', 'inaccessible', 'placeholder'].some(key => typeof value[key] !== 'boolean') ||
      typeof value.captionExt !== 'string' || !isRecord(value.alternateCaptionExtensionCounts) ||
      Object.values(value.alternateCaptionExtensionCounts).some(count => typeof count !== 'number' || !Number.isSafeInteger(count) || count < 0) ||
      (value.unknownReason !== undefined && typeof value.unknownReason !== 'string')) throw new Error('Worker returned an invalid dataset check');
  const summary = scanAdvisorDataset(dataset, index, 1, 'Awaiting worker check');
  for (const key of counts) summary[key] = Number(value[key]);
  summary.captionExt = value.captionExt;
  summary.truncated = value.truncated === true;
  summary.inaccessible = value.inaccessible === true;
  summary.placeholder = value.placeholder === true;
  summary.unknownReason = typeof value.unknownReason === 'string' ? value.unknownReason : undefined;
  summary.alternateCaptionExtensionCounts = Object.fromEntries(Object.entries(value.alternateCaptionExtensionCounts).map(([key, count]) => [key, Number(count)]));
  return summary;
}

/** Paths stay with their owning machine; the target receives dataset names only. */
export async function runTrainingPreflight(jobConfig: JobConfig, gpuIds: string | null, workerId = 'local'): Promise<AdvisorResult> {
  const datasetChecks: Record<number, DatasetScanSummary> = {};
  let target: NonNullable<AdvisorResult['target']> = { workerId, state: 'checked', detail: 'Dataset paths checked on the local worker.' };
  if (!isLocalWorker(workerId)) {
    target = { workerId, state: 'unknown', detail: 'Target execution readiness has not been verified.' };
    try {
      const worker = await getRemoteWorker(workerId);
      const health = await fetchWorkerHealth(worker);
      target = { workerId, state: health.ok && health.executionReady === true ? 'checked' : 'unknown', detail: health.executionDetail };
      for (const [index, dataset] of (jobConfig.config.process[0]?.datasets || []).entries()) {
        const remote = parseRemoteDatasetRef(dataset.folder_path);
        if (!remote) continue; // Local source media will be transferred at launch.
        if (remote.workerID !== workerId) {
          datasetChecks[index] = scanAdvisorDataset(dataset, index, 1, 'Import this dataset from its owning worker before checking the selected training target.');
          continue;
        }
        try {
          const scan: unknown = await remoteJson(worker, '/api/training-advisor/dataset', { method: 'POST',
            body: JSON.stringify({ dataset_name: remote.datasetName, caption_ext: dataset.caption_ext }), timeoutMs: 5000 });
          datasetChecks[index] = parseDatasetScan(scan, dataset, index);
        } catch {
          datasetChecks[index] = scanAdvisorDataset(dataset, index, 1, 'The target worker could not check this dataset. It may be offline or require an app update.');
        }
      }
    } catch {
      target.detail = 'The target worker is unavailable. Remote dataset and execution checks remain incomplete.';
    }
  }
  const result = analyzeTrainingAdvisor(jobConfig, { gpuIds, datasetChecks, skipControlPaths: !isLocalWorker(workerId) });
  if (!isLocalWorker(workerId) && (jobConfig.config.process[0]?.datasets || []).some(dataset => !parseRemoteDatasetRef(dataset.folder_path))) {
    target = { workerId, state: 'unknown', detail: `${target.detail} Local source datasets were checked here; transferred files and control paths still need verification on the target.` };
  }
  if (target.state === 'unknown') {
    result.findings.push({ id: 'target.unchecked', severity: 'info', stage: 'preflight', category: 'config',
      title: 'Target checks incomplete', message: target.detail, recommendation: 'Check the selected worker and verify transferred datasets before training.' });
    result.summary.info++;
    if (!result.summary.critical && !result.summary.warning) result.summary.text = 'Target checks are incomplete';
  }
  return { ...result, target };
}
