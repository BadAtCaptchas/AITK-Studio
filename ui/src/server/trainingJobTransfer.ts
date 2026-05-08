import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';

export const TRAINING_JOB_EXPORT_FORMAT = 'ai-toolkit-training-job-export';
export const TRAINING_JOB_EXPORT_VERSION = 1;

export type TrainingJobCheckpointExportMode = 'latest' | 'all';

export type DatasetArchiveMapping = {
  archivePath: string;
  originalPath: string;
  targetConfigPaths: string[];
  isDirectory: boolean;
};

export type ModelReference = {
  configPath: string;
  value: string;
  isLocal: boolean;
  existsOnSource: boolean;
};

export type TrainingJobExportManifest = {
  format: typeof TRAINING_JOB_EXPORT_FORMAT;
  version: typeof TRAINING_JOB_EXPORT_VERSION;
  exportedAt: string;
  source: {
    app: 'ai-toolkit';
    jobId: string;
    jobName: string;
  };
  options: {
    includeDatasets: boolean;
    includeBaseModels: false;
    checkpointMode: TrainingJobCheckpointExportMode;
  };
  training: {
    archivePath: 'training';
    dbStep: number;
    latestCheckpointPath: string | null;
    latestCheckpointStep: number | null;
    optimizerIncluded: boolean;
    status: string;
  };
  datasets: {
    included: boolean;
    mappings: DatasetArchiveMapping[];
  };
  models: {
    references: ModelReference[];
  };
  warnings: string[];
};

export type DatasetReference = {
  configPath: string;
  value: string;
};

const DATASET_PATH_FIELDS = [
  'folder_path',
  'dataset_path',
  'control_path',
  'control_path_1',
  'control_path_2',
  'control_path_3',
  'mask_path',
  'unconditional_path',
  'inpaint_path',
  'clip_image_path',
];

const MODEL_REFERENCE_PATHS = [
  ['model', 'name_or_path'],
  ['model', 'refiner_name_or_path'],
  ['model', 'vae_path'],
  ['model', 'lora_path'],
  ['model', 'assistant_lora_path'],
  ['model', 'inference_lora_path'],
  ['model', 'unet_path'],
  ['model', 'te_name_or_path'],
  ['model', 'extras_name_or_path'],
  ['model', 'accuracy_recovery_adapter'],
  ['network', 'pretrained_lora_path'],
  ['network', 'network_weights'],
  ['adapter', 'name_or_path'],
  ['adapter', 'image_encoder_path'],
  ['adapter', 'text_encoder_path'],
  ['train', 'adapter_assist_name_or_path'],
];

export function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value));
}

export function safeNameSegment(value: string, fallback = 'item') {
  const cleaned = value
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_')
    .replace(/\s+/g, '_')
    .replace(/^\.+/, '')
    .slice(0, 120);
  return cleaned || fallback;
}

export function makeExportFileName(jobName: string) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  return `${safeNameSegment(jobName, 'job')}_${timestamp}.aitk.zip`;
}

export function resolveConfigPath(value: string) {
  return path.isAbsolute(value) ? path.resolve(value) : path.resolve(TOOLKIT_ROOT, value);
}

export function isRemoteReference(value: string) {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(value);
}

export function looksLikeLocalPath(value: string, existsLocally = false) {
  if (!value || isRemoteReference(value)) return false;
  return (
    existsLocally ||
    path.isAbsolute(value) ||
    value.startsWith('./') ||
    value.startsWith('../') ||
    value.startsWith('~') ||
    value.includes('\\') ||
    /\.(safetensors|ckpt|pt|pth|bin)$/i.test(value)
  );
}

export function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

export function collectDatasetReferences(jobConfig: any): DatasetReference[] {
  const refs: DatasetReference[] = [];
  const processes = Array.isArray(jobConfig?.config?.process) ? jobConfig.config.process : [];

  processes.forEach((processConfig: any, processIndex: number) => {
    const datasets = Array.isArray(processConfig?.datasets) ? processConfig.datasets : [];
    datasets.forEach((dataset: any, datasetIndex: number) => {
      DATASET_PATH_FIELDS.forEach(field => {
        const value = dataset?.[field];
        const basePath = `config.process[${processIndex}].datasets[${datasetIndex}].${field}`;
        if (typeof value === 'string' && value.trim()) {
          refs.push({ configPath: basePath, value });
        } else if (Array.isArray(value)) {
          value.forEach((item, itemIndex) => {
            if (typeof item === 'string' && item.trim()) {
              refs.push({ configPath: `${basePath}[${itemIndex}]`, value: item });
            }
          });
        }
      });
    });
  });

  return refs;
}

export async function collectDatasetArchiveMappings(jobConfig: any, includeDatasets: boolean, datasetsRoot: string) {
  const warnings: string[] = [];
  const mappings: DatasetArchiveMapping[] = [];
  if (!includeDatasets) {
    return { mappings, warnings };
  }

  const grouped = new Map<string, { absolutePath: string; refs: DatasetReference[]; isDirectory: boolean }>();
  for (const ref of collectDatasetReferences(jobConfig)) {
    if (isRemoteReference(ref.value)) continue;

    const absolutePath = resolveConfigPath(ref.value);
    let stat: fs.Stats;
    try {
      stat = await fsp.stat(absolutePath);
    } catch {
      warnings.push(`Dataset path not found and was not included: ${ref.value}`);
      continue;
    }

    const realPath = await fsp.realpath(absolutePath);
    if (!isPathInside(datasetsRoot, realPath)) {
      warnings.push(`Dataset path is outside the datasets folder and was not included: ${ref.value}`);
      continue;
    }

    const existing = grouped.get(realPath);
    if (existing) {
      existing.refs.push(ref);
    } else {
      grouped.set(realPath, {
        absolutePath: realPath,
        refs: [ref],
        isDirectory: stat.isDirectory(),
      });
    }
  }

  const usedArchiveNames = new Set<string>();
  for (const group of grouped.values()) {
    const basename = safeNameSegment(path.basename(group.absolutePath), 'dataset');
    let archiveName = basename;
    let suffix = 2;
    while (usedArchiveNames.has(archiveName)) {
      archiveName = `${basename}_${suffix}`;
      suffix += 1;
    }
    usedArchiveNames.add(archiveName);
    mappings.push({
      archivePath: path.posix.join('datasets', archiveName),
      originalPath: group.absolutePath,
      targetConfigPaths: group.refs.map(ref => ref.configPath),
      isDirectory: group.isDirectory,
    });
  }

  return { mappings, warnings };
}

export function collectModelReferences(jobConfig: any): ModelReference[] {
  const references: ModelReference[] = [];
  const processes = Array.isArray(jobConfig?.config?.process) ? jobConfig.config.process : [];

  processes.forEach((processConfig: any, processIndex: number) => {
    MODEL_REFERENCE_PATHS.forEach(([section, field]) => {
      const value = processConfig?.[section]?.[field];
      if (typeof value !== 'string' || !value.trim()) return;

      const resolved = resolveConfigPath(value);
      const existsOnSource = fs.existsSync(resolved);
      references.push({
        configPath: `config.process[${processIndex}].${section}.${field}`,
        value,
        isLocal: looksLikeLocalPath(value, existsOnSource),
        existsOnSource,
      });
    });
  });

  return references;
}

export async function listFilesRecursive(root: string, shouldInclude: (absolutePath: string, relativePath: string) => boolean) {
  const files: string[] = [];

  async function walk(current: string) {
    const entries = await fsp.readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const absolutePath = path.join(current, entry.name);
      const relativePath = path.relative(root, absolutePath);
      if (!shouldInclude(absolutePath, relativePath)) continue;
      if (entry.isDirectory()) {
        await walk(absolutePath);
      } else if (entry.isFile()) {
        files.push(relativePath);
      }
    }
  }

  if (fs.existsSync(root)) {
    await walk(root);
  }

  return files;
}

export function shouldIncludeTrainingExportPath(_absolutePath: string, relativePath: string) {
  const normalized = relativePath.replace(/\\/g, '/');
  const basename = path.basename(relativePath);
  if (!normalized) return true;
  if (basename.endsWith('.aitk.zip')) return false;
  if (basename.endsWith('.tmp')) return false;
  if (basename === 'samples.zip') return false;
  if (basename.startsWith('.aitk-export-')) return false;
  return true;
}

export function isCheckpointExportPath(relativePath: string) {
  const normalized = relativePath.replace(/\\/g, '/');
  return !normalized.includes('/') && normalized.toLowerCase().endsWith('.safetensors');
}

export async function findLatestCheckpoint(trainingFolder: string, dbStep: number) {
  if (!fs.existsSync(trainingFolder)) {
    return { relativePath: null as string | null, step: null as number | null };
  }

  const entries = await fsp.readdir(trainingFolder, { withFileTypes: true });
  const candidates: { relativePath: string; mtimeMs: number; step: number | null }[] = [];

  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.safetensors')) continue;
    const absolutePath = path.join(trainingFolder, entry.name);
    const stat = await fsp.stat(absolutePath);
    const stepMatch = entry.name.match(/_(\d{1,9})(?:\D|$)/);
    candidates.push({
      relativePath: entry.name,
      mtimeMs: stat.mtimeMs,
      step: stepMatch ? Number(stepMatch[1]) : dbStep || null,
    });
  }

  if (candidates.length === 0) {
    return { relativePath: null, step: null };
  }

  candidates.sort((a, b) => a.mtimeMs - b.mtimeMs);
  const latest = candidates[candidates.length - 1];
  return { relativePath: latest.relativePath, step: latest.step };
}

function parseConfigPath(configPath: string) {
  const parts: Array<string | number> = [];
  const re = /([^[.\]]+)|\[(\d+)\]/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(configPath)) !== null) {
    if (match[1] !== undefined) parts.push(match[1]);
    if (match[2] !== undefined) parts.push(Number(match[2]));
  }
  return parts;
}

export function setConfigPathValue(target: any, configPath: string, value: unknown) {
  const parts = parseConfigPath(configPath);
  if (parts.length === 0) return;

  let cursor = target;
  for (let i = 0; i < parts.length - 1; i++) {
    if (cursor == null) return;
    cursor = cursor[parts[i] as any];
  }

  if (cursor != null) {
    cursor[parts[parts.length - 1] as any] = value;
  }
}

export function rewriteJobConfigForTarget(
  rawJobConfig: any,
  options: {
    jobName: string;
    trainingFolder: string;
    sqliteDbPath: string;
    datasetPathByConfigPath?: Map<string, string>;
  },
) {
  const jobConfig = cloneJson(rawJobConfig);
  if (!jobConfig.config) jobConfig.config = {};
  jobConfig.config.name = options.jobName;

  const processes = Array.isArray(jobConfig.config.process) ? jobConfig.config.process : [];
  processes.forEach((processConfig: any) => {
    processConfig.training_folder = options.trainingFolder;
    processConfig.sqlite_db_path = options.sqliteDbPath;
  });

  options.datasetPathByConfigPath?.forEach((targetPath, configPath) => {
    setConfigPathValue(jobConfig, configPath, targetPath);
  });

  return jobConfig;
}

export async function renameImportedTrainingFiles(trainingFolder: string, sourceName: string, targetName: string) {
  if (sourceName === targetName || !fs.existsSync(trainingFolder)) return;

  const entries = await fsp.readdir(trainingFolder);
  for (const entry of entries) {
    let replacement: string | null = null;
    if (entry === sourceName || entry.startsWith(`${sourceName}_`) || entry.startsWith(`${sourceName}.`)) {
      replacement = `${targetName}${entry.slice(sourceName.length)}`;
    } else {
      const criticPrefix = `CRITIC_${sourceName}_`;
      if (entry.startsWith(criticPrefix)) {
        replacement = `CRITIC_${targetName}_${entry.slice(criticPrefix.length)}`;
      }
    }

    if (replacement && replacement !== entry) {
      const from = path.join(trainingFolder, entry);
      const to = path.join(trainingFolder, replacement);
      if (!fs.existsSync(to)) {
        await fsp.rename(from, to);
      }
    }
  }
}

function renameImportedTopLevelName(relativePath: string, sourceName: string, targetName: string) {
  if (sourceName === targetName) return relativePath;
  const normalized = relativePath.replace(/\\/g, '/');
  const parts = normalized.split('/');
  const first = parts[0];

  if (first === sourceName || first.startsWith(`${sourceName}_`) || first.startsWith(`${sourceName}.`)) {
    parts[0] = `${targetName}${first.slice(sourceName.length)}`;
  } else {
    const criticPrefix = `CRITIC_${sourceName}_`;
    if (first.startsWith(criticPrefix)) {
      parts[0] = `CRITIC_${targetName}_${first.slice(criticPrefix.length)}`;
    }
  }

  return parts.join('/');
}

export async function refreshImportedLatestCheckpoint(
  trainingFolder: string,
  latestCheckpointArchivePath: string | null,
  sourceName: string,
  targetName: string,
) {
  if (!latestCheckpointArchivePath) return;

  const relativePath = latestCheckpointArchivePath.replace(/^training[\\/]/, '');
  const finalRelativePath = renameImportedTopLevelName(relativePath, sourceName, targetName);
  const checkpointPath = path.join(trainingFolder, ...finalRelativePath.split('/'));
  if (!fs.existsSync(checkpointPath) || !fs.statSync(checkpointPath).isFile()) return;

  const tempPath = `${checkpointPath}.aitk-latest.tmp`;
  await fsp.copyFile(checkpointPath, tempPath);
  await fsp.rm(checkpointPath, { force: true });
  await fsp.rename(tempPath, checkpointPath);
}

export async function nextAvailablePath(parent: string, preferredName: string) {
  const safePreferredName = safeNameSegment(preferredName, 'imported');
  let candidateName = safePreferredName;
  let candidatePath = path.join(parent, candidateName);
  let suffix = 2;
  while (fs.existsSync(candidatePath)) {
    candidateName = `${safePreferredName}_${suffix}`;
    candidatePath = path.join(parent, candidateName);
    suffix += 1;
  }
  return candidatePath;
}

export function validateArchiveEntryName(fileName: string) {
  const normalized = fileName.replace(/\\/g, '/');
  if (!normalized || normalized.startsWith('/') || normalized.includes('\0')) {
    throw new Error(`Invalid archive entry: ${fileName}`);
  }
  if (normalized.split('/').some(part => part === '..')) {
    throw new Error(`Archive entry escapes import folder: ${fileName}`);
  }
  return normalized;
}
