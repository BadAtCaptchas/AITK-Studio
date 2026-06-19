import { spawn } from 'child_process';
import fs from 'fs';
import fsp from 'fs/promises';
import os from 'os';
import path from 'path';
import { TOOLKIT_ROOT } from '../paths';
import type { DatasetSummary } from '../types';
import { makeRemoteDatasetRef } from '../utils/remoteDatasetRefs';
import { listDatasetSummaries } from './encryptedDatasets';
import { prepareHfTokenEnv } from './hfTokenEnv';
import { getToolkitPythonPath } from './pythonPath';
import { getHFToken } from './settings';
import { isOfflineModeEnabled } from './networkPolicy';
import { nextAvailablePath, safeNameSegment } from './trainingJobTransfer';

export type HfDatasetCaptionMode = 'auto' | 'none' | 'column';
export type HfDatasetImportAction = 'preview' | 'import';

export type HfDatasetImportRequest = {
  action: HfDatasetImportAction;
  worker_id?: string;
  dataset: string;
  config?: string;
  split?: string;
  imageColumn?: string;
  captionMode: HfDatasetCaptionMode;
  captionColumn?: string;
  outputName?: string;
  maxRows?: number;
};

export type HfDatasetImportStats = {
  datasetID: string;
  config: string;
  split: string;
  imageColumn: string;
  captionColumn: string | null;
  imagesWritten: number;
  captionsWritten: number;
  rowsScanned: number;
  rowsSkipped: number;
  warnings: string[];
};

export type HfDatasetImportResult = {
  dataset: DatasetSummary;
  path: string;
  renamed: boolean;
  imported: HfDatasetImportStats;
};

export type HfDatasetPreviewResult = {
  datasetID: string;
  configs: string[];
  splits: string[];
  selectedConfig: string;
  selectedSplit: string;
  rowCount?: number | null;
  features: Array<{ name: string; kind: string }>;
  imageColumns: string[];
  textColumns: string[];
  suggestedImageColumn: string | null;
  suggestedCaptionColumn: string | null;
  samples: Array<Record<string, unknown>>;
};

const CAPTION_PRIORITY = ['caption', 'captions', 'prompt', 'text', 'description', 'title'];
const HF_DATASET_ID_RE =
  /^(?:[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?\/)?[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/;

function trimString(value: unknown) {
  return typeof value === 'string' ? value.trim() : '';
}

function emptyToUndefined(value: unknown) {
  const trimmed = trimString(value);
  return trimmed || undefined;
}

function parseMaxRows(value: unknown) {
  if (value == null || value === '') return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return undefined;
  return Math.floor(parsed);
}

export function normalizeHfDatasetID(value: unknown) {
  let raw = trimString(value);
  if (!raw) {
    throw new Error('Hugging Face dataset is required.');
  }

  raw = raw.replace(/^https?:\/\/(?:www\.)?huggingface\.co\//i, '');
  raw = raw.replace(/^huggingface\.co\//i, '');
  raw = raw.split(/[?#]/, 1)[0].replace(/^\/+|\/+$/g, '');
  if (raw.toLowerCase().startsWith('datasets/')) {
    raw = raw.slice('datasets/'.length);
  }

  if (!HF_DATASET_ID_RE.test(raw) || raw.includes('..') || raw.includes('//')) {
    throw new Error('Enter a valid Hugging Face dataset ID or dataset URL.');
  }
  return raw;
}

export function rankHfCaptionColumns(columns: string[]) {
  const unique = Array.from(new Set(columns.filter(Boolean)));
  const byLower = new Map(unique.map(column => [column.toLowerCase(), column]));
  const ranked: string[] = [];
  CAPTION_PRIORITY.forEach(column => {
    const match = byLower.get(column);
    if (match && !ranked.includes(match)) ranked.push(match);
  });
  unique.forEach(column => {
    if (!ranked.includes(column)) ranked.push(column);
  });
  return ranked;
}

export function normalizeHfDatasetImportRequest(value: unknown): HfDatasetImportRequest {
  const body = value && typeof value === 'object' ? (value as Record<string, unknown>) : {};
  const action = body.action === 'import' ? 'import' : body.action === 'preview' ? 'preview' : null;
  if (!action) {
    throw new Error('Invalid Hugging Face import action.');
  }

  const captionMode = body.captionMode === 'none' || body.captionMode === 'column' ? body.captionMode : 'auto';
  const request: HfDatasetImportRequest = {
    action,
    worker_id: emptyToUndefined(body.worker_id),
    dataset: normalizeHfDatasetID(body.dataset),
    config: emptyToUndefined(body.config),
    split: emptyToUndefined(body.split),
    imageColumn: emptyToUndefined(body.imageColumn),
    captionMode,
    captionColumn: emptyToUndefined(body.captionColumn),
    outputName: emptyToUndefined(body.outputName),
    maxRows: parseMaxRows(body.maxRows),
  };

  if (captionMode === 'column' && !request.captionColumn) {
    throw new Error('Choose a caption column or set captions to Auto or None.');
  }

  return request;
}

function trimOutput(value: string, maxLength = 6000) {
  return value.length > maxLength ? value.slice(value.length - maxLength) : value;
}

function parseJsonResult(stdout: string) {
  const jsonLine = stdout.trim().split(/\r?\n/).filter(Boolean).pop();
  if (!jsonLine) {
    throw new Error('Hugging Face dataset importer did not return a result.');
  }
  return JSON.parse(jsonLine);
}

async function runHfDatasetScript<T>(payload: Record<string, unknown>, timeoutMs: number): Promise<T> {
  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-hf-dataset-'));
  const inputPath = path.join(tempDir, 'input.json');
  const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'import_hf_dataset.py');
  const token = await getHFToken();
  const preparedHfEnv = await prepareHfTokenEnv({
    token,
    tokenFilePrefix: 'import-hf-dataset',
  });

  try {
    await fsp.writeFile(inputPath, JSON.stringify(payload), 'utf8');
    const stdout = await new Promise<string>((resolve, reject) => {
      const subprocess = spawn(getToolkitPythonPath(), ['-u', scriptPath, '--input', inputPath], {
        cwd: TOOLKIT_ROOT,
        env: {
          ...preparedHfEnv.env,
          HF_HUB_ENABLE_HF_TRANSFER: process.platform === 'win32' ? '0' : '1',
        },
        windowsHide: true,
      });

      let stdoutText = '';
      let stderrText = '';
      let timedOut = false;
      let settled = false;
      let killTimer: ReturnType<typeof setTimeout> | null = null;
      const timeoutTimer = setTimeout(() => {
        timedOut = true;
        subprocess.kill('SIGTERM');
        killTimer = setTimeout(() => subprocess.kill('SIGKILL'), 5000);
      }, timeoutMs);
      const cleanupTimers = () => {
        clearTimeout(timeoutTimer);
        if (killTimer) clearTimeout(killTimer);
      };
      const fail = (error: Error) => {
        if (settled) return;
        settled = true;
        cleanupTimers();
        reject(error);
      };
      const finish = (value: string) => {
        if (settled) return;
        settled = true;
        cleanupTimers();
        resolve(value);
      };

      subprocess.stdout?.on('data', chunk => {
        stdoutText = trimOutput(stdoutText + chunk.toString());
      });
      subprocess.stderr?.on('data', chunk => {
        stderrText = trimOutput(stderrText + chunk.toString());
      });
      subprocess.on('error', fail);
      subprocess.on('close', code => {
        if (timedOut) {
          fail(new Error(`Hugging Face dataset import timed out after ${Math.round(timeoutMs / 1000)} seconds.`));
          return;
        }
        if (code === 0) {
          finish(stdoutText);
          return;
        }
        const details = trimOutput([stderrText, stdoutText].filter(Boolean).join('\n').trim());
        fail(new Error(details || 'Failed to import Hugging Face dataset.'));
      });
    });

    return parseJsonResult(stdout) as T;
  } finally {
    await preparedHfEnv.cleanup();
    await fsp.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
  }
}

function defaultOutputName(request: HfDatasetImportRequest) {
  const datasetName = request.dataset.replace(/[^\w.-]+/g, '_');
  const splitSuffix = request.split && request.split !== 'train' ? `_${request.split}` : '';
  return safeNameSegment(`${datasetName}${splitSuffix}`, 'hf_dataset');
}

export async function previewHfDatasetImport(rawRequest: unknown) {
  const request = normalizeHfDatasetImportRequest(rawRequest);
  if (await isOfflineModeEnabled()) {
    throw new Error('Hugging Face dataset preview is blocked while offline mode is enabled.');
  }
  return runHfDatasetScript<HfDatasetPreviewResult>(
    {
      ...request,
      action: 'preview',
    },
    3 * 60 * 1000,
  );
}

export async function importHfDataset(datasetsRoot: string, rawRequest: unknown): Promise<HfDatasetImportResult> {
  const request = normalizeHfDatasetImportRequest(rawRequest);
  if (await isOfflineModeEnabled()) {
    throw new Error('Hugging Face dataset import is blocked while offline mode is enabled.');
  }
  await fsp.mkdir(datasetsRoot, { recursive: true });

  const importID = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const workRoot = path.join(datasetsRoot, `.aitk-hf-dataset-import-${importID}`);
  const outputPath = path.join(workRoot, 'dataset');

  try {
    await fsp.mkdir(workRoot, { recursive: true });
    const imported = await runHfDatasetScript<HfDatasetImportStats>(
      {
        ...request,
        action: 'import',
        outputPath,
      },
      24 * 60 * 60 * 1000,
    );

    if (!fs.existsSync(outputPath) || !fs.statSync(outputPath).isDirectory()) {
      throw new Error('Hugging Face dataset importer did not create an output folder.');
    }

    const preferredName = safeNameSegment(request.outputName || defaultOutputName(request), 'hf_dataset');
    const targetPath = await nextAvailablePath(datasetsRoot, preferredName);
    await fsp.rename(outputPath, targetPath);
    const importedName = path.basename(targetPath);
    const allDatasets = await listDatasetSummaries(datasetsRoot);
    const dataset = allDatasets.find(item => item.name === importedName) || {
      name: importedName,
      encrypted: false,
      source: 'local' as const,
      worker_id: 'local',
      worker_name: 'Local',
      ref: `aitk-dataset://local/${encodeURIComponent(importedName)}`,
      path: targetPath,
    };

    return {
      dataset,
      path: targetPath,
      renamed: importedName !== preferredName,
      imported,
    };
  } finally {
    await fsp.rm(workRoot, { recursive: true, force: true }).catch(() => undefined);
  }
}

export function decorateRemoteHfDatasetImportResult(
  worker: { id: string; name: string },
  result: HfDatasetImportResult,
): HfDatasetImportResult {
  return {
    ...result,
    dataset: {
      ...result.dataset,
      source: 'remote',
      worker_id: worker.id,
      worker_name: worker.name,
      ref: makeRemoteDatasetRef(worker.id, result.dataset.name),
      path: undefined,
    },
  };
}
