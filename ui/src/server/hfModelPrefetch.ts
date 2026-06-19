import { spawn } from 'child_process';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';
import { getHFToken } from './settings';
import { prepareHfTokenEnv } from './hfTokenEnv';
import { getToolkitPythonPath } from './pythonPath';
import type { ModelReference } from './trainingJobTransfer';
import { isOfflineModeEnabled } from './networkPolicy';

type HfModelPrefetchResult = {
  handledValues: string[];
  downloads?: Array<{ value: string; path: string; kind: string; cached?: boolean }>;
  warnings?: string[];
};

const DEFAULT_MAX_PREFETCH_REFERENCES = 20;
const DEFAULT_PREFETCH_TIMEOUT_MS = 30 * 60 * 1000;

function trimOutput(value: string, maxLength = 6000) {
  return value.length > maxLength ? value.slice(value.length - maxLength) : value;
}

export function normalizeModelReferenceValue(value: string) {
  return value.trim().replace(/\\/g, '/');
}

function parsePositiveInteger(value: string | undefined, fallback: number) {
  if (!value) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

function getMaxPrefetchReferences() {
  return parsePositiveInteger(process.env.AITK_MODEL_PREFETCH_MAX_REFERENCES, DEFAULT_MAX_PREFETCH_REFERENCES);
}

function getPrefetchTimeoutMs() {
  return parsePositiveInteger(process.env.AITK_MODEL_PREFETCH_TIMEOUT_MS, DEFAULT_PREFETCH_TIMEOUT_MS);
}

async function runModelPrefetchScript(inputPath: string): Promise<HfModelPrefetchResult> {
  const pythonPath = getToolkitPythonPath();
  const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'prefetch_hf_models.py');
  const token = await getHFToken();
  const timeoutMs = getPrefetchTimeoutMs();
  const preparedHfEnv = await prepareHfTokenEnv({
    token,
    tokenFilePrefix: 'import-model-prefetch',
  });

  try {
    const stdout = await new Promise<string>((resolve, reject) => {
      const subprocess = spawn(pythonPath, ['-u', scriptPath, '--input', inputPath], {
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
        if (killTimer !== null) {
          clearTimeout(killTimer);
        }
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
          fail(new Error(`Model reference download timed out after ${Math.round(timeoutMs / 1000)} seconds.`));
          return;
        }
        if (code === 0) {
          finish(stdoutText);
          return;
        }
        const details = trimOutput([stderrText, stdoutText].filter(Boolean).join('\n').trim());
        fail(new Error(`Failed to download model references.${details ? `\n${details}` : ''}`));
      });
    });

    const jsonLine = stdout.trim().split(/\r?\n/).filter(Boolean).pop();
    if (!jsonLine) {
      throw new Error('Model prefetch did not return a result.');
    }
    return JSON.parse(jsonLine) as HfModelPrefetchResult;
  } finally {
    await preparedHfEnv.cleanup();
  }
}

export async function prefetchModelReferences(references: ModelReference[]): Promise<HfModelPrefetchResult> {
  if (await isOfflineModeEnabled()) {
    return {
      handledValues: [],
      warnings: ['Model reference downloads are blocked while offline mode is enabled.'],
    };
  }

  if (!references.length) {
    return { handledValues: [], warnings: [] };
  }

  const maxReferences = getMaxPrefetchReferences();
  const seen = new Set<string>();
  const normalizedReferences: ModelReference[] = [];
  for (const reference of references) {
    const value = normalizeModelReferenceValue(reference.value || '');
    if (!value || seen.has(value)) continue;
    seen.add(value);
    normalizedReferences.push({
      ...reference,
      value,
    });
  }

  const warnings: string[] = [];
  const limitedReferences = normalizedReferences.slice(0, maxReferences);
  if (normalizedReferences.length > limitedReferences.length) {
    warnings.push(
      `Only the first ${limitedReferences.length} unique model references were downloaded. ${normalizedReferences.length - limitedReferences.length} additional references were skipped.`,
    );
  }

  if (!limitedReferences.length) {
    return { handledValues: [], warnings };
  }

  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-model-prefetch-'));
  const inputPath = path.join(tempDir, 'references.json');
  try {
    await fs.writeFile(
      inputPath,
      JSON.stringify({
        references: limitedReferences,
      }),
      'utf8',
    );
    const result = await runModelPrefetchScript(inputPath);
    return {
      handledValues: Array.isArray(result.handledValues) ? result.handledValues.map(normalizeModelReferenceValue) : [],
      downloads: Array.isArray(result.downloads) ? result.downloads : [],
      warnings: [...warnings, ...(Array.isArray(result.warnings) ? result.warnings : [])],
    };
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}
