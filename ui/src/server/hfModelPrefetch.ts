import { spawn } from 'child_process';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';
import { getHFToken } from './settings';
import { prepareHfTokenEnv } from './hfTokenEnv';
import { getToolkitPythonPath } from './pythonPath';
import type { ModelReference } from './trainingJobTransfer';

type HfModelPrefetchResult = {
  handledValues: string[];
  downloads?: Array<{ value: string; path: string; kind: string; cached?: boolean }>;
  warnings?: string[];
};

function trimOutput(value: string, maxLength = 6000) {
  return value.length > maxLength ? value.slice(value.length - maxLength) : value;
}

async function runModelPrefetchScript(inputPath: string): Promise<HfModelPrefetchResult> {
  const pythonPath = getToolkitPythonPath();
  const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'prefetch_hf_models.py');
  const token = await getHFToken();
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
      subprocess.stdout?.on('data', chunk => {
        stdoutText += chunk.toString();
      });
      subprocess.stderr?.on('data', chunk => {
        stderrText += chunk.toString();
      });
      subprocess.on('error', reject);
      subprocess.on('close', code => {
        if (code === 0) {
          resolve(stdoutText);
          return;
        }
        const details = trimOutput([stderrText, stdoutText].filter(Boolean).join('\n').trim());
        reject(new Error(`Failed to download model references during import.${details ? `\n${details}` : ''}`));
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
  if (!references.length) {
    return { handledValues: [], warnings: [] };
  }

  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-model-prefetch-'));
  const inputPath = path.join(tempDir, 'references.json');
  try {
    await fs.writeFile(inputPath, JSON.stringify({ references }), 'utf8');
    const result = await runModelPrefetchScript(inputPath);
    return {
      handledValues: Array.isArray(result.handledValues) ? result.handledValues : [],
      downloads: Array.isArray(result.downloads) ? result.downloads : [],
      warnings: Array.isArray(result.warnings) ? result.warnings : [],
    };
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}
