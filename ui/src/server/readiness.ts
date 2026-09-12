import { execFile } from 'child_process';
import fs from 'fs/promises';
import { constants } from 'fs';
import { promisify } from 'util';
import path from 'path';
import { TOOLKIT_ROOT } from '../paths';
import { getToolkitPythonPath } from './pythonPath';
import { getDatasetsRoot, getTrainingFolder } from './settings';
import { getModelsRoot } from './settings';
import { db } from './db';
import { isRecord } from './commandInput';
const execute = promisify(execFile);
type Check = { name: string; ok: boolean; detail: string; action?: string | null };
let cached: { key: string; expires: number; promise: Promise<Check[]> } | undefined;
async function runtimeChecks(): Promise<Check[]> {
  let executable: string;
  try {
    executable = getToolkitPythonPath();
  } catch (error) {
    return [
      {
        name: 'Python',
        ok: false,
        detail: error instanceof Error ? error.message : 'Python unavailable',
        action: 'Set AITK_PYTHON_PATH to a supported isolated interpreter.',
      },
    ];
  }
  const key = [
    executable,
    (await fs.stat(executable)).mtimeMs,
    (await fs.stat(path.join(TOOLKIT_ROOT, 'requirements_base.txt'))).mtimeMs,
  ].join(':');
  if (cached?.key === key && cached.expires > Date.now()) return cached.promise;
  const promise = (async () => {
    let stdout: string;
    try {
      stdout = (
        await execute(executable, [path.join(TOOLKIT_ROOT, 'scripts/environment_doctor.py')], {
          cwd: TOOLKIT_ROOT,
          timeout: 90_000,
          maxBuffer: 128 * 1024,
          windowsHide: true,
          env: { ...process.env, HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', NO_ALBUMENTATIONS_UPDATE: '1' },
        })
      ).stdout;
    } catch (error) {
      if (!isRecord(error) || typeof error.stdout !== 'string' || !error.stdout.trim())
        return [
          {
            name: 'Runtime doctor',
            ok: false,
            detail: 'Runtime diagnostics could not finish within 90 seconds.',
            action: 'Run scripts/environment_doctor.py with the selected interpreter for detailed output.',
          },
        ];
      stdout = error.stdout;
    }
    const report: unknown = JSON.parse(stdout);
    if (!isRecord(report) || !Array.isArray(report.checks)) throw new Error('Invalid doctor output');
    return report.checks.filter(
      (check): check is Check =>
        isRecord(check) &&
        typeof check.name === 'string' &&
        typeof check.ok === 'boolean' &&
        typeof check.detail === 'string',
    );
  })();
  cached = { key, expires: Date.now() + 60_000, promise };
  return promise;
}
export async function getReadiness() {
  const checks = [...(await runtimeChecks())];
  for (const [name, resolve] of [
    ['Datasets', getDatasetsRoot],
    ['Training output', getTrainingFolder],
    ['Models', getModelsRoot],
  ] as const) {
    try {
      const root = await resolve();
      await fs.access(root, constants.R_OK | constants.W_OK);
      checks.push({ name, ok: true, detail: root });
    } catch {
      checks.push({
        name,
        ok: false,
        detail: 'Configured folder is missing or inaccessible.',
        action: 'Review Storage settings and folder permissions.',
      });
    }
  }
  const heartbeat = await db.runtime.get('worker-heartbeat');
  const value = heartbeat?.value;
  const fresh = isRecord(value) && typeof value.at === 'number' && Date.now() - value.at < 15_000;
  checks.push({
    name: 'Queue worker',
    ok: fresh,
    detail: fresh ? 'Supervisor heartbeat is current.' : 'No recent supervisor heartbeat.',
    action: fresh ? null : 'Start the managed app stack; inspect worker startup logs.',
  });
  return {
    version: 1,
    checkedAt: new Date().toISOString(),
    ready: checks.every(check => check.ok),
    checks,
    freshness: (await db.runtime.get('job-read-model:freshness'))?.value ?? null,
  };
}
