import fs from 'fs';
import path from 'path';
import { execFile, spawnSync } from 'child_process';
import { promisify } from 'util';
import { TOOLKIT_ROOT } from '../paths';

const execute = promisify(execFile);
let resolved: { key: string; executable: string; expires: number } | undefined;
const preflights = new Map<string, { expires: number; promise: Promise<void> }>();

export function getToolkitPythonPath(): string {
  const windows = process.platform === 'win32';
  const configured = process.env.AITK_PYTHON_PATH?.trim();
  const local = ['.venv', 'venv'].find(name => fs.existsSync(path.join(TOOLKIT_ROOT, name)));
  const prefix = process.env.VIRTUAL_ENV || process.env.CONDA_PREFIX;
  const candidate = configured || (local
    ? path.join(TOOLKIT_ROOT, local, windows ? 'Scripts/python.exe' : 'bin/python')
    : prefix ? path.join(prefix, windows ? (process.env.VIRTUAL_ENV ? 'Scripts/python.exe' : 'python.exe') : 'bin/python') : windows ? 'python.exe' : 'python3');
  if (local && !configured && !fs.existsSync(candidate)) {
    throw new Error(`The selected ${local} environment has no Python executable. Repair it or set AITK_PYTHON_PATH explicitly.`);
  }
  const key = `${candidate}:${fs.existsSync(candidate) ? fs.statSync(candidate).mtimeMs : ''}`;
  if (resolved?.key === key && Date.now() < resolved.expires) return resolved.executable;
  const result = spawnSync(candidate, ['-c', 'import sys,json; print(json.dumps({"executable":sys.executable,"version":list(sys.version_info[:2])}))'], {
    cwd: TOOLKIT_ROOT, windowsHide: true, encoding: 'utf8', timeout: 5_000, maxBuffer: 32_768,
  });
  if (result.error || result.status !== 0) throw new Error('Python cannot run. Activate Python 3.12 or set AITK_PYTHON_PATH to a working environment.');
  const value: unknown = JSON.parse(result.stdout);
  if (!value || typeof value !== 'object' || !('executable' in value) || typeof value.executable !== 'string' ||
      !('version' in value) || !Array.isArray(value.version) || value.version[0] !== 3 || ![11, 12].includes(value.version[1])) {
    throw new Error('Unsupported Python interpreter. Use Python 3.12 or the Python 3.11 DGX profile.');
  }
  resolved = { key, executable: value.executable, expires: Date.now() + 60_000 };
  return value.executable;
}

export async function assertPythonRuntimeReady(arch?: string): Promise<void> {
  const executable = getToolkitPythonPath();
  const key = [arch || '', executable, fs.statSync(executable).mtimeMs, ...['requirements.txt', 'requirements_base.txt', 'requirements_torch_macos.txt', 'ui/src/domain/modelCapabilities.json'].map(
    name => fs.statSync(path.join(TOOLKIT_ROOT, name)).mtimeMs,
  )].join(':');
  const cached = preflights.get(key);
  if (cached && Date.now() < cached.expires) return cached.promise;
  for (const [oldKey, entry] of preflights) if (entry.expires <= Date.now()) preflights.delete(oldKey);
  const promise = execute(executable, ['-c', 'import sys, torch, yaml, safetensors, diffusers, transformers; from toolkit.job import get_job; from toolkit.util.get_model import get_model_class; from toolkit.config_modules import ModelConfig; get_model_class(ModelConfig(name_or_path="preflight/no-weights", arch=sys.argv[1])) if len(sys.argv)>1 else None', ...(arch ? [arch] : [])], {
    cwd: TOOLKIT_ROOT, windowsHide: true, timeout: 45_000, maxBuffer: 64 * 1024,
    env: { ...process.env, HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', NO_ALBUMENTATIONS_UPDATE: '1' },
  }).then(() => undefined).catch(() => {
    preflights.delete(key);
    throw new Error('Python runtime preflight failed. Run the environment doctor for this interpreter before starting a job.');
  });
  preflights.set(key, { expires: Date.now() + 5 * 60_000, promise });
  return promise;
}
