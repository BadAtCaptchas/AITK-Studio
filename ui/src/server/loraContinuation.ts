import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { createHash } from 'crypto';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { TOOLKIT_ROOT } from '../paths';
import { getDataRoot, getModelsRoot, getTrainingFolder } from './settings';
import { getToolkitPythonPath } from './pythonPath';
import { isRecord } from './commandInput';
import { parseLoraInspection, validSha256, type LoraInspection } from '../domain/loraContinuation';
import { parseRemoteLoraRef } from '../utils/remoteLoraRefs';

export const continuationCapabilities = { version: 1, models: ['krea2', 'krea2:turbo'], modes: ['none', 'replay', 'teacher', 'replay_teacher'] } as const;
export const MAX_CONTINUATION_BYTES = 2 * 1024 ** 3;
const execute = promisify(execFile);
const inside = (root: string, file: string) => {
  const relative = path.relative(root, file);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
};

export async function hashContinuationAsset(file: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

export async function resolveContinuationSource(value: unknown): Promise<string> {
  if (typeof value !== 'string' || !value.trim() || value.length > 4096 || !value.toLowerCase().endsWith('.safetensors'))
    throw new Error('Select a local .safetensors LoRA from the library or upload one.');
  const file = await fsp.realpath(path.resolve(value));
  const roots = await Promise.all([getDataRoot(), getTrainingFolder(), getModelsRoot()]);
  const canonical = await Promise.all(roots.map(root => fsp.realpath(root).catch(() => null)));
  if (!canonical.some(root => root && inside(root, file))) throw new Error('LoRA is outside configured data, training and model storage. Upload it first.');
  const stat = await fsp.stat(file);
  if (!stat.isFile() || stat.size <= 0 || stat.size > MAX_CONTINUATION_BYTES) throw new Error('LoRA must be a file no larger than 2 GiB.');
  return file;
}

export async function inspectContinuationSource(value: unknown, signal?: AbortSignal): Promise<LoraInspection> {
  if (!isRecord(value)) throw new Error('Invalid LoRA inspection request.');
  const remote = parseRemoteLoraRef(value.path);
  if (remote) {
    const { getRemoteWorker, remoteJson } = await import('./remoteClient');
    const result: unknown = await remoteJson(await getRemoteWorker(remote.workerID), '/api/training/lora/inspect', {
      method: 'POST', body: JSON.stringify({ ...value, path: remote.path }), signal, timeoutMs: 150_000, headerTimeoutMs: 150_000,
    });
    const inspection = parseLoraInspection(result);
    if (value.sha256 !== undefined && inspection.sha256 !== value.sha256) throw new Error('Worker source checksum differs from the inspected LoRA.');
    return inspection;
  }
  const source = await resolveContinuationSource(value.path);
  const args = ['-m', 'toolkit.lora_continuation', '--path', source, '--validate-krea2-topology'];
  if (value.scale !== undefined) {
    if (typeof value.scale !== 'number' || !Number.isFinite(value.scale) || value.scale <= 0) throw new Error('Source scale must be positive.');
    args.push('--scale', String(value.scale));
  }
  if (value.sha256 !== undefined) {
    if (!validSha256(value.sha256)) throw new Error('Invalid source checksum.');
    args.push('--sha256', value.sha256);
  }
  const { stdout } = await execute(getToolkitPythonPath(), args, {
    cwd: TOOLKIT_ROOT, windowsHide: true, timeout: 120_000, maxBuffer: 4 * 1024 * 1024, signal,
    env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1', PYTORCH_NVML_BASED_CUDA_CHECK: '0', HF_HUB_OFFLINE: '1', PYTHONIOENCODING: 'utf-8' },
  });
  const result: unknown = JSON.parse(stdout);
  return parseLoraInspection(result);
}

export function continuationProcesses(value: unknown): Record<string, unknown>[] {
  if (!isRecord(value) || !isRecord(value.config) || !Array.isArray(value.config.process)) return [];
  return value.config.process.filter(isRecord).filter(p => isRecord(p.network) && p.network.continuation !== undefined);
}

export async function validateContinuationSources(value: unknown, jobFolder?: string, targetWorkerID?: string): Promise<void> {
  for (const process of continuationProcesses(value)) {
    if (!isRecord(process.network) || !isRecord(process.network.continuation)) throw new Error('Invalid continuation configuration.');
    const source = process.network.continuation;
    const remote = parseRemoteLoraRef(process.network.pretrained_lora_path);
    if (remote && targetWorkerID !== undefined && remote.workerID !== targetWorkerID)
      throw new Error('This LoRA belongs to another worker. Select that worker or upload the file to the new destination.');
    const pinned = jobFolder && validSha256(source.source_sha256) ? path.join(jobFolder, '.continuation', `${source.source_sha256}.safetensors`) : undefined;
    await inspectContinuationSource({ path: pinned && fs.existsSync(pinned) ? pinned : process.network.pretrained_lora_path, scale: source.source_scale, sha256: source.source_sha256 });
  }
}

export type ContinuationArchiveAsset = { archivePath: string; configPath: string; sha256: string; size: number; workerSource?: string };

export async function collectContinuationAssets(config: unknown, jobFolder?: string, targetWorkerID?: string): Promise<{ assets: ContinuationArchiveAsset[]; files: { sourcePath: string; archivePath: string }[] }> {
  const assets: ContinuationArchiveAsset[] = [], files: { sourcePath: string; archivePath: string }[] = [];
  if (!isRecord(config) || !isRecord(config.config) || !Array.isArray(config.config.process)) return { assets, files };
  for (const [index, process] of config.config.process.entries()) {
    if (!isRecord(process) || !isRecord(process.network) || !isRecord(process.network.continuation)) continue;
    const remote = parseRemoteLoraRef(process.network.pretrained_lora_path);
    if (remote) {
      if (remote.workerID !== targetWorkerID) throw new Error('Upload the source LoRA to the selected execution worker first.');
      const inspection = await inspectContinuationSource({ path: process.network.pretrained_lora_path,
        scale: process.network.continuation.source_scale, sha256: process.network.continuation.source_sha256 });
      assets.push({ archivePath: `training/.continuation/${inspection.sha256}.safetensors`, configPath: `config.process[${index}].network.pretrained_lora_path`,
        sha256: inspection.sha256, size: inspection.size_bytes, workerSource: remote.path });
    }
    const fields: [string, unknown, unknown][] = [['network.pretrained_lora_path', process.network.pretrained_lora_path, process.network.continuation.source_sha256]];
    let pinnedAssistant: unknown = process.network.continuation.assistant_sha256 ?? undefined;
    if (jobFolder) {
      const provenancePath = path.join(jobFolder, '.continuation', 'provenance.json');
      const provenanceStat = await fsp.stat(provenancePath).catch(() => null);
      if (provenanceStat) {
        if (!provenanceStat.isFile() || provenanceStat.size > 1024 * 1024) throw new Error('Invalid continuation provenance.');
        const provenance: unknown = JSON.parse(await fsp.readFile(provenancePath, 'utf8'));
        if (!isRecord(provenance) || provenance.source_sha256 !== process.network.continuation.source_sha256) throw new Error('Continuation provenance does not match its source.');
        if (pinnedAssistant !== undefined && provenance.assistant_sha256 !== pinnedAssistant)
          throw new Error('Continuation training-adapter checksum changed; create a new job.');
        pinnedAssistant = provenance.assistant_sha256 ?? pinnedAssistant;
      }
    }
    if (isRecord(process.model) && typeof process.model.assistant_lora_path === 'string') {
      // Hub references resolve and are pinned by the worker; local auxiliaries must travel.
      const auxiliary = process.model.assistant_lora_path;
      if (validSha256(pinnedAssistant) || path.isAbsolute(auxiliary) || path.win32.isAbsolute(auxiliary) || fs.existsSync(auxiliary))
        fields.push(['model.assistant_lora_path', auxiliary, pinnedAssistant]);
    }
    for (const [field, value, expected] of fields) {
      if (remote && field === 'network.pretrained_lora_path') continue;
      const pinned = jobFolder && validSha256(expected) ? path.join(jobFolder, '.continuation', `${expected}.safetensors`) : undefined;
      const sourcePath = await resolveContinuationSource(pinned && fs.existsSync(pinned) ? pinned : value);
      const sha256 = await hashContinuationAsset(sourcePath);
      if (expected !== undefined && expected !== sha256) throw new Error('Continuation source changed before transfer.');
      const archivePath = `training/.continuation/${sha256}.safetensors`;
      assets.push({ archivePath, configPath: `config.process[${index}].${field}`, sha256, size: (await fsp.stat(sourcePath)).size });
      if (!files.some(file => file.archivePath === archivePath)) files.push({ sourcePath, archivePath });
    }
  }
  return { assets, files };
}

export async function rewriteImportedContinuationAssets(config: unknown, assets: unknown, extractionRoot: string, destination: string): Promise<void> {
  if (assets === undefined) {
    if (continuationProcesses(config).length) throw new Error('Continuation archive is missing its source assets.');
    return;
  }
  if (!Array.isArray(assets) || assets.length > 64 || !isRecord(config) || !isRecord(config.config) || !Array.isArray(config.config.process))
    throw new Error('Invalid continuation archive assets.');
  const seen = new Set<string>();
  const changes: (() => void)[] = [];
  for (const asset of assets) {
    if (!isRecord(asset) || !validSha256(asset.sha256) || asset.archivePath !== `training/.continuation/${asset.sha256}.safetensors` ||
        typeof asset.size !== 'number' || !Number.isSafeInteger(asset.size) || asset.size <= 0 || asset.size > MAX_CONTINUATION_BYTES || typeof asset.configPath !== 'string') throw new Error('Invalid continuation asset.');
    const match = /^config\.process\[(\d+)\]\.(network\.pretrained_lora_path|model\.assistant_lora_path)$/.exec(asset.configPath);
    if (!match || seen.has(asset.configPath)) throw new Error('Invalid or duplicate continuation path mapping.');
    seen.add(asset.configPath);
    const process = config.config.process[Number(match[1])];
    if (!isRecord(process) || !isRecord(process.network) || !isRecord(process.network.continuation)) throw new Error('Missing continuation process.');
    if (asset.workerSource !== undefined) {
      const source = await resolveContinuationSource(asset.workerSource);
      if ((await fsp.stat(source)).size !== asset.size || await hashContinuationAsset(source) !== asset.sha256)
        throw new Error('Worker source adapter changed since inspection.');
      const staged = path.join(extractionRoot, asset.archivePath);
      await fsp.mkdir(path.dirname(staged), { recursive: true });
      if (!inside(await fsp.realpath(extractionRoot), await fsp.realpath(path.dirname(staged)))) throw new Error('Unsafe worker source staging path.');
      if (!fs.existsSync(staged)) await fsp.copyFile(source, staged, fs.constants.COPYFILE_EXCL);
    }
    const file = await fsp.realpath(path.join(extractionRoot, asset.archivePath));
    if (!inside(await fsp.realpath(extractionRoot), file) || (await fsp.stat(file)).size !== asset.size || await hashContinuationAsset(file) !== asset.sha256)
      throw new Error('Transferred continuation asset failed checksum validation.');
    const [section, field] = match[2].split('.');
    const target = process[section];
    if (!isRecord(target)) throw new Error('Missing continuation configuration section.');
    if (section === 'network' && process.network.continuation.source_sha256 !== asset.sha256) throw new Error('Source hash does not match the imported job.');
    const continuation = process.network.continuation;
    if (section === 'model' && continuation.assistant_sha256 !== undefined && continuation.assistant_sha256 !== asset.sha256)
      throw new Error('Auxiliary hash does not match the imported job.');
    changes.push(() => {
      if (section === 'model') continuation.assistant_sha256 = asset.sha256;
      target[field] = path.join(destination, '.continuation', `${asset.sha256}.safetensors`);
    });
  }
  for (const [index, process] of config.config.process.entries()) {
    if (isRecord(process) && isRecord(process.network) && process.network.continuation && !seen.has(`config.process[${index}].network.pretrained_lora_path`))
      throw new Error('Missing source adapter in continuation archive.');
  }
  for (const change of changes) change();
}
