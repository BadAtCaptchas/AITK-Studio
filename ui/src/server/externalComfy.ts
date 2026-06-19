import { db } from './db';
import { flushCache } from './settings';
import fs from 'fs';
import path from 'path';
import { getTrainingFolder } from './settings';
import {
  extractTriggerWordsFromMetadata,
  listUploadedLoras,
  mergeTriggerWords,
  readSafetensorsMetadata,
  splitTriggerWords,
} from './loraLibrary';
import { classTypesFromWorkflow, requiredIdeogramModels, type IdeogramRequiredModel } from '../utils/ideogramWorkflow';
import { assertUrlAllowedByOfflineMode, guardedFetch } from './networkPolicy';

export const COMFY_EXTERNAL_URL_KEY = 'COMFY_EXTERNAL_URL';
export const COMFY_EXTERNAL_LORA_DIR_KEY = 'COMFY_EXTERNAL_LORA_DIR';
export const DEFAULT_EXTERNAL_COMFY_URL = 'http://127.0.0.1:8188';

const LORA_JOB_TYPES = new Set(['lora', 'locon', 'lokr', 'lorm']);

export type FetchLike = typeof fetch;

export type ComfyImageRef = {
  filename: string;
  subfolder: string;
  type: string;
};

export type ComfyPreflightItem = {
  id: string;
  label: string;
  status: 'found' | 'missing' | 'unknown';
  detail: string;
};

export type ComfyPreflightResult = {
  ok: boolean;
  serverUrl: string;
  connected: boolean;
  systemStats: unknown;
  nodes: ComfyPreflightItem[];
  models: ComfyPreflightItem[];
  error: string | null;
};

export type ToolkitLoraSummary = {
  id: string;
  label: string;
  path: string;
  filename: string;
  source: 'job' | 'uploaded';
  sizeBytes: number;
  updatedAt: string;
  triggerWords: string[];
};

export class ExternalComfyError extends Error {
  status: number;

  constructor(message: string, status = 500) {
    super(message);
    this.name = 'ExternalComfyError';
    this.status = status;
  }
}

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function safeJsonText(value: unknown) {
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function normalizeExternalComfyUrl(value: unknown) {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) return '';
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new ExternalComfyError('ComfyUI URL must be a valid http(s) URL.', 400);
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new ExternalComfyError('ComfyUI URL must use http or https.', 400);
  }
  url.hash = '';
  url.search = '';
  return url.toString().replace(/\/+$/, '');
}

export function normalizeExternalComfyLoraDir(value: unknown) {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) return '';
  const resolved = path.resolve(raw);
  if (resolved === path.parse(resolved).root) {
    throw new ExternalComfyError('External ComfyUI LoRA folder cannot be the filesystem root.', 400);
  }
  return resolved;
}

export async function getSavedExternalComfyUrl() {
  const row = await db.settings.get(COMFY_EXTERNAL_URL_KEY);
  return normalizeExternalComfyUrl(row?.value || DEFAULT_EXTERNAL_COMFY_URL);
}

export async function getSavedExternalComfyLoraDir() {
  const row = await db.settings.get(COMFY_EXTERNAL_LORA_DIR_KEY);
  return normalizeExternalComfyLoraDir(row?.value || '');
}

export async function saveExternalComfyUrl(value: unknown) {
  const serverUrl = normalizeExternalComfyUrl(value || DEFAULT_EXTERNAL_COMFY_URL);
  await db.settings.upsert(COMFY_EXTERNAL_URL_KEY, serverUrl);
  flushCache();
  return serverUrl;
}

export async function saveExternalComfyLoraDir(value: unknown) {
  const loraDir = normalizeExternalComfyLoraDir(value || '');
  await db.settings.upsert(COMFY_EXTERNAL_LORA_DIR_KEY, loraDir);
  flushCache();
  return loraDir;
}

export async function resolveExternalComfyUrl(value?: unknown) {
  const normalized = normalizeExternalComfyUrl(value || '');
  if (normalized) return normalized;
  const saved = await getSavedExternalComfyUrl();
  return saved;
}

function parseJobConfig(jobConfig: string) {
  try {
    return JSON.parse(jobConfig);
  } catch {
    return null;
  }
}

function isLoraTrainingJob(jobConfig: any) {
  const networkType = String(jobConfig?.config?.process?.[0]?.network?.type || '').toLowerCase();
  return LORA_JOB_TYPES.has(networkType);
}

async function getSafeJobFolder(trainingRoot: string, jobName: string) {
  const root = await fs.promises.realpath(trainingRoot).catch(() => null);
  if (!root) return null;
  const folder = path.resolve(root, jobName);
  const relativePath = path.relative(root, folder);
  if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) return null;
  const stat = await fs.promises.stat(folder).catch(() => null);
  return stat?.isDirectory() ? folder : null;
}

export async function listToolkitLoras(): Promise<ToolkitLoraSummary[]> {
  const trainingRoot = await getTrainingFolder();
  const jobs = await db.jobs.list({ job_type: 'train', project_id: null });
  const loras: ToolkitLoraSummary[] = [];

  for (const job of jobs) {
    if (job.worker_id && job.worker_id !== 'local') continue;
    const jobConfig = parseJobConfig(job.job_config);
    if (!jobConfig || !isLoraTrainingJob(jobConfig)) continue;
    const jobFolder = await getSafeJobFolder(trainingRoot, job.name);
    if (!jobFolder) continue;

    const entries = await fs.promises.readdir(jobFolder, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.toLowerCase().endsWith('.safetensors')) continue;
      const filePath = path.join(jobFolder, entry.name);
      const stat = await fs.promises.stat(filePath).catch(() => null);
      if (!stat) continue;
      const metadata = await readSafetensorsMetadata(filePath);
      const triggerWords = mergeTriggerWords(
        extractTriggerWordsFromMetadata(metadata),
        splitTriggerWords(jobConfig?.config?.process?.[0]?.trigger_word),
      );
      loras.push({
        id: `${job.id}:${entry.name}`,
        label: `${job.name} / ${entry.name}`,
        path: filePath,
        filename: entry.name,
        source: 'job',
        updatedAt: stat.mtime.toISOString(),
        sizeBytes: stat.size,
        triggerWords,
      });
    }
  }

  const uploaded = await listUploadedLoras();
  loras.push(
    ...uploaded.map(lora => ({
      id: lora.id,
      label: lora.label,
      path: lora.path,
      filename: lora.filename,
      source: lora.source,
      updatedAt: lora.updatedAt,
      sizeBytes: lora.sizeBytes,
      triggerWords: lora.triggerWords,
    })),
  );
  return loras.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
}

function comfyUrl(serverUrl: string, pathname: string, query?: Record<string, string | number | null | undefined>) {
  const url = new URL(pathname, `${serverUrl}/`);
  Object.entries(query || {}).forEach(([key, value]) => {
    if (value == null) return;
    url.searchParams.set(key, String(value));
  });
  return url;
}

async function readErrorDetail(response: Response) {
  const text = await response.text().catch(() => '');
  return text ? `${response.status}: ${text.slice(0, 1000)}` : `${response.status} ${response.statusText}`;
}

export async function comfyRequest<T = unknown>({
  serverUrl,
  path,
  method = 'GET',
  body,
  query,
  fetchImpl = fetch,
  expectJson = true,
  signal,
}: {
  serverUrl: string;
  path: string;
  method?: 'GET' | 'POST';
  body?: unknown;
  query?: Record<string, string | number | null | undefined>;
  fetchImpl?: FetchLike;
  expectJson?: boolean;
  signal?: AbortSignal;
}): Promise<T> {
  const url = comfyUrl(serverUrl, path, query);
  let response: Response;
  try {
    const init: RequestInit = {
      method,
      signal,
      headers: body == null ? undefined : { 'Content-Type': 'application/json' },
      body: body == null ? undefined : JSON.stringify(body),
    };
    response =
      fetchImpl === fetch
        ? await guardedFetch(url, init, 'External ComfyUI request')
        : await (async () => {
            await assertUrlAllowedByOfflineMode(url, 'External ComfyUI request');
            return fetchImpl(url, init);
          })();
  } catch (error) {
    throw new ExternalComfyError(
      `Could not reach ComfyUI at ${serverUrl}: ${error instanceof Error ? error.message : String(error)}`,
      502,
    );
  }

  if (!response.ok) {
    throw new ExternalComfyError(
      `ComfyUI request failed for ${path}: ${await readErrorDetail(response)}`,
      response.status,
    );
  }
  if (!expectJson) {
    return (await response.arrayBuffer()) as T;
  }
  if (response.status === 204) return null as T;
  return (await response.json()) as T;
}

export async function getComfySystemStats(serverUrl: string, fetchImpl?: FetchLike) {
  return comfyRequest({ serverUrl, path: '/system_stats', fetchImpl });
}

export async function getComfyObjectInfo(serverUrl: string, fetchImpl?: FetchLike) {
  return comfyRequest<Record<string, unknown>>({ serverUrl, path: '/object_info', fetchImpl });
}

function inputAllowedValues(objectInfo: Record<string, unknown>, classType: string, inputName: string) {
  const node = objectInfo[classType];
  if (!isRecord(node)) return null;
  const input = node.input;
  if (!isRecord(input)) return null;
  const required = isRecord(input.required) ? input.required[inputName] : undefined;
  const optional = isRecord(input.optional) ? input.optional[inputName] : undefined;
  const spec = required ?? optional;
  if (Array.isArray(spec) && Array.isArray(spec[0])) {
    return spec[0].map(item => String(item));
  }
  return null;
}

export function loraNamesFromObjectInfo(objectInfo: Record<string, unknown>) {
  return inputAllowedValues(objectInfo, 'LoraLoader', 'lora_name') || [];
}

export async function listExternalComfyLoras(serverUrl: string, fetchImpl?: FetchLike) {
  try {
    const objectInfo = await getComfyObjectInfo(serverUrl, fetchImpl);
    const objectInfoNames = loraNamesFromObjectInfo(objectInfo);
    if (objectInfoNames.length > 0) {
      return {
        source: 'object_info' as const,
        loras: Array.from(new Set(objectInfoNames)).sort((a, b) => a.localeCompare(b)),
      };
    }
  } catch {
    // Fall back to ComfyUI's model listing route below when object_info is unavailable.
  }
  try {
    const modelNames = await comfyRequest<string[]>({ serverUrl, path: '/models/loras', fetchImpl });
    return {
      source: 'models' as const,
      loras: Array.from(new Set((Array.isArray(modelNames) ? modelNames : []).map(String))).sort((a, b) =>
        a.localeCompare(b),
      ),
    };
  } catch {
    return { source: 'none' as const, loras: [] };
  }
}

function nodePreflightItems(
  workflow: Record<string, unknown>,
  objectInfo: Record<string, unknown>,
): ComfyPreflightItem[] {
  return classTypesFromWorkflow(workflow).map(classType => ({
    id: `node:${classType}`,
    label: `Node: ${classType}`,
    status: Object.prototype.hasOwnProperty.call(objectInfo, classType) ? 'found' : 'missing',
    detail: Object.prototype.hasOwnProperty.call(objectInfo, classType) ? 'Available' : 'Missing custom node',
  }));
}

function modelPreflightItem(model: IdeogramRequiredModel, objectInfo: Record<string, unknown>): ComfyPreflightItem {
  const allowed = inputAllowedValues(objectInfo, model.classType, model.inputName);
  if (!allowed) {
    return {
      id: `model:${model.id}`,
      label: `${model.label} (${model.value})`,
      status: 'unknown',
      detail: `Could not read ${model.classType}.${model.inputName} choices`,
    };
  }
  const found = allowed.includes(model.value);
  return {
    id: `model:${model.id}`,
    label: `${model.label} (${model.value})`,
    status: found ? 'found' : 'missing',
    detail: found ? 'Found' : 'Missing model file',
  };
}

export async function runIdeogramComfyPreflight({
  serverUrl,
  workflow,
  models,
  fetchImpl,
}: {
  serverUrl: string;
  workflow: Record<string, unknown>;
  models?: IdeogramRequiredModel[];
  fetchImpl?: FetchLike;
}): Promise<ComfyPreflightResult> {
  let systemStats: unknown = null;
  try {
    systemStats = await getComfySystemStats(serverUrl, fetchImpl);
    const objectInfo = await getComfyObjectInfo(serverUrl, fetchImpl);
    const nodes = nodePreflightItems(workflow, objectInfo);
    const modelItems = (models || requiredIdeogramModels()).map(model => modelPreflightItem(model, objectInfo));
    const ok = [...nodes, ...modelItems].every(item => item.status === 'found');
    return {
      ok,
      serverUrl,
      connected: true,
      systemStats,
      nodes,
      models: modelItems,
      error: null,
    };
  } catch (error) {
    return {
      ok: false,
      serverUrl,
      connected: false,
      systemStats,
      nodes: [],
      models: [],
      error: error instanceof Error ? error.message : safeJsonText(error),
    };
  }
}

export async function queueComfyPrompt({
  serverUrl,
  workflow,
  clientId,
  fetchImpl,
}: {
  serverUrl: string;
  workflow: Record<string, unknown>;
  clientId: string;
  fetchImpl?: FetchLike;
}) {
  const response = await comfyRequest<{ prompt_id?: string; number?: number }>({
    serverUrl,
    path: '/prompt',
    method: 'POST',
    body: { prompt: workflow, client_id: clientId },
    fetchImpl,
  });
  if (!response?.prompt_id) {
    throw new ExternalComfyError(`ComfyUI did not return a prompt_id: ${safeJsonText(response)}`, 502);
  }
  return {
    promptId: response.prompt_id,
    queueNumber: response.number ?? null,
    clientId,
  };
}

export async function getComfyHistory(serverUrl: string, promptId: string, fetchImpl?: FetchLike) {
  if (!promptId.trim()) throw new ExternalComfyError('promptId is required.', 400);
  return comfyRequest({ serverUrl, path: `/history/${encodeURIComponent(promptId.trim())}`, fetchImpl });
}

export function normalizeComfyHistoryEntry(history: unknown, promptId: string) {
  if (!history) return null;
  if (isRecord(history) && isRecord(history[promptId])) return history[promptId];
  if (isRecord(history) && (isRecord(history.outputs) || isRecord(history.status))) return history;
  return null;
}

export function workflowFromHistoryEntry(entry: unknown) {
  if (!isRecord(entry)) return null;
  const prompt = entry.prompt;
  if (Array.isArray(prompt) && isRecord(prompt[2])) return prompt[2] as Record<string, unknown>;
  if (isRecord(prompt)) return prompt as Record<string, unknown>;
  if (isRecord(entry.workflow)) return entry.workflow as Record<string, unknown>;
  return null;
}

export function imageRefsFromHistoryEntry(entry: unknown): ComfyImageRef[] {
  const outputs = isRecord(entry) && isRecord(entry.outputs) ? entry.outputs : {};
  return Object.values(outputs).flatMap(output => {
    if (!isRecord(output) || !Array.isArray(output.images)) return [];
    return output.images.flatMap(image => {
      if (!isRecord(image) || typeof image.filename !== 'string' || !image.filename) return [];
      return [
        {
          filename: image.filename,
          subfolder: typeof image.subfolder === 'string' ? image.subfolder : '',
          type: typeof image.type === 'string' ? image.type : 'output',
        },
      ];
    });
  });
}

export async function getComfyViewImage({
  serverUrl,
  filename,
  subfolder = '',
  type = 'output',
  fetchImpl,
}: {
  serverUrl: string;
  filename: string;
  subfolder?: string;
  type?: string;
  fetchImpl?: FetchLike;
}) {
  if (!filename.trim()) throw new ExternalComfyError('filename is required.', 400);
  return comfyRequest<ArrayBuffer>({
    serverUrl,
    path: '/view',
    query: { filename, subfolder, type },
    fetchImpl,
    expectJson: false,
  });
}

export async function interruptComfy(serverUrl: string, fetchImpl?: FetchLike) {
  return comfyRequest({ serverUrl, path: '/interrupt', method: 'POST', body: {}, fetchImpl });
}

function staysWithin(parent: string, child: string) {
  const relativePath = path.relative(parent, child);
  return relativePath === '' || (!relativePath.startsWith('..') && !path.isAbsolute(relativePath));
}

export async function copyToolkitLoraToExternalComfy({
  toolkitPath,
  loraDir,
  knownLoras,
}: {
  toolkitPath: string;
  loraDir?: string;
  knownLoras?: ToolkitLoraSummary[];
}) {
  const destinationRoot = normalizeExternalComfyLoraDir(loraDir || (await getSavedExternalComfyLoraDir()));
  if (!destinationRoot) throw new ExternalComfyError('External ComfyUI LoRA folder is not configured.', 400);

  const destinationRootReal = await fs.promises.realpath(destinationRoot).catch(() => null);
  if (!destinationRootReal) throw new ExternalComfyError('External ComfyUI LoRA folder does not exist.', 400);
  const destinationStat = await fs.promises.stat(destinationRootReal).catch(() => null);
  if (!destinationStat?.isDirectory())
    throw new ExternalComfyError('External ComfyUI LoRA folder is not a directory.', 400);

  const requestedReal = await fs.promises.realpath(path.resolve(toolkitPath || '')).catch(() => null);
  if (!requestedReal) throw new ExternalComfyError('Toolkit LoRA file was not found.', 404);
  const known = knownLoras || (await listToolkitLoras());
  const knownLora = await Promise.all(
    known.map(async lora => ({
      lora,
      realPath: await fs.promises.realpath(lora.path).catch(() => ''),
    })),
  ).then(entries => entries.find(entry => entry.realPath === requestedReal)?.lora);
  if (!knownLora) throw new ExternalComfyError('Only known Toolkit LoRAs can be copied to external ComfyUI.', 403);
  if (!requestedReal.toLowerCase().endsWith('.safetensors')) {
    throw new ExternalComfyError('Only .safetensors LoRA files can be copied.', 400);
  }

  const filename = path.basename(requestedReal);
  const destination = path.join(destinationRootReal, filename);
  if (!staysWithin(destinationRootReal, destination)) {
    throw new ExternalComfyError('Resolved LoRA destination escaped the configured folder.', 400);
  }
  if (await fs.promises.stat(destination).catch(() => null)) {
    throw new ExternalComfyError(`External ComfyUI already has ${filename}.`, 409);
  }
  await fs.promises.copyFile(requestedReal, destination);
  return {
    filename,
    destination,
    copied: true,
    lora: knownLora,
  };
}
