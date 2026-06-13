import { db } from './db';
import { flushCache } from './settings';
import {
  classTypesFromWorkflow,
  requiredIdeogramModels,
  type IdeogramRequiredModel,
} from '../utils/ideogramWorkflow';

export const COMFY_EXTERNAL_URL_KEY = 'COMFY_EXTERNAL_URL';
export const DEFAULT_EXTERNAL_COMFY_URL = 'http://127.0.0.1:8188';

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

export async function getSavedExternalComfyUrl() {
  const row = await db.settings.get(COMFY_EXTERNAL_URL_KEY);
  return normalizeExternalComfyUrl(row?.value || DEFAULT_EXTERNAL_COMFY_URL);
}

export async function saveExternalComfyUrl(value: unknown) {
  const serverUrl = normalizeExternalComfyUrl(value || DEFAULT_EXTERNAL_COMFY_URL);
  await db.settings.upsert(COMFY_EXTERNAL_URL_KEY, serverUrl);
  flushCache();
  return serverUrl;
}

export async function resolveExternalComfyUrl(value?: unknown) {
  const normalized = normalizeExternalComfyUrl(value || '');
  if (normalized) return normalized;
  const saved = await getSavedExternalComfyUrl();
  return saved;
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
    response = await fetchImpl(url, {
      method,
      signal,
      headers: body == null ? undefined : { 'Content-Type': 'application/json' },
      body: body == null ? undefined : JSON.stringify(body),
    });
  } catch (error) {
    throw new ExternalComfyError(
      `Could not reach ComfyUI at ${serverUrl}: ${error instanceof Error ? error.message : String(error)}`,
      502,
    );
  }

  if (!response.ok) {
    throw new ExternalComfyError(`ComfyUI request failed for ${path}: ${await readErrorDetail(response)}`, response.status);
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

function nodePreflightItems(workflow: Record<string, unknown>, objectInfo: Record<string, unknown>): ComfyPreflightItem[] {
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
