import { assertGlobalPayload, ObsoleteWorkspaceError } from '../utils/obsoleteWorkspaceGuard';
import type { JobStartRequest } from '../types';

export class CommandInputError extends Error {
  constructor(
    message: string,
    readonly status = 400,
    readonly code = 'INVALID_COMMAND',
  ) {
    super(message);
  }
}
export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
export function commandError(error: unknown): { error: string; code: string; status: number } | null {
  if (error instanceof CommandInputError || error instanceof ObsoleteWorkspaceError) {
    return { error: error.message, code: error.code, status: error.status };
  }
  return null;
}
const parsedCommands = new WeakMap<Request, Promise<Record<string, unknown>>>();
export function readJsonCommand(
  request: Request,
  options: { allowEmpty?: boolean; maxBytes?: number; timeoutMs?: number } = {},
): Promise<Record<string, unknown>> {
  const existing = parsedCommands.get(request);
  if (existing) return existing;
  const result = parseJsonCommand(request, options);
  parsedCommands.set(request, result);
  return result;
}

/** Applies one bounded parse before legacy handlers can catch and misclassify input errors. */
export function withCommandBoundary<R extends Request, A extends unknown[]>(
  handler: (request: R, ...args: A) => Promise<Response>,
) {
  return async (request: R, ...args: A): Promise<Response> => {
    try {
      if (
        !['GET', 'HEAD', 'OPTIONS'].includes(request.method) &&
        !/^(multipart\/form-data|application\/octet-stream)(?:\s*;|$)/i.test(request.headers.get('content-type') || '')
      ) {
        await readJsonCommand(request, {
          allowEmpty: request.method === 'DELETE' || request.body === null || !request.headers.get('content-type'),
          maxBytes: new URL(request.url).pathname === '/api/auth' ? 4096 : undefined,
        });
      }
      return await handler(request, ...args);
    } catch (error) {
      const invalid = commandError(error);
      if (!invalid) throw error;
      return Response.json({ error: invalid.error, code: invalid.code }, { status: invalid.status });
    }
  };
}

async function parseJsonCommand(
  request: Request,
  options: { allowEmpty?: boolean; maxBytes?: number; timeoutMs?: number } = {},
): Promise<Record<string, unknown>> {
  const limit = options.maxBytes ?? 2 * 1024 * 1024;
  const declared = request.headers.get('content-length');
  if (declared && (!/^\d+$/.test(declared) || !Number.isSafeInteger(Number(declared))))
    throw new CommandInputError('Invalid Content-Length');
  if (declared && Number(declared) > limit)
    throw new CommandInputError('JSON command is too large', 413, 'COMMAND_TOO_LARGE');
  const reader = request.body?.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let text = '',
    bytes = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(new CommandInputError('JSON command timed out', 408)), options.timeoutMs ?? 15000);
  });
  try {
    if (reader)
      for (;;) {
        const chunk = await Promise.race([reader.read(), deadline]);
        if (chunk.done) break;
        bytes += chunk.value.byteLength;
        if (bytes > limit) throw new CommandInputError('JSON command is too large', 413, 'COMMAND_TOO_LARGE');
        text += decoder.decode(chunk.value, { stream: true });
      }
    text += decoder.decode();
  } catch (error) {
    if (error instanceof CommandInputError) throw error;
    throw new CommandInputError('Cannot read JSON command');
  } finally {
    clearTimeout(timer);
    void reader?.cancel().catch(() => undefined);
    reader?.releaseLock();
  }
  if (bytes === 0 && options.allowEmpty) return {};
  if (!/^application\/(?:json|[a-z0-9.+-]+\+json)(?:\s*;|$)/i.test(request.headers.get('content-type') || '')) {
    throw new CommandInputError('Expected application/json', 415);
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new CommandInputError('Malformed JSON');
  }
  if (!isRecord(value)) throw new CommandInputError('Expected a JSON object');
  const pending: Array<{ value: unknown; depth: number }> = [{ value, depth: 0 }];
  let nodes = 0;
  while (pending.length) {
    const item = pending.pop()!;
    if (++nodes > 100_000 || item.depth > 32) throw new CommandInputError('JSON command is too complex', 413);
    if (Array.isArray(item.value) || isRecord(item.value)) {
      const entries = Object.values(item.value);
      if (entries.length > 10_000) throw new CommandInputError('JSON collection is too large', 413);
      for (const child of entries) pending.push({ value: child, depth: item.depth + 1 });
    }
  }
  return assertGlobalPayload(value);
}

export function parseJobStartCommand(value: Record<string, unknown>): JobStartRequest {
  const allowed = new Set(['encryptedDatasetKeys', 'durableEncryptedDatasetKeys', 'background', 'idempotencyKey']);
  if (Object.keys(value).some(key => !allowed.has(key))) throw new CommandInputError('Unknown job start field');
  for (const key of ['durableEncryptedDatasetKeys', 'background']) {
    if (value[key] !== undefined && typeof value[key] !== 'boolean') throw new CommandInputError(`Invalid ${key}`);
  }
  const keys = value.encryptedDatasetKeys;
  if (
    keys !== undefined &&
    (!Array.isArray(keys) ||
      keys.length > 1000 ||
      keys.some(
        key =>
          !isRecord(key) ||
          typeof key.datasetPath !== 'string' ||
          !key.datasetPath ||
          typeof key.keyB64 !== 'string' ||
          !/^[a-zA-Z0-9+/]{43}=$/.test(key.keyB64),
      ))
  )
    throw new CommandInputError('Invalid encryptedDatasetKeys');
  if (
    value.idempotencyKey !== undefined &&
    (typeof value.idempotencyKey !== 'string' || !/^[a-zA-Z0-9_-]{8,128}$/.test(value.idempotencyKey))
  ) {
    throw new CommandInputError('Invalid idempotencyKey');
  }
  return {
    ...(Array.isArray(keys)
      ? {
          encryptedDatasetKeys: keys.map(key => ({ datasetPath: String(key.datasetPath), keyB64: String(key.keyB64) })),
        }
      : {}),
    background: value.background === true,
    durableEncryptedDatasetKeys: value.durableEncryptedDatasetKeys === true,
    ...(typeof value.idempotencyKey === 'string' ? { idempotencyKey: value.idempotencyKey } : {}),
  };
}
