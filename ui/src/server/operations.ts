import { createHash, randomUUID } from 'crypto';
import { db } from './db';
import { withProcessLease } from './processLease';
import { isRecord } from './commandInput';

export type OperationInput =
  | { kind: 'remote-start'; jobID: string; durableKeys: boolean; needsEphemeralKeys: boolean; configHash: string }
  | {
      kind: 'dataset-import' | 'job-import';
      uploadID: string;
      root: string;
      chunked: boolean;
      chunksTotal: number;
      expectedBytes: number | null;
      preferredName: string | null;
      gpuIds: string | null;
    };
export type OperationState = 'queued' | 'running' | 'retrying' | 'needs-keys' | 'completed' | 'failed' | 'canceled';
export type Operation = {
  version: 1;
  id: string;
  input: OperationInput;
  resource: string;
  state: OperationState;
  phase: string;
  createdAt: string;
  updatedAt: string;
  attempts: number;
  nextAttemptAt: number;
  cancelRequested: boolean;
  checkpoint: Record<string, unknown>;
  progress: Record<string, unknown>;
  result: unknown;
  error: string | null;
};
const terminal = new Set<OperationState>(['completed', 'failed', 'canceled']);
function decode(value: unknown): Operation {
  if (
    !isRecord(value) ||
    value.version !== 1 ||
    typeof value.id !== 'string' ||
    !isRecord(value.input) ||
    !['remote-start', 'dataset-import', 'job-import'].includes(String(value.input.kind)) ||
    !isRecord(value.checkpoint) ||
    !isRecord(value.progress) ||
    typeof value.state !== 'string'
  )
    throw new Error('Invalid persisted operation');
  return value as Operation;
}
export function operationID(kind: OperationInput['kind'], key: string): string {
  return createHash('sha256')
    .update(JSON.stringify(['aitk-operation-v1', kind, key]))
    .digest('hex');
}
export async function getOperation(id: string): Promise<Operation | null> {
  if (!/^[a-f0-9]{64}$/.test(id)) return null;
  const row = await db.runtime.get(`operation:${id}`);
  return row ? decode(row.value) : null;
}
export async function updateOperation(
  id: string,
  patch: Partial<
    Pick<
      Operation,
      | 'state'
      | 'phase'
      | 'attempts'
      | 'nextAttemptAt'
      | 'cancelRequested'
      | 'checkpoint'
      | 'progress'
      | 'result'
      | 'error'
    >
  >,
): Promise<Operation> {
  for (let retry = 0; retry < 20; retry++) {
    const row = await db.runtime.get(`operation:${id}`);
    if (!row) throw new Error('Operation no longer exists');
    const current = decode(row.value);
    if (terminal.has(current.state)) return current;
    if (patch.cancelRequested && current.checkpoint.commitStarted === true)
      throw Object.assign(
        new Error(
          'Publication or remote start has begun. Wait for completion, then use the job stop or delete command.',
        ),
        { status: 409 },
      );
    if (current.cancelRequested && (patch.phase || patch.checkpoint))
      throw Object.assign(new Error('Operation canceled'), { code: 'OPERATION_CANCELED' });
    const next = {
      ...current,
      ...patch,
      checkpoint: { ...current.checkpoint, ...patch.checkpoint },
      progress: { ...current.progress, ...patch.progress },
      updatedAt: new Date().toISOString(),
    };
    if (await db.runtime.compareAndSwap(row.key, row.version, next)) return next;
  }
  throw new Error('Operation update is busy');
}
export async function activeOperationForResource(resource: string): Promise<Operation | null> {
  const row = await db.runtime.get(`operation-resource:${resource}`);
  if (!row || typeof row.value !== 'string') return null;
  const operation = await getOperation(row.value);
  return operation && !terminal.has(operation.state) ? operation : null;
}

/** The request only accepts work. The supervised worker owns execution and recovery. */
export async function enqueueOperation(input: OperationInput, key: string = randomUUID()): Promise<Operation> {
  const resource =
    input.kind === 'remote-start' ? `job:${input.jobID}` : `${input.kind}:${input.root}:${input.uploadID}`;
  return withProcessLease(`operation-resource:${resource}`, async () => {
    const existing = await activeOperationForResource(resource);
    if (existing) {
      if (JSON.stringify(existing.input) !== JSON.stringify(input))
        throw Object.assign(new Error('An operation with different input already owns this resource'), { status: 409 });
      await db.runtime.compareAndSwap(`operation-pending:${existing.id}`, null, existing.id);
      return existing;
    }
    const id = operationID(input.kind, key);
    const prior = await getOperation(id);
    if (prior) {
      if (JSON.stringify(prior.input) !== JSON.stringify(input))
        throw new Error('Idempotency key was already used with different input');
      if (!terminal.has(prior.state)) await db.runtime.compareAndSwap(`operation-pending:${prior.id}`, null, prior.id);
      return prior;
    }
    const now = new Date().toISOString();
    const operation: Operation = {
      version: 1,
      id,
      input,
      resource,
      state: 'queued',
      phase: 'prepared',
      createdAt: now,
      updatedAt: now,
      attempts: 0,
      nextAttemptAt: 0,
      cancelRequested: false,
      checkpoint: {},
      progress: {},
      result: null,
      error: null,
    };
    await db.runtime.compareAndSwap(`operation:${id}`, null, operation);
    const owner = await db.runtime.get(`operation-resource:${resource}`);
    if (!(await db.runtime.compareAndSwap(`operation-resource:${resource}`, owner?.version ?? null, id)))
      throw new Error('Operation resource changed');
    // Separate pending index keeps historical operations out of the worker's bounded scan.
    await db.runtime.compareAndSwap(`operation-pending:${id}`, null, id);
    return operation;
  });
}

export async function runPendingOperations(
  execute: (operation: Operation) => Promise<unknown>,
  cleanup?: (operation: Operation) => Promise<void>,
): Promise<void> {
  // Repair acceptance interrupted between linking the resource and writing the pending index.
  const repair = await db.runtime.get('operation-repair-cursor');
  const after = typeof repair?.value === 'string' ? repair.value : '';
  const resources = await db.runtime.list('operation-resource:', 32, after);
  for (const resource of resources) {
    if (typeof resource.value !== 'string') continue;
    const operation = await getOperation(resource.value);
    if (operation && !terminal.has(operation.state) && operation.state !== 'needs-keys')
      await db.runtime.compareAndSwap(`operation-pending:${operation.id}`, null, operation.id);
  }
  await db.runtime.compareAndSwap(
    'operation-repair-cursor',
    repair?.version ?? null,
    resources.length === 32 ? resources.at(-1)!.key : '',
  );
  const pendingCursor = await db.runtime.get('operation-pending-cursor');
  const pending = await db.runtime.list(
    'operation-pending:',
    32,
    typeof pendingCursor?.value === 'string' ? pendingCursor.value : '',
  );
  await db.runtime.compareAndSwap(
    'operation-pending-cursor',
    pendingCursor?.version ?? null,
    pending.length === 32 ? pending.at(-1)!.key : '',
  );
  await Promise.all(
    [0, 1].map(async slot => {
      for (const row of pending.filter((_row, index) => index % 2 === slot)) {
        if (typeof row.value !== 'string') continue;
        const operation = await getOperation(row.value);
        if (!operation) {
          await db.runtime.delete(row.key, row.version);
          continue;
        }
        if (terminal.has(operation.state)) {
          await cleanup?.(operation);
          await db.runtime.delete(row.key, row.version);
          continue;
        }
        if (operation.state === 'needs-keys' && !operation.cancelRequested) {
          await db.runtime.delete(row.key, row.version);
          continue;
        }
        if (operation.nextAttemptAt > Date.now() && !operation.cancelRequested) continue;
        try {
          await withProcessLease(`operation:${operation.id}`, async lease => {
            const current = await getOperation(operation.id);
            if (!current || terminal.has(current.state)) return;
            if (current.cancelRequested) {
              await updateOperation(current.id, {
                state: 'canceled',
                error: 'Canceled before the next operation phase',
              });
              return;
            }
            await updateOperation(current.id, { state: 'running', attempts: current.attempts + 1, error: null });
            try {
              const result = await execute(current);
              await lease.assertOwned();
              await updateOperation(current.id, { state: 'completed', phase: 'confirmed', result, error: null });
            } catch (error) {
              const message = error instanceof Error ? error.message : 'Operation failed';
              const needsKeys = isRecord(error) && error.code === 'OPERATION_KEYS_REQUIRED';
              const canceled = isRecord(error) && error.code === 'OPERATION_CANCELED';
              const state = canceled
                ? 'canceled'
                : needsKeys
                  ? 'needs-keys'
                  : current.attempts < 4
                    ? 'retrying'
                    : 'failed';
              await updateOperation(current.id, {
                state,
                error: needsKeys ? 'Unlock the datasets again to continue this operation.' : message,
                nextAttemptAt: Date.now() + Math.min(60_000, 1000 * 2 ** current.attempts),
              });
            }
          });
        } catch (error) {
          if (!isRecord(error) || error.status !== 409)
            console.error(
              `Operation ${operation.id} ownership check failed`,
              error instanceof Error ? error.message : 'unknown error',
            );
        }
      }
    }),
  );
}

export async function checkpointOperation(
  id: string,
  phase: string,
  checkpoint: Record<string, unknown> = {},
): Promise<Operation> {
  const operation = await getOperation(id);
  if (!operation || operation.cancelRequested)
    throw Object.assign(new Error('Operation canceled'), { code: 'OPERATION_CANCELED' });
  return updateOperation(id, { phase, checkpoint });
}

/** CAS makes cancellation and the first irreversible side effect mutually exclusive. */
export async function beginOperationCommit(id: string, phase: string): Promise<void> {
  const operation = await updateOperation(id, { phase, checkpoint: { commitStarted: true } });
  if (operation.cancelRequested || operation.state === 'canceled')
    throw Object.assign(new Error('Operation canceled'), { code: 'OPERATION_CANCELED' });
}

export function publicOperation(operation: Operation) {
  // Input and phase checkpoints are internal; never expose secret handoff metadata.
  return {
    id: operation.id,
    state: operation.state,
    phase: operation.phase,
    createdAt: operation.createdAt,
    updatedAt: operation.updatedAt,
    attempts: operation.attempts,
    canCancel: !terminal.has(operation.state) && operation.checkpoint.commitStarted !== true,
    progress: operation.progress,
    result: operation.result,
    error: operation.error,
  };
}
