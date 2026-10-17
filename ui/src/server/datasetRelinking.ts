import fs from 'fs/promises';
import path from 'path';
import { isRecord } from './commandInput';
import { db } from './db';
import { claimJobMaintenance } from './jobAttempts';
import { getOperation } from './operations';
import { resolveDatasetScope } from './datasetScope';
import { resolveConfigPath } from './trainingJobTransfer';
import { makeRemoteDatasetRef, parseRemoteDatasetRef } from '../utils/remoteDatasetRefs';
import { getRemoteWorker, isLocalWorker, remoteJson, withoutRemoteRedirects } from './remoteClient';

export type DatasetReplacement = { field: string; from: string; to: string };
export type RelinkPreview = {
  id: string;
  name: string;
  revision: string;
  remoteRevision?: string;
  status: string;
  replacements: DatasetReplacement[];
};
const fields = new Set([
  'folder_path',
  'dataset_path',
  'control_path',
  'mask_path',
  'unconditional_path',
  'inpaint_path',
  'clip_image_path',
]);

export async function replaceDatasetReferences(
  config: unknown,
  replace: (value: string) => Promise<string | null>,
): Promise<DatasetReplacement[]> {
  const changes: DatasetReplacement[] = [];
  if (!isRecord(config) || !isRecord(config.config) || !Array.isArray(config.config.process)) return changes;
  async function entry(owner: Record<string, unknown>, key: string, field: string) {
    const value = owner[key];
    if (typeof value === 'string') {
      const next = await replace(value);
      if (next && next !== value) {
        owner[key] = next;
        changes.push({ field, from: value, to: next });
      }
    } else if (Array.isArray(value)) {
      for (let index = 0; index < value.length; index++) {
        const current: unknown = value[index];
        if (typeof current !== 'string') continue;
        const next = await replace(current);
        if (next && next !== current) {
          value[index] = next;
          changes.push({ field: `${field}[${index}]`, from: current, to: next });
        }
      }
    }
  }
  for (const [p, process] of config.config.process.entries()) {
    if (!isRecord(process)) continue;
    if (Array.isArray(process.datasets))
      for (const [d, dataset] of process.datasets.entries()) {
        if (!isRecord(dataset)) continue;
        for (const key of fields) await entry(dataset, key, `config.process[${p}].datasets[${d}].${key}`);
        if (changes.some(change => change.field.startsWith(`config.process[${p}].datasets[${d}].`)))
          delete dataset.encrypted;
      }
    if (isRecord(process.caption))
      await entry(process.caption, 'path_to_caption', `config.process[${p}].caption.path_to_caption`);
  }
  return changes;
}

export async function relinkDecryptedDataset(
  operationID: string,
  workerID: string,
  selections?: Array<{ id: string; revision: string; remoteRevision?: string }>,
) {
  const remote = !isLocalWorker(workerID);
  const worker = remote ? await getRemoteWorker(workerID) : null;
  if (worker && new URL(worker.base_url).protocol !== 'https:') throw new Error('Remote migration requires HTTPS');
  const operation: unknown = worker
    ? await remoteJson(
        worker,
        `/api/operations/${encodeURIComponent(operationID)}`,
        withoutRemoteRedirects({ method: 'GET' }),
      )
    : await getOperation(operationID);
  if (
    !isRecord(operation) ||
    operation.state !== 'completed' ||
    !isRecord(operation.result) ||
    typeof operation.result.sourceName !== 'string' ||
    typeof operation.result.datasetName !== 'string'
  )
    throw new Error('Complete dataset decryption before relinking jobs');
  if (!remote && (!isRecord(operation.input) || operation.input.kind !== 'dataset-decrypt'))
    throw new Error('Not a dataset decryption operation');
  if (remote && operation.kind !== 'dataset-decrypt') throw new Error('Not a dataset decryption operation');
  const { sourceName, datasetName } = operation.result;
  const { datasetsRoot } = await resolveDatasetScope();
  if (!remote && isRecord(operation.input) && operation.input.root !== datasetsRoot)
    throw new Error('Dataset root changed since migration');
  const source = remote ? '' : await fs.realpath(path.join(datasetsRoot, sourceName));
  const destination = remote ? '' : await fs.realpath(path.join(datasetsRoot, datasetName));
  const remotePreview: unknown = worker
    ? await remoteJson(
        worker,
        '/api/datasets/relink',
        withoutRemoteRedirects({ method: 'POST', body: JSON.stringify({ action: 'preview', operationID }) }),
      )
    : null;
  const remoteJobs: unknown[] = isRecord(remotePreview) && Array.isArray(remotePreview.jobs) ? remotePreview.jobs : [];
  const replace = async (value: string): Promise<string | null> => {
    const ref = parseRemoteDatasetRef(value);
    if (ref)
      return ref.workerID === workerID && ref.datasetName === sourceName
        ? makeRemoteDatasetRef(workerID, datasetName)
        : null;
    if (remote || /^[a-z][a-z0-9+.-]*:\/\//i.test(value)) return null;
    const canonical = await fs.realpath(resolveConfigPath(value)).catch(() => null);
    return canonical === source ? destination : null;
  };
  const jobs = selections
    ? await Promise.all(selections.map(selection => db.jobs.findById(selection.id)))
    : await db.jobs.list({});
  const previews: RelinkPreview[] = [];
  const outcomes: Array<{ id: string; success: boolean; error?: string }> = [];
  for (const job of jobs) {
    if (!job) continue;
    try {
      const config: unknown = JSON.parse(job.job_config);
      const candidate = remoteJobs.find(value => isRecord(value) && value.id === job.remote_job_id);
      const remoteRevision =
        isRecord(candidate) && typeof candidate.revision === 'string' ? candidate.revision : undefined;
      const remoteReplacements: unknown[] =
        isRecord(candidate) && Array.isArray(candidate.replacements) ? candidate.replacements : [];
      const replacements = await replaceDatasetReferences(config, async value => {
        const direct = await replace(value);
        if (direct || job.worker_id !== workerID || !job.remote_job_id) return direct;
        const replacement = remoteReplacements.find(change => isRecord(change) && change.from === value);
        return isRecord(replacement) && typeof replacement.to === 'string' ? replacement.to : null;
      });
      if (!replacements.length) continue;
      if (!selections) {
        previews.push({
          id: job.id,
          name: job.name,
          status: job.status,
          revision: new Date(job.updated_at).toISOString(),
          remoteRevision,
          replacements,
        });
        continue;
      }
      const selected = selections.find(selection => selection.id === job.id);
      if (selected?.revision !== new Date(job.updated_at).toISOString())
        throw new Error('Job changed since preview. Refresh and try again.');
      // Remote mirrors are updated through the owning worker's same conditional migration operation.
      const editing = await claimJobMaintenance(job, 'editing');
      if (!editing) throw new Error('Job is active or changed while relinking');
      try {
        if (job.remote_job_id) {
          if (!worker || job.worker_id !== workerID) throw new Error('Relink this job on its owning worker');
          if (!remoteRevision || selected?.remoteRevision !== remoteRevision)
            throw new Error('Remote job changed since preview. Refresh and try again.');
          const applied: unknown = await remoteJson(
            worker,
            '/api/datasets/relink',
            withoutRemoteRedirects({
              method: 'POST',
              body: JSON.stringify({
                action: 'apply',
                operationID,
                jobs: [{ id: job.remote_job_id, revision: remoteRevision }],
              }),
            }),
          );
          if (
            !isRecord(applied) ||
            !Array.isArray(applied.outcomes) ||
            !applied.outcomes.some(
              (value: unknown) => isRecord(value) && value.id === job.remote_job_id && value.success === true,
            )
          )
            throw new Error('Worker could not relink this job; refresh its state');
        }
        const saved = await db.jobs.updateIf(
          job.id,
          { attempt_id: editing.attempt_id ?? null, status: 'editing', updated_at: new Date(editing.updated_at) },
          {
            job_config: JSON.stringify(config),
            status: job.status === 'queued' ? 'stopped' : job.status,
            stop: false,
            return_to_queue: false,
          },
        );
        if (!saved) throw new Error('Job changed while relinking');
        outcomes.push({ id: job.id, success: true });
      } catch (error) {
        await db.jobs.updateIf(
          job.id,
          { attempt_id: editing.attempt_id ?? null, status: 'editing' },
          { status: job.status === 'queued' ? 'stopped' : job.status },
        );
        throw error;
      }
    } catch (error) {
      if (selections)
        outcomes.push({ id: job.id, success: false, error: error instanceof Error ? error.message : 'Relink failed' });
    }
  }
  if (selections)
    for (const selected of selections)
      if (!outcomes.some(outcome => outcome.id === selected.id))
        outcomes.push({
          id: selected.id,
          success: false,
          error: 'Job missing or no longer references the source dataset',
        });
  return { jobs: previews, outcomes };
}
