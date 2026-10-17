import { readJsonCommand, withCommandBoundary, isRecord, CommandInputError } from '@/server/commandInput';
import { resolveDatasetScope } from '@/server/datasetScope';
import { resolveDatasetMutationPath } from '@/server/datasetMutationPaths';
import { readEncryptedManifest, validateEncryptedCatalogKey, cleanDatasetName } from '@/server/encryptedDatasets';
import { activeOperationForResource, enqueueOperation, publicOperation, updateOperation } from '@/server/operations';
import { rememberOperationKeys } from '@/server/operationKeys';
import { getRemoteWorker, isLocalWorker, remoteJson, withoutRemoteRedirects } from '@/server/remoteClient';
import { db } from '@/server/db';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import fs from 'fs/promises';
import path from 'path';

export const POST = withCommandBoundary(async (request: Request): Promise<Response> => {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    if (
      typeof body.datasetName !== 'string' ||
      typeof body.outputName !== 'string' ||
      typeof body.keyB64 !== 'string' ||
      !/^[A-Za-z0-9+/]{43}=$/.test(body.keyB64)
    )
      throw new CommandInputError('Dataset, destination and a valid key are required');
    if (
      cleanDatasetName(body.datasetName) !== body.datasetName ||
      cleanDatasetName(body.outputName) !== body.outputName ||
      !body.outputName ||
      body.datasetName.toLowerCase() === body.outputName.toLowerCase()
    )
      throw new CommandInputError('Choose a different valid destination name');
    const workerID = typeof body.worker_id === 'string' ? body.worker_id : 'local';
    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      if (new URL(worker.base_url).protocol !== 'https:')
        throw new CommandInputError('Remote decryption requires HTTPS');
      const capability: unknown = await remoteJson(
        worker,
        '/api/capabilities',
        withoutRemoteRedirects({ method: 'GET' }),
      );
      if (!isRecord(capability) || capability.datasetDecryption !== 1)
        throw new CommandInputError('Upgrade this worker before decrypting datasets', 409);
      const result: unknown = await remoteJson(
        worker,
        '/api/datasets/decrypt',
        withoutRemoteRedirects({
          method: 'POST',
          body: JSON.stringify({ datasetName: body.datasetName, outputName: body.outputName, keyB64: body.keyB64 }),
        }),
      );
      if (!isRecord(result) || typeof result.id !== 'string') throw new Error('Invalid worker migration response');
      return Response.json({ ...result, worker_id: workerID }, { status: 202 });
    }
    const { datasetsRoot } = await resolveDatasetScope();
    const source = await resolveDatasetMutationPath(datasetsRoot, body.datasetName);
    const destination = await resolveDatasetMutationPath(datasetsRoot, body.outputName, { allowMissingLeaf: true });
    if (
      await fs.lstat(destination.entryPath).then(
        () => true,
        () => false,
      )
    ) {
      const prior = await activeOperationForResource(`dataset-decrypt:${datasetsRoot}:${body.datasetName}`);
      const metadata: unknown = await fs
        .readFile(path.join(destination.canonicalPath, '.aitk_dataset_metadata.json'), 'utf8')
        .then(JSON.parse)
        .catch(() => null);
      const resumingPublication =
        prior?.input.kind === 'dataset-decrypt' &&
        prior.input.outputName === body.outputName &&
        isRecord(metadata) &&
        isRecord(metadata.decryption) &&
        metadata.decryption.operationID === prior.id;
      if (!resumingPublication) throw new CommandInputError('Output dataset already exists', 409);
    }
    validateEncryptedCatalogKey(await readEncryptedManifest(source.canonicalPath), body.keyB64);
    const operation = await enqueueOperation({
      kind: 'dataset-decrypt',
      root: datasetsRoot,
      sourceName: body.datasetName,
      outputName: body.outputName,
    });
    rememberOperationKeys(operation.id, [{ datasetPath: body.datasetName, keyB64: body.keyB64 }]);
    if (operation.state === 'needs-keys')
      await updateOperation(operation.id, { state: 'queued', error: null, nextAttemptAt: 0 });
    await db.runtime.compareAndSwap(`operation-pending:${operation.id}`, null, operation.id);
    return Response.json({ ...publicOperation(operation), worker_id: 'local' }, { status: 202 });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Dataset decryption failed' },
      { status: error instanceof CommandInputError ? error.status : 400 },
    );
  }
});
