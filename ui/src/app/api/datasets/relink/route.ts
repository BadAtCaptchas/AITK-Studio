import { CommandInputError, isRecord, readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { relinkDecryptedDataset } from '@/server/datasetRelinking';

export const POST = withCommandBoundary(async (request: Request) => {
  const body = await readJsonCommand(request);
  if (
    typeof body.operationID !== 'string' ||
    !/^[a-f0-9]{64}$/.test(body.operationID) ||
    !['preview', 'apply'].includes(String(body.action))
  )
    throw new CommandInputError('Invalid relinking request');
  let selections: Array<{ id: string; revision: string; remoteRevision?: string }> | undefined;
  if (body.action === 'apply') {
    if (!Array.isArray(body.jobs) || body.jobs.length > 1000) throw new CommandInputError('Select jobs to relink');
    selections = body.jobs.map((value: unknown) => {
      if (
        !isRecord(value) ||
        typeof value.id !== 'string' ||
        typeof value.revision !== 'string' ||
        !Number.isFinite(Date.parse(value.revision))
      )
        throw new CommandInputError('Expected job revisions are required');
      if (
        value.remoteRevision !== undefined &&
        (typeof value.remoteRevision !== 'string' || !Number.isFinite(Date.parse(value.remoteRevision)))
      )
        throw new CommandInputError('Invalid remote revision');
      return {
        id: value.id,
        revision: value.revision,
        remoteRevision: typeof value.remoteRevision === 'string' ? value.remoteRevision : undefined,
      };
    });
    if (new Set(selections.map(value => value.id)).size !== selections.length)
      throw new CommandInputError('Duplicate job selection');
  }
  return Response.json(
    await relinkDecryptedDataset(
      body.operationID,
      typeof body.worker_id === 'string' ? body.worker_id : 'local',
      selections,
    ),
  );
});
