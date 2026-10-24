import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { createRequestedArchive, type ArchiveRequest } from '@/server/archiveDownloads';
import { DatasetScopeError } from '@/server/datasetScope';


function isArchiveRequest(value: unknown): value is ArchiveRequest {
  if (!value || typeof value !== 'object') return false;
  const target = (value as { zipTarget?: unknown }).zipTarget;
  return target === 'samples' || target === 'dataset' || target === 'dataset_captions';
}

async function postCommand(request: Request) {
  try {
    const body: unknown = assertGlobalPayload(await readJsonCommand(request));
    if (!isArchiveRequest(body)) return Response.json({ error: 'Invalid archive request' }, { status: 400 });
    return Response.json({ ok: true, ...(await createRequestedArchive(body)) });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to create archive';
    const status =
      error instanceof DatasetScopeError ? error.status : /not found|empty|no captions/i.test(message) ? 404 : 400;
    return Response.json({ error: message }, { status });
  }
}

export const POST = withCommandBoundary(postCommand);
