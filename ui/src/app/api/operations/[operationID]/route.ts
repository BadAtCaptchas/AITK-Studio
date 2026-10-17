import { getRemoteWorker, isLocalWorker, remoteJson, withoutRemoteRedirects } from '@/server/remoteClient';
import { getOperation, publicOperation, updateOperation } from '@/server/operations';
import { db } from '@/server/db';

export async function GET(request: Request, { params }: { params: Promise<{ operationID: string }> }) {
  const workerID = new URL(request.url).searchParams.get('worker_id');
  if (workerID && !isLocalWorker(workerID)) {
    const worker = await getRemoteWorker(workerID);
    if (new URL(worker.base_url).protocol !== 'https:')
      return Response.json({ error: 'Migration requires HTTPS' }, { status: 400 });
    return Response.json(
      await remoteJson(
        worker,
        '/api/operations/' + encodeURIComponent((await params).operationID),
        withoutRemoteRedirects({ method: 'GET' }),
      ),
    );
  }
  const operation = await getOperation((await params).operationID);
  return operation
    ? Response.json(publicOperation(operation), { headers: { 'Cache-Control': 'no-store' } })
    : Response.json({ error: 'Operation not found' }, { status: 404 });
}
export async function DELETE(request: Request, { params }: { params: Promise<{ operationID: string }> }) {
  const workerID = new URL(request.url).searchParams.get('worker_id');
  if (workerID && !isLocalWorker(workerID)) {
    const worker = await getRemoteWorker(workerID);
    if (new URL(worker.base_url).protocol !== 'https:')
      return Response.json({ error: 'Migration requires HTTPS' }, { status: 400 });
    return Response.json(
      await remoteJson(
        worker,
        '/api/operations/' + encodeURIComponent((await params).operationID),
        withoutRemoteRedirects({ method: 'DELETE' }),
      ),
    );
  }
  const operation = await getOperation((await params).operationID);
  if (!operation) return Response.json({ error: 'Operation not found' }, { status: 404 });
  let updated;
  try {
    updated = await updateOperation(operation.id, { cancelRequested: true });
  } catch (error) {
    if (error instanceof Error && 'status' in error && error.status === 409)
      return Response.json({ error: error.message }, { status: 409 });
    throw error;
  }
  await db.runtime.compareAndSwap(`operation-pending:${operation.id}`, null, operation.id);
  return Response.json(publicOperation(updated), { status: 202 });
}
