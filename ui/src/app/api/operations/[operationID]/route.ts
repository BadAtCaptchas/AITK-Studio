import { getOperation, publicOperation, updateOperation } from '@/server/operations';
import { db } from '@/server/db';

export async function GET(_request: Request, { params }: { params: Promise<{ operationID: string }> }) {
  const operation = await getOperation((await params).operationID);
  return operation
    ? Response.json(publicOperation(operation), { headers: { 'Cache-Control': 'no-store' } })
    : Response.json({ error: 'Operation not found' }, { status: 404 });
}
export async function DELETE(_request: Request, { params }: { params: Promise<{ operationID: string }> }) {
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
