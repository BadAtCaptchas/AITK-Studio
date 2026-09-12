import { internalOperationAuthorized, readOperationKeys } from '@/server/operationKeys';

export async function GET(request: Request, { params }: { params: Promise<{ operationID: string }> }) {
  if (!internalOperationAuthorized(request.headers)) return Response.json({ error: 'Unauthorized' }, { status: 401 });
  const value = await readOperationKeys((await params).operationID);
  return Response.json(value ? { encryptedDatasetKeys: value } : { error: 'Unlock required' }, {
    status: value ? 200 : 409,
    headers: { 'Cache-Control': 'no-store' },
  });
}
