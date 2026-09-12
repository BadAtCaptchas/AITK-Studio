import contract from '@/domain/modelCapabilities.json';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { isRecord } from '@/server/commandInput';

export async function GET(request: Request): Promise<Response> {
  const workerID = new URL(request.url).searchParams.get('worker_id') || 'local';
  if (!isLocalWorker(workerID)) {
    try {
      const remote: unknown = await remoteJson(await getRemoteWorker(workerID), '/api/capabilities');
      if (!isRecord(remote) || remote.version !== contract.version || !Array.isArray(remote.choices))
        throw new Error('Worker capability version does not match');
      return Response.json(remote, { headers: { 'Cache-Control': 'no-store' } });
    } catch (error) {
      return Response.json(
        { error: error instanceof Error ? error.message : 'Worker capabilities unavailable' },
        { status: 502 },
      );
    }
  }
  return Response.json(
    { ...contract, platform: process.platform, deviceBackend: process.platform === 'darwin' ? 'mps' : 'cuda' },
    { headers: { 'Cache-Control': 'private, max-age=60' } },
  );
}
