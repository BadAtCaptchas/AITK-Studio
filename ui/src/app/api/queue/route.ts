import { getQueueFreshness, listQueuesForQueueApi } from '@/server/queueSync';

export async function GET(request: Request) {
  try {
    const localOnly = new URL(request.url).searchParams.get('local_only') === '1';
    const [queues, freshness] = await Promise.all([listQueuesForQueueApi(localOnly), localOnly ? [] : getQueueFreshness()]);
    return Response.json({ queues, freshness }, { headers: { 'Cache-Control': 'no-store' } });
  } catch (error) {
    console.error(error);
    return Response.json({ error: 'Failed to fetch queue' }, { status: 500 });
  }
}
