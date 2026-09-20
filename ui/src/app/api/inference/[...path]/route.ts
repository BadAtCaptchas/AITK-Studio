import { isRequestAuthenticated } from '@/utils/authSession';
import { proxyInferenceRequest } from '@/server/inferenceProxy';

export const dynamic = 'force-dynamic';
async function proxy(request: Request, { params }: { params: Promise<{ path: string[] }> }): Promise<Response> {
  if (!await isRequestAuthenticated(request, process.env.AI_TOOLKIT_AUTH)) return Response.json({ error: 'Unauthorized' }, { status: 401 });
  return proxyInferenceRequest(request, (await params).path);
}
export const GET = proxy;
export const POST = proxy;
