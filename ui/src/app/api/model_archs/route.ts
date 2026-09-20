import { listExtensionUiModules } from '@/server/extensionUi';
import { isRequestAuthenticated } from '@/utils/authSession';
export const dynamic = 'force-dynamic';
export async function GET(request: Request) {
  if (!(await isRequestAuthenticated(request, process.env.AI_TOOLKIT_AUTH)))
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  return Response.json(await listExtensionUiModules(), { headers: { 'Cache-Control': 'no-store' } });
}
