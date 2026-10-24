import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { ExternalComfyError, interruptComfy, resolveExternalComfyUrl } from '@/server/externalComfy';


function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return Response.json({ error: error.message }, { status: error.status });
  }
  return Response.json({ error: error instanceof Error ? error.message : 'External ComfyUI interrupt failed.' }, { status: 500 });
}

async function postCommand(request: Request) {
  try {
    const body = await readJsonCommand(request);
    const serverUrl = await resolveExternalComfyUrl(body?.server_url ?? body?.serverUrl);
    return Response.json({ serverUrl, result: await interruptComfy(serverUrl) });
  } catch (error) {
    return errorResponse(error);
  }
}

export const POST = withCommandBoundary(postCommand);
