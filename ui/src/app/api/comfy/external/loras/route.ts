import {
  ExternalComfyError,
  getSavedExternalComfyLoraDir,
  listExternalComfyLoras,
  listToolkitLoras,
  resolveExternalComfyUrl,
} from '@/server/externalComfy';


function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return Response.json({ error: error.message }, { status: error.status });
  }
  return Response.json({ error: error instanceof Error ? error.message : 'External ComfyUI LoRA discovery failed.' }, { status: 500 });
}

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const serverUrl = await resolveExternalComfyUrl(url.searchParams.get('server_url') || url.searchParams.get('serverUrl'));
    const [external, toolkitLoras, loraDir] = await Promise.all([
      listExternalComfyLoras(serverUrl),
      listToolkitLoras(),
      getSavedExternalComfyLoraDir(),
    ]);
    return Response.json({
      serverUrl,
      externalLoras: external.loras,
      externalSource: external.source,
      toolkitLoras,
      loraDir,
    });
  } catch (error) {
    return errorResponse(error);
  }
}
