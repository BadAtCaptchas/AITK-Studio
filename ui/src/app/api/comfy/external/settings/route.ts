import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import {
  ExternalComfyError,
  getSavedExternalComfyLoraDir,
  getSavedExternalComfyUrl,
  saveExternalComfyLoraDir,
  saveExternalComfyUrl,
} from '@/server/externalComfy';


function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return Response.json({ error: error.message }, { status: error.status });
  }
  return Response.json(
    { error: error instanceof Error ? error.message : 'External ComfyUI settings failed.' },
    { status: 500 },
  );
}

export async function GET() {
  try {
    const [serverUrl, loraDir] = await Promise.all([getSavedExternalComfyUrl(), getSavedExternalComfyLoraDir()]);
    return Response.json({ serverUrl, loraDir });
  } catch (error) {
    return errorResponse(error);
  }
}

async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const serverUrl = await saveExternalComfyUrl(body?.server_url ?? body?.serverUrl ?? '');
    const hasLoraDir =
      Object.prototype.hasOwnProperty.call(body || {}, 'lora_dir') ||
      Object.prototype.hasOwnProperty.call(body || {}, 'loraDir');
    const loraDir = hasLoraDir
      ? await saveExternalComfyLoraDir(body?.lora_dir ?? body?.loraDir ?? '')
      : await getSavedExternalComfyLoraDir();
    return Response.json({ serverUrl, loraDir });
  } catch (error) {
    return errorResponse(error);
  }
}

export const POST = withCommandBoundary(postCommand);
