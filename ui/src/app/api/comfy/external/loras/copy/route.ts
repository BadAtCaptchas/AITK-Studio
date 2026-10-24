import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { copyToolkitLoraToExternalComfy, ExternalComfyError } from '@/server/externalComfy';


function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return Response.json({ error: error.message }, { status: error.status });
  }
  return Response.json(
    { error: error instanceof Error ? error.message : 'External ComfyUI LoRA copy failed.' },
    { status: 500 },
  );
}

async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const toolkitPath =
      typeof body?.toolkitPath === 'string'
        ? body.toolkitPath
        : typeof body?.toolkit_path === 'string'
          ? body.toolkit_path
          : '';
    const loraDir =
      typeof body?.loraDir === 'string' ? body.loraDir : typeof body?.lora_dir === 'string' ? body.lora_dir : undefined;
    const result = await copyToolkitLoraToExternalComfy({ toolkitPath, loraDir });
    return Response.json(result);
  } catch (error) {
    return errorResponse(error);
  }
}

export const POST = withCommandBoundary(postCommand);
