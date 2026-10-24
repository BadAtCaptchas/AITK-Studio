import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { randomUUID } from 'crypto';
import {
  ExternalComfyError,
  queueComfyPrompt,
  resolveExternalComfyUrl,
  runIdeogramComfyPreflight,
} from '@/server/externalComfy';
import {
  buildIdeogramComfyWorkflow,
  requiredIdeogramModels,
  type IdeogramWorkflowState,
} from '@/utils/ideogramWorkflow';


function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return Response.json({ error: error.message }, { status: error.status });
  }
  return Response.json(
    { error: error instanceof Error ? error.message : 'External ComfyUI prompt failed.' },
    { status: 500 },
  );
}

async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const serverUrl = await resolveExternalComfyUrl(body?.server_url ?? body?.serverUrl);
    const state = isRecord(body?.state) ? (body.state as IdeogramWorkflowState) : undefined;
    const workflow = isRecord(body?.workflow) ? body.workflow : buildIdeogramComfyWorkflow(state);
    const clientId = typeof body?.clientId === 'string' && body.clientId.trim() ? body.clientId.trim() : randomUUID();

    const preflight = await runIdeogramComfyPreflight({
      serverUrl,
      workflow,
      models: requiredIdeogramModels(state),
    });
    if (!preflight.ok) {
      return Response.json(
        { error: preflight.error || 'External ComfyUI preflight failed.', preflight },
        { status: 400 },
      );
    }

    const queued = await queueComfyPrompt({ serverUrl, workflow, clientId });
    return Response.json({ ...queued, serverUrl });
  } catch (error) {
    return errorResponse(error);
  }
}

export const POST = withCommandBoundary(postCommand);
