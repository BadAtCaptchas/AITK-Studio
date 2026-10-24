import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import {
  deleteIdeogramWorkflowHistoryEntry,
  listIdeogramWorkflowHistory,
  setIdeogramWorkflowHistoryFavorite,
  upsertIdeogramWorkflowHistoryEntry,
} from '@/server/ideogramWorkflowHistory';
import { buildIdeogramComfyWorkflow, type IdeogramWorkflowState } from '@/utils/ideogramWorkflow';


function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function errorResponse(error: unknown) {
  return Response.json(
    { error: error instanceof Error ? error.message : 'Ideogram workflow history failed.' },
    { status: 500 },
  );
}

export async function GET() {
  try {
    return Response.json({ entries: await listIdeogramWorkflowHistory() });
  } catch (error) {
    return errorResponse(error);
  }
}

async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    if (!isRecord(body?.state)) {
      return Response.json({ error: 'state is required.' }, { status: 400 });
    }
    const state = body.state as IdeogramWorkflowState;
    const workflow = isRecord(body?.workflow) ? body.workflow : buildIdeogramComfyWorkflow(state);
    const result = await upsertIdeogramWorkflowHistoryEntry({
      id: typeof body?.id === 'string' ? body.id : undefined,
      title: typeof body?.title === 'string' ? body.title : undefined,
      promptId:
        typeof body?.promptId === 'string'
          ? body.promptId
          : typeof body?.prompt_id === 'string'
            ? body.prompt_id
            : undefined,
      serverUrl:
        typeof body?.serverUrl === 'string'
          ? body.serverUrl
          : typeof body?.server_url === 'string'
            ? body.server_url
            : undefined,
      state,
      workflow,
      images: Array.isArray(body?.images) ? body.images : [],
      favorite: typeof body?.favorite === 'boolean' ? body.favorite : undefined,
      status: body.status === 'queued' || body.status === 'completed' || body.status === 'error' || body.status === 'imported' || body.status === 'canceled' ? body.status : undefined,
    });
    return Response.json(result);
  } catch (error) {
    return errorResponse(error);
  }
}

async function patchCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const id = typeof body?.id === 'string' ? body.id.trim() : '';
    if (!id) return Response.json({ error: 'id is required.' }, { status: 400 });
    const result = await setIdeogramWorkflowHistoryFavorite(id, Boolean(body?.favorite));
    if (!result.entry) return Response.json({ error: 'History entry not found.' }, { status: 404 });
    return Response.json(result);
  } catch (error) {
    return errorResponse(error);
  }
}

async function deleteCommand(request: Request) {
  try {
    const url = new URL(request.url);
    const id = url.searchParams.get('id')?.trim() || '';
    if (!id) return Response.json({ error: 'id is required.' }, { status: 400 });
    const result = await deleteIdeogramWorkflowHistoryEntry(id);
    if (!result.deleted) return Response.json({ error: 'History entry not found.' }, { status: 404 });
    return Response.json(result);
  } catch (error) {
    return errorResponse(error);
  }
}

export const POST = withCommandBoundary(postCommand);
export const PATCH = withCommandBoundary(patchCommand);
export const DELETE = withCommandBoundary(deleteCommand);
