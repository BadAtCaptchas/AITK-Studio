import { NextResponse } from 'next/server';
import {
  deleteIdeogramWorkflowHistoryEntry,
  listIdeogramWorkflowHistory,
  setIdeogramWorkflowHistoryFavorite,
  upsertIdeogramWorkflowHistoryEntry,
} from '@/server/ideogramWorkflowHistory';
import { buildIdeogramComfyWorkflow, type IdeogramWorkflowState } from '@/utils/ideogramWorkflow';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function errorResponse(error: unknown) {
  return NextResponse.json({ error: error instanceof Error ? error.message : 'Ideogram workflow history failed.' }, { status: 500 });
}

export async function GET() {
  try {
    return NextResponse.json({ entries: await listIdeogramWorkflowHistory() });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    if (!isRecord(body?.state)) {
      return NextResponse.json({ error: 'state is required.' }, { status: 400 });
    }
    const state = body.state as IdeogramWorkflowState;
    const workflow = isRecord(body?.workflow) ? body.workflow : buildIdeogramComfyWorkflow(state);
    const result = await upsertIdeogramWorkflowHistoryEntry({
      id: typeof body?.id === 'string' ? body.id : undefined,
      title: typeof body?.title === 'string' ? body.title : undefined,
      promptId: typeof body?.promptId === 'string' ? body.promptId : typeof body?.prompt_id === 'string' ? body.prompt_id : undefined,
      serverUrl: typeof body?.serverUrl === 'string' ? body.serverUrl : typeof body?.server_url === 'string' ? body.server_url : undefined,
      state,
      workflow,
      images: Array.isArray(body?.images) ? body.images : [],
      favorite: typeof body?.favorite === 'boolean' ? body.favorite : undefined,
      status: body?.status,
    });
    return NextResponse.json(result);
  } catch (error) {
    return errorResponse(error);
  }
}

export async function PATCH(request: Request) {
  try {
    const body = await request.json();
    const id = typeof body?.id === 'string' ? body.id.trim() : '';
    if (!id) return NextResponse.json({ error: 'id is required.' }, { status: 400 });
    const result = await setIdeogramWorkflowHistoryFavorite(id, Boolean(body?.favorite));
    if (!result.entry) return NextResponse.json({ error: 'History entry not found.' }, { status: 404 });
    return NextResponse.json(result);
  } catch (error) {
    return errorResponse(error);
  }
}

export async function DELETE(request: Request) {
  try {
    const url = new URL(request.url);
    const id = url.searchParams.get('id')?.trim() || '';
    if (!id) return NextResponse.json({ error: 'id is required.' }, { status: 400 });
    const result = await deleteIdeogramWorkflowHistoryEntry(id);
    if (!result.deleted) return NextResponse.json({ error: 'History entry not found.' }, { status: 404 });
    return NextResponse.json(result);
  } catch (error) {
    return errorResponse(error);
  }
}
