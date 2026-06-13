import { NextResponse } from 'next/server';
import {
  ExternalComfyError,
  getComfyHistory,
  imageRefsFromHistoryEntry,
  normalizeComfyHistoryEntry,
  resolveExternalComfyUrl,
  workflowFromHistoryEntry,
} from '@/server/externalComfy';
import { parseIdeogramComfyWorkflow } from '@/utils/ideogramWorkflow';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }
  return NextResponse.json({ error: error instanceof Error ? error.message : 'External ComfyUI import failed.' }, { status: 500 });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const promptId = typeof body?.promptId === 'string' ? body.promptId.trim() : '';
    if (!promptId) {
      return NextResponse.json({ error: 'promptId is required.' }, { status: 400 });
    }
    const serverUrl = await resolveExternalComfyUrl(body?.server_url ?? body?.serverUrl);
    const history = await getComfyHistory(serverUrl, promptId);
    const entry = normalizeComfyHistoryEntry(history, promptId);
    if (!entry) {
      return NextResponse.json({ error: `No ComfyUI history entry found for prompt ${promptId}.` }, { status: 404 });
    }
    const workflow = workflowFromHistoryEntry(entry);
    if (!workflow) {
      return NextResponse.json({ error: 'ComfyUI history entry does not include an API workflow.' }, { status: 422 });
    }
    const imported = parseIdeogramComfyWorkflow(workflow);
    return NextResponse.json({
      ...imported,
      promptId,
      serverUrl,
      images: imageRefsFromHistoryEntry(entry),
      status: typeof entry === 'object' && entry && 'status' in entry ? (entry as Record<string, unknown>).status : null,
    });
  } catch (error) {
    return errorResponse(error);
  }
}
