import { randomUUID } from 'crypto';
import { NextResponse } from 'next/server';
import {
  ExternalComfyError,
  queueComfyPrompt,
  resolveExternalComfyUrl,
  runIdeogramComfyPreflight,
} from '@/server/externalComfy';
import { buildIdeogramComfyWorkflow, requiredIdeogramModels, type IdeogramWorkflowState } from '@/utils/ideogramWorkflow';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }
  return NextResponse.json({ error: error instanceof Error ? error.message : 'External ComfyUI prompt failed.' }, { status: 500 });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
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
      return NextResponse.json({ error: preflight.error || 'External ComfyUI preflight failed.', preflight }, { status: 400 });
    }

    const queued = await queueComfyPrompt({ serverUrl, workflow, clientId });
    return NextResponse.json({ ...queued, serverUrl });
  } catch (error) {
    return errorResponse(error);
  }
}
