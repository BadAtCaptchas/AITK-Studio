import { NextResponse } from 'next/server';
import {
  ExternalComfyError,
  getSavedExternalComfyLoraDir,
  listExternalComfyLoras,
  listToolkitLoras,
  resolveExternalComfyUrl,
} from '@/server/externalComfy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }
  return NextResponse.json({ error: error instanceof Error ? error.message : 'External ComfyUI LoRA discovery failed.' }, { status: 500 });
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
    return NextResponse.json({
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
