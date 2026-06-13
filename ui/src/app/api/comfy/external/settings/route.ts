import { NextResponse } from 'next/server';
import { ExternalComfyError, getSavedExternalComfyUrl, saveExternalComfyUrl } from '@/server/externalComfy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }
  return NextResponse.json({ error: error instanceof Error ? error.message : 'External ComfyUI settings failed.' }, { status: 500 });
}

export async function GET() {
  try {
    return NextResponse.json({ serverUrl: await getSavedExternalComfyUrl() });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const serverUrl = await saveExternalComfyUrl(body?.server_url ?? body?.serverUrl ?? '');
    return NextResponse.json({ serverUrl });
  } catch (error) {
    return errorResponse(error);
  }
}
