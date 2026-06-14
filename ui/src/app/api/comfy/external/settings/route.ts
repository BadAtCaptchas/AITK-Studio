import { NextResponse } from 'next/server';
import {
  ExternalComfyError,
  getSavedExternalComfyLoraDir,
  getSavedExternalComfyUrl,
  saveExternalComfyLoraDir,
  saveExternalComfyUrl,
} from '@/server/externalComfy';

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
    const [serverUrl, loraDir] = await Promise.all([getSavedExternalComfyUrl(), getSavedExternalComfyLoraDir()]);
    return NextResponse.json({ serverUrl, loraDir });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const serverUrl = await saveExternalComfyUrl(body?.server_url ?? body?.serverUrl ?? '');
    const hasLoraDir = Object.prototype.hasOwnProperty.call(body || {}, 'lora_dir') || Object.prototype.hasOwnProperty.call(body || {}, 'loraDir');
    const loraDir = hasLoraDir
      ? await saveExternalComfyLoraDir(body?.lora_dir ?? body?.loraDir ?? '')
      : await getSavedExternalComfyLoraDir();
    return NextResponse.json({ serverUrl, loraDir });
  } catch (error) {
    return errorResponse(error);
  }
}
