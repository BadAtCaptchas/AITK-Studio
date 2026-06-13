import { NextResponse } from 'next/server';
import { ExternalComfyError, getComfyViewImage, resolveExternalComfyUrl } from '@/server/externalComfy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }
  return NextResponse.json({ error: error instanceof Error ? error.message : 'External ComfyUI image fetch failed.' }, { status: 500 });
}

function imageContentType(filename: string) {
  const lower = filename.toLowerCase();
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
  if (lower.endsWith('.webp')) return 'image/webp';
  return 'image/png';
}

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const filename = url.searchParams.get('filename') || '';
    const subfolder = url.searchParams.get('subfolder') || '';
    const type = url.searchParams.get('type') || 'output';
    const serverUrl = await resolveExternalComfyUrl(url.searchParams.get('server_url') || url.searchParams.get('serverUrl'));
    const image = await getComfyViewImage({ serverUrl, filename, subfolder, type });
    return new Response(image, {
      headers: {
        'Content-Type': imageContentType(filename),
        'Cache-Control': 'no-store',
      },
    });
  } catch (error) {
    return errorResponse(error);
  }
}
