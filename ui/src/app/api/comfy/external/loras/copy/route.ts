import { NextResponse } from 'next/server';
import { copyToolkitLoraToExternalComfy, ExternalComfyError } from '@/server/externalComfy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function errorResponse(error: unknown) {
  if (error instanceof ExternalComfyError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }
  return NextResponse.json({ error: error instanceof Error ? error.message : 'External ComfyUI LoRA copy failed.' }, { status: 500 });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const toolkitPath =
      typeof body?.toolkitPath === 'string' ? body.toolkitPath : typeof body?.toolkit_path === 'string' ? body.toolkit_path : '';
    const loraDir = typeof body?.loraDir === 'string' ? body.loraDir : typeof body?.lora_dir === 'string' ? body.lora_dir : undefined;
    const result = await copyToolkitLoraToExternalComfy({ toolkitPath, loraDir });
    return NextResponse.json(result);
  } catch (error) {
    return errorResponse(error);
  }
}
