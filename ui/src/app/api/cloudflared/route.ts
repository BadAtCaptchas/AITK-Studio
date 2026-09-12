import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { NextRequest, NextResponse } from 'next/server';
import { downloadCloudflared, getCloudflaredStatus, startCloudflared, stopCloudflared } from '@/server/cloudflared';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json(await getCloudflaredStatus());
}

async function postCommand(request: NextRequest) {
  try {
    const body = await readJsonCommand(request);
    return NextResponse.json(await startCloudflared({ autoDownload: Boolean(body?.autoDownload) }));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to start cloudflared' },
      { status: 400 },
    );
  }
}

async function putCommand() {
  try {
    const download = await downloadCloudflared();
    return NextResponse.json({
      download,
      status: await getCloudflaredStatus(),
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to download cloudflared' },
      { status: 400 },
    );
  }
}

async function deleteCommand() {
  return NextResponse.json(await stopCloudflared());
}

export const POST = withCommandBoundary(postCommand);
export const PUT = withCommandBoundary(putCommand);
export const DELETE = withCommandBoundary(deleteCommand);
