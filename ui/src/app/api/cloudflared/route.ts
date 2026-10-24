import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { downloadCloudflared, getCloudflaredStatus, startCloudflared, stopCloudflared } from '@/server/cloudflared';


export async function GET() {
  return Response.json(await getCloudflaredStatus());
}

async function postCommand(request: Request) {
  try {
    const body = await readJsonCommand(request);
    return Response.json(await startCloudflared({ autoDownload: Boolean(body?.autoDownload) }));
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to start cloudflared' },
      { status: 400 },
    );
  }
}

async function putCommand() {
  try {
    const download = await downloadCloudflared();
    return Response.json({
      download,
      status: await getCloudflaredStatus(),
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to download cloudflared' },
      { status: 400 },
    );
  }
}

async function deleteCommand() {
  return Response.json(await stopCloudflared());
}

export const POST = withCommandBoundary(postCommand);
export const PUT = withCommandBoundary(putCommand);
export const DELETE = withCommandBoundary(deleteCommand);
