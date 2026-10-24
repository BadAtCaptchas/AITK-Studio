import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { getRepoUpdateStatus, requestRepoUpdateCheck } from '@/server/updater';


export async function GET() {
  return Response.json(await getRepoUpdateStatus());
}

async function postCommand(request: Request) {
  try {
    const body = await readJsonCommand(request);
    const action = body?.action === 'apply' || body?.action === 'restart' ? body.action : 'check';
    return Response.json(await requestRepoUpdateCheck(action));
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to request update check' },
      { status: 500 },
    );
  }
}

export const POST = withCommandBoundary(postCommand);
