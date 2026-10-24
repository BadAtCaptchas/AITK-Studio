import { getComfyManagedInstallStatus, startComfyManagedInstall } from '@/server/comfyManagedInstall';


export async function GET() {
  return Response.json(await getComfyManagedInstallStatus());
}

export async function POST() {
  try {
    return Response.json(await startComfyManagedInstall());
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to start managed ComfyUI install' },
      { status: 400 },
    );
  }
}
