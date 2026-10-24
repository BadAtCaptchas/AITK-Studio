import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { JobStartError } from '@/server/jobStart';
import { restartTrainingJobFromScratch, TrainingJobRestartError } from '@/server/trainingJobRestart';
import type { JobStartRequest } from '@/types';
import { isRequestAuthenticated } from '@/utils/authSession';

async function ensureApiAccess(request: Request): Promise<Response | null> {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) {
    return null;
  }

  if (!(await isRequestAuthenticated(request, tokenToUse))) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

function handleRestartError(error: unknown) {
  if (error instanceof JobStartError || error instanceof TrainingJobRestartError) {
    return Response.json(error.payload, { status: error.status });
  }
  throw error;
}

async function postCommand(request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const accessResponse = await ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  let body: JobStartRequest = {};
  try {
    body = assertGlobalPayload(await readJsonCommand(request));
  } catch {
    body = {};
  }

  const { jobID } = await params;

  try {
    return Response.json(
      await restartTrainingJobFromScratch(jobID, {
        encryptedDatasetKeys: body.encryptedDatasetKeys,
        durableEncryptedDatasetKeys: body.durableEncryptedDatasetKeys === true,
      }),
    );
  } catch (error) {
    return handleRestartError(error);
  }
}

export const POST = withCommandBoundary(postCommand);
