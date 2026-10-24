import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import {
  getSecureCaptionSystemPrompt,
  normalizeSecureCaptionSystemPrompt,
  setSecureCaptionSystemPrompt,
  validateSecureCaptionDataset,
} from '@/server/secureCaptionSettings';


function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Failed to update dataset system prompt';
}

export async function GET(request: Request) {
  try {
    const datasetName = new URL(request.url).searchParams.get('datasetName') || '';
    const dataset = await validateSecureCaptionDataset(datasetName);
    const systemPrompt = await getSecureCaptionSystemPrompt(dataset.datasetName);
    return Response.json({ datasetName: dataset.datasetName, systemPrompt });
  } catch (error) {
    return Response.json({ error: errorMessage(error) }, { status: 400 });
  }
}

async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const dataset = await validateSecureCaptionDataset(String(body?.datasetName || ''));
    const systemPrompt = await setSecureCaptionSystemPrompt(
      dataset.datasetName,
      normalizeSecureCaptionSystemPrompt(body?.systemPrompt),
    );
    return Response.json({ ok: true, datasetName: dataset.datasetName, systemPrompt });
  } catch (error) {
    return Response.json({ error: errorMessage(error) }, { status: 400 });
  }
}

export const POST = withCommandBoundary(postCommand);
