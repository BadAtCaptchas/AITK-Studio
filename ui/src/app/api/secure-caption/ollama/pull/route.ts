import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import {
  decryptSecureCaptionJson,
  encryptSecureCaptionJson,
  getSecureCaptionBearerToken,
  type SecureCaptionEnvelope,
} from '@/server/secureCaptionCrypto';
import { startOllamaModelPull } from '@/server/ollama';


type SecureOllamaPullRequest = {
  model: string;
};

async function postCommand(request: Request) {
  try {
    const token = getSecureCaptionBearerToken(request);
    if (!token) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const envelope = assertGlobalPayload(await readJsonCommand(request)) as SecureCaptionEnvelope;
    const payload = decryptSecureCaptionJson<SecureOllamaPullRequest>(token, 'request', envelope);
    const pullStatus = await startOllamaModelPull(payload.model);
    const responseEnvelope = encryptSecureCaptionJson(token, 'response', envelope.jobId, envelope.itemId, pullStatus);

    return Response.json(responseEnvelope);
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Secure Ollama pull failed' },
      { status: 400 },
    );
  }
}

export const POST = withCommandBoundary(postCommand);
