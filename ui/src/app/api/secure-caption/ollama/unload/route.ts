import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import {
  decryptSecureCaptionJson,
  encryptSecureCaptionJson,
  getSecureCaptionBearerToken,
  type SecureCaptionEnvelope,
} from '@/server/secureCaptionCrypto';
import { unloadOllamaModel } from '@/server/ollama';


type SecureOllamaUnloadRequest = {
  model: string;
};

async function postCommand(request: Request) {
  try {
    const token = getSecureCaptionBearerToken(request);
    if (!token) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const envelope = assertGlobalPayload(await readJsonCommand(request)) as SecureCaptionEnvelope;
    const payload = decryptSecureCaptionJson<SecureOllamaUnloadRequest>(token, 'request', envelope);
    await unloadOllamaModel(payload.model);
    const responseEnvelope = encryptSecureCaptionJson(token, 'response', envelope.jobId, envelope.itemId, {
      unloaded: true,
    });

    return Response.json(responseEnvelope);
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Secure Ollama unload failed' },
      { status: 400 },
    );
  }
}

export const POST = withCommandBoundary(postCommand);
