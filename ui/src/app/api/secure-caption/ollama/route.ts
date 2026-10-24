import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import {
  decryptSecureCaptionJson,
  encryptSecureCaptionJson,
  getSecureCaptionBearerToken,
  type SecureCaptionEnvelope,
} from '@/server/secureCaptionCrypto';
import { generateOllamaImageCaption } from '@/server/ollama';


type SecureOllamaCaptionRequest = {
  model: string;
  prompt: string;
  systemPrompt?: string;
  imageBase64: string;
  maxNewTokens?: number;
};

async function postCommand(request: Request) {
  try {
    const token = getSecureCaptionBearerToken(request);
    if (!token) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const envelope = assertGlobalPayload(await readJsonCommand(request)) as SecureCaptionEnvelope;
    const payload = decryptSecureCaptionJson<SecureOllamaCaptionRequest>(token, 'request', envelope);
    const caption = await generateOllamaImageCaption({
      model: payload.model,
      prompt: payload.prompt,
      systemPrompt: payload.systemPrompt,
      imageBase64: payload.imageBase64,
      maxNewTokens: payload.maxNewTokens,
    });
    const responseEnvelope = encryptSecureCaptionJson(token, 'response', envelope.jobId, envelope.itemId, {
      caption,
    });

    return Response.json(responseEnvelope);
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Secure Ollama caption failed' },
      { status: 400 },
    );
  }
}

export const POST = withCommandBoundary(postCommand);
