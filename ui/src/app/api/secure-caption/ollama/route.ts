import { NextRequest, NextResponse } from 'next/server';
import {
  decryptSecureCaptionJson,
  encryptSecureCaptionJson,
  getSecureCaptionBearerToken,
  type SecureCaptionEnvelope,
} from '@/server/secureCaptionCrypto';
import { generateOllamaImageCaption } from '@/server/ollama';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type SecureOllamaCaptionRequest = {
  model: string;
  prompt: string;
  systemPrompt?: string;
  imageBase64: string;
  maxNewTokens?: number;
};

export async function POST(request: NextRequest) {
  try {
    const token = getSecureCaptionBearerToken(request);
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const envelope = (await request.json()) as SecureCaptionEnvelope;
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

    return NextResponse.json(responseEnvelope);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Secure Ollama caption failed' },
      { status: 400 },
    );
  }
}
