import { NextRequest, NextResponse } from 'next/server';
import {
  decryptSecureCaptionJson,
  encryptSecureCaptionJson,
  getSecureCaptionBearerToken,
  type SecureCaptionEnvelope,
} from '@/server/secureCaptionCrypto';
import { startOllamaModelPull } from '@/server/ollama';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type SecureOllamaPullRequest = {
  model: string;
};

export async function POST(request: NextRequest) {
  try {
    const token = getSecureCaptionBearerToken(request);
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const envelope = (await request.json()) as SecureCaptionEnvelope;
    const payload = decryptSecureCaptionJson<SecureOllamaPullRequest>(token, 'request', envelope);
    const pullStatus = await startOllamaModelPull(payload.model);
    const responseEnvelope = encryptSecureCaptionJson(
      token,
      'response',
      envelope.jobId,
      envelope.itemId,
      pullStatus,
    );

    return NextResponse.json(responseEnvelope);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Secure Ollama pull failed' },
      { status: 400 },
    );
  }
}
