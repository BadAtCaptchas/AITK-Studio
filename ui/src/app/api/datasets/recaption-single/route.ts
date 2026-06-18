import { NextRequest, NextResponse } from 'next/server';
import { encryptedOpenRouterUploadImageDataUrl, plainOpenRouterImageDataUrl, positiveNumberFromValue } from '@/server/openRouterImageData';
import { assertProjectScopeEnabled, DatasetScopeError } from '@/server/datasetScope';
import { getOpenRouterApiKey } from '@/server/settings';
import { generateSingleImageRecaption } from '@/server/datasetSingleRecaption';

export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get('content-type') || '';
    let provider = '';
    let model = '';
    let prompt = '';
    let systemPrompt = '';
    let outputFormat = '';
    let existingCaption = '';
    let remoteWorkerId = '';
    let maxNewTokens: number | null = null;
    let imageDataUrl = '';

    if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData();
      await assertProjectScopeEnabled(formData.get('project_id'));
      provider = String(formData.get('provider') || '');
      model = String(formData.get('model') || '');
      prompt = String(formData.get('prompt') || '');
      systemPrompt = String(formData.get('systemPrompt') || '');
      outputFormat = String(formData.get('outputFormat') || '');
      existingCaption = String(formData.get('existingCaption') || '');
      remoteWorkerId = String(formData.get('remoteWorkerId') || '');
      maxNewTokens = positiveNumberFromValue(formData.get('maxNewTokens'));
      imageDataUrl = await encryptedOpenRouterUploadImageDataUrl(formData, 'Recaption');
    } else {
      const body = await request.json();
      provider = typeof body?.provider === 'string' ? body.provider : '';
      model = typeof body?.model === 'string' ? body.model : '';
      prompt = typeof body?.prompt === 'string' ? body.prompt : '';
      systemPrompt = typeof body?.systemPrompt === 'string' ? body.systemPrompt : '';
      outputFormat = typeof body?.outputFormat === 'string' ? body.outputFormat : '';
      existingCaption = typeof body?.existingCaption === 'string' ? body.existingCaption : '';
      remoteWorkerId = typeof body?.remoteWorkerId === 'string' ? body.remoteWorkerId : '';
      maxNewTokens = positiveNumberFromValue(body?.maxNewTokens);
      imageDataUrl = await plainOpenRouterImageDataUrl(body?.imgPath, 'Recaption', body?.project_id);
    }

    return NextResponse.json(
      await generateSingleImageRecaption({
        provider,
        model,
        prompt,
        systemPrompt,
        outputFormat,
        existingCaption,
        remoteWorkerId,
        maxNewTokens,
        imageDataUrl,
        openRouterApiKey: await getOpenRouterApiKey(),
      }),
    );
  } catch (error) {
    const status = error instanceof DatasetScopeError ? error.status : 400;
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to recaption image.' },
      { status },
    );
  }
}
