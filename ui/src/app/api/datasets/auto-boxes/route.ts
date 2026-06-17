import { NextRequest, NextResponse } from 'next/server';
import {
  boolFromValue,
  encryptedOpenRouterUploadImageDataUrl,
  plainOpenRouterImageDataUrl,
  positiveNumberFromValue,
} from '@/server/openRouterImageData';
import { generateOpenRouterBoxPatches } from '@/server/openRouterBoxes';
import { generateOllamaBoxPatches, generateRemoteOllamaBoxPatches } from '@/server/ollamaVision';
import { getOpenRouterApiKey } from '@/server/settings';

export const runtime = 'nodejs';

type StudioBoxProvider = 'openrouter' | 'ollama' | 'remote_ollama';

function normalizeProvider(value: unknown): StudioBoxProvider {
  return value === 'ollama' || value === 'remote_ollama' ? value : 'openrouter';
}

export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get('content-type') || '';
    let caption = '';
    let model: string | null = null;
    let provider: StudioBoxProvider = 'openrouter';
    let remoteWorkerId: string | null = null;
    let refine = false;
    let imageWidth: number | null = null;
    let imageHeight: number | null = null;
    let imageDataUrl = '';

    if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData();
      caption = String(formData.get('caption') || '');
      model = String(formData.get('model') || '');
      provider = normalizeProvider(formData.get('provider'));
      remoteWorkerId = String(formData.get('remoteWorkerId') || '');
      refine = boolFromValue(formData.get('refine'));
      imageWidth = positiveNumberFromValue(formData.get('imageWidth'));
      imageHeight = positiveNumberFromValue(formData.get('imageHeight'));
      if (!imageWidth || !imageHeight) {
        throw new Error('Image width and height are required for Auto Boxes.');
      }
      imageDataUrl = await encryptedOpenRouterUploadImageDataUrl(formData, 'Auto Boxes');
    } else {
      const body = await request.json();
      caption = typeof body?.caption === 'string' ? body.caption : '';
      model = typeof body?.model === 'string' ? body.model : '';
      provider = normalizeProvider(body?.provider);
      remoteWorkerId = typeof body?.remoteWorkerId === 'string' ? body.remoteWorkerId : '';
      refine = body?.refine === true;
      imageWidth = positiveNumberFromValue(body?.imageWidth);
      imageHeight = positiveNumberFromValue(body?.imageHeight);
      if (!imageWidth || !imageHeight) {
        throw new Error('Image width and height are required for Auto Boxes.');
      }
      imageDataUrl = await plainOpenRouterImageDataUrl(body?.imgPath, 'Auto Boxes', body?.project_id);
    }

    const imageSize = { width: imageWidth, height: imageHeight };
    if (provider === 'ollama') {
      return NextResponse.json(await generateOllamaBoxPatches({ imageDataUrl, caption, model, refine, imageSize }));
    }
    if (provider === 'remote_ollama') {
      return NextResponse.json(
        await generateRemoteOllamaBoxPatches({
          imageDataUrl,
          caption,
          model,
          refine,
          imageSize,
          remoteWorkerId,
        }),
      );
    }

    return NextResponse.json(
      await generateOpenRouterBoxPatches({
        apiKey: await getOpenRouterApiKey(),
        imageDataUrl,
        caption,
        model,
        refine,
        imageSize,
      }),
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to create boxes.' },
      { status: 400 },
    );
  }
}
