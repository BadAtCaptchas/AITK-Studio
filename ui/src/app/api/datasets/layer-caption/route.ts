import { NextRequest, NextResponse } from 'next/server';
import {
  encryptedOpenRouterUploadImageDataUrl,
  nonNegativeIntegerFromValue,
  plainOpenRouterImageDataUrl,
  positiveNumberFromValue,
} from '@/server/openRouterImageData';
import { generateOpenRouterLayerCaption } from '@/server/openRouterLayerCaption';
import { generateOllamaLayerCaption, generateRemoteOllamaLayerCaption } from '@/server/ollamaVision';
import { getOpenRouterApiKey } from '@/server/settings';

export const runtime = 'nodejs';

type StudioBoxProvider = 'openrouter' | 'ollama' | 'remote_ollama';

function normalizeProvider(value: unknown): StudioBoxProvider {
  return value === 'ollama' || value === 'remote_ollama' ? value : 'openrouter';
}

function requiredElementIndex(value: unknown) {
  const elementIndex = nonNegativeIntegerFromValue(value);
  if (elementIndex == null) throw new Error('elementIndex is required.');
  return elementIndex;
}

export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get('content-type') || '';
    let caption = '';
    let model: string | null = null;
    let provider: StudioBoxProvider = 'openrouter';
    let remoteWorkerId: string | null = null;
    let elementIndex = 0;
    let imageWidth: number | null = null;
    let imageHeight: number | null = null;
    let imageDataUrl = '';

    if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData();
      caption = String(formData.get('caption') || '');
      model = String(formData.get('model') || '');
      provider = normalizeProvider(formData.get('provider'));
      remoteWorkerId = String(formData.get('remoteWorkerId') || '');
      elementIndex = requiredElementIndex(formData.get('elementIndex'));
      imageWidth = positiveNumberFromValue(formData.get('imageWidth'));
      imageHeight = positiveNumberFromValue(formData.get('imageHeight'));
      if (!imageWidth || !imageHeight) {
        throw new Error('Image width and height are required for Caption Layer.');
      }
      imageDataUrl = await encryptedOpenRouterUploadImageDataUrl(formData, 'Caption Layer');
    } else {
      const body = await request.json();
      caption = typeof body?.caption === 'string' ? body.caption : '';
      model = typeof body?.model === 'string' ? body.model : '';
      provider = normalizeProvider(body?.provider);
      remoteWorkerId = typeof body?.remoteWorkerId === 'string' ? body.remoteWorkerId : '';
      elementIndex = requiredElementIndex(body?.elementIndex);
      imageWidth = positiveNumberFromValue(body?.imageWidth);
      imageHeight = positiveNumberFromValue(body?.imageHeight);
      if (!imageWidth || !imageHeight) {
        throw new Error('Image width and height are required for Caption Layer.');
      }
      imageDataUrl = await plainOpenRouterImageDataUrl(body?.imgPath, 'Caption Layer');
    }

    const imageSize = { width: imageWidth, height: imageHeight };
    if (provider === 'ollama') {
      return NextResponse.json(await generateOllamaLayerCaption({ imageDataUrl, caption, elementIndex, model, imageSize }));
    }
    if (provider === 'remote_ollama') {
      return NextResponse.json(
        await generateRemoteOllamaLayerCaption({
          imageDataUrl,
          caption,
          elementIndex,
          model,
          imageSize,
          remoteWorkerId,
        }),
      );
    }

    return NextResponse.json(
      await generateOpenRouterLayerCaption({
        apiKey: await getOpenRouterApiKey(),
        imageDataUrl,
        caption,
        elementIndex,
        model,
        imageSize,
      }),
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to caption selected layer.' },
      { status: 400 },
    );
  }
}
