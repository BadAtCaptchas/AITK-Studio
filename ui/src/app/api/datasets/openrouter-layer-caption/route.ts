import { NextRequest, NextResponse } from 'next/server';
import {
  encryptedOpenRouterUploadImageDataUrl,
  nonNegativeIntegerFromValue,
  plainOpenRouterImageDataUrl,
  positiveNumberFromValue,
} from '@/server/openRouterImageData';
import { generateOpenRouterLayerCaption } from '@/server/openRouterLayerCaption';
import { getOpenRouterApiKey } from '@/server/settings';

export const runtime = 'nodejs';

function requiredElementIndex(value: unknown) {
  const elementIndex = nonNegativeIntegerFromValue(value);
  if (elementIndex == null) throw new Error('elementIndex is required.');
  return elementIndex;
}

export async function POST(request: NextRequest) {
  try {
    const apiKey = await getOpenRouterApiKey();
    const contentType = request.headers.get('content-type') || '';
    let caption = '';
    let model: string | null = null;
    let elementIndex = 0;
    let imageWidth: number | null = null;
    let imageHeight: number | null = null;
    let imageDataUrl = '';

    if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData();
      caption = String(formData.get('caption') || '');
      model = String(formData.get('model') || '');
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
      elementIndex = requiredElementIndex(body?.elementIndex);
      imageWidth = positiveNumberFromValue(body?.imageWidth);
      imageHeight = positiveNumberFromValue(body?.imageHeight);
      if (!imageWidth || !imageHeight) {
        throw new Error('Image width and height are required for Caption Layer.');
      }
      imageDataUrl = await plainOpenRouterImageDataUrl(body?.imgPath, 'Caption Layer', body?.project_id);
    }

    const result = await generateOpenRouterLayerCaption({
      apiKey,
      imageDataUrl,
      caption,
      elementIndex,
      model,
      imageSize: { width: imageWidth, height: imageHeight },
    });

    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to caption selected layer with OpenRouter.' },
      { status: 400 },
    );
  }
}
