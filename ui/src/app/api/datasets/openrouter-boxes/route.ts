import { NextRequest, NextResponse } from 'next/server';
import {
  boolFromValue,
  encryptedOpenRouterUploadImageDataUrl,
  plainOpenRouterImageDataUrl,
  positiveNumberFromValue,
} from '@/server/openRouterImageData';
import { generateOpenRouterBoxPatches } from '@/server/openRouterBoxes';
import { getOpenRouterApiKey } from '@/server/settings';
import { assertProjectScopeEnabled, DatasetScopeError } from '@/server/datasetScope';

export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  try {
    const apiKey = await getOpenRouterApiKey();
    const contentType = request.headers.get('content-type') || '';
    let caption = '';
    let model: string | null = null;
    let refine = false;
    let imageWidth: number | null = null;
    let imageHeight: number | null = null;
    let imageDataUrl = '';

    if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData();
      await assertProjectScopeEnabled(formData.get('project_id'));
      caption = String(formData.get('caption') || '');
      model = String(formData.get('model') || '');
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
      refine = body?.refine === true;
      imageWidth = positiveNumberFromValue(body?.imageWidth);
      imageHeight = positiveNumberFromValue(body?.imageHeight);
      if (!imageWidth || !imageHeight) {
        throw new Error('Image width and height are required for Auto Boxes.');
      }
      imageDataUrl = await plainOpenRouterImageDataUrl(body?.imgPath, 'Auto Boxes', body?.project_id);
    }

    const result = await generateOpenRouterBoxPatches({
      apiKey,
      imageDataUrl,
      caption,
      model,
      refine,
      imageSize: { width: imageWidth, height: imageHeight },
    });

    return NextResponse.json(result);
  } catch (error) {
    const status = error instanceof DatasetScopeError ? error.status : 400;
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to create boxes with OpenRouter.' },
      { status },
    );
  }
}
