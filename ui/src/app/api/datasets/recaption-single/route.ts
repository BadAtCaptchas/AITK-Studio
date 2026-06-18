import { NextRequest, NextResponse } from 'next/server';
import { encryptedOpenRouterUploadImageDataUrl, plainOpenRouterImageDataUrl, positiveNumberFromValue } from '@/server/openRouterImageData';
import { assertProjectScopeEnabled, DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';
import { getOpenRouterApiKey } from '@/server/settings';
import { generateSingleImageRecaption } from '@/server/datasetSingleRecaption';
import { resolveDatasetFolder } from '@/server/encryptedDatasets';
import { readDatasetRootCaption } from '@/server/datasetRootCaption';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';

export const runtime = 'nodejs';

async function rootPromptForDataset(datasetName: string, projectID: unknown) {
  if (!datasetName.trim()) return '';
  try {
    const { datasetsRoot } = await resolveDatasetScope(projectID);
    const datasetFolder = resolveDatasetFolder(datasetsRoot, datasetName);
    const rootCaption = await readDatasetRootCaption(datasetFolder);
    return rootCaption.found ? rootCaption.systemPrompt : '';
  } catch {
    return '';
  }
}

export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get('content-type') || '';
    let provider = '';
    let model = '';
    let prompt = '';
    let systemPrompt = '';
    let outputFormat = '';
    let existingCaption = '';
    let datasetName = '';
    let projectID: unknown = null;
    let imgPath: unknown = null;
    let remoteWorkerId = '';
    let maxNewTokens: number | null = null;
    let imageDataUrl = '';

    if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData();
      projectID = formData.get('project_id');
      await assertProjectScopeEnabled(projectID);
      provider = String(formData.get('provider') || '');
      model = String(formData.get('model') || '');
      prompt = String(formData.get('prompt') || '');
      systemPrompt = String(formData.get('systemPrompt') || '');
      outputFormat = String(formData.get('outputFormat') || '');
      existingCaption = String(formData.get('existingCaption') || '');
      datasetName = String(formData.get('datasetName') || '');
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
      datasetName = typeof body?.datasetName === 'string' ? body.datasetName : '';
      projectID = body?.project_id;
      imgPath = body?.imgPath;
      remoteWorkerId = typeof body?.remoteWorkerId === 'string' ? body.remoteWorkerId : '';
      maxNewTokens = positiveNumberFromValue(body?.maxNewTokens);
      imageDataUrl = await plainOpenRouterImageDataUrl(imgPath, 'Recaption', projectID);
    }

    if (!systemPrompt.trim() && datasetName && !parseRemoteDatasetAssetRef(typeof imgPath === 'string' ? imgPath : null)) {
      systemPrompt = await rootPromptForDataset(datasetName, projectID);
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
