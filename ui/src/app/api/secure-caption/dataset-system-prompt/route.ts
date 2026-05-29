import { NextRequest, NextResponse } from 'next/server';
import {
  getSecureCaptionSystemPrompt,
  normalizeSecureCaptionSystemPrompt,
  setSecureCaptionSystemPrompt,
  validateSecureCaptionDataset,
} from '@/server/secureCaptionSettings';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Failed to update dataset system prompt';
}

export async function GET(request: NextRequest) {
  try {
    const datasetName = request.nextUrl.searchParams.get('datasetName') || '';
    const dataset = await validateSecureCaptionDataset(datasetName);
    const systemPrompt = await getSecureCaptionSystemPrompt(dataset.datasetName);
    return NextResponse.json({ datasetName: dataset.datasetName, systemPrompt });
  } catch (error) {
    return NextResponse.json({ error: errorMessage(error) }, { status: 400 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const dataset = await validateSecureCaptionDataset(String(body?.datasetName || ''));
    const systemPrompt = await setSecureCaptionSystemPrompt(
      dataset.datasetName,
      normalizeSecureCaptionSystemPrompt(body?.systemPrompt),
    );
    return NextResponse.json({ ok: true, datasetName: dataset.datasetName, systemPrompt });
  } catch (error) {
    return NextResponse.json({ error: errorMessage(error) }, { status: 400 });
  }
}
