import { NextRequest, NextResponse } from 'next/server';
import {
  getTrainingJobExportProgress,
  requestTrainingJobExportCancellation,
} from '@/server/trainingJobExportProgress';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ jobID: string; exportID: string }> },
) {
  const { jobID, exportID } = await params;
  const progress = getTrainingJobExportProgress(exportID);

  if (!progress || progress.jobID !== jobID) {
    return NextResponse.json({ error: 'Export not found' }, { status: 404 });
  }

  return NextResponse.json(progress);
}

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ jobID: string; exportID: string }> },
) {
  const { jobID, exportID } = await params;
  const progress = getTrainingJobExportProgress(exportID);

  if (!progress || progress.jobID !== jobID) {
    return NextResponse.json({ error: 'Export not found' }, { status: 404 });
  }

  const updated = requestTrainingJobExportCancellation(exportID);
  return NextResponse.json(updated || progress);
}
