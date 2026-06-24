import { NextResponse } from 'next/server';
import { DatasetScopeError } from '@/server/datasetScope';
import { DatasetTransferError, transferProjectDatasetsToGlobal } from '@/server/datasetTransfer';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => null);
    const payload = isRecord(body) ? body : {};
    const result = await transferProjectDatasetsToGlobal({
      sourceProjectID: payload.source_project_id,
      operation: payload.operation,
      all: payload.all,
      datasetNames: payload.dataset_names,
    });

    return NextResponse.json(result);
  } catch (error) {
    const status =
      error instanceof DatasetTransferError || error instanceof DatasetScopeError
        ? error.status
        : typeof (error as { status?: unknown })?.status === 'number'
          ? (error as { status: number }).status
          : 400;
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to transfer project datasets' },
      { status },
    );
  }
}
