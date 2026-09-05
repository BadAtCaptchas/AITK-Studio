import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { NextRequest, NextResponse } from 'next/server';
import { createRequestedArchive, type ArchiveRequest } from '@/server/archiveDownloads';
import { DatasetScopeError } from '@/server/datasetScope';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function isArchiveRequest(value: unknown): value is ArchiveRequest {
  if (!value || typeof value !== 'object') return false;
  const target = (value as { zipTarget?: unknown }).zipTarget;
  return target === 'samples' || target === 'dataset' || target === 'dataset_captions';
}

export async function POST(request: NextRequest) {
  try {
    const body: unknown = assertGlobalPayload(await request.json());
    if (!isArchiveRequest(body)) return NextResponse.json({ error: 'Invalid archive request' }, { status: 400 });
    return NextResponse.json({ ok: true, ...(await createRequestedArchive(body)) });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to create archive';
    const status =
      error instanceof DatasetScopeError ? error.status : /not found|empty|no captions/i.test(message) ? 404 : 400;
    return NextResponse.json({ error: message }, { status });
  }
}
