/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { findEncryptedDatasetRoot } from '@/server/encryptedDatasets';

export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return new NextResponse(null, { status: 499 });
  }

  if (request.signal.aborted) {
    return new NextResponse(null, { status: 499 });
  }

  const { imgPaths } = body as { imgPaths?: string[] };
  if (!Array.isArray(imgPaths)) {
    return NextResponse.json({ error: 'imgPaths must be an array' }, { status: 400 });
  }

  const allowedRoot = path.resolve(await getDatasetsRoot());
  const captions: Record<string, string> = {};

  for (const imgPath of imgPaths) {
    if (typeof imgPath !== 'string') continue;
    const resolvedFilePath = path.resolve(imgPath);
    const relativeFilePath = path.relative(allowedRoot, resolvedFilePath);
    const isAllowed =
      relativeFilePath !== '' && !relativeFilePath.startsWith('..') && !path.isAbsolute(relativeFilePath);
    if (!isAllowed) continue;
    if (findEncryptedDatasetRoot(resolvedFilePath, allowedRoot)) continue;

    const captionPath = resolvedFilePath.replace(/\.[^/.]+$/, '') + '.txt';
    try {
      captions[imgPath] = fs.existsSync(captionPath) ? fs.readFileSync(captionPath, 'utf-8') : '';
    } catch {
      captions[imgPath] = '';
    }
  }

  return NextResponse.json({ captions });
}
