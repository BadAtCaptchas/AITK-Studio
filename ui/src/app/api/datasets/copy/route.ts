import { NextResponse } from 'next/server';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { resolveDatasetDirectoryInsideRoot, isPathInside } from '@/server/remoteCaptionSecurity';
import { resolveDatasetScope } from '@/server/datasetScope';

function safeDatasetCopyName(baseName: string, suffix: string) {
  const safeBase = baseName
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80);
  const safeSuffix = suffix
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 40);
  const stamp = new Date().toISOString().replace(/-|:|T|Z|\./g, '').slice(0, 14);
  return `${safeBase || 'dataset'}_${safeSuffix || 'copy'}_${stamp}`;
}

async function uniqueDatasetPath(datasetsRoot: string, requestedName: string) {
  const root = path.resolve(datasetsRoot);
  let candidateName = requestedName;
  let candidatePath = path.resolve(root, candidateName);
  let counter = 2;
  while (fs.existsSync(candidatePath)) {
    candidateName = `${requestedName}_${counter}`;
    candidatePath = path.resolve(root, candidateName);
    counter += 1;
  }
  if (!isPathInside(root, candidatePath) || candidatePath === root) {
    throw new Error('Invalid dataset copy destination');
  }
  return { name: candidateName, path: candidatePath };
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const datasetPath = typeof body?.datasetPath === 'string' ? body.datasetPath : '';
    if (!datasetPath.trim()) {
      return NextResponse.json({ error: 'datasetPath is required' }, { status: 400 });
    }

    const { datasetsRoot } = await resolveDatasetScope(body?.project_id);
    const sourcePath = await resolveDatasetDirectoryInsideRoot(path.resolve(datasetPath), datasetsRoot);
    const sourceName = path.basename(sourcePath);
    const requestedName =
      typeof body?.name === 'string' && body.name.trim()
        ? body.name.trim()
        : safeDatasetCopyName(sourceName, typeof body?.suffix === 'string' ? body.suffix : 'copy');
    const destination = await uniqueDatasetPath(datasetsRoot, requestedName);

    await fsp.cp(sourcePath, destination.path, {
      recursive: true,
      force: false,
      errorOnExist: true,
      verbatimSymlinks: false,
    });

    return NextResponse.json(destination);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to copy dataset' },
      { status: 400 },
    );
  }
}
