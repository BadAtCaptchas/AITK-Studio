import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, caption } = body;
    const datasetsPath = await getDatasetsRoot();
    const datasetsRoot = path.resolve(datasetsPath);
    const resolvedImagePath = path.resolve(imgPath);
    const relativeImagePath = path.relative(datasetsRoot, resolvedImagePath);

    // make sure the resolved image path is in the dataset path
    if (relativeImagePath.startsWith('..') || path.isAbsolute(relativeImagePath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    if (!fs.existsSync(resolvedImagePath)) {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }

    // check for caption
    const captionPath = resolvedImagePath.replace(/\.[^/.]+$/, '') + '.txt';
    // save caption to file
    fs.writeFileSync(captionPath, caption);

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
