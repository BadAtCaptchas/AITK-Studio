import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath } = body;
    const datasetsPath = fs.realpathSync(await getDatasetsRoot());
    const trainingPath = fs.realpathSync(await getTrainingFolder());

    const normalizedInputPath = path.resolve(imgPath);
    const isWithin = (rootPath: string, targetPath: string) => {
      const relativePath = path.relative(rootPath, targetPath);
      return relativePath === '' || (!relativePath.startsWith('..') && !path.isAbsolute(relativePath));
    };

    // make sure the requested file is under dataset or training roots
    if (!isWithin(datasetsPath, normalizedInputPath) && !isWithin(trainingPath, normalizedInputPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // make sure it is an image
    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|flac|ogg)$/i.test(imgPath.toLowerCase())) {
      return NextResponse.json({ error: 'Not an image' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    if (!fs.existsSync(normalizedInputPath)) {
      return NextResponse.json({ success: true });
    }

    const realImagePath = fs.realpathSync(normalizedInputPath);
    if (!isWithin(datasetsPath, realImagePath) && !isWithin(trainingPath, realImagePath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // delete it and return success
    fs.unlinkSync(realImagePath);

    // check for caption
    const captionPath = realImagePath.replace(/\.[^/.]+$/, '') + '.txt';
    if (fs.existsSync(captionPath)) {
      // delete caption file
      fs.unlinkSync(captionPath);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
