import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const tokenToUse = process.env.AI_TOOLKIT_AUTH || null;
    if (tokenToUse) {
      const token = request.headers.get('Authorization')?.split(' ')[1];
      if (!token || token !== tokenToUse) {
        return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
      }
    }

    const body = await request.json();
    const { imgPath } = body;
    if (typeof imgPath !== 'string') {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();
    const normalizedImgPath = path.resolve(imgPath);
    const allowedRoots = [datasetsPath, trainingPath].map((root) => path.resolve(root) + path.sep);
    const isWithinAllowedRoot = allowedRoots.some((root) => normalizedImgPath.startsWith(root));

    // make sure the dataset path is in the image path
    if (!isWithinAllowedRoot) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // make sure it is an image
    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|flac|ogg)$/i.test(normalizedImgPath.toLowerCase())) {
      return NextResponse.json({ error: 'Not an image' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    if (!fs.existsSync(normalizedImgPath)) {
      return NextResponse.json({ success: true });
    }

    // delete it and return success
    fs.unlinkSync(normalizedImgPath);

    // check for caption
    const captionPath = normalizedImgPath.replace(/\.[^/.]+$/, '') + '.txt';
    if (fs.existsSync(captionPath)) {
      // delete caption file
      fs.unlinkSync(captionPath);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
