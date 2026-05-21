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
    // Client aborted the request before body was fully sent
    return new NextResponse(null, { status: 499 });
  }

  if (request.signal.aborted) {
    return new NextResponse(null, { status: 499 });
  }

  const { imgPath } = body;
  console.log('Received POST request for caption:', imgPath);
  try {
    // Decode the path
    const filepath = imgPath;
    console.log('Decoded image path:', filepath);

    // Get allowed directories
    const allowedDir = await getDatasetsRoot();

    const resolvedFilePath = path.resolve(filepath);
    const allowedRoot = path.resolve(allowedDir);
    const relativeFilePath = path.relative(allowedRoot, resolvedFilePath);
    const isAllowed = relativeFilePath !== '' && !relativeFilePath.startsWith('..') && !path.isAbsolute(relativeFilePath);

    if (!isAllowed) {
      console.warn(`Access denied: ${filepath} not in ${allowedDir}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    if (findEncryptedDatasetRoot(resolvedFilePath, allowedRoot)) {
      return new NextResponse('Encrypted captions are not served through this route', { status: 403 });
    }

    // caption name is the filepath without extension but with .txt
    const captionPath = resolvedFilePath.replace(/\.[^/.]+$/, '') + '.txt';

    // Check if file exists
    if (!fs.existsSync(captionPath)) {
      // send back blank string if caption file does not exist
      return new NextResponse('');
    }

    // Read caption file
    const caption = fs.readFileSync(captionPath, 'utf-8');

    // Return caption
    return new NextResponse(caption);
  } catch (error) {
    console.error('Error getting caption:', error);
    return new NextResponse('Error getting caption', { status: 500 });
  }
}
