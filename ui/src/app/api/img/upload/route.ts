// src/app/api/datasets/upload/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { getDataRoot } from '@/server/settings';
import {v4 as uuidv4} from 'uuid';

const MAX_REQUEST_BYTES = 5 * 1024 * 1024 * 1024; // 5GB
const MAX_FILE_BYTES = 1 * 1024 * 1024 * 1024; // 1GB per file

export async function POST(request: NextRequest) {
  try {
    const contentLength = Number(request.headers.get('content-length') || 0);
    if (contentLength > MAX_REQUEST_BYTES) {
      return NextResponse.json({ error: 'Request body too large' }, { status: 413 });
    }

    const dataRoot = await getDataRoot();
    if (!dataRoot) {
      return NextResponse.json({ error: 'Data root path not found' }, { status: 500 });
    }
    const imgRoot = join(dataRoot, 'images');


    const formData = await request.formData();
    const files = formData.getAll('files');

    if (!files || files.length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }

    // make it recursive if it doesn't exist
    await mkdir(imgRoot, { recursive: true });
    let totalBytes = 0;
    const savedFiles = await Promise.all(
      files.map(async (file: any) => {
        if (typeof file?.size === 'number' && file.size > MAX_FILE_BYTES) {
          throw new Error('File too large');
        }

        const bytes = await file.arrayBuffer();
        totalBytes += bytes.byteLength;
        if (totalBytes > MAX_REQUEST_BYTES) {
          throw new Error('Request body too large');
        }
        const buffer = Buffer.from(bytes);

        const extension = file.name.split('.').pop() || 'jpg';

        // Clean filename and ensure it's unique
        const fileName = `${uuidv4()}`; // Use UUID for unique file names
        const filePath = join(imgRoot, `${fileName}.${extension}`);

        await writeFile(filePath, buffer);
        return filePath;
      }),
    );

    return NextResponse.json({
      message: 'Files uploaded successfully',
      files: savedFiles,
    });
  } catch (error) {
    console.error('Upload error:', error);
    if (error instanceof Error) {
      if (error.message === 'File too large' || error.message === 'Request body too large') {
        return NextResponse.json({ error: error.message }, { status: 413 });
      }
    }
    return NextResponse.json({ error: 'Error uploading files' }, { status: 500 });
  }
}

// Increase payload size limit (default is 4mb)
export const config = {
  api: {
    bodyParser: false,
    responseLimit: '50mb',
  },
};
