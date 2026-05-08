/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

export async function GET(request: NextRequest, { params }: { params: { filePath: string } }) {
  const { filePath } = await params;
  try {
    // Decode the path
    const decodedFilePath = decodeURIComponent(filePath);

    // Get allowed directories
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const allowedDirs = [datasetRoot, trainingRoot];

    // Check if file exists
    if (!fs.existsSync(decodedFilePath)) {
      console.warn(`File not found: ${decodedFilePath}`);
      return new NextResponse('File not found', { status: 404 });
    }

    const resolvedFilePath = fs.realpathSync(decodedFilePath);

    // Security check: Ensure canonical path is contained in canonical allowed directories
    const isAllowed = allowedDirs.some(allowedDir => {
      if (!allowedDir || !fs.existsSync(allowedDir)) {
        return false;
      }

      const resolvedAllowedDir = fs.realpathSync(allowedDir);
      const relativePath = path.relative(resolvedAllowedDir, resolvedFilePath);
      return relativePath !== '' && !relativePath.startsWith('..') && !path.isAbsolute(relativePath);
    });

    if (!isAllowed) {
      console.warn(`Access denied: ${decodedFilePath} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Get file info
    const stat = fs.statSync(decodedFilePath);
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    // Get filename for Content-Disposition
    const filename = path.basename(decodedFilePath);

    // Determine content type
    const ext = path.extname(decodedFilePath).toLowerCase();
    const contentTypeMap: { [key: string]: string } = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.webp': 'image/webp',
      '.svg': 'image/svg+xml',
      '.bmp': 'image/bmp',
      '.safetensors': 'application/octet-stream',
      '.zip': 'application/zip',
      // Videos
      '.mp4': 'video/mp4',
      '.avi': 'video/x-msvideo',
      '.mov': 'video/quicktime',
      '.mkv': 'video/x-matroska',
      '.wmv': 'video/x-ms-wmv',
      '.m4v': 'video/x-m4v',
      '.flv': 'video/x-flv',
      // Audio
      '.mp3': 'audio/mpeg',
      '.wav': 'audio/wav',
      '.flac': 'audio/flac',
      '.ogg': 'audio/ogg',
    };

    const contentType = contentTypeMap[ext];
    if (!contentType) {
      return new NextResponse('File type not allowed', { status: 403 });
    }

    // Get range header for partial content support
    const range = request.headers.get('range');

    // Common headers for better download handling
    const commonHeaders = {
      'Content-Type': contentType,
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'public, max-age=86400',
      'Content-Disposition': `attachment; filename="${encodeURIComponent(filename)}"`,
      'X-Content-Type-Options': 'nosniff',
    };

    if (range) {
      // Parse range header
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : Math.min(start + 10 * 1024 * 1024, stat.size - 1); // 10MB chunks
      const chunkSize = end - start + 1;

      const fileStream = fs.createReadStream(resolvedFilePath, {
        start,
        end,
        highWaterMark: 64 * 1024, // 64KB buffer
      });

      return new NextResponse(fileStream as any, {
        status: 206,
        headers: {
          ...commonHeaders,
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Content-Length': String(chunkSize),
        },
      });
    } else {
      // For full file download, read directly without streaming wrapper
      const fileStream = fs.createReadStream(resolvedFilePath, {
        highWaterMark: 64 * 1024, // 64KB buffer
      });

      return new NextResponse(fileStream as any, {
        headers: {
          ...commonHeaders,
          'Content-Length': String(stat.size),
        },
      });
    }
  } catch (error) {
    console.error('Error serving file:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
