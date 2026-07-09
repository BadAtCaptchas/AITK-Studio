/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { isRegisteredProjectPath } from '@/server/projectMediaSecurity';

async function realpathIfExists(filePath: string) {
  try {
    return await fs.promises.realpath(filePath);
  } catch {
    return null;
  }
}

function parseRange(value: string | null, size: number) {
  if (!value) return null;
  const match = /^bytes=(\d*)-(\d*)$/i.exec(value.trim());
  if (!match || (!match[1] && !match[2])) return 'invalid' as const;
  let start: number;
  let end: number;
  if (!match[1]) {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return 'invalid' as const;
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] ? Number(match[2]) : Math.min(start + 10 * 1024 * 1024, size - 1);
  }
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || start >= size) {
    return 'invalid' as const;
  }
  return { start, end: Math.min(end, size - 1) };
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ filePath: string[] }> }) {
  const { filePath } = await params;
  try {
    // Decode the path
    const decodedFilePath = decodeURIComponent(filePath.join('/'));

    // Get allowed directories
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const allowedDirs = [datasetRoot, trainingRoot].filter((dir): dir is string => !!dir);

    const resolvedFilePath = await realpathIfExists(decodedFilePath);
    if (!resolvedFilePath) {
      console.warn(`File not found: ${decodedFilePath}`);
      return new NextResponse('File not found', { status: 404 });
    }

    if (await isRegisteredProjectPath(resolvedFilePath)) {
      return new NextResponse('Project files require a signed project-relative URL', { status: 403 });
    }

    // Security check: Ensure canonical path is contained in canonical allowed directories
    const allowedChecks = await Promise.all(
      allowedDirs.map(async allowedDir => {
        const resolvedAllowedDir = await realpathIfExists(allowedDir);
        if (!resolvedAllowedDir) return false;
        const relativePath = path.relative(resolvedAllowedDir, resolvedFilePath);
        return relativePath !== '' && !relativePath.startsWith('..') && !path.isAbsolute(relativePath);
      }),
    );
    const isAllowed = allowedChecks.some(Boolean);

    if (!isAllowed) {
      console.warn(`Access denied: ${decodedFilePath} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Get file info
    const stat = await fs.promises.stat(resolvedFilePath);
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    // Get filename for Content-Disposition
    const filename = path.basename(resolvedFilePath);

    // Determine content type
    const ext = path.extname(resolvedFilePath).toLowerCase();
    const contentTypeMap: { [key: string]: string } = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.webp': 'image/webp',
      '.jxl': 'image/jxl',
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

    // Common headers for better download handling
    const commonHeaders = {
      'Content-Type': contentType,
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'private, no-cache, must-revalidate',
      'Content-Disposition': `attachment; filename="${encodeURIComponent(filename)}"`,
      'X-Content-Type-Options': 'nosniff',
    };

    const requestedRange = parseRange(request.headers.get('range'), stat.size);
    if (requestedRange === 'invalid') {
      return new NextResponse(null, {
        status: 416,
        headers: { ...commonHeaders, 'Content-Range': `bytes */${stat.size}` },
      });
    }
    if (requestedRange) {
      const chunkSize = requestedRange.end - requestedRange.start + 1;

      const fileStream = fs.createReadStream(resolvedFilePath, {
        start: requestedRange.start,
        end: requestedRange.end,
        highWaterMark: 64 * 1024, // 64KB buffer
      });

      return new NextResponse(fileStream as any, {
        status: 206,
        headers: {
          ...commonHeaders,
          'Content-Range': `bytes ${requestedRange.start}-${requestedRange.end}/${stat.size}`,
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
