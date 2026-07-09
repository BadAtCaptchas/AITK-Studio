import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { Readable } from 'stream';
import { NextResponse } from 'next/server';
import {
  normalizeProjectAssetPath,
  verifyProjectAssetSignature,
  type ProjectAssetDisposition,
} from '@/server/projectAssetUrls';
import { getProjectRoots, isPathInside, resolveProject } from '@/server/projects';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const CONTENT_TYPES: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.jxl': 'image/jxl',
  '.bmp': 'image/bmp',
  '.avif': 'image/avif',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
  '.m4a': 'audio/mp4',
  '.txt': 'text/plain; charset=utf-8',
  '.json': 'application/json',
  '.yaml': 'text/yaml; charset=utf-8',
  '.yml': 'text/yaml; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.safetensors': 'application/octet-stream',
  '.ckpt': 'application/octet-stream',
  '.pt': 'application/octet-stream',
  '.pth': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.gguf': 'application/octet-stream',
};

function parseRange(value: string | null, size: number) {
  if (!value) return null;
  const match = /^bytes=(\d*)-(\d*)$/.exec(value.trim());
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
    end = match[2] ? Number(match[2]) : size - 1;
  }
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || start >= size) {
    return 'invalid' as const;
  }
  return { start, end: Math.min(end, size - 1) };
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ projectID: string; filePath: string[] }> },
) {
  try {
    const { projectID: encodedProjectID, filePath } = await params;
    const projectID = decodeURIComponent(encodedProjectID);
    const url = new URL(request.url);
    const queryProjectID = url.searchParams.get('project_id') || '';
    const queryPath = normalizeProjectAssetPath(url.searchParams.get('path') || '');
    const routePath = normalizeProjectAssetPath(filePath.map(decodeURIComponent).join('/'));
    const dispositionValue = url.searchParams.get('disposition');
    const disposition: ProjectAssetDisposition = dispositionValue === 'attachment' ? 'attachment' : 'inline';
    const expires = Number(url.searchParams.get('expires') || '');
    const signature = url.searchParams.get('sig') || '';
    if (
      queryProjectID !== projectID ||
      queryPath !== routePath ||
      !verifyProjectAssetSignature({ projectID, relativePath: queryPath, disposition, expires, signature })
    ) {
      return new NextResponse('Invalid or expired project asset link', { status: 403 });
    }

    const project = await resolveProject(projectID, { intent: 'read' });
    const roots = await getProjectRoots(project);
    const root = await fsp.realpath(roots.root);
    const candidate = path.resolve(root, ...queryPath.split('/'));
    const target = await fsp.realpath(candidate).catch(() => null);
    if (!target || !isPathInside(root, target) || target === root) {
      return new NextResponse('Project asset not found', { status: 404 });
    }
    const lstat = await fsp.lstat(candidate);
    const stat = await fsp.stat(target);
    if (lstat.isSymbolicLink() || !stat.isFile()) return new NextResponse('Project asset not found', { status: 404 });

    const contentType = CONTENT_TYPES[path.extname(target).toLowerCase()] || 'application/octet-stream';
    const filename = path.basename(target).replace(/[\r\n"]/g, '_');
    const baseHeaders: Record<string, string> = {
      'Content-Type': contentType,
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'private, max-age=300',
      'Content-Disposition': `${disposition}; filename="${filename}"; filename*=UTF-8''${encodeURIComponent(filename)}`,
      'X-Content-Type-Options': 'nosniff',
    };
    const requestedRange = parseRange(request.headers.get('range'), stat.size);
    if (requestedRange === 'invalid') {
      return new NextResponse(null, { status: 416, headers: { ...baseHeaders, 'Content-Range': `bytes */${stat.size}` } });
    }
    if (requestedRange) {
      const length = requestedRange.end - requestedRange.start + 1;
      const stream = fs.createReadStream(target, { start: requestedRange.start, end: requestedRange.end });
      return new NextResponse(Readable.toWeb(stream) as ReadableStream, {
        status: 206,
        headers: {
          ...baseHeaders,
          'Content-Length': String(length),
          'Content-Range': `bytes ${requestedRange.start}-${requestedRange.end}/${stat.size}`,
        },
      });
    }
    return new NextResponse(Readable.toWeb(fs.createReadStream(target)) as ReadableStream, {
      headers: { ...baseHeaders, 'Content-Length': String(stat.size) },
    });
  } catch {
    return new NextResponse('Project asset not found', { status: 404 });
  }
}
