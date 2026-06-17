import fsp from 'fs/promises';
import path from 'path';
import { NextResponse } from 'next/server';
import { ensureProjectFolders, isPathInside, resolveProject } from '@/server/projects';

const TEXT_EXTENSIONS = new Set(['.txt', '.caption', '.json', '.jsonc', '.yaml', '.yml', '.md', '.toml', '.log', '.csv']);
const MEDIA_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.jxl', '.gif', '.bmp', '.mp4', '.mp3', '.wav', '.flac', '.ogg']);
const MAX_TEXT_PREVIEW_BYTES = 1024 * 1024;

function ensureApiAccess(request: Request): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) return null;

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

function safeName(value: unknown) {
  if (typeof value !== 'string') return '';
  return value
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_')
    .replace(/^\.+/, '')
    .slice(0, 120);
}

async function resolveTarget(projectID: string, rawPath: unknown) {
  const project = await resolveProject(decodeURIComponent(projectID));
  const roots = await ensureProjectFolders(project);
  const root = await fsp.realpath(roots.root).catch(() => path.resolve(roots.root));
  const requested = typeof rawPath === 'string' && rawPath.trim() ? rawPath : root;
  const target = path.isAbsolute(requested) ? path.resolve(requested) : path.resolve(root, requested);
  const realTarget = await fsp.realpath(target).catch(() => target);
  if (!isPathInside(root, realTarget)) {
    const error = new Error('Path is outside the project sandbox');
    (error as any).status = 400;
    throw error;
  }
  return { project, roots, root, target: realTarget };
}

async function childrenForDirectory(root: string, folder: string) {
  const entries = await fsp.readdir(folder, { withFileTypes: true }).catch(() => []);
  const children = [];
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const absolutePath = path.join(folder, entry.name);
    if (!isPathInside(root, absolutePath)) continue;
    const stat = await fsp.stat(absolutePath).catch(() => null);
    if (!stat) continue;
    children.push({
      name: entry.name,
      path: absolutePath,
      relativePath: path.relative(root, absolutePath),
      kind: entry.isDirectory() ? 'folder' : 'file',
      size: stat.size,
      updatedAt: stat.mtime.toISOString(),
    });
  }
  return children.sort((a, b) => Number(b.kind === 'folder') - Number(a.kind === 'folder') || a.name.localeCompare(b.name));
}

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;

  try {
    const { projectID } = await params;
    const rawPath = new URL(request.url).searchParams.get('path');
    const { project, roots, root, target } = await resolveTarget(projectID, rawPath);
    const stat = await fsp.stat(target);
    const ext = path.extname(target).toLowerCase();
    const payload: any = {
      project,
      roots,
      item: {
        name: path.basename(target),
        path: target,
        relativePath: path.relative(root, target),
        kind: stat.isDirectory() ? 'folder' : 'file',
        size: stat.size,
        updatedAt: stat.mtime.toISOString(),
      },
    };

    if (stat.isDirectory()) {
      payload.children = await childrenForDirectory(root, target);
      return NextResponse.json(payload);
    }

    payload.downloadUrl = `/api/files/${encodeURIComponent(target)}`;
    if (MEDIA_EXTENSIONS.has(ext)) {
      payload.mediaUrl = `/api/img/${encodeURIComponent(target)}`;
    }
    if (TEXT_EXTENSIONS.has(ext) && stat.size <= MAX_TEXT_PREVIEW_BYTES) {
      payload.content = await fsp.readFile(target, 'utf-8');
    }
    return NextResponse.json(payload);
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to read project file' }, { status: error?.status || 500 });
  }
}

export async function PATCH(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;

  try {
    const { projectID } = await params;
    const body = await request.json();
    const { root, target } = await resolveTarget(projectID, body?.path);
    if (target === root) return NextResponse.json({ error: 'Project root cannot be renamed' }, { status: 400 });
    const name = safeName(body?.newName);
    if (!name) return NextResponse.json({ error: 'New name is required' }, { status: 400 });
    const destination = path.join(path.dirname(target), name);
    if (!isPathInside(root, destination)) return NextResponse.json({ error: 'Invalid destination' }, { status: 400 });
    await fsp.rename(target, destination);
    return NextResponse.json({ success: true, path: destination });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to rename project file' }, { status: error?.status || 500 });
  }
}

export async function DELETE(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;

  try {
    const { projectID } = await params;
    const body = await request.json();
    const { root, target } = await resolveTarget(projectID, body?.path);
    if (target === root) return NextResponse.json({ error: 'Project root cannot be deleted' }, { status: 400 });
    await fsp.rm(target, { recursive: true, force: true });
    return NextResponse.json({ success: true });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to delete project file' }, { status: error?.status || 500 });
  }
}
