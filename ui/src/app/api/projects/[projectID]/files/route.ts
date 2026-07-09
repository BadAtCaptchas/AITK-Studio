import fsp from 'fs/promises';
import path from 'path';
import { NextResponse } from 'next/server';
import { createProjectAssetUrl } from '@/server/projectAssetUrls';
import { getProjectRoots, isPathInside, resolveProject } from '@/server/projects';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from '@/server/settings';

const TEXT_EXTENSIONS = new Set(['.txt', '.caption', '.json', '.jsonc', '.yaml', '.yml', '.md', '.toml', '.log', '.csv']);
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.jxl', '.gif', '.bmp']);
const VIDEO_EXTENSIONS = new Set(['.mp4', '.webm', '.mov', '.mkv']);
const AUDIO_EXTENSIONS = new Set(['.mp3', '.wav', '.flac', '.ogg', '.m4a']);
const MAX_TEXT_PREVIEW_BYTES = 1024 * 1024;
const MAX_SEARCH_RESULTS = 250;
const MAX_SEARCH_VISITS = 4000;
const RESERVED_ROOT_ZONES = new Set(['datasets', 'configs', 'runs', 'outputs', 'models', 'assets', 'notes', 'cache']);
const DOMAIN_MANAGED_ZONES = new Set(['datasets', 'runs', 'models']);

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

async function resolveTarget(projectID: string, rawPath: unknown, intent: 'read' | 'write' = 'read') {
  const project = await resolveProject(decodeURIComponent(projectID), { intent });
  const roots = await getProjectRoots(project);
  const root = await fsp.realpath(roots.root).catch(() => path.resolve(roots.root));
  const requested = typeof rawPath === 'string' && rawPath.trim() ? rawPath : root;
  const target = path.isAbsolute(requested) ? path.resolve(requested) : path.resolve(root, requested);
  const lstat = await fsp.lstat(target).catch(() => null);
  if (lstat?.isSymbolicLink()) {
    const error = new Error('Symbolic links are not available through the project file browser');
    (error as Error & { status?: number }).status = 400;
    throw error;
  }
  const realTarget = await fsp.realpath(target).catch(() => target);
  if (!isPathInside(root, realTarget)) {
    const error = new Error('Path is outside the project sandbox');
    (error as Error & { status?: number }).status = 400;
    throw error;
  }
  return { project, roots, root, target: realTarget };
}

function pathSegments(root: string, target: string) {
  return path
    .relative(root, target)
    .split(path.sep)
    .filter(Boolean);
}

function relativeProjectPath(root: string, target: string) {
  return path.relative(root, target).replace(/\\/g, '/');
}

function assertGenericMutationAllowed(root: string, target: string) {
  const segments = pathSegments(root, target);
  if (segments.length === 0) {
    const error = new Error('Project root cannot be changed');
    (error as Error & { status?: number; code?: string }).status = 400;
    (error as Error & { status?: number; code?: string }).code = 'PROJECT_ROOT_PROTECTED';
    throw error;
  }
  const zone = segments[0].toLowerCase();
  if (segments.length === 1 && RESERVED_ROOT_ZONES.has(zone)) {
    const error = new Error('Managed project zones cannot be renamed or deleted');
    (error as Error & { status?: number; code?: string }).status = 409;
    (error as Error & { status?: number; code?: string }).code = 'PROJECT_ZONE_PROTECTED';
    throw error;
  }
  if (DOMAIN_MANAGED_ZONES.has(zone)) {
    const error = new Error(`Use the ${zone} workspace to rename or delete this item safely`);
    (error as Error & { status?: number; code?: string }).status = 409;
    (error as Error & { status?: number; code?: string }).code = 'PROJECT_DOMAIN_PATH_PROTECTED';
    throw error;
  }
}

async function childrenForDirectory(root: string, folder: string) {
  const entries = await fsp.readdir(folder, { withFileTypes: true }).catch(() => []);
  const children = [];
  for (const entry of entries) {
    if (entry.name.startsWith('.') || entry.isSymbolicLink()) continue;
    const absolutePath = path.join(folder, entry.name);
    if (!isPathInside(root, absolutePath)) continue;
    const stat = await fsp.stat(absolutePath).catch(() => null);
    if (!stat) continue;
    children.push({
      name: entry.name,
      path: relativeProjectPath(root, absolutePath),
      relativePath: relativeProjectPath(root, absolutePath),
      kind: entry.isDirectory() ? 'folder' : 'file',
      size: stat.size,
      updatedAt: stat.mtime.toISOString(),
    });
  }
  return children.sort((a, b) => Number(b.kind === 'folder') - Number(a.kind === 'folder') || a.name.localeCompare(b.name));
}

async function searchProjectFiles(root: string, query: string) {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return { results: [], truncated: false };

  const results: Awaited<ReturnType<typeof childrenForDirectory>> = [];
  const queue = [root];
  let visits = 0;
  while (queue.length > 0 && results.length < MAX_SEARCH_RESULTS && visits < MAX_SEARCH_VISITS) {
    const folder = queue.shift() as string;
    const entries = await childrenForDirectory(root, folder);
    for (const item of entries) {
      visits += 1;
      if (`${item.name}\n${item.relativePath}\n${item.kind}`.toLowerCase().includes(normalizedQuery)) {
        results.push(item);
        if (results.length >= MAX_SEARCH_RESULTS) break;
      }
      if (item.kind === 'folder' && visits < MAX_SEARCH_VISITS) queue.push(path.resolve(root, item.path));
    }
  }
  return { results, truncated: queue.length > 0 || visits >= MAX_SEARCH_VISITS };
}

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const searchParams = new URL(request.url).searchParams;
    const rawPath = searchParams.get('path');
    const { project, root, target } = await resolveTarget(projectID, rawPath);
    const stat = await fsp.stat(target);
    const ext = path.extname(target).toLowerCase();
    const relativePath = relativeProjectPath(root, target);
    const payload: Record<string, unknown> = {
      project: { id: project.id, slug: project.slug, name: project.name },
      item: {
        name: path.basename(target),
        path: relativePath,
        relativePath,
        kind: stat.isDirectory() ? 'folder' : 'file',
        size: stat.size,
        updatedAt: stat.mtime.toISOString(),
      },
    };

    const search = searchParams.get('search');
    if (stat.isDirectory() && search?.trim()) {
      const searchResult = await searchProjectFiles(root, search);
      payload.children = searchResult.results;
      payload.search = search;
      payload.truncated = searchResult.truncated;
      return NextResponse.json(payload);
    }

    if (stat.isDirectory()) {
      payload.children = await childrenForDirectory(root, target);
      return NextResponse.json(payload);
    }

    payload.downloadUrl = createProjectAssetUrl(project.id, relativePath, 'attachment');
    if (IMAGE_EXTENSIONS.has(ext)) {
      payload.mediaUrl = createProjectAssetUrl(project.id, relativePath, 'inline');
      payload.mediaKind = 'image';
    } else if (VIDEO_EXTENSIONS.has(ext)) {
      payload.mediaUrl = createProjectAssetUrl(project.id, relativePath, 'inline');
      payload.mediaKind = 'video';
    } else if (AUDIO_EXTENSIONS.has(ext)) {
      payload.mediaUrl = createProjectAssetUrl(project.id, relativePath, 'inline');
      payload.mediaKind = 'audio';
    }
    if (TEXT_EXTENSIONS.has(ext) && stat.size <= MAX_TEXT_PREVIEW_BYTES) {
      payload.content = await fsp.readFile(target, 'utf-8');
    }
    return NextResponse.json(payload);
  } catch (error: unknown) {
    const known = error as Error & { status?: number };
    return NextResponse.json({ error: known?.message || 'Failed to read project file' }, { status: known?.status || 500 });
  }
}

export async function PATCH(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const body = await request.json();
    const { root, target } = await resolveTarget(projectID, body?.path, 'write');
    assertGenericMutationAllowed(root, target);
    const name = safeName(body?.newName);
    if (!name) return NextResponse.json({ error: 'New name is required' }, { status: 400 });
    const destination = path.join(path.dirname(target), name);
    if (!isPathInside(root, destination)) return NextResponse.json({ error: 'Invalid destination' }, { status: 400 });
    const destinationExists = await fsp.access(destination).then(() => true).catch(() => false);
    if (destinationExists) return NextResponse.json({ error: 'An item with that name already exists', code: 'PROJECT_PATH_COLLISION' }, { status: 409 });
    await fsp.rename(target, destination);
    return NextResponse.json({ success: true, path: relativeProjectPath(root, destination) });
  } catch (error: unknown) {
    const known = error as Error & { status?: number; code?: string };
    return NextResponse.json({ error: known?.message || 'Failed to rename project file', code: known?.code }, { status: known?.status || 500 });
  }
}

export async function DELETE(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const body = await request.json();
    const { root, target } = await resolveTarget(projectID, body?.path, 'write');
    assertGenericMutationAllowed(root, target);
    const stat = await fsp.stat(target);
    if (stat.isDirectory()) {
      const entries = await fsp.readdir(target);
      if (entries.length > 0 && body?.recursive !== true) {
        return NextResponse.json(
          { error: 'Folder is not empty. Confirm recursive deletion to continue.', code: 'PROJECT_RECURSIVE_CONFIRMATION_REQUIRED' },
          { status: 409 },
        );
      }
    }
    await fsp.rm(target, { recursive: stat.isDirectory(), force: false });
    return NextResponse.json({ success: true });
  } catch (error: unknown) {
    const known = error as Error & { status?: number; code?: string };
    return NextResponse.json({ error: known?.message || 'Failed to delete project file', code: known?.code }, { status: known?.status || 500 });
  }
}
