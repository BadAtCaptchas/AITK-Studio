import fsp from 'fs/promises';
import path from 'path';
import { NextResponse } from 'next/server';
import { ensureProjectFolders, isPathInside, resolveProject } from '@/server/projects';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from '@/server/settings';

function ensureApiAccess(request: Request): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) return null;

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

async function listArtifacts(root: string, maxEntries = 250) {
  const rootReal = await fsp.realpath(root).catch(() => path.resolve(root));
  const entries: Array<{
    name: string;
    path: string;
    relativePath: string;
    kind: 'file' | 'folder';
    size: number;
    updatedAt: string;
  }> = [];
  const stack = [''];

  while (stack.length > 0 && entries.length < maxEntries) {
    const relativeDir = stack.shift() as string;
    const dir = path.join(rootReal, relativeDir);
    const children = await fsp.readdir(dir, { withFileTypes: true }).catch(() => []);
    children.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name));
    for (const child of children) {
      if (entries.length >= maxEntries || child.name.startsWith('.')) break;
      const absolutePath = path.join(dir, child.name);
      if (!isPathInside(rootReal, absolutePath)) continue;
      const stat = await fsp.stat(absolutePath).catch(() => null);
      if (!stat) continue;
      const relativePath = path.relative(rootReal, absolutePath);
      entries.push({
        name: child.name,
        path: absolutePath,
        relativePath,
        kind: child.isDirectory() ? 'folder' : 'file',
        size: stat.size,
        updatedAt: stat.mtime.toISOString(),
      });
      if (child.isDirectory() && relativePath.split(path.sep).length < 4) {
        stack.push(relativePath);
      }
    }
  }

  return entries;
}

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const project = await resolveProject(decodeURIComponent(projectID));
    const roots = await ensureProjectFolders(project);
    const artifacts = await listArtifacts(roots.root);
    return NextResponse.json({ project, roots, artifacts });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to list project artifacts' }, { status: 500 });
  }
}
