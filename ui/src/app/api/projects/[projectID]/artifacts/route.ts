import fsp from 'fs/promises';
import path from 'path';
import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { createProjectAssetUrl } from '@/server/projectAssetUrls';
import { getProjectRoots, isPathInside, resolveProject } from '@/server/projects';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from '@/server/settings';
import { isRequestAuthenticated } from '@/utils/authSession';

type ArtifactKind = 'image' | 'video' | 'audio' | 'model' | 'file';
type ArtifactSource = 'generated' | 'training-sample' | 'model' | 'file';

type ProjectArtifact = {
  id: string;
  projectId: string;
  name: string;
  relativePath: string;
  portableRef: string;
  zone: 'outputs' | 'runs' | 'models';
  kind: ArtifactKind;
  source: ArtifactSource;
  size: number;
  updatedAt: string;
  sourceJobId: string | null;
  sourceJobName: string | null;
  availability: 'local';
  workerID: 'local';
  previewUrl: string | null;
  downloadUrl: string;
};

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.jxl', '.gif', '.bmp']);
const VIDEO_EXTENSIONS = new Set(['.mp4', '.webm', '.mov', '.mkv']);
const AUDIO_EXTENSIONS = new Set(['.mp3', '.wav', '.flac', '.ogg', '.m4a']);
const MODEL_EXTENSIONS = new Set(['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf']);
const MAX_SCAN_ENTRIES = 6000;

async function ensureApiAccess(request: Request): Promise<NextResponse | null> {
  const token = process.env.AI_TOOLKIT_AUTH;
  if (!token) return null;
  if (!(await isRequestAuthenticated(request, token))) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  return null;
}

function positiveInteger(value: string | null, fallback: number, max: number) {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? Math.min(parsed, max) : fallback;
}

function artifactKind(filePath: string): ArtifactKind {
  const extension = path.extname(filePath).toLowerCase();
  if (IMAGE_EXTENSIONS.has(extension)) return 'image';
  if (VIDEO_EXTENSIONS.has(extension)) return 'video';
  if (AUDIO_EXTENSIONS.has(extension)) return 'audio';
  if (MODEL_EXTENSIONS.has(extension)) return 'model';
  return 'file';
}

function sourceFor(zone: ProjectArtifact['zone'], filePath: string, kind: ArtifactKind): ArtifactSource {
  if (kind === 'model' || zone === 'models') return 'model';
  if (zone === 'outputs') return 'generated';
  if (IMAGE_EXTENSIONS.has(path.extname(filePath).toLowerCase())) return 'training-sample';
  return 'file';
}

function portableRef(projectID: string, relativePath: string) {
  return `aitk-project://${projectID}/${relativePath
    .split(/[\\/]+/)
    .filter(Boolean)
    .map(segment => encodeURIComponent(segment))
    .join('/')}`;
}

async function scanFiles(
  projectRoot: string,
  zoneRoot: string,
  zone: ProjectArtifact['zone'],
  allow: (kind: ArtifactKind, filePath: string) => boolean,
) {
  const rootReal = await fsp.realpath(projectRoot).catch(() => path.resolve(projectRoot));
  const zoneReal = await fsp.realpath(zoneRoot).catch(() => path.resolve(zoneRoot));
  if (!isPathInside(rootReal, zoneReal)) return [];

  const files: Array<{ absolutePath: string; relativePath: string; zone: ProjectArtifact['zone']; kind: ArtifactKind; size: number; updatedAt: string }> = [];
  const queue = [zoneReal];
  let visited = 0;
  while (queue.length > 0 && visited < MAX_SCAN_ENTRIES) {
    const folder = queue.shift() as string;
    const entries = await fsp.readdir(folder, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (visited >= MAX_SCAN_ENTRIES) break;
      visited += 1;
      if (entry.name.startsWith('.')) continue;
      const absolutePath = path.join(folder, entry.name);
      if (!isPathInside(rootReal, absolutePath)) continue;
      const stat = await fsp.lstat(absolutePath).catch(() => null);
      if (!stat || stat.isSymbolicLink()) continue;
      if (stat.isDirectory()) {
        queue.push(absolutePath);
        continue;
      }
      if (!stat.isFile()) continue;
      const kind = artifactKind(absolutePath);
      if (!allow(kind, absolutePath)) continue;
      files.push({
        absolutePath,
        relativePath: path.relative(rootReal, absolutePath),
        zone,
        kind,
        size: stat.size,
        updatedAt: stat.mtime.toISOString(),
      });
    }
  }
  return files;
}

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = await ensureApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const project = await resolveProject(decodeURIComponent(projectID), { intent: 'read' });
    const roots = await getProjectRoots(project);
    const searchParams = new URL(request.url).searchParams;
    const requestedKind = searchParams.get('kind') || 'all';
    const query = (searchParams.get('query') || '').trim().toLowerCase();
    const sourceFilter = searchParams.get('source') || 'all';
    const limit = positiveInteger(searchParams.get('limit'), 60, 200);
    const cursor = Math.max(0, Number(searchParams.get('cursor') || 0) || 0);
    const jobs = await db.jobs.list({ project_id: project.id });
    const jobByName = new Map(jobs.map(job => [job.name.toLowerCase(), job]));

    const scans: Array<Promise<Awaited<ReturnType<typeof scanFiles>>>> = [];
    if (requestedKind === 'all' || requestedKind === 'outputs') {
      scans.push(scanFiles(roots.root, roots.outputs, 'outputs', kind => kind === 'image' || kind === 'video' || kind === 'audio'));
      scans.push(
        scanFiles(roots.root, roots.runs, 'runs', (kind, filePath) => {
          if (kind !== 'image' && kind !== 'video' && kind !== 'audio') return false;
          const normalized = filePath.replace(/\\/g, '/').toLowerCase();
          return normalized.includes('/samples/') || normalized.includes('/sample/') || normalized.includes('/output/');
        }),
      );
    }
    if (requestedKind === 'all' || requestedKind === 'models') {
      scans.push(scanFiles(roots.root, roots.models, 'models', kind => kind === 'model'));
      scans.push(scanFiles(roots.root, roots.runs, 'runs', kind => kind === 'model'));
    }

    const scanned = (await Promise.all(scans)).flat();
    const artifacts: ProjectArtifact[] = scanned.map(file => {
      const zoneRelative = path.relative(roots[file.zone], file.absolutePath);
      const firstSegment = zoneRelative.split(path.sep).filter(Boolean)[0]?.toLowerCase() || '';
      const sourceJob = file.zone === 'runs' ? jobByName.get(firstSegment) || null : null;
      const source = sourceFor(file.zone, file.absolutePath, file.kind);
      return {
        id: file.relativePath.replace(/\\/g, '/'),
        projectId: project.id,
        name: path.basename(file.absolutePath),
        relativePath: file.relativePath,
        portableRef: portableRef(project.id, file.relativePath),
        zone: file.zone,
        kind: file.kind,
        source,
        size: file.size,
        updatedAt: file.updatedAt,
        sourceJobId: sourceJob?.id || null,
        sourceJobName: sourceJob?.name || null,
        availability: 'local',
        workerID: 'local',
        previewUrl: ['image', 'video', 'audio'].includes(file.kind)
          ? createProjectAssetUrl(project.id, file.relativePath, 'inline')
          : null,
        downloadUrl: createProjectAssetUrl(project.id, file.relativePath, 'attachment'),
      };
    });

    const filtered = artifacts
      .filter(item => sourceFilter === 'all' || item.source === sourceFilter)
      .filter(item => !query || `${item.name}\n${item.relativePath}\n${item.sourceJobName || ''}`.toLowerCase().includes(query))
      .sort((left, right) => new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime());
    const page = filtered.slice(cursor, cursor + limit);
    const nextCursor = cursor + page.length < filtered.length ? cursor + page.length : null;

    return NextResponse.json({
      project: { id: project.id, slug: project.slug, name: project.name },
      artifacts: page,
      total: filtered.length,
      nextCursor,
      truncated: scanned.length >= MAX_SCAN_ENTRIES,
    });
  } catch (error: unknown) {
    const known = error as Error & { status?: number };
    return NextResponse.json({ error: known?.message || 'Failed to list project artifacts' }, { status: known?.status || 500 });
  }
}
