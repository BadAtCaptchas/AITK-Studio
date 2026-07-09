import fsp from 'fs/promises';
import path from 'path';
import type { Job, Project } from '@/types';
import { db } from './db';
import { createProjectAssetUrl } from './projectAssetUrls';
import { getProjectRoots, isPathInside } from './projects';

export type IndexedProjectArtifactKind = 'image' | 'video' | 'audio' | 'model' | 'file';
export type IndexedProjectArtifactSource = 'generated' | 'training-sample' | 'model' | 'file';
export type IndexedProjectArtifactZone = 'outputs' | 'runs' | 'models';

export type IndexedProjectArtifact = {
  id: string;
  projectId: string;
  name: string;
  relativePath: string;
  portableRef: string;
  zone: IndexedProjectArtifactZone;
  kind: IndexedProjectArtifactKind;
  source: IndexedProjectArtifactSource;
  size: number;
  updatedAt: string;
  sourceJobId: string | null;
  sourceJobName: string | null;
  availability: 'local';
  workerID: 'local';
  previewUrl: string | null;
  downloadUrl: string;
};

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.jxl', '.gif', '.bmp', '.avif']);
const VIDEO_EXTENSIONS = new Set(['.mp4', '.webm', '.mov', '.mkv']);
const AUDIO_EXTENSIONS = new Set(['.mp3', '.wav', '.flac', '.ogg', '.m4a']);
const MODEL_EXTENSIONS = new Set(['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf']);

function artifactKind(filePath: string): IndexedProjectArtifactKind {
  const extension = path.extname(filePath).toLowerCase();
  if (IMAGE_EXTENSIONS.has(extension)) return 'image';
  if (VIDEO_EXTENSIONS.has(extension)) return 'video';
  if (AUDIO_EXTENSIONS.has(extension)) return 'audio';
  if (MODEL_EXTENSIONS.has(extension)) return 'model';
  return 'file';
}

function artifactSource(
  zone: IndexedProjectArtifactZone,
  filePath: string,
  kind: IndexedProjectArtifactKind,
): IndexedProjectArtifactSource {
  if (kind === 'model' || zone === 'models') return 'model';
  if (zone === 'outputs') return 'generated';
  if (IMAGE_EXTENSIONS.has(path.extname(filePath).toLowerCase())) return 'training-sample';
  return 'file';
}

function projectPortableRef(projectID: string, relativePath: string) {
  return `aitk-project://${projectID}/${relativePath
    .split(/[\\/]+/)
    .filter(Boolean)
    .map(segment => encodeURIComponent(segment))
    .join('/')}`;
}

async function scanZone(options: {
  projectRoot: string;
  zoneRoot: string;
  zone: IndexedProjectArtifactZone;
  maxEntries: number;
  allow: (kind: IndexedProjectArtifactKind, filePath: string) => boolean;
}) {
  const projectReal = await fsp.realpath(options.projectRoot).catch(() => path.resolve(options.projectRoot));
  const zoneReal = await fsp.realpath(options.zoneRoot).catch(() => path.resolve(options.zoneRoot));
  if (!isPathInside(projectReal, zoneReal)) return [];

  const files: Array<{
    absolutePath: string;
    relativePath: string;
    zone: IndexedProjectArtifactZone;
    kind: IndexedProjectArtifactKind;
    size: number;
    updatedAt: string;
  }> = [];
  const queue = [zoneReal];
  let visited = 0;
  while (queue.length > 0 && visited < options.maxEntries) {
    const folder = queue.shift() as string;
    const entries = await fsp.readdir(folder, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (visited >= options.maxEntries) break;
      visited += 1;
      if (entry.name.startsWith('.')) continue;
      const absolutePath = path.join(folder, entry.name);
      if (!isPathInside(projectReal, absolutePath)) continue;
      const stat = await fsp.lstat(absolutePath).catch(() => null);
      if (!stat || stat.isSymbolicLink()) continue;
      if (stat.isDirectory()) {
        queue.push(absolutePath);
        continue;
      }
      if (!stat.isFile()) continue;
      const kind = artifactKind(absolutePath);
      if (!options.allow(kind, absolutePath)) continue;
      files.push({
        absolutePath,
        relativePath: path.relative(projectReal, absolutePath),
        zone: options.zone,
        kind,
        size: stat.size,
        updatedAt: stat.mtime.toISOString(),
      });
    }
  }
  return files;
}

function sourceJobFor(file: { zone: IndexedProjectArtifactZone; relativePath: string }, jobs: Job[]) {
  if (file.zone !== 'runs') return null;
  const pathSegments = file.relativePath.split(/[\\/]+/).filter(Boolean);
  const runsIndex = pathSegments.indexOf('runs');
  const jobName = runsIndex >= 0 ? pathSegments[runsIndex + 1] : null;
  return jobName ? jobs.find(job => job.name.toLowerCase() === jobName.toLowerCase()) || null : null;
}

export async function indexProjectArtifacts(
  project: Project,
  options: { kind?: 'outputs' | 'models' | 'all'; maxEntries?: number } = {},
): Promise<IndexedProjectArtifact[]> {
  const roots = await getProjectRoots(project);
  const kind = options.kind ?? 'all';
  const maxEntries = Math.max(100, Math.min(options.maxEntries ?? 6_000, 20_000));
  const scans: Array<ReturnType<typeof scanZone>> = [];

  if (kind === 'all' || kind === 'outputs') {
    scans.push(
      scanZone({
        projectRoot: roots.root,
        zoneRoot: roots.outputs,
        zone: 'outputs',
        maxEntries,
        allow: fileKind => ['image', 'video', 'audio'].includes(fileKind),
      }),
      scanZone({
        projectRoot: roots.root,
        zoneRoot: roots.runs,
        zone: 'runs',
        maxEntries,
        allow: (fileKind, filePath) =>
          ['image', 'video', 'audio'].includes(fileKind) &&
          /\/(?:samples?|outputs?)\//i.test(filePath.replace(/\\/g, '/')),
      }),
    );
  }
  if (kind === 'all' || kind === 'models') {
    scans.push(
      scanZone({
        projectRoot: roots.root,
        zoneRoot: roots.models,
        zone: 'models',
        maxEntries,
        allow: fileKind => fileKind === 'model',
      }),
      scanZone({
        projectRoot: roots.root,
        zoneRoot: roots.runs,
        zone: 'runs',
        maxEntries,
        allow: fileKind => fileKind === 'model',
      }),
    );
  }

  const [jobs, scanned] = await Promise.all([db.jobs.list({ project_id: project.id }), Promise.all(scans)]);
  return scanned
    .flat()
    .map(file => {
      const sourceJob = sourceJobFor(file, jobs);
      const source = artifactSource(file.zone, file.absolutePath, file.kind);
      return {
        id: file.relativePath.replace(/\\/g, '/'),
        projectId: project.id,
        name: path.basename(file.absolutePath),
        relativePath: file.relativePath,
        portableRef: projectPortableRef(project.id, file.relativePath),
        zone: file.zone,
        kind: file.kind,
        source,
        size: file.size,
        updatedAt: file.updatedAt,
        sourceJobId: sourceJob?.id || null,
        sourceJobName: sourceJob?.name || null,
        availability: 'local' as const,
        workerID: 'local' as const,
        previewUrl: ['image', 'video', 'audio'].includes(file.kind)
          ? createProjectAssetUrl(project.id, file.relativePath, 'inline')
          : null,
        downloadUrl: createProjectAssetUrl(project.id, file.relativePath, 'attachment'),
      };
    })
    .sort((left, right) => new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime());
}
