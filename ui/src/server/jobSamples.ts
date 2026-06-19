import fs from 'fs';
import path from 'path';
import { db } from './db';
import { getJobTrainingRoot, getProjectRoots } from './projects';
import type { Job } from '@/types';

export const sampleContentTypeMap: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.jxl': 'image/jxl',
  '.mp4': 'video/mp4',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
};

const allowedSampleExtensions = new Set(Object.keys(sampleContentTypeMap));

function isPathInsideRoot(root: string, filepath: string) {
  const relativePath = path.relative(root, filepath);
  return relativePath === '' || (!relativePath.startsWith('..') && !path.isAbsolute(relativePath));
}

function isSafeSampleName(filename: string) {
  return (
    filename.length > 0 &&
    filename !== '.' &&
    filename !== '..' &&
    filename === path.basename(filename) &&
    !filename.includes('/') &&
    !filename.includes('\\')
  );
}

function configuredOutputFolderCandidates(jobConfigJson: string) {
  const candidates: string[] = [];

  try {
    const jobConfig = JSON.parse(jobConfigJson);
    const processes = Array.isArray(jobConfig?.config?.process) ? jobConfig.config.process : [];
    for (const processConfig of processes) {
      const outputFolder = processConfig?.output_folder;
      if (typeof outputFolder !== 'string' || !outputFolder.trim()) continue;
      candidates.push(outputFolder);
      candidates.push(path.join(outputFolder, 'samples'));
    }
  } catch {
    return candidates;
  }

  return candidates;
}

async function realpathIfExists(filepath: string) {
  return fs.promises.realpath(path.resolve(filepath)).catch(() => null);
}

async function getAllowedSampleParents(job: Job, trainingFolder: string) {
  const parents: string[] = [];
  const canonicalTrainingFolder = await realpathIfExists(trainingFolder);
  if (canonicalTrainingFolder) {
    parents.push(canonicalTrainingFolder);
  }

  if (job.project_id) {
    const project = await db.projects.findById(job.project_id);
    if (project) {
      const roots = await getProjectRoots(project);
      const canonicalProjectRoot = await realpathIfExists(roots.root);
      if (canonicalProjectRoot) {
        parents.push(canonicalProjectRoot);
      }
    }
  }

  return parents;
}

export async function getJobSampleRoots(job: Job) {
  const trainingFolder = await getJobTrainingRoot(job);
  const allowedParents = await getAllowedSampleParents(job, trainingFolder);
  if (allowedParents.length === 0) return [];

  const candidates = [
    path.join(trainingFolder, job.name, 'samples'),
    ...configuredOutputFolderCandidates(job.job_config),
  ];

  const seen = new Set<string>();
  const roots: string[] = [];

  for (const candidate of candidates) {
    const canonicalCandidate = await realpathIfExists(candidate);
    if (!canonicalCandidate || seen.has(canonicalCandidate)) continue;
    if (!allowedParents.some(parent => isPathInsideRoot(parent, canonicalCandidate))) continue;

    seen.add(canonicalCandidate);
    roots.push(canonicalCandidate);
  }

  return roots;
}

export async function listJobSampleUrls(job: Job) {
  const roots = await getJobSampleRoots(job);
  const seen = new Set<string>();
  const samples: string[] = [];

  for (const root of roots) {
    const entries = await fs.promises.readdir(root, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (!entry.isFile() || !allowedSampleExtensions.has(path.extname(entry.name).toLowerCase())) continue;
      if (seen.has(entry.name)) continue;

      seen.add(entry.name);
      samples.push(`/api/jobs/${encodeURIComponent(job.id)}/samples/${encodeURIComponent(entry.name)}`);
    }
  }

  return samples.sort();
}

export async function resolveJobSampleFile(job: Job, filename: string) {
  if (!isSafeSampleName(filename)) return null;
  const ext = path.extname(filename).toLowerCase();
  const contentType = sampleContentTypeMap[ext];
  if (!contentType) return null;

  const roots = await getJobSampleRoots(job);
  for (const root of roots) {
    const filepath = path.resolve(root, filename);
    const canonicalPath = await realpathIfExists(filepath);
    if (!canonicalPath || !isPathInsideRoot(root, canonicalPath)) continue;

    const stat = await fs.promises.stat(canonicalPath).catch(() => null);
    if (!stat || !stat.isFile()) continue;

    return { path: canonicalPath, stat, contentType };
  }

  return null;
}
