import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import archiver from 'archiver';
import { db } from './db';
import { resolveDatasetScope, isPathInside } from './datasetScope';
import { resolveDatasetFolder } from './encryptedDatasets';
import { createProjectAssetUrl } from './projectAssetUrls';
import { getJobTrainingRoot, getProjectRoots } from './projects';
import { getRemoteWorker, isLocalWorker, remoteJson } from './remoteClient';
import { makeRemoteAssetRef } from './remoteAssets';

const CAPTION_EXTENSIONS = new Set(['.txt', '.json', '.caption']);

export type ArchiveRequest = {
  zipTarget: 'samples' | 'dataset' | 'dataset_captions';
  jobID?: string;
  jobName?: string;
  datasetName?: string;
  project_id?: string | null;
};

export type ArchiveResult = { zipPath: string; fileName: string; downloadUrl?: string };

function safeName(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value === path.basename(value) && value !== '.' && value !== '..';
}

async function collectFiles(root: string, captionsOnly: boolean): Promise<string[]> {
  const output: string[] = [];
  for (const entry of await fsp.readdir(root, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;
    const candidate = path.join(root, entry.name);
    if (entry.isDirectory()) output.push(...(await collectFiles(candidate, captionsOnly)));
    else if (entry.isFile() && (!captionsOnly || CAPTION_EXTENSIONS.has(path.extname(entry.name).toLowerCase()))) {
      output.push(candidate);
    }
  }
  return output;
}

async function writeArchive(outputPath: string, entries: Array<{ source: string; name: string }>) {
  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  await fsp.rm(outputPath, { force: true });
  await new Promise<void>((resolve, reject) => {
    const output = fs.createWriteStream(outputPath);
    const archive = archiver('zip', { zlib: { level: 9 } });
    output.once('close', resolve);
    output.once('error', reject);
    archive.once('error', reject);
    archive.pipe(output);
    for (const entry of entries) archive.file(entry.source, { name: entry.name });
    void archive.finalize().catch(reject);
  });
}

async function datasetArchive(request: ArchiveRequest): Promise<ArchiveResult> {
  if (!safeName(request.datasetName)) throw new Error('Invalid datasetName');
  const scope = await resolveDatasetScope(request.project_id, { intent: 'read' });
  const canonicalRoot = await fsp.realpath(scope.datasetsRoot);
  const canonicalFolder = await fsp.realpath(resolveDatasetFolder(scope.datasetsRoot, request.datasetName)).catch(() => null);
  if (!canonicalFolder || !isPathInside(canonicalRoot, canonicalFolder)) throw new Error('Dataset not found');

  const captionsOnly = request.zipTarget === 'dataset_captions';
  const files = await collectFiles(canonicalFolder, captionsOnly);
  if (files.length === 0) throw new Error(captionsOnly ? 'No captions found' : 'Dataset is empty');
  const fileName = `${request.datasetName}_${captionsOnly ? 'captions' : 'dataset'}.zip`;
  const outputPath = path.join(canonicalRoot, '.zips', fileName);
  await writeArchive(
    outputPath,
    files.map(source => ({ source, name: path.join(request.datasetName!, path.relative(canonicalFolder, source)) })),
  );

  if (scope.project) {
    const projectRoot = (await getProjectRoots(scope.project)).root;
    return {
      zipPath: outputPath,
      fileName,
      downloadUrl: createProjectAssetUrl(scope.project.id, path.relative(projectRoot, outputPath), 'attachment'),
    };
  }
  return { zipPath: outputPath, fileName };
}

async function resolveSampleJob(request: ArchiveRequest) {
  if (request.jobID) return db.jobs.findById(request.jobID);
  if (!safeName(request.jobName)) return null;
  const matches = (await db.jobs.list()).filter(job => !job.project_id && job.name === request.jobName);
  return matches.length === 1 ? matches[0] : null;
}

async function sampleArchive(request: ArchiveRequest): Promise<ArchiveResult> {
  const job = await resolveSampleJob(request);
  if (!job) throw new Error('Job not found or ambiguous');
  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) throw new Error('Remote job is unavailable');
    const worker = await getRemoteWorker(job.worker_id);
    const result = await remoteJson<ArchiveResult>(worker, '/api/zip', {
      method: 'POST',
      body: JSON.stringify({ zipTarget: 'samples', jobID: job.remote_job_id, jobName: job.name }),
    });
    return {
      zipPath: makeRemoteAssetRef(job.id, 'file', result.zipPath),
      fileName: result.fileName,
    };
  }

  const trainingRoot = await getJobTrainingRoot(job);
  const jobRoot = path.resolve(trainingRoot, job.name);
  if (!isPathInside(path.resolve(trainingRoot), jobRoot)) throw new Error('Invalid job output path');
  const samplesRoot = await fsp.realpath(path.join(jobRoot, 'samples')).catch(() => null);
  if (!samplesRoot || !isPathInside(jobRoot, samplesRoot)) throw new Error('Samples folder not found');
  const files = await collectFiles(samplesRoot, false);
  if (files.length === 0) throw new Error('Samples folder is empty');
  const outputPath = path.join(jobRoot, 'samples.zip');
  await writeArchive(outputPath, files.map(source => ({ source, name: path.join('samples', path.relative(samplesRoot, source)) })));

  if (job.project_id) {
    const project = await db.projects.findById(job.project_id);
    if (!project) throw new Error('Project not found');
    const projectRoot = (await getProjectRoots(project)).root;
    return {
      zipPath: outputPath,
      fileName: 'samples.zip',
      downloadUrl: createProjectAssetUrl(project.id, path.relative(projectRoot, outputPath), 'attachment'),
    };
  }
  return { zipPath: outputPath, fileName: 'samples.zip' };
}

export async function createRequestedArchive(request: ArchiveRequest): Promise<ArchiveResult> {
  return request.zipTarget === 'samples' ? sampleArchive(request) : datasetArchive(request);
}
