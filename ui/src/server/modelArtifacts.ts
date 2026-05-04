import crypto from 'crypto';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { getTrainingFolder } from './settings';
import { db, type ModelArtifactInput } from './db';
import { collectModelReferences, resolveConfigPath, safeNameSegment } from './trainingJobTransfer';
import type { Job } from '../types';

function stableID(parts: Array<string | null | undefined>) {
  return crypto.createHash('sha1').update(parts.filter(Boolean).join('|')).digest('hex');
}

function parseCheckpointStep(fileName: string, fallback: number | null = null) {
  const match = fileName.match(/(?:_|step-?)(\d{1,9})(?:\D|$)/i) || fileName.match(/(\d{4,9})(?:\D|$)/);
  return match ? Number(match[1]) : fallback;
}

function statInfo(filePath: string) {
  try {
    const stat = fs.statSync(filePath);
    return {
      exists: true,
      size: stat.size,
      modified_at: stat.mtime,
    };
  } catch {
    return {
      exists: false,
      size: null,
      modified_at: null,
    };
  }
}

function artifactName(value: string) {
  const base = path.basename(value.replace(/[\\/]+$/, ''));
  return safeNameSegment(base || value, 'model');
}

function parseJobConfig(job: Job) {
  try {
    return JSON.parse(job.job_config);
  } catch {
    return null;
  }
}

async function indexJobArtifacts(job: Job, trainingRoot: string): Promise<ModelArtifactInput[]> {
  const artifacts: ModelArtifactInput[] = [];
  const jobConfig = parseJobConfig(job);

  if (jobConfig) {
    for (const ref of collectModelReferences(jobConfig)) {
      const absolutePath = resolveConfigPath(ref.value);
      const stats = statInfo(absolutePath);
      artifacts.push({
        id: stableID(['config-ref', job.id, ref.configPath, ref.value]),
        kind: ref.isLocal ? 'base_model_local' : 'base_model_remote',
        name: artifactName(ref.value),
        path: ref.value,
        source: 'job_config',
        job_id: job.id,
        step: null,
        exists: ref.isLocal ? stats.exists : true,
        size: stats.size,
        modified_at: stats.modified_at,
        metadata: JSON.stringify({
          configPath: ref.configPath,
          absolutePath: ref.isLocal ? absolutePath : null,
          remote: !ref.isLocal,
        }),
      });
    }
  }

  const jobFolder = path.join(trainingRoot, job.name);
  try {
    const entries = await fsp.readdir(jobFolder, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      const filePath = path.join(jobFolder, entry.name);
      const lower = entry.name.toLowerCase();
      if (lower.endsWith('.safetensors')) {
        const stats = statInfo(filePath);
        artifacts.push({
          id: stableID(['checkpoint', job.id, filePath]),
          kind: 'checkpoint',
          name: entry.name.replace(/\.safetensors$/i, ''),
          path: filePath,
          source: 'training_folder',
          job_id: job.id,
          step: parseCheckpointStep(entry.name, job.step || null),
          exists: stats.exists,
          size: stats.size,
          modified_at: stats.modified_at,
          metadata: JSON.stringify({ fileName: entry.name }),
        });
      }
      if (lower.endsWith('.aitk.zip')) {
        const stats = statInfo(filePath);
        artifacts.push({
          id: stableID(['export', job.id, filePath]),
          kind: 'export',
          name: entry.name,
          path: filePath,
          source: 'training_folder',
          job_id: job.id,
          step: job.step || null,
          exists: stats.exists,
          size: stats.size,
          modified_at: stats.modified_at,
          metadata: JSON.stringify({ fileName: entry.name }),
        });
      }
    }
  } catch {
    // Missing job folders are represented by absent checkpoint artifacts.
  }

  return artifacts;
}

export async function reindexModelArtifacts() {
  const [jobs, trainingRoot] = await Promise.all([db.jobs.list({}), getTrainingFolder()]);
  const grouped = await Promise.all(jobs.map(job => indexJobArtifacts(job, trainingRoot)));
  const artifacts = grouped.flat();
  await db.modelArtifacts.upsertMany(artifacts);
  return artifacts;
}

export async function listModelArtifacts() {
  return db.modelArtifacts.list();
}
