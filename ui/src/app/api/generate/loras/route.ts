import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { db } from '@/server/db';
import { getTrainingFolder } from '@/server/settings';

const LORA_JOB_TYPES = new Set(['lora', 'locon', 'lokr', 'lorm']);

type LoraModelSummary = {
  name_or_path?: string;
  arch?: string;
  quantize?: boolean;
  quantize_te?: boolean;
  qtype?: string;
  qtype_te?: string;
  low_vram?: boolean;
  layer_offloading?: boolean;
  layer_offloading_transformer_percent?: number;
  layer_offloading_text_encoder_percent?: number;
  model_kwargs?: Record<string, unknown>;
  extras_name_or_path?: string;
  vae_path?: string;
  refiner_name_or_path?: string;
  te_name_or_path?: string;
  quantize_kwargs?: Record<string, unknown>;
};

function parseJobConfig(jobConfig: string) {
  try {
    return JSON.parse(jobConfig);
  } catch {
    return null;
  }
}

function isLoraTrainingJob(jobConfig: any) {
  const networkType = String(jobConfig?.config?.process?.[0]?.network?.type || '').toLowerCase();
  return LORA_JOB_TYPES.has(networkType);
}

function getModelSummary(jobConfig: any): LoraModelSummary {
  const model = jobConfig?.config?.process?.[0]?.model || {};
  return {
    name_or_path: model.name_or_path,
    arch: model.arch,
    quantize: model.quantize,
    quantize_te: model.quantize_te,
    qtype: model.qtype,
    qtype_te: model.qtype_te,
    low_vram: model.low_vram,
    layer_offloading: model.layer_offloading,
    layer_offloading_transformer_percent: model.layer_offloading_transformer_percent,
    layer_offloading_text_encoder_percent: model.layer_offloading_text_encoder_percent,
    model_kwargs: model.model_kwargs,
    extras_name_or_path: model.extras_name_or_path,
    vae_path: model.vae_path,
    refiner_name_or_path: model.refiner_name_or_path,
    te_name_or_path: model.te_name_or_path,
    quantize_kwargs: model.quantize_kwargs,
  };
}

async function getSafeJobFolder(trainingRoot: string, jobName: string) {
  const root = await fs.promises.realpath(trainingRoot).catch(() => null);
  if (!root) return null;

  const folder = path.resolve(root, jobName);
  const relativePath = path.relative(root, folder);
  if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    return null;
  }

  const stat = await fs.promises.stat(folder).catch(() => null);
  if (!stat?.isDirectory()) {
    return null;
  }

  return folder;
}

export async function GET() {
  try {
    const trainingRoot = await getTrainingFolder();
    const jobs = await db.jobs.list({ job_type: 'train' });
    const loras = [];

    for (const job of jobs) {
      if (job.worker_id && job.worker_id !== 'local') continue;

      const jobConfig = parseJobConfig(job.job_config);
      if (!jobConfig || !isLoraTrainingJob(jobConfig)) continue;

      const jobFolder = await getSafeJobFolder(trainingRoot, job.name);
      if (!jobFolder) continue;

      const entries = await fs.promises.readdir(jobFolder, { withFileTypes: true }).catch(() => []);
      for (const entry of entries) {
        if (!entry.isFile() || !entry.name.toLowerCase().endsWith('.safetensors')) continue;

        const filePath = path.join(jobFolder, entry.name);
        const stat = await fs.promises.stat(filePath).catch(() => null);
        if (!stat) continue;

        loras.push({
          id: `${job.id}:${entry.name}`,
          label: `${job.name} / ${entry.name}`,
          path: filePath,
          filename: entry.name,
          jobId: job.id,
          jobName: job.name,
          jobStatus: job.status,
          updatedAt: stat.mtime.toISOString(),
          sizeBytes: stat.size,
          model: getModelSummary(jobConfig),
        });
      }
    }

    loras.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
    return NextResponse.json({ loras });
  } catch (error) {
    console.error('Error listing generated LoRAs:', error);
    return NextResponse.json({ error: 'Failed to list generated LoRAs' }, { status: 500 });
  }
}
