import { execFile } from 'child_process';
import fs from 'fs';
import fsp from 'fs/promises';
import os from 'os';
import path from 'path';
import { promisify } from 'util';
import { db } from './db';
import { getDatasetsRoot, getTrainingFolder } from './settings';
import { TOOLKIT_ROOT } from '../paths';

const execFileAsync = promisify(execFile);
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp']);

function safeJsonParse<T>(value: string, fallback: T): T {
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function pythonPath() {
  const candidates =
    process.platform === 'win32'
      ? [
          path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe'),
          path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe'),
        ]
      : [path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python'), path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python')];
  return candidates.find(candidate => fs.existsSync(candidate)) ?? 'python';
}

async function listImages(root: string) {
  const out: string[] = [];
  async function walk(current: string) {
    const entries = await fsp.readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const absolutePath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        await walk(absolutePath);
      } else if (entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
        out.push(absolutePath);
      }
    }
  }
  if (fs.existsSync(root)) await walk(root);
  return out.sort();
}

function samplePromptByBasename(jobConfig: any) {
  const samples = jobConfig?.config?.process?.[0]?.sample?.samples;
  if (!Array.isArray(samples)) return new Map<string, string>();
  const prompts = new Map<string, string>();
  samples.forEach((sample: any, index: number) => {
    prompts.set(String(index), String(sample?.prompt ?? ''));
  });
  return prompts;
}

async function buildEvaluationItems(runID: string, jobIDs: string[], referencePath: string | null) {
  const trainingRoot = await getTrainingFolder();
  const items = [];
  for (const jobID of jobIDs) {
    const job = await db.jobs.findById(jobID);
    if (!job) continue;
    const jobConfig = safeJsonParse(job.job_config, {});
    const prompts = samplePromptByBasename(jobConfig);
    const samples = await listImages(path.join(trainingRoot, job.name, 'samples'));
    for (const [index, samplePath] of samples.entries()) {
      const prompt = prompts.get(String(index)) ?? '';
      items.push({
        run_id: runID,
        item_type: 'sample',
        item_id: `${jobID}:${index}`,
        sample_path: samplePath,
        reference_path: referencePath,
        metrics: JSON.stringify({ prompt }),
        status: 'queued',
      });
    }
  }
  await db.evaluations.createItems(items);
}

export async function createEvaluationRun(input: {
  name?: string;
  jobIds: string[];
  artifactIds?: string[];
  referencePath?: string | null;
}) {
  const referencePath =
    input.referencePath && !path.isAbsolute(input.referencePath)
      ? path.join(await getDatasetsRoot(), input.referencePath)
      : input.referencePath || null;
  const run = await db.evaluations.createRun({
    name: input.name || `Evaluation ${new Date().toLocaleString()}`,
    job_ids: JSON.stringify(input.jobIds),
    artifact_ids: JSON.stringify(input.artifactIds ?? []),
    reference_path: referencePath,
  });
  await buildEvaluationItems(run.id, input.jobIds, referencePath);
  return run;
}

async function runEvaluator(samples: Array<{ id: string; sample_path: string; reference_path: string | null; prompt: string }>) {
  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-eval-'));
  const inputPath = path.join(tempDir, 'input.json');
  const outputPath = path.join(tempDir, 'output.json');
  await fsp.writeFile(inputPath, JSON.stringify({ samples }, null, 2), 'utf-8');

  const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'evaluate_samples.py');
  await execFileAsync(pythonPath(), [scriptPath, '--input', inputPath, '--output', outputPath], {
    cwd: TOOLKIT_ROOT,
    timeout: 30 * 60 * 1000,
    maxBuffer: 1024 * 1024 * 8,
  });
  const result = JSON.parse(await fsp.readFile(outputPath, 'utf-8')) as {
    summary: Record<string, unknown>;
    items: Array<{ id: string; metrics: Record<string, unknown>; errors: string[] }>;
  };
  await fsp.rm(tempDir, { recursive: true, force: true });
  return result;
}

export async function processQueuedEvaluationRuns() {
  const [run] = await db.evaluations.listRuns({ status: 'queued', limit: 1 });
  if (!run) return;

  await db.evaluations.updateRun(run.id, { status: 'running', error: null } as any);
  try {
    const items = await db.evaluations.listItems(run.id);
    const samples = items
      .filter(item => item.sample_path)
      .map(item => ({
        id: item.id,
        sample_path: item.sample_path as string,
        reference_path: item.reference_path,
        prompt: safeJsonParse<{ prompt?: string }>(item.metrics, {}).prompt ?? '',
      }));

    if (samples.length === 0) {
      throw new Error('No generated sample images found for this evaluation.');
    }

    const result = await runEvaluator(samples);
    const byID = new Map(result.items.map(item => [item.id, item]));
    for (const item of items) {
      const evaluated = byID.get(item.id);
      if (!evaluated) {
        await db.evaluations.updateItem(item.id, {
          status: 'unavailable',
          error: 'Evaluator returned no result for this item.',
        } as any);
        continue;
      }
      await db.evaluations.updateItem(item.id, {
        status: evaluated.errors.length ? 'completed_with_warnings' : 'completed',
        metrics: JSON.stringify(evaluated.metrics),
        error: evaluated.errors.join('\n') || null,
      } as any);
    }
    await db.evaluations.updateRun(run.id, {
      status: 'completed',
      metrics: JSON.stringify(result.summary),
      completed_at: new Date(),
      error: null,
    } as any);
  } catch (error) {
    await db.evaluations.updateRun(run.id, {
      status: 'failed',
      error: error instanceof Error ? error.message : String(error),
      completed_at: new Date(),
    } as any);
  }
}
