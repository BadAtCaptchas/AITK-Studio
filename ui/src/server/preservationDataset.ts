import fs from 'fs/promises';
import path from 'path';
import { db } from './db';
import { resolveDatasetScope } from './datasetScope';
import { resolveJobSampleFile } from './jobSamples';
import { readSampleMetadata } from './sampleMetadata';
import { publishDatasetDirectory } from './publishDatasetDirectory';
import { hashContinuationAsset } from './loraContinuation';
import { isRecord } from './commandInput';
import { validSha256 } from '../domain/loraContinuation';
import { getRemoteWorker, isLocalWorker, remoteJson } from './remoteClient';
import { makeRemoteDatasetRef } from '../utils/remoteDatasetRefs';
import { readPreservationPublication, type PreservationReceipt } from './preservationPublication';

export async function createPreservationDataset(value: unknown): Promise<PreservationReceipt> {
  if (!isRecord(value) || typeof value.job_id !== 'string' || !/^[a-zA-Z0-9_-]{1,128}$/.test(value.job_id) ||
      typeof value.name !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9 _-]{0,79}$/.test(value.name) ||
      typeof value.request_id !== 'string' || !/^[a-f0-9-]{36}$/.test(value.request_id) ||
      !Array.isArray(value.samples) || value.samples.length === 0 || value.samples.length > 512 ||
      !value.samples.every(item => typeof item === 'string' && item === path.basename(item) && !/[\\/\0]/.test(item)))
    throw new Error('Provide a generation job, selected sample filenames, dataset name and request ID.');
  if (new Set(value.samples).size !== value.samples.length) throw new Error('Duplicate preservation samples.');
  if (value.name !== value.name.trim() || /^(con|prn|aux|nul|com[0-9]|lpt[0-9])$/i.test(value.name)) throw new Error('Choose another dataset name.');
  const job = await db.jobs.findById(value.job_id);
  if (!job || job.job_type !== 'generate') throw new Error('Select a generation job.');
  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) throw new Error('Generation job has no remote identity.');
    const worker = await getRemoteWorker(job.worker_id!);
    const result: unknown = await remoteJson(worker, '/api/training/lora/preservation', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...value, job_id: job.remote_job_id }),
    });
    if (!isRecord(result) || typeof result.name !== 'string' || result.name !== value.name ||
        !validSha256(result.source_sha256) || typeof result.count !== 'number' || result.count !== value.samples.length)
      throw new Error('Invalid worker preservation response.');
    return { name: result.name, path: makeRemoteDatasetRef(worker.id, result.name), source_sha256: result.source_sha256, count: result.count };
  }
  const { datasetsRoot } = await resolveDatasetScope();
  await fs.mkdir(datasetsRoot, { recursive: true });
  const root = await fs.realpath(datasetsRoot);
  const destination = path.join(root, value.name);
  const requestID = value.request_id;
  const samples = value.samples;
  const readPublished = () => readPreservationPublication(destination, requestID, job.id, samples);
  const existing = await readPublished();
  if (existing) return existing;
  const staging = await fs.mkdtemp(path.join(root, '.aitk-preservation-'));
  try {
    const examples = [];
    let sourceHash: string | undefined, sourceBase: string | undefined, totalBytes = 0;
    for (const [index, filename] of value.samples.entries()) {
      const sample = await resolveJobSampleFile(job, filename);
      if (!sample || !['.png', '.jpg', '.jpeg', '.webp'].includes(path.extname(filename).toLowerCase())) throw new Error('Select ordinary image samples only.');
      totalBytes += sample.stat.size;
      if (totalBytes > 2 * 1024 ** 3 || sample.stat.size > 64 * 1024 ** 2) throw new Error('Preservation dataset exceeds the size limit.');
      const metadata = await readSampleMetadata(sample.path);
      if (!metadata || !validSha256(metadata.source_lora_sha256) || !metadata.source_base || !metadata.prompt.trim())
        throw new Error('Samples need recorded source-LoRA provenance and captions. Generate new candidates with this version of AITK.');
      if (!metadata.media_sha256 && job.status !== 'completed')
        throw new Error('Older samples can be curated after generation completes. New candidates include image checksums.');
      sourceHash ??= metadata.source_lora_sha256;
      sourceBase ??= metadata.source_base;
      if (sourceHash !== metadata.source_lora_sha256 || sourceBase !== metadata.source_base) throw new Error('All preservation examples must use the same source LoRA and base.');
      const image = `${String(index).padStart(4, '0')}${path.extname(filename).toLowerCase()}`;
      const caption = `${String(index).padStart(4, '0')}.txt`;
      await fs.copyFile(sample.path, path.join(staging, image));
      const imageHash = await hashContinuationAsset(path.join(staging, image));
      const latest = await resolveJobSampleFile(job, filename);
      if (!latest || latest.path !== sample.path || latest.stat.size !== sample.stat.size || latest.stat.mtimeMs !== sample.stat.mtimeMs ||
          latest.stat.ctimeMs !== sample.stat.ctimeMs || latest.stat.ino !== sample.stat.ino ||
          JSON.stringify(await readSampleMetadata(sample.path)) !== JSON.stringify(metadata) ||
          (metadata.media_sha256 && metadata.media_sha256 !== imageHash))
        throw new Error('Sample changed during curation; refresh the samples and try again.');
      await fs.writeFile(path.join(staging, caption), metadata.prompt, { flag: 'wx' });
      examples.push({ image, caption, image_sha256: imageHash,
        caption_sha256: await hashContinuationAsset(path.join(staging, caption)), source_sample: filename, metadata });
    }
    if (!sourceHash) throw new Error('No preservation examples selected.');
    const manifest = { version: 1, request_id: value.request_id, source_sha256: sourceHash, source_base: sourceBase,
      generation_job: job.id, selected_samples: value.samples, created_at: new Date().toISOString(), examples, evaluation_examples: [] };
    await fs.writeFile(path.join(staging, '.aitk-preservation.json'), JSON.stringify(manifest, null, 2), { flag: 'wx' });
    try {
      await publishDatasetDirectory(staging, destination);
    } catch (error) {
      // A concurrent retry may have published the same request first. Never
      // replace its directory, but acknowledge its verified snapshot.
      const published = await readPublished();
      if (published) return published;
      throw error;
    }
    return { path: destination, name: value.name, source_sha256: sourceHash, count: examples.length };
  } finally {
    // Only this operation's private staging directory is disposable.
    if (path.dirname(staging) === root) await fs.rm(staging, { recursive: true, force: true });
  }
}
