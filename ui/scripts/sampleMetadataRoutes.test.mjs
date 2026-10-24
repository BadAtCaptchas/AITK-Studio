import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import archiver from 'archiver';
import yauzl from 'yauzl';
import ts from 'typescript';
import { loadTrainingSource } from './trainingMigrationFixture.mjs';

function load(relative, dependencies) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const result = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, result, result.exports);
  return result.exports;
}
const sampleMetadata = loadTrainingSource('utils/sampleMetadata.ts');
const remoteAssets = loadTrainingSource('server/remoteAssets.ts');
const recorded = { version: 1, prompt: 'historical', negative_prompt: '', seed: 97, width: 512, height: 512,
  guidance_scale: 4, guidance_rescale: 0, sample_steps: 20, network_multiplier: 1, num_frames: 1, fps: 15,
  step: 100, prompt_index: 0, sampler: 'flowmatch', iteration_id: 'first' };

test('remote sample metadata is validated and rekeyed to the controller mirror reference', async () => {
  const sample = '/api/jobs/worker-job/samples/one.png';
  let response = { samples: [sample], metadata: { [sample]: recorded } };
  const remotePaths = [];
  const { GET } = load('../src/app/api/jobs/[jobID]/samples/route.ts', { '@/server/db': { db: { jobs: { findById: async () => ({ id: 'mirror', worker_id: 'worker', remote_job_id: 'worker-job' }) } } },
    '@/server/remoteClient': { isLocalWorker: () => false, getRemoteWorker: async () => ({}), remotePollingJson: async (_worker, remotePath) => { remotePaths.push(remotePath); return response; }, isRemoteJobMissingError: () => false },
    '@/server/remoteAssets': remoteAssets, '@/server/jobSamples': {}, '@/utils/sampleMetadata': sampleMetadata,
  });
  const request = (query = '?metadata=1') => GET(new Request('http://localhost/api/jobs/mirror/samples' + query), { params: Promise.resolve({ jobID: 'mirror' }) });
  const result = await (await request()).json();
  const ref = remoteAssets.makeRemoteAssetRef('mirror', 'img', sample);
  assert.deepEqual(result, { samples: [ref], metadata: { [ref]: recorded } });
  assert.equal(remotePaths.at(-1), '/api/jobs/worker-job/samples?metadata=1');
  assert.deepEqual(await (await request('')).json(), { samples: [ref] });
  assert.equal(remotePaths.at(-1), '/api/jobs/worker-job/samples');
  response = { samples: [sample] };
  assert.deepEqual((await (await request()).json()).metadata, {}, 'older workers remain supported');
  response = { samples: [sample], metadata: { [sample]: { ...recorded, seed: 'invalid' } } };
  assert.deepEqual((await (await request()).json()).metadata, {});
});

test('dashboard sample polling does not read metadata sidecars', async () => {
  let metadataReads = 0;
  const samples = ['/api/jobs/local-job/samples/one.png'];
  const metadata = { [samples[0]]: recorded };
  const { GET } = load('../src/app/api/jobs/[jobID]/samples/route.ts', { '@/server/db': { db: { jobs: { findById: async () => ({ id: 'local-job', worker_id: 'local' }) } } },
    '@/server/remoteClient': { isLocalWorker: () => true }, '@/server/remoteAssets': remoteAssets,
    '@/server/jobSamples': { listJobSampleUrls: async () => samples, listJobSampleMetadata: async () => { metadataReads++; return metadata; } },
    '@/utils/sampleMetadata': sampleMetadata,
  });
  const request = query => GET(new Request('http://localhost/api/jobs/local-job/samples' + query), { params: Promise.resolve({ jobID: 'local-job' }) });
  assert.deepEqual(await (await request('')).json(), { samples });
  assert.equal(metadataReads, 0);
  assert.deepEqual(await (await request('?metadata=1')).json(), { samples, metadata });
  assert.equal(metadataReads, 1);
});

test('sample ZIP exports include provenance and exclude staging directories', async t => {
  const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-sample-export-'));
  t.after(() => fsp.rm(root, { recursive: true, force: true }));
  const samples = path.join(root, 'job-storage', 'samples');
  await fsp.mkdir(path.join(samples, '.tmp'), { recursive: true });
  await fsp.writeFile(path.join(samples, 'one.png'), 'media fixture');
  await fsp.writeFile(path.join(samples, 'one.png.metadata.json'), JSON.stringify(recorded));
  await fsp.writeFile(path.join(samples, '.tmp', 'partial.png'), 'uncommitted fixture');
  const { createRequestedArchive } = load('../src/server/archiveDownloads.ts', {
    '../utils/jobIdentity': { jobStorageKey: () => 'job-storage' }, fs, 'fs/promises': fsp, path, archiver,
    './db': { db: { jobs: { findById: async () => ({ id: 'job', worker_id: 'local' }) } } },
    './datasetScope': { isPathInside: (parent, child) => { const rel = path.relative(parent, child); return rel !== '..' && !rel.startsWith('..' + path.sep) && !path.isAbsolute(rel); } },
    './encryptedDatasets': {}, './trainingPaths': { getJobTrainingRoot: async () => root },
    './remoteClient': { isLocalWorker: () => true }, './remoteAssets': remoteAssets, './layeredImages': {},
  });
  const result = await createRequestedArchive({ zipTarget: 'samples', jobID: 'job' });
  const names = await new Promise((resolve, reject) => {
    yauzl.open(result.zipPath, { lazyEntries: true }, (error, zip) => {
      if (error) return reject(error);
      const entries = [];
      zip.on('entry', entry => { entries.push(entry.fileName); zip.readEntry(); });
      zip.on('error', reject);
      zip.on('close', () => resolve(entries));
      zip.readEntry();
    });
  });
  assert.deepEqual(names.sort(), ['samples/one.png', 'samples/one.png.metadata.json']);
  const outside = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-sample-outside-'));
  t.after(() => fsp.rm(outside, { recursive: true, force: true }));
  const moved = path.join(outside, 'job-storage');
  await fsp.rename(path.join(root, 'job-storage'), moved);
  await fsp.symlink(moved, path.join(root, 'job-storage'), 'junction');
  await assert.rejects(createRequestedArchive({ zipTarget: 'samples', jobID: 'job' }), /Invalid job output path/);

});
