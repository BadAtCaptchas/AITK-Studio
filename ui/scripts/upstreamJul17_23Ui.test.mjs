import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { startSettledPollLoop } from '../dist/src/utils/basic.js';
import { SharedAbortableRequestPool } from '../dist/src/utils/sharedAbortableRequest.js';
import {
  DEFAULT_SQLITE_JOURNAL_MODE,
  resolveSqliteJournalMode,
} from './sqlite-journal-mode.mjs';
import {
  getJobValidationConfigErrors,
  getValidationConfigErrors,
} from '../dist/src/utils/validationConfig.js';

const scriptsDirectory = path.dirname(fileURLToPath(import.meta.url));
const uiDirectory = path.dirname(scriptsDirectory);
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const read = relativePath => readFileSync(path.join(uiDirectory, relativePath), 'utf8');

test('settled poll loops never overlap slow requests', async () => {
  let active = 0;
  let maximumActive = 0;
  let calls = 0;

  const stop = startSettledPollLoop(async () => {
    calls += 1;
    active += 1;
    maximumActive = Math.max(maximumActive, active);
    await delay(20);
    active -= 1;
  }, 1);

  await delay(72);
  stop();
  await delay(25);

  assert.equal(maximumActive, 1);
  assert.ok(calls >= 2, `expected at least two polls, received ${calls}`);
});

test('settled poll loops abort active work and support one-shot loading', async () => {
  let observedAbort = false;
  const stop = startSettledPollLoop(
    signal =>
      new Promise(resolve => {
        signal.addEventListener(
          'abort',
          () => {
            observedAbort = true;
            resolve();
          },
          { once: true },
        );
      }),
    1,
  );

  await delay(5);
  stop();
  await delay(5);
  assert.equal(observedAbort, true);

  let oneShotCalls = 0;
  const stopOneShot = startSettledPollLoop(() => {
    oneShotCalls += 1;
  }, null);
  await delay(10);
  stopOneShot();
  assert.equal(oneShotCalls, 1);
});

test('shared request subscribers deduplicate transport and abort it only after the last subscriber leaves', async () => {
  let factoryCalls = 0;
  let transportAborts = 0;
  const pending = new Map();
  const pool = new SharedAbortableRequestPool((key, signal) => {
    factoryCalls += 1;
    return new Promise((resolve, reject) => {
      pending.set(key, { resolve, reject });
      signal.addEventListener(
        'abort',
        () => {
          transportAborts += 1;
          const error = new Error('Transport aborted');
          error.name = 'AbortError';
          reject(error);
        },
        { once: true },
      );
    });
  });

  const firstController = new AbortController();
  const secondController = new AbortController();
  const first = pool.subscribe('local', firstController.signal);
  const second = pool.subscribe('local', secondController.signal);
  await delay(0);

  assert.equal(factoryCalls, 1);
  firstController.abort();
  await assert.rejects(first, error => error?.name === 'AbortError');
  assert.equal(transportAborts, 0);

  pending.get('local').resolve({ worker: 'local' });
  assert.deepEqual(await second, { worker: 'local' });
  assert.equal(transportAborts, 0);

  const loneController = new AbortController();
  const abandoned = pool.subscribe('remote', loneController.signal);
  await delay(0);
  loneController.abort();
  await assert.rejects(abandoned, error => error?.name === 'AbortError');
  await delay(0);
  assert.equal(transportAborts, 1);

  const replacement = pool.subscribe('remote');
  await delay(0);
  assert.equal(factoryCalls, 3);
  pending.get('remote').resolve({ worker: 'remote' });
  assert.deepEqual(await replacement, { worker: 'remote' });
});

test('sqlite journal mode override is normalized and safely falls back to WAL', () => {
  const warnings = [];
  assert.equal(resolveSqliteJournalMode(undefined, message => warnings.push(message)), DEFAULT_SQLITE_JOURNAL_MODE);
  assert.equal(resolveSqliteJournalMode(' delete ', message => warnings.push(message)), 'DELETE');
  assert.equal(resolveSqliteJournalMode('memory', message => warnings.push(message)), 'MEMORY');
  assert.equal(resolveSqliteJournalMode('not-a-mode', message => warnings.push(message)), DEFAULT_SQLITE_JOURNAL_MODE);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /Invalid AI_TOOLKIT_DB_JOURNAL_MODE/);
});

test('validation uploads and previews retain project-space boundaries', () => {
  const uploadRoute = read('src/app/api/img/upload/route.ts');
  const controlImage = read('src/components/SampleControlImage.tsx');
  const simpleJob = read('src/app/jobs/new/SimpleJob.tsx');
  const jobConfig = read('src/app/jobs/new/jobConfig.ts');
  const streamedUploads = read('src/utils/streamedUploads.ts');

  assert.match(uploadRoute, /resolveOptionalProject\(projectID \|\| null, \{ intent: 'write' \}\)/);
  assert.match(uploadRoute, /getProjectValidationRoot\(project\)/);
  assert.match(controlImage, /\/api\/projects\/\$\{encodeURIComponent\(projectID\)\}\/files/);
  assert.match(controlImage, /params: \{ path: src \}/);
  assert.match(streamedUploads, /'X-AITK-Project-ID'/);
  assert.match(jobConfig, /resolution: 1024/);
  assert.match(jobConfig, /validate_every_n_steps: 1/);
  assert.match(jobConfig, /validation_sigmas: \[0\.5\]/);
  assert.match(simpleJob, /objectCopy\(defaultValidationConfig\)/);
  assert.match(simpleJob, /projectID=\{projectID\}/);
  assert.doesNotMatch(simpleJob, /!item\.prompt\.trim\(\)/);
});

test('validation settings are rejected at the real save boundary', () => {
  const valid = {
    validation_items: [{ image_path: 'assets/validation/held-out.png', prompt: '' }],
    resolution: 1024,
    validate_every_n_steps: 1,
    validation_sigmas: [0.5],
  };
  assert.deepEqual(getValidationConfigErrors(valid), []);
  assert.match(getValidationConfigErrors({ ...valid, validation_items: [] })[0], /held-out image/);
  assert.match(getValidationConfigErrors({ ...valid, resolution: 32 })[0], /resolution/);
  assert.match(getValidationConfigErrors({ ...valid, validate_every_n_steps: 0 })[0], /cadence/);
  assert.match(getValidationConfigErrors({ ...valid, validation_sigmas: [Number.NaN] })[0], /sigmas/);
  assert.equal(
    getJobValidationConfigErrors({
      config: { process: [{ train: { validation_config: { ...valid, validation_items: [{ image_path: '' }] } } }] },
    }).length,
    1,
  );

  const trainingForm = read('src/app/jobs/new/TrainingFormContent.tsx');
  const jobsRoute = read('src/app/api/jobs/route.ts');
  assert.match(trainingForm, /getValidationConfigErrors\(trainConfig\?\.validation_config\)/);
  assert.match(jobsRoute, /getJobValidationConfigErrors\(projectJobConfig\)/);
});

test('updates to linked remote project jobs resync and relink portable project inputs', () => {
  const jobsRoute = read('src/app/api/jobs/route.ts');

  assert.match(jobsRoute, /import \{ prepareProjectJobReplica \} from '@\/server\/projectSync'/);
  assert.match(
    jobsRoute,
    /if \(existing\.project_id\) \{[\s\S]*prepareProjectJobReplica\(\{[\s\S]*job_config: JSON\.stringify\(projectJobConfig\)/,
  );
  assert.match(jobsRoute, /remote_job_id: linked\.remoteJob\.id/);
});

test('sample grids and validation loss defaults include the upstream behavior', () => {
  const samples = read('src/components/SampleImages.tsx');
  const lossGraph = read('src/components/JobLossGraph.tsx');

  assert.match(samples, /gridTemplateColumns: `repeat\(\$\{gridCols\}, minmax\(0, 1fr\)\)`/);
  assert.doesNotMatch(samples, /grid-cols-40/);
  assert.match(lossGraph, /key === defaultKey \|\| key === 'val\/loss'/);
  assert.match(lossGraph, /persistedEnabledRef\.current\?\.\[key\] \?\?/);
});

test('network polling hooks use the settled loop instead of setInterval', () => {
  const pollingFiles = [
    'src/hooks/useJob.tsx',
    'src/hooks/useJobByRef.tsx',
    'src/hooks/useJobsList.tsx',
    'src/hooks/useFilesList.tsx',
    'src/hooks/useSampleImages.tsx',
    'src/hooks/useCPUInfo.tsx',
    'src/hooks/useGPUInfo.tsx',
    'src/hooks/useJobLog.tsx',
    'src/hooks/useJobLossLog.tsx',
    'src/hooks/useJobMetrics.tsx',
    'src/hooks/useJobDownloadProgress.tsx',
    'src/hooks/useJobComfyInstallProgress.tsx',
    'src/hooks/useTensorBoardStatus.tsx',
    'src/components/OstrisCloudBalance.tsx',
  ];

  for (const file of pollingFiles) {
    const source = read(file);
    assert.match(source, /usePollLoop/, `${file} should use the settled poll loop`);
    assert.doesNotMatch(source, /setInterval\s*\(/, `${file} should not use setInterval`);
  }
});

test('polling hooks guard request identity and clear data when their request scope changes', () => {
  const gpu = read('src/hooks/useGPUInfo.tsx');
  const metrics = read('src/hooks/useJobMetrics.tsx');
  const job = read('src/hooks/useJob.tsx');
  const files = read('src/hooks/useFilesList.tsx');
  const samples = read('src/hooks/useSampleImages.tsx');
  const jobs = read('src/hooks/useJobsList.tsx');

  assert.match(gpu, /SharedAbortableRequestPool/);
  assert.match(gpu, /signal => fetchGpuInfo\(\{ signal \}\)/);
  assert.match(metrics, /controller\.signal\.aborted \|\|\s+abortRef\.current !== controller/);
  assert.match(metrics, /if \(abortRef\.current === controller\)/);
  assert.match(metrics, /activeScopeRef\.current !== currentRequestScope/);
  assert.match(metrics, /signal => refreshMetrics\(\{ full: needsFullRefreshRef\.current, signal \}\)/);

  assert.match(job, /setJob\(null\);\s+setStatus\('idle'\);/);
  assert.match(job, /\[jobID, projectID\]/);
  assert.match(files, /activeJobIDRef\.current !== requestJobID/);
  assert.match(files, /setFiles\(\[\]\);/);
  assert.match(samples, /activeJobIDRef\.current !== requestJobID/);
  assert.match(samples, /setSampleImages\(\[\]\);/);
  assert.match(jobs, /activeScopeRef\.current !== currentRequestScope/);
  assert.match(jobs, /setJobs\(\[\]\);/);
});
