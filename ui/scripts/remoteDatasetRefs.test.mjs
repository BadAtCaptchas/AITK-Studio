import assert from 'node:assert/strict';
import test from 'node:test';

import {
  makeRemoteDatasetRef,
  shouldImportRemoteDatasetForWorker,
} from '../dist/src/utils/remoteDatasetRefs.js';
import {
  collectDatasetArchiveMappings,
  collectSameWorkerRemoteDatasetReferences,
  rewriteSameWorkerRemoteDatasetRefs,
} from '../dist/src/server/trainingJobTransfer.js';

function jobConfig(folderPath) {
  return {
    config: {
      process: [
        {
          datasets: [
            {
              folder_path: folderPath,
              control_path_1: [folderPath],
            },
          ],
        },
      ],
    },
  };
}

test('same-worker plain remote datasets are not imported on central save', () => {
  const ref = makeRemoteDatasetRef('worker-1', 'cats');

  assert.equal(shouldImportRemoteDatasetForWorker(ref, 'worker-1', false), false);
  assert.equal(shouldImportRemoteDatasetForWorker(ref, 'worker-2', false), true);
  assert.equal(shouldImportRemoteDatasetForWorker(ref, 'local', false), true);
  assert.equal(shouldImportRemoteDatasetForWorker(ref, 'worker-1', true), true);
});

test('same-worker remote dataset refs rewrite to worker-local paths', () => {
  const ref = makeRemoteDatasetRef('worker-1', 'cats');
  const config = jobConfig(ref);

  const refs = collectSameWorkerRemoteDatasetReferences(config, 'worker-1');
  assert.deepEqual(
    refs.map(item => item.configPath),
    [
      'config.process[0].datasets[0].folder_path',
      'config.process[0].datasets[0].control_path_1[0]',
    ],
  );

  const rewritten = rewriteSameWorkerRemoteDatasetRefs(config, {
    workerID: 'worker-1',
    workerName: 'Remote One',
    datasets: [{ name: 'cats', encrypted: false, path: '/remote/datasets/cats' }],
  });

  assert.equal(rewritten.config.process[0].datasets[0].folder_path, '/remote/datasets/cats');
  assert.deepEqual(rewritten.config.process[0].datasets[0].control_path_1, ['/remote/datasets/cats']);
  assert.equal(config.config.process[0].datasets[0].folder_path, ref);
});

test('remote refs are skipped by central dataset archive collection', async () => {
  const ref = makeRemoteDatasetRef('worker-1', 'cats');
  const result = await collectDatasetArchiveMappings(jobConfig(ref), true, process.cwd());

  assert.deepEqual(result.mappings, []);
  assert.deepEqual(result.warnings, []);
});

test('different-worker refs are left for import or transfer handling', () => {
  const ref = makeRemoteDatasetRef('worker-2', 'cats');
  const config = jobConfig(ref);

  assert.deepEqual(collectSameWorkerRemoteDatasetReferences(config, 'worker-1'), []);
  assert.equal(
    rewriteSameWorkerRemoteDatasetRefs(config, {
      workerID: 'worker-1',
      datasets: [{ name: 'cats', encrypted: false, path: '/remote/datasets/cats' }],
    }),
    config,
  );
});

test('encrypted same-worker refs fail clearly if they survive client import', () => {
  const ref = makeRemoteDatasetRef('worker-1', 'locked');

  assert.throws(
    () =>
      rewriteSameWorkerRemoteDatasetRefs(jobConfig(ref), {
        workerID: 'worker-1',
        workerName: 'Remote One',
        datasets: [{ name: 'locked', encrypted: true, path: '/remote/datasets/locked' }],
      }),
    /Encrypted remote dataset "locked" must be imported/,
  );
});

test('missing same-worker dataset paths fail clearly', () => {
  const ref = makeRemoteDatasetRef('worker-1', 'cats');

  assert.throws(
    () =>
      rewriteSameWorkerRemoteDatasetRefs(jobConfig(ref), {
        workerID: 'worker-1',
        workerName: 'Remote One',
        datasets: [{ name: 'cats', encrypted: false }],
      }),
    /did not report a local path/,
  );
});
