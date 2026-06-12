import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { syncRemoteDatasetsForJobConfig } from '../dist/src/server/remoteDatasetSync.js';

function makeWorker() {
  return {
    id: 'worker-1',
    name: 'Remote One',
    base_url: 'https://worker.example',
    api_token: 'token',
    enabled: true,
    last_status: 'ready',
    last_error: null,
    last_checked_at: null,
    capabilities: '{}',
    gpus: '[]',
    created_at: new Date(),
    updated_at: new Date(),
  };
}

function makeConfig(folderPath, extra = {}) {
  return {
    config: {
      process: [
        {
          datasets: [
            {
              folder_path: folderPath,
              control_path_1: [folderPath],
              ...extra,
            },
          ],
        },
      ],
    },
  };
}

function makeTempDatasets() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-remote-sync-'));
  const cats = path.join(root, 'cats');
  const dogs = path.join(root, 'dogs');
  fs.mkdirSync(path.join(cats, 'subdir'), { recursive: true });
  fs.mkdirSync(dogs, { recursive: true });
  fs.writeFileSync(path.join(cats, 'image.txt'), 'caption');
  fs.writeFileSync(path.join(cats, 'subdir', 'nested.txt'), 'nested');
  fs.writeFileSync(path.join(dogs, 'image.txt'), 'caption');
  return { root, cats, dogs };
}

function makeDeps(root, remoteDatasets = []) {
  const calls = {
    archives: [],
    uploads: [],
    removed: [],
    progress: [],
  };

  const deps = {
    getDatasetsRoot: async () => root,
    listRemoteDatasets: async () => remoteDatasets,
    createDatasetArchive: async (datasetName, datasetFolder, outputPath) => {
      calls.archives.push({ datasetName, datasetFolder, outputPath });
      await fs.promises.mkdir(path.dirname(outputPath), { recursive: true });
      await fs.promises.writeFile(outputPath, `archive:${datasetName}`);
    },
    uploadDatasetArchive: async (_worker, zipPath, preferredName, onProgress) => {
      calls.uploads.push({ zipPath, preferredName });
      const size = fs.statSync(zipPath).size;
      onProgress?.({ loaded: 0, total: size });
      onProgress?.({ loaded: size, total: size });
      return {
        dataset: { name: preferredName, encrypted: false, path: `/remote/datasets/${preferredName}` },
        path: `/remote/datasets/${preferredName}`,
        renamed: false,
      };
    },
    stat: targetPath => fs.promises.stat(targetPath),
    realpath: targetPath => fs.promises.realpath(targetPath),
    rmPath: async (targetPath, options) => {
      calls.removed.push(targetPath);
      await fs.promises.rm(targetPath, options);
    },
  };

  return { deps, calls };
}

test('remote dataset sync reuses same-named worker dataset without uploading', async () => {
  const { root, cats } = makeTempDatasets();
  const { deps, calls } = makeDeps(root, [{ name: 'cats', encrypted: false, path: '/remote/datasets/cats' }]);

  try {
    const result = await syncRemoteDatasetsForJobConfig(makeConfig(cats), makeWorker(), { deps });

    assert.equal(result.jobConfig.config.process[0].datasets[0].folder_path, '/remote/datasets/cats');
    assert.deepEqual(result.jobConfig.config.process[0].datasets[0].control_path_1, ['/remote/datasets/cats']);
    assert.equal(result.mappings[0].uploaded, false);
    assert.equal(calls.uploads.length, 0);
    assert.equal(calls.archives.length, 0);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('remote dataset sync uploads missing dataset once and rewrites duplicate refs', async () => {
  const { root, cats } = makeTempDatasets();
  const { deps, calls } = makeDeps(root, []);

  try {
    const result = await syncRemoteDatasetsForJobConfig(makeConfig(cats), makeWorker(), {
      deps,
      onProgress: progress => calls.progress.push(progress),
    });

    assert.equal(calls.archives.length, 1);
    assert.equal(calls.uploads.length, 1);
    assert.equal(calls.uploads[0].preferredName, 'cats');
    assert.equal(result.jobConfig.config.process[0].datasets[0].folder_path, '/remote/datasets/cats');
    assert.deepEqual(result.jobConfig.config.process[0].datasets[0].control_path_1, ['/remote/datasets/cats']);
    assert.equal(result.mappings[0].uploaded, true);
    assert.ok(calls.progress.some(progress => progress.status === 'uploading-dataset'));
    assert.ok(calls.progress.some(progress => progress.status === 'importing-dataset'));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('remote dataset sync uploads multiple missing datasets for one training job', async () => {
  const { root, cats, dogs } = makeTempDatasets();
  const { deps, calls } = makeDeps(root, []);

  const config = {
    config: {
      process: [
        {
          datasets: [
            { folder_path: cats, control_path_1: [cats] },
            { folder_path: dogs, control_path_1: [dogs] },
          ],
        },
      ],
    },
  };

  try {
    const result = await syncRemoteDatasetsForJobConfig(config, makeWorker(), {
      deps,
      onProgress: progress => calls.progress.push(progress),
    });

    assert.equal(calls.archives.length, 2);
    assert.deepEqual(calls.uploads.map(upload => upload.preferredName).sort(), ['cats', 'dogs']);
    assert.equal(result.jobConfig.config.process[0].datasets[0].folder_path, '/remote/datasets/cats');
    assert.equal(result.jobConfig.config.process[0].datasets[1].folder_path, '/remote/datasets/dogs');
    assert.deepEqual(result.jobConfig.config.process[0].datasets[0].control_path_1, ['/remote/datasets/cats']);
    assert.deepEqual(result.jobConfig.config.process[0].datasets[1].control_path_1, ['/remote/datasets/dogs']);
    assert.equal(result.mappings.length, 2);
    assert.equal(result.mappings.every(mapping => mapping.uploaded), true);
    assert.ok(calls.progress.filter(progress => progress.status === 'importing-dataset').length >= 2);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('remote dataset sync preserves subpaths inside uploaded dataset', async () => {
  const { root, cats } = makeTempDatasets();
  const subdir = path.join(cats, 'subdir');
  const { deps } = makeDeps(root, [{ name: 'cats', encrypted: false, path: '/remote/datasets/cats' }]);

  try {
    const result = await syncRemoteDatasetsForJobConfig(makeConfig(subdir), makeWorker(), { deps });

    assert.equal(result.jobConfig.config.process[0].datasets[0].folder_path, '/remote/datasets/cats/subdir');
    assert.deepEqual(result.jobConfig.config.process[0].datasets[0].control_path_1, ['/remote/datasets/cats/subdir']);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('remote dataset sync warns and leaves outside dataset paths unchanged', async () => {
  const { root } = makeTempDatasets();
  const outsideRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-outside-'));
  const outsideDataset = path.join(outsideRoot, 'outside');
  fs.mkdirSync(outsideDataset, { recursive: true });
  const { deps, calls } = makeDeps(root, []);

  try {
    const result = await syncRemoteDatasetsForJobConfig(makeConfig(outsideDataset), makeWorker(), { deps });

    assert.deepEqual(result.jobConfig, makeConfig(outsideDataset));
    assert.match(result.warnings[0], /outside the datasets folder/);
    assert.equal(calls.uploads.length, 0);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(outsideRoot, { recursive: true, force: true });
  }
});
