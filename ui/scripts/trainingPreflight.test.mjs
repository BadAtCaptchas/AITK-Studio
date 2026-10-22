import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import test from 'node:test';
import { isAdvisorJobConfig, readWorkerDatasetCheck, runTrainingPreflight } from '../dist/src/server/trainingPreflight.js';
const require = createRequire(import.meta.url);
const remote = require('../dist/src/server/remoteClient.js');
const configFor = folder_path => ({ config: { process: [{ datasets: [{ folder_path, caption_ext: 'txt' }], train: {}, sample: {}, model: {}, save: {}, logging: {} }] } });

test('preflight rejects malformed consumed fields before any filesystem or worker check', () => {
  for (const mutate of [
    process => { process.model.arch = {}; },
    process => { process.model.name_or_path = { toString: 'not callable' }; },
    process => { process.sample.samples = 'not an array'; },
    process => { process.sample.samples = [null]; },
    process => { process.datasets[0].folder_path = {}; },
    process => { process.datasets[0].controls = [1]; },
    process => { process.train.lr = []; },
  ]) {
    const config = configFor('unused');
    mutate(config.config.process[0]);
    assert.equal(isAdvisorJobConfig(config), false);
  }
});

test('worker dataset checks reject traversal and expose names without absolute paths', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-target-preflight-'));
  fs.mkdirSync(path.join(root, 'images'));
  fs.writeFileSync(path.join(root, 'images', 'one.webm'), 'video');
  fs.writeFileSync(path.join(root, 'images', 'one.txt'), 'caption');
  try {
    for (const name of ['../outside', '..', 'C:\\private', 'a/b']) await assert.rejects(readWorkerDatasetCheck({ dataset_name: name }, async () => root), /Invalid dataset/);
    const scan = await readWorkerDatasetCheck({ dataset_name: 'images' }, async () => root);
    assert.equal(scan.path, 'images');
    assert.equal(scan.mediaFiles, 1);
    assert.equal(scan.missingCaptions, 0);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('target preflight sends only same-worker dataset names and marks unavailable checks unknown', async () => {
  const original = { getRemoteWorker: remote.getRemoteWorker, fetchWorkerHealth: remote.fetchWorkerHealth, remoteJson: remote.remoteJson };
  remote.getRemoteWorker = async id => ({ id });
  remote.fetchWorkerHealth = async () => ({ ok: true, executionReady: true, executionDetail: 'Ready' });
  const requests = [];
  remote.remoteJson = async (_worker, route, init) => {
    requests.push({ route, body: JSON.parse(init.body) });
    return { captionExt: '.txt', mediaFiles: 1, captionFiles: 1, missingCaptions: 0, emptyCaptions: 0,
      captionExtensionMismatches: 0, alternateCaptionExtensionCounts: {}, truncated: false, inaccessible: false, placeholder: false };
  };
  try {
    const config = configFor('aitk-dataset://remote/target/images');
    assert.equal(isAdvisorJobConfig(config), true);
    assert.equal(isAdvisorJobConfig({ config: { process: [{ datasets: [null] }] } }), false);
    const checked = await runTrainingPreflight(config, '0', 'target');
    assert.equal(checked.datasetStats.mediaFiles, 1);
    assert.equal(checked.datasetStats.uncheckedDatasets, 0);
    assert.equal(checked.target.state, 'checked');
    assert.deepEqual(requests[0].body, { dataset_name: 'images', caption_ext: 'txt' });
    assert.equal(JSON.stringify(requests).includes('folder_path'), false);
    remote.remoteJson = async () => { throw new Error('old worker'); };
    const unchecked = await runTrainingPreflight(config, '0', 'target');
    assert.equal(unchecked.datasetStats.uncheckedDatasets, 1);
    assert.ok(unchecked.findings.some(item => item.id === 'dataset.0.unchecked'));
    remote.fetchWorkerHealth = async () => { throw new Error('offline'); };
    const offline = await runTrainingPreflight(config, '0', 'target');
    assert.equal(offline.target.state, 'unknown');
    assert.equal(offline.datasetStats.uncheckedDatasets, 1);
  } finally { Object.assign(remote, original); }
});
