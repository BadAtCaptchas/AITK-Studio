import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  buildInitialRemoteCaptionState,
  buildRemoteOllamaCaptionJobConfig,
  getRemoteCaptionState,
  isRemoteCaptionDispatchConfig,
  setRemoteCaptionState,
} = require('../dist/src/server/remoteCaptionJobs.js');

test('detects and rewrites remote Ollama caption dispatch config', () => {
  const centralConfig = {
    job: 'extension',
    config: {
      name: 'central_caption',
      process: [
        {
          type: 'SecureRemoteOllamaCaptioner',
          device: 'cpu',
          caption: {
            model_name_or_path: 'llava:latest',
            path_to_caption: '/datasets/cats',
            caption_extension: 'txt',
            remote_worker_id: 'worker-1',
          },
        },
      ],
    },
  };

  assert.equal(isRemoteCaptionDispatchConfig(centralConfig), true);

  const remoteConfig = buildRemoteOllamaCaptionJobConfig(centralConfig, {
    remoteDatasetPath: '/remote/datasets/cats_remote',
    remoteJobName: 'remote_caption',
  });

  assert.equal(remoteConfig.config.name, 'remote_caption');
  assert.equal(remoteConfig.config.process[0].type, 'OllamaCaptioner');
  assert.equal(remoteConfig.config.process[0].caption.path_to_caption, '/remote/datasets/cats_remote');
  assert.equal('remote_worker_id' in remoteConfig.config.process[0].caption, false);
  assert.equal(centralConfig.config.process[0].type, 'SecureRemoteOllamaCaptioner');
});

test('stores remote caption state in job config', () => {
  const job = { id: 'job_1234567890', name: 'caption_job' };
  const state = buildInitialRemoteCaptionState({
    job,
    worker: { id: 'worker-1', name: 'DGX' },
    originalDatasetPath: '/datasets/cats',
    encrypted: false,
    durableEncryptedKeys: false,
    captionExtension: 'txt',
    recaption: true,
  });

  const jobConfig = setRemoteCaptionState({ job: 'extension', config: { process: [] } }, state);
  const recovered = getRemoteCaptionState(jobConfig);

  assert.equal(recovered.originalDatasetName, 'cats');
  assert.equal(recovered.remoteWorkerId, 'worker-1');
  assert.equal(recovered.downloadStatus, 'dispatching');
  assert.equal(recovered.recaption, true);
});
