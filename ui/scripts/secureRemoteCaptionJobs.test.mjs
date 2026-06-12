import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  getDirectRemoteOllamaWorkerId,
  getSecureRemoteOllamaWorkerId,
  isAnyRemoteOllamaCaptionJob,
  isDirectRemoteOllamaCaptionJob,
  isSecureRemoteOllamaCaptionJob,
  rewriteDirectRemoteOllamaCaptionersForLocalOllama,
} = require('../dist/src/server/secureRemoteCaptionJobs.js');

test('detects secure remote Ollama caption jobs', () => {
  const jobConfig = {
    config: {
      process: [
        {
          type: 'SecureRemoteOllamaCaptioner',
          caption: {
            remote_worker_id: 'remote-worker-1',
          },
        },
      ],
    },
  };

  assert.equal(getSecureRemoteOllamaWorkerId(jobConfig), 'remote-worker-1');
  assert.equal(isSecureRemoteOllamaCaptionJob(jobConfig), true);
  assert.equal(isAnyRemoteOllamaCaptionJob(jobConfig), true);
});

test('detects direct remote Ollama caption jobs', () => {
  const jobConfig = {
    config: {
      process: [
        {
          type: 'SecureRemoteOllamaCaptioner',
          caption: {
            remote_ollama_worker_id: 'ollama-worker-1',
          },
        },
      ],
    },
  };

  assert.equal(getDirectRemoteOllamaWorkerId(jobConfig), 'ollama-worker-1');
  assert.equal(isDirectRemoteOllamaCaptionJob(jobConfig), true);
  assert.equal(isAnyRemoteOllamaCaptionJob(jobConfig), true);
});

test('ignores local or non-secure caption jobs', () => {
  assert.equal(
    isSecureRemoteOllamaCaptionJob({
      config: {
        process: [{ type: 'SecureRemoteOllamaCaptioner', caption: { remote_worker_id: 'local' } }],
      },
    }),
    false,
  );
  assert.equal(
    isSecureRemoteOllamaCaptionJob({
      config: {
        process: [{ type: 'Qwen3VLCaptioner', caption: { remote_worker_id: 'remote-worker-1' } }],
      },
    }),
    false,
  );
});

test('rewrites direct remote Ollama captioners to local Ollama runtime', () => {
  const jobConfig = {
    config: {
      process: [
        {
          type: 'SecureRemoteOllamaCaptioner',
          caption: {
            remote_ollama_worker_id: 'ollama-worker-1',
          },
        },
      ],
    },
  };

  rewriteDirectRemoteOllamaCaptionersForLocalOllama(jobConfig);

  assert.equal(jobConfig.config.process[0].type, 'OllamaCaptioner');
  assert.equal(jobConfig.config.process[0].caption.remote_ollama_worker_id, 'ollama-worker-1');
});
