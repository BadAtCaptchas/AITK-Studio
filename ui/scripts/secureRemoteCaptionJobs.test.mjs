import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  getSecureRemoteOllamaWorkerId,
  isSecureRemoteOllamaCaptionJob,
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
