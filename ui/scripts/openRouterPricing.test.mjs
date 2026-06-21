import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

import { getOpenRouterCaptionPricing } from '../dist/src/server/openRouterPricing.js';

const require = createRequire(import.meta.url);
const dbModule = require('../dist/src/server/db.js');

const originalFetch = globalThis.fetch;
const originalSettings = dbModule.db.settings;
const envKeys = ['AITK_OFFLINE_MODE', 'AI_TOOLKIT_OFFLINE_MODE'];
const originalEnv = Object.fromEntries(envKeys.map(key => [key, process.env[key]]));

function restoreEnv() {
  for (const key of envKeys) {
    if (originalEnv[key] === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = originalEnv[key];
    }
  }
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  dbModule.db.settings = originalSettings;
  restoreEnv();
});

test('OpenRouter caption pricing does not fetch while offline and still uses fallback pricing', async () => {
  process.env.AITK_OFFLINE_MODE = '1';
  let fetchCalls = 0;
  globalThis.fetch = async () => {
    fetchCalls += 1;
    throw new Error('fetch must not be called while offline');
  };
  dbModule.db.settings = {
    async get() {
      return null;
    },
  };

  const fallback = await getOpenRouterCaptionPricing('x-ai/grok-4.3');
  const unknown = await getOpenRouterCaptionPricing('unknown/model');

  assert.equal(fetchCalls, 0);
  assert.equal(fallback?.source, 'fallback');
  assert.equal(unknown, null);
});
