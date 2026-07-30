import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const uiRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const {
  normalizeTelemetrySetting,
  telemetryChildProcessEnv,
} = require('../dist/src/utils/telemetry.js');

test('telemetry is opt-in and disabled for missing or false values', () => {
  assert.equal(normalizeTelemetrySetting(undefined), false);
  assert.equal(normalizeTelemetrySetting('false'), false);
  assert.equal(normalizeTelemetrySetting('0'), false);
  assert.equal(normalizeTelemetrySetting(true), true);
  assert.equal(normalizeTelemetrySetting('enabled'), true);
});

test('disabled child environment blocks Diffusers and Hugging Face telemetry', () => {
  assert.deepEqual(telemetryChildProcessEnv(false), {
    AITK_TELEMETRY_ENABLED: '0',
    DISABLE_TELEMETRY: 'YES',
    HF_HUB_DISABLE_TELEMETRY: '1',
  });
});

test('enabled child environment explicitly opts libraries into telemetry', () => {
  assert.deepEqual(telemetryChildProcessEnv(true), {
    AITK_TELEMETRY_ENABLED: '1',
    DISABLE_TELEMETRY: 'NO',
    HF_HUB_DISABLE_TELEMETRY: '0',
  });
});

test('settings API and UI expose telemetry as an opt-in setting', () => {
  const routeSource = fs.readFileSync(path.join(uiRoot, 'src/app/api/settings/route.ts'), 'utf8');
  const hookSource = fs.readFileSync(path.join(uiRoot, 'src/hooks/useSettings.tsx'), 'utf8');
  const pageSource = fs.readFileSync(path.join(uiRoot, 'src/app/settings/page.tsx'), 'utf8');

  assert.match(routeSource, /normalizeBooleanSetting\(TELEMETRY_ENABLED, false\)/);
  assert.match(hookSource, /TELEMETRY_ENABLED: 'false'/);
  assert.match(pageSource, /Library telemetry/);
  assert.match(pageSource, /settings\.TELEMETRY_ENABLED === 'true'/);
});
