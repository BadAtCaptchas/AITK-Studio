import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { cached, invalidateCache } from '../dist/src/server/apiCache.js';
import { assignSeriesColors } from '../dist/src/utils/seriesColors.js';
import { TerminalEmulator } from '../dist/src/utils/terminalEmulator.js';

const scriptsDirectory = path.dirname(fileURLToPath(import.meta.url));
const uiDirectory = path.dirname(scriptsDirectory);

const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

test('terminal emulation preserves control sequences split across chunks', () => {
  const terminal = new TerminalEmulator();
  terminal.write('Progress 10%\rProgress 20%\n');
  terminal.write('\u001b[3');
  terminal.write('1mred\u001b[0m\nabc\bX');

  assert.equal(terminal.toString(), 'Progress 20%\nred\nabX');

  terminal.write('\r\u001b[2Kdone');
  assert.equal(terminal.toString(), 'Progress 20%\nred\ndone');
  terminal.reset();
  assert.equal(terminal.toString(), '');
});

test('telemetry cache coalesces requests, expires, and retries failures', async () => {
  invalidateCache('test-telemetry');
  let calls = 0;
  const fetcher = async () => {
    calls += 1;
    await delay(5);
    return calls;
  };

  const [first, second] = await Promise.all([
    cached('test-telemetry', fetcher, 20),
    cached('test-telemetry', fetcher, 20),
  ]);
  assert.deepEqual([first, second], [1, 1]);
  assert.equal(calls, 1);

  await delay(30);
  assert.equal(await cached('test-telemetry', fetcher, 20), 2);

  invalidateCache('test-telemetry-failure');
  let attempts = 0;
  await assert.rejects(
    cached('test-telemetry-failure', async () => {
      attempts += 1;
      throw new Error('temporary');
    }),
    /temporary/,
  );
  assert.equal(
    await cached('test-telemetry-failure', async () => {
      attempts += 1;
      return 'recovered';
    }),
    'recovered',
  );
  assert.equal(attempts, 2);
});

test('series colors remain tied to their stable series position', () => {
  const keys = ['loss/main', 'loss/prior', 'loss/regularization'];
  const colors = assignSeriesColors(keys, ['blue', 'green']);
  const activeKeys = keys.filter(key => key !== 'loss/main');

  assert.equal(colors['loss/prior'], 'green');
  assert.equal(colors['loss/regularization'], 'blue');
  assert.deepEqual(activeKeys.map(key => colors[key]), ['green', 'blue']);
});

test('sample-now and caption controls retain project, remote, and database safeguards', () => {
  const read = relativePath => readFileSync(path.join(uiDirectory, relativePath), 'utf8');
  const route = read('src/app/api/jobs/[jobID]/sample_now/route.ts');
  const prepareDb = read('scripts/prepare-db.mjs');
  const schema = read('prisma/schema.prisma');
  const captionOptions = read('src/helpers/captionOptions.ts');
  const captionEditor = read('src/components/CaptionSimpleJob.tsx');
  const logHook = read('src/hooks/useJobLog.tsx');

  assert.match(route, /isRequestAuthenticated/);
  assert.match(route, /assertProjectJobEnabled\(job, 'write'\)/);
  assert.match(route, /remote_job_id/);
  assert.match(route, /sample_now`/);
  assert.match(route, /sample_now: true/);
  assert.match(prepareDb, /sample_now BOOLEAN NOT NULL DEFAULT false/);
  assert.match(prepareDb, /sample_now: \{ \$exists: false \}/);
  assert.match(schema, /sample_now\s+Boolean\s+@default\(false\)/);
  assert.match(captionOptions, /'caption\.thinking'/);
  assert.match(captionEditor, /caption\.thinking/);
  assert.match(logHook, /terminalRef\.current\?\.reset\(\)/);
  assert.match(logHook, /offsetRef\.current = payload\.offset/);
});
