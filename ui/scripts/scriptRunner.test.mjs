import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { buildScriptInvocation, ScriptValidationError } = require('../dist/src/server/scriptRunner.js');

function makeFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-script-runner-'));
  const jobDir = path.join(root, 'job');
  fs.mkdirSync(jobDir, { recursive: true });
  const loraPath = path.join(jobDir, 'input.safetensors');
  fs.writeFileSync(loraPath, 'placeholder');
  return {
    root,
    jobDir,
    loraPath,
    outputPath: path.join(jobDir, 'merged.safetensors'),
  };
}

test('buildScriptInvocation accepts allowlisted merge_loras request', () => {
  const fixture = makeFixture();

  const invocation = buildScriptInvocation(
    {
      script: 'merge_loras.py',
      args: {
        loras: JSON.stringify([{ path: fixture.loraPath, strength: 0.5 }]),
        output: fixture.outputPath,
      },
    },
    fixture.root,
  );

  assert.equal(path.basename(invocation.scriptPath), 'merge_loras.py');
  assert.deepEqual(invocation.args.slice(0, 2), ['--loras', JSON.stringify([{ path: fs.realpathSync.native(fixture.loraPath), strength: 0.5 }])]);
  assert.ok(invocation.args.includes('--output'));
  assert.ok(invocation.args.includes(fixture.outputPath));
});

test('buildScriptInvocation rejects path traversal script names', () => {
  const fixture = makeFixture();

  assert.throws(
    () =>
      buildScriptInvocation(
        {
          script: '../merge_loras.py',
          args: { loras: JSON.stringify([{ path: fixture.loraPath }]), output: fixture.outputPath },
        },
        fixture.root,
      ),
    ScriptValidationError,
  );
});

test('buildScriptInvocation rejects unknown scripts', () => {
  const fixture = makeFixture();

  assert.throws(
    () =>
      buildScriptInvocation(
        {
          script: 'test_script.py',
          args: { loras: JSON.stringify([{ path: fixture.loraPath }]), output: fixture.outputPath },
        },
        fixture.root,
      ),
    /Unknown script/,
  );
});

test('buildScriptInvocation rejects invalid args', () => {
  const fixture = makeFixture();

  assert.throws(
    () =>
      buildScriptInvocation(
        {
          script: 'merge_loras.py',
          args: ['--loras', fixture.loraPath],
        },
        fixture.root,
      ),
    /args must be an object/,
  );
});

test('buildScriptInvocation rejects LoRA paths outside the training folder', () => {
  const fixture = makeFixture();
  const outside = path.join(os.tmpdir(), `outside-${Date.now()}.safetensors`);
  fs.writeFileSync(outside, 'placeholder');

  assert.throws(
    () =>
      buildScriptInvocation(
        {
          script: 'merge_loras.py',
          args: { loras: JSON.stringify([{ path: outside }]), output: fixture.outputPath },
        },
        fixture.root,
      ),
    /LoRA inputs must be inside/,
  );
});
