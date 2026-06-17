import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

function readSource(relativePath) {
  return fs.readFileSync(path.join(process.cwd(), relativePath), 'utf8');
}

function archBlock(source, name) {
  const start = source.indexOf(`name: '${name}'`);
  assert.notEqual(start, -1, `missing ${name} arch block`);
  const end = source.indexOf('\n  },', start);
  assert.notEqual(end, -1, `missing end of ${name} arch block`);
  return source.slice(start, end);
}

test('LTX UI profiles prefer shifted logit-normal and expose strategy settings', () => {
  const options = readSource('src/app/jobs/new/options.ts');

  for (const arch of ['ltx2', 'ltx2.3']) {
    const block = archBlock(options, arch);
    assert.match(
      block,
      /'config\.process\[0\]\.train\.timestep_type': \['shifted_logit_normal', 'sigmoid'\]/,
    );
    assert.match(block, /'train\.ltx_strategy'/);
  }
});

test('LTX strategy and shifted sampler are present in editor, docs, and types', () => {
  const simpleJob = readSource('src/app/jobs/new/SimpleJob.tsx');
  const phases = readSource('src/app/jobs/new/TrainingPhasesEditor.tsx');
  const docs = readSource('src/docs.tsx');
  const types = readSource('src/types.ts');

  assert.match(simpleJob, /value: 'shifted_logit_normal'/);
  assert.match(phases, /value: 'shifted_logit_normal'/);
  assert.match(docs, /'train\.ltx_strategy'/);
  assert.match(types, /ltx_strategy\?: LTXStrategyConfig/);
  assert.match(types, /'first_frame' \| 'prefix' \| 'suffix'/);
});
