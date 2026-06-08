import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import {
  expectedFluxGuidanceBypass,
  getFluxGuidanceBypassPolicy,
} from '../dist/src/utils/fluxGuidancePolicy.js';

function archBlock(source, name) {
  const start = source.indexOf(`name: '${name}'`);
  assert.notEqual(start, -1, `missing ${name} arch block`);
  const end = source.indexOf('\n  },', start);
  assert.notEqual(end, -1, `missing end of ${name} arch block`);
  return source.slice(start, end);
}

test('Flux guidance policy distinguishes official FLUX and Flex defaults', () => {
  assert.equal(
    getFluxGuidanceBypassPolicy({ arch: 'flux', name_or_path: 'black-forest-labs/FLUX.1-dev' }),
    'forbidden',
  );
  assert.equal(
    getFluxGuidanceBypassPolicy({ arch: 'flux', name_or_path: 'black-forest-labs/FLUX.1-schnell' }),
    'forbidden',
  );
  assert.equal(
    getFluxGuidanceBypassPolicy({ arch: 'flux_kontext', name_or_path: 'black-forest-labs/FLUX.1-Kontext-dev' }),
    'forbidden',
  );
  assert.equal(
    getFluxGuidanceBypassPolicy({ arch: 'flex1', name_or_path: 'ostris/Flex.1-alpha' }),
    'required',
  );
  assert.equal(
    getFluxGuidanceBypassPolicy({ arch: 'flex2', name_or_path: 'ostris/Flex.2-preview' }),
    'required',
  );
  assert.equal(
    expectedFluxGuidanceBypass({ arch: 'flux', name_or_path: 'black-forest-labs/FLUX.1-dev' }),
    false,
  );
  assert.equal(
    expectedFluxGuidanceBypass({ arch: 'flex1', name_or_path: 'ostris/Flex.1-alpha' }),
    true,
  );
});

test('Flux UI arch defaults set or clear guidance bypass explicitly', () => {
  const source = fs.readFileSync(path.join(process.cwd(), 'src/app/jobs/new/options.ts'), 'utf8');

  assert.match(
    archBlock(source, 'flux'),
    /'config\.process\[0\]\.train\.bypass_guidance_embedding': \[false, false\]/,
  );
  assert.match(
    archBlock(source, 'flux_kontext'),
    /'config\.process\[0\]\.train\.bypass_guidance_embedding': \[false, false\]/,
  );
  assert.match(
    archBlock(source, 'flex1'),
    /'config\.process\[0\]\.train\.bypass_guidance_embedding': \[true, false\]/,
  );
  assert.match(
    archBlock(source, 'flex2'),
    /'config\.process\[0\]\.train\.bypass_guidance_embedding': \[true, false\]/,
  );
});
