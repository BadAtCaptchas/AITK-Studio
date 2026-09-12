import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import YAML from 'yaml';
import { configContractErrors } from '../dist/src/domain/configContract.js';

test('guided JSON and raw YAML share validation and preserve advanced fields', () => {
  const cases = JSON.parse(
    fs.readFileSync(new URL('../../tests/fixtures/config_contract_cases.json', import.meta.url), 'utf8'),
  );
  for (const fixture of cases) {
    const original = JSON.stringify(fixture.config);
    assert.equal(configContractErrors(fixture.config).length === 0, fixture.valid, fixture.name);
    assert.equal(
      configContractErrors(YAML.parse(YAML.stringify(fixture.config))).length === 0,
      fixture.valid,
      fixture.name + ' raw YAML',
    );
    assert.equal(JSON.stringify(fixture.config), original, 'Validation must not discard advanced options');
  }
});

test('execution worker context overrides controller-side device strings', () => {
  const config = {
    config: { process: [{ type: 'diffusion_trainer', device: 'cuda', model: { arch: 'flux', qtype: 'nvfp4' } }] },
  };
  assert.equal(configContractErrors(config, { deviceBackend: 'cuda' }).length, 0);
  assert.ok(configContractErrors(config, { deviceBackend: 'mps' }).some(error => error.includes('CUDA worker')));
});
