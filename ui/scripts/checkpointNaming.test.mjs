import assert from 'node:assert/strict';
import test from 'node:test';
import { applyJobOutputIdentity } from '../dist/src/utils/jobIdentity.js';

test('checkpoint name is the LoRA name while every process keeps the stable storage folder', () => {
  const config = { config: { name: 'job-old-id', process: [{ type: 'diffusion_trainer' }] } };
  const job = { name: 'My LoRA', storage_key: 'job-743465cc-0702-4124-a4f8-45618ca27a27' };
  applyJobOutputIdentity(config, job);
  assert.equal(config.config.name, 'My LoRA');
  assert.equal(config.config.process[0].name, job.storage_key);

  applyJobOutputIdentity(config, { ...job, name: 'Renamed LoRA' });
  assert.equal(config.config.name, 'Renamed LoRA');
  assert.equal(config.config.process[0].name, job.storage_key);
});

test('legacy jobs without a storage key keep their existing named folder', () => {
  const config = { config: { name: 'stale name', process: [{ type: 'diffusion_trainer' }] } };
  applyJobOutputIdentity(config, { name: 'Legacy LoRA', storage_key: null });
  assert.equal(config.config.name, 'Legacy LoRA');
  assert.equal(config.config.process[0].name, 'Legacy LoRA');
});
