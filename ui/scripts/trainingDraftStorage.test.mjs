import assert from 'node:assert/strict';
import test from 'node:test';
import { loadTrainingSource, trainingFixture } from './trainingMigrationFixture.mjs';
const { saveStoredTrainingDraft, readStoredTrainingDraft, removeStoredTrainingDraft } = loadTrainingSource('utils/trainingDraftStorage.ts');
function fixture() {
  const values = new Map();
  const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
  const draft = { config: trainingFixture(), gpuIDs: '0', workerID: 'local' };
  return { values, storage, draft, value: { draft, baseline: JSON.stringify(draft) } };
}

test('draft persistence restores independent new/edit/clone/migrate configurations and clears saved keys', () => {
  const f = fixture();
  const keys = ['new', 'edit:one', 'clone:one', 'migrate:one'];
  for (const key of keys) {
    const value = structuredClone(f.value);
    value.draft.config.config.name = key;
    assert.equal(saveStoredTrainingDraft(key, value, f.storage), true);
  }
  for (const key of keys) assert.equal(readStoredTrainingDraft(key, f.storage).draft.config.config.name, key);
  removeStoredTrainingDraft('new', f.storage);
  assert.equal(readStoredTrainingDraft('new', f.storage), undefined);
  assert.ok(readStoredTrainingDraft('edit:one', f.storage));
});

test('credentials and decrypted payloads are removed recursively without losing token training settings', () => {
  const f = fixture();
  f.draft.config.config.process[0].model.model_kwargs = { api_key: 'secret-one', hf_token: 'secret-two',
    nested: { rawKey: 'secret-three', decryptedCatalog: 'secret-four', authorization: 'secret-five' }, token_dropout_rate: 0.2,
    endpoint: 'https://user:secret-seven@example.com/model?token=secret-eight&revision=main&X-Amz-Signature=signed-secret&X-Amz-Security-Token=security-secret&sig=azure-secret' };
  f.draft.config.config.process[0].watermark = { secret: 'secret-six', enabled: true };
  f.draft.migration = { source: structuredClone(f.draft.config), changes: [{ path: 'model.hf_token', before: 'secret-nine', after: '', reason: 'reset' }] };
  f.value.baseline = JSON.stringify(f.draft);
  assert.equal(saveStoredTrainingDraft('migrate:one', f.value, f.storage), true);
  const serialized = [...f.values.values()][0];
  assert.doesNotMatch(serialized, /secret-(one|two|three|four|five|six|seven|eight|nine)/);
  assert.doesNotMatch(serialized, /signed-secret|security-secret|azure-secret/);
  const restored = readStoredTrainingDraft('migrate:one', f.storage).draft;
  assert.equal(restored.config.config.process[0].model.model_kwargs.token_dropout_rate, 0.2);
  assert.equal(restored.migration.source.config.name, f.draft.config.config.name);
});

test('malformed versions, oversized drafts and blocked storage fail gracefully', () => {
  const f = fixture();
  for (const raw of ['{bad', '{"version":2}', '{"version":1,"draft":{}}', ' '.repeat(2_000_001)]) {
    f.storage.setItem('aitk:training-draft:v1:new', raw);
    assert.equal(readStoredTrainingDraft('new', f.storage), undefined);
  }
  const blocked = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('quota'); }, removeItem() { throw new Error('blocked'); } };
  assert.equal(saveStoredTrainingDraft('new', f.value, blocked), false);
  assert.equal(readStoredTrainingDraft('new', blocked), undefined);
  assert.doesNotThrow(() => removeStoredTrainingDraft('new', blocked));
});
