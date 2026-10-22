import assert from 'node:assert/strict';
import test from 'node:test';
import { loadTrainingSource, trainingFixture } from './trainingMigrationFixture.mjs';

const { createConfigEditorSession, updateConfigEditorSession, syncConfigEditorSession,
  readConfigEditorSession, rememberConfigEditorSession, clearConfigEditorSession } = loadTrainingSource('utils/configEditorSession.ts');
const { isEditableTrainingConfig } = loadTrainingSource('utils/trainingValidation.ts');
const validate = value => {
  if (!isEditableTrainingConfig(value)) throw new Error('Not an editable training configuration');
  return value;
};

test('unfinished YAML survives parent updates and keeps its last applied configuration', () => {
  const config = trainingFixture();
  const session = createConfigEditorSession(config);
  const applied = session.appliedConfig;
  const invalid = `${session.text}\nbroken: [`;
  assert.deepEqual(updateConfigEditorSession(session, invalid, validate), { valid: false });
  assert.equal(session.dirty, true);
  assert.equal(session.valid, false);
  assert.equal(session.appliedConfig, applied);
  config.config.name = 'parent-update';
  assert.equal(syncConfigEditorSession(session, config), false);
  assert.equal(session.text, invalid);
  assert.ok(session.errorLine > 1);
});

test('valid Config edits round-trip advanced values without removing text or comments on rerender', () => {
  const config = trainingFixture();
  const session = createConfigEditorSession(config);
  const text = session.text.replace('0.00037', '0.00042') + '\n# keep this note in memory\n';
  const result = updateConfigEditorSession(session, text, validate);
  assert.equal(result.valid, true);
  assert.equal(result.config.config.process[0].train.lr, 0.00042);
  assert.deepEqual(result.config.config.process[0].train.optimizer_params, config.config.process[0].train.optimizer_params);
  assert.equal(syncConfigEditorSession(session, result.config), false);
  assert.equal(session.text, text);
  assert.equal(session.dirty, true);
  result.config.config.name = 'edited-in-form';
  assert.equal(syncConfigEditorSession(session, result.config), true);
  assert.match(session.text, /edited-in-form/);
});

test('invalid editable structures stay recoverable and restoring the original clears raw-only dirty state', () => {
  const session = createConfigEditorSession(trainingFixture());
  const original = session.text;
  assert.equal(updateConfigEditorSession(session, 'config: {}', validate).valid, false);
  assert.match(session.error, /Not an editable/);
  assert.equal(updateConfigEditorSession(session, original, validate).valid, true);
  assert.equal(session.error, '');
  assert.equal(session.dirty, false);
});

test('raw sessions and editor positions stay separate for new, edit and migration drafts and clear after save', () => {
  for (const key of ['new', 'edit:one', 'migrate:one']) {
    const session = createConfigEditorSession(trainingFixture());
    session.text += `\n# ${key}`;
    session.viewState = { fixture: key };
    rememberConfigEditorSession(key, session);
  }
  assert.match(readConfigEditorSession('edit:one').text, /# edit:one/);
  assert.deepEqual(readConfigEditorSession('migrate:one').viewState, { fixture: 'migrate:one' });
  clearConfigEditorSession('edit:one');
  assert.equal(readConfigEditorSession('edit:one'), undefined);
  assert.ok(readConfigEditorSession('new'));
  clearConfigEditorSession('new'); clearConfigEditorSession('migrate:one');
});
