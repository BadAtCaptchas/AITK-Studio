import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import ts from 'typescript';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
function evaluate(source, dependencies = {}) {
  const evaluatedModule = { exports: {} };
  const code = ts.transpileModule(source, {
    fileName: 'settings.ts',
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(name in dependencies, `Unexpected dependency: ${name}`);
    return dependencies[name];
  }, evaluatedModule, evaluatedModule.exports);
  return evaluatedModule.exports;
}

// Exercise the actual settings route with an isolated store, never the user's DB.
function settingsFixture() {
  const store = new Map();
  const source = fs.readFileSync(path.join(root, 'src/server/settings.ts'), 'utf8');
  const ast = ts.createSourceFile('settings.ts', source, ts.ScriptTarget.Latest, true);
  const normalizer = ast.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === 'normalizeBooleanSetting');
  assert.ok(normalizer);
  const dependencies = {
    'next/server': { NextResponse: { json: (body, options) => ({ body, status: options?.status ?? 200 }) } },
    '@/paths': { defaultTrainFolder: '/training', defaultDatasetsFolder: '/datasets', defaultModelsFolder: '/models', defaultProjectsFolder: '/projects' },
    '@/server/settings': { ...evaluate(normalizer.getText(ast)), flushCache() {}, PROJECTS_ENABLED_KEY: 'PROJECTS_ENABLED' },
    '@/server/pathContainment': { normalizeStoragePathSetting: async (value, fallback) => value || fallback },
    '@/server/db': { db: { settings: {
      list: async () => [...store].map(([key, value]) => ({ key, value })),
      get: async key => store.has(key) ? { value: store.get(key) } : null,
      upsertMany: async values => { for (const [key, value] of Object.entries(values)) store.set(key, value); },
    } } },
    '@/server/encryptedDatasetSecrets': { isEncryptedDatasetSecretSettingKey: () => false },
    '@/server/secureCaptionSettings': { isSecureCaptionSystemPromptSettingKey: () => false },
    '@/server/remoteOllamaWorkers': { isRemoteOllamaWorkersSettingKey: () => false },
    '@/server/datasetWatchers': { isDatasetWatchersSettingKey: () => false },
    '@/server/networkPolicy': { getOfflineModeState: async () => ({ enabled: false, lockedByEnv: false }), OFFLINE_MODE_SETTING_KEY: 'OFFLINE_MODE' },
    '@/server/externalComfy': { DEFAULT_EXTERNAL_COMFY_URL: 'http://127.0.0.1:8188', normalizeExternalComfyLoraDir: value => value, normalizeExternalComfyUrl: value => value },
    '@/server/ideogramWorkflowHistory': { IDEOGRAM_WORKFLOW_HISTORY_KEY: 'history' },
    '@/utils/authSession': { isRequestAuthenticated: async () => true },
    '@/utils/telemetry': { TELEMETRY_ENABLED_SETTING_KEY: 'TELEMETRY_ENABLED' },
    '@/server/modelsPath': { modelsPathFromEnv: () => null, resolveModelsPathState: async ({ defaultRoot, settingValue }) => ({ path: settingValue || defaultRoot, lockedByEnv: false }) },
  };
  const route = evaluate(fs.readFileSync(path.join(root, 'src/app/api/settings/route.ts'), 'utf8'), dependencies);
  return { store, get: () => route.GET({}), post: body => route.POST({ json: async () => body }) };
}

test('new installations default to guided training', async () => {
  const fixture = settingsFixture();
  const result = await fixture.get();
  assert.equal(result.status, 200);
  assert.equal(result.body.TRAINING_LEGACY_VIEW, 'false');
});

test('legacy preference survives saving and reading, and can be disabled', async () => {
  const fixture = settingsFixture();
  assert.equal((await fixture.post({ TRAINING_LEGACY_VIEW: 'true' })).status, 200);
  assert.equal((await fixture.get()).body.TRAINING_LEGACY_VIEW, 'true');
  assert.equal((await fixture.post({ TRAINING_LEGACY_VIEW: 'false' })).status, 200);
  assert.equal((await fixture.get()).body.TRAINING_LEGACY_VIEW, 'false');
});

test('older settings clients do not reset the legacy preference when omitting it', async () => {
  const fixture = settingsFixture();
  fixture.store.set('TRAINING_LEGACY_VIEW', 'true');
  assert.equal((await fixture.post({})).status, 200);
  assert.equal((await fixture.get()).body.TRAINING_LEGACY_VIEW, 'true');
});

test('malformed preference falls back to guided and does not persist arbitrary data', async () => {
  const fixture = settingsFixture();
  assert.equal((await fixture.post({ TRAINING_LEGACY_VIEW: { invalid: true } })).status, 200);
  assert.equal(fixture.store.get('TRAINING_LEGACY_VIEW'), 'false');
});
