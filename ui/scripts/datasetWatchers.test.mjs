import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const dbModule = require('../dist/src/server/db.js');
const settingsModule = require('../dist/src/server/settings.js');
const {
  DATASET_WATCHER_IMPORT_MANIFEST,
  datasetWatcherPathCandidates,
  getDatasetWatcherStatuses,
  listDatasetWatchers,
  runDatasetWatcherOnce,
  saveDatasetWatcher,
} = require('../dist/src/server/datasetWatchers.js');

const originalSettings = dbModule.db.settings;
const originalProjects = dbModule.db.projects;
const originalFetch = globalThis.fetch;
const tempRoots = [];

afterEach(async () => {
  dbModule.db.settings = originalSettings;
  dbModule.db.projects = originalProjects;
  settingsModule.flushCache();
  globalThis.fetch = originalFetch;
  globalThis.__aitkDatasetWatcherPending = new Map();
  await Promise.all(tempRoots.splice(0).map(root => fs.rm(root, { recursive: true, force: true })));
});

function installSettingsStore(initial = {}) {
  const store = new Map(Object.entries(initial));
  dbModule.db.settings = {
    async get(key) {
      return store.has(key) ? { key, value: store.get(key) } : null;
    },
    async upsert(key, value) {
      store.set(key, value);
      return { key, value };
    },
    async list() {
      return [...store.entries()].map(([key, value]) => ({ key, value }));
    },
    async delete(key) {
      store.delete(key);
    },
  };
  settingsModule.flushCache();
  return store;
}

function installProjectStore(projects) {
  dbModule.db.projects = {
    async findById(id) {
      return projects.find(project => project.id === id) || null;
    },
    async findBySlug(slug) {
      return projects.find(project => project.slug === slug) || null;
    },
  };
}

async function makeWorkspace() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-watchers-'));
  tempRoots.push(root);
  const datasetsRoot = path.join(root, 'datasets');
  const dataset = path.join(datasetsRoot, 'cats');
  const source = path.join(root, 'source');
  await fs.mkdir(dataset, { recursive: true });
  await fs.mkdir(source, { recursive: true });
  installSettingsStore({
    DATASETS_FOLDER: datasetsRoot,
    OPENROUTER_API_KEY: 'test-openrouter-key',
  });
  return { root, datasetsRoot, dataset, source };
}

async function makeProjectWorkspace() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-watchers-project-'));
  tempRoots.push(root);
  const datasetsRoot = path.join(root, 'datasets');
  const projectsRoot = path.join(root, 'projects');
  const project = {
    id: 'project-1',
    slug: 'project-one',
    name: 'Project One',
    description: '',
    badge_asset: null,
    root_path: path.join(projectsRoot, 'project-one'),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  const projectDataset = path.join(project.root_path, 'datasets', 'cats');
  const globalDataset = path.join(datasetsRoot, 'cats');
  const source = path.join(root, 'source');
  await fs.mkdir(projectDataset, { recursive: true });
  await fs.mkdir(globalDataset, { recursive: true });
  await fs.mkdir(source, { recursive: true });
  installSettingsStore({
    DATASETS_FOLDER: datasetsRoot,
    PROJECTS_FOLDER: projectsRoot,
    PROJECTS_ENABLED: 'true',
    OPENROUTER_API_KEY: 'test-openrouter-key',
  });
  installProjectStore([project]);
  return { project, projectDataset, globalDataset, source };
}

async function runStableImport(watcher, start = 1_000) {
  await runDatasetWatcherOnce(watcher, { now: start, stableMs: 0 });
  return runDatasetWatcherOnce(watcher, { now: start + 1, stableMs: 0 });
}

function openRouterFetchReturning(content) {
  return async url => {
    assert.equal(String(url), 'https://openrouter.ai/api/v1/chat/completions');
    return new Response(
      JSON.stringify({
        choices: [{ message: { content } }],
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  };
}

test('watcher path candidates translate common Windows and WSL path forms', () => {
  assert.ok(
    datasetWatcherPathCandidates('C:\\Users\\me\\Pictures', {
      platform: 'linux',
      env: { WSL_DISTRO_NAME: 'Ubuntu' },
    }).includes('/mnt/c/Users/me/Pictures'),
  );
  assert.ok(
    datasetWatcherPathCandidates('/mnt/d/media', {
      platform: 'win32',
      env: { WSL_DISTRO_NAME: 'Ubuntu' },
    }).includes('D:\\media'),
  );
  assert.ok(
    datasetWatcherPathCandidates('\\\\wsl$\\Ubuntu\\home\\me\\media', {
      platform: 'linux',
      env: { WSL_DISTRO_NAME: 'Ubuntu' },
    }).includes('/home/me/media'),
  );
});

test('watcher creation rejects missing and overlapping source folders', async () => {
  const { root, datasetsRoot } = await makeWorkspace();

  await assert.rejects(
    () =>
      saveDatasetWatcher({
        datasetName: 'cats',
        sourcePath: path.join(root, 'missing'),
      }),
    /not found|not accessible/i,
  );

  await assert.rejects(
    () =>
      saveDatasetWatcher({
        datasetName: 'cats',
        sourcePath: datasetsRoot,
      }),
    /overlap/i,
  );
});

test('watcher copies only new media, skips caption sidecars, and preserves relative paths', async () => {
  const { dataset, source } = await makeWorkspace();

  const watcher = await saveDatasetWatcher({
    datasetName: 'cats',
    sourcePath: source,
    includeSubfolders: true,
    preserveRelativePaths: true,
  });
  await fs.mkdir(path.join(source, 'nested'), { recursive: true });
  await fs.writeFile(path.join(source, 'nested', 'a.jpg'), 'image-bytes');
  await fs.writeFile(path.join(source, 'nested', 'a.txt'), 'caption that should not be copied');

  const first = await runStableImport(watcher);
  assert.equal(first.lastImportedCount, 1);
  assert.equal(await fs.readFile(path.join(dataset, 'nested', 'a.jpg'), 'utf-8'), 'image-bytes');
  await assert.rejects(() => fs.stat(path.join(dataset, 'nested', 'a.txt')), /ENOENT/);

  globalThis.__aitkDatasetWatcherPending = new Map();
  const second = await runStableImport(watcher, 2_000);
  assert.equal(second.lastImportedCount, 0);
  await assert.rejects(() => fs.stat(path.join(dataset, 'nested', 'a_2.jpg')), /ENOENT/);

  const manifest = JSON.parse(await fs.readFile(path.join(dataset, DATASET_WATCHER_IMPORT_MANIFEST), 'utf-8'));
  assert.equal(Object.keys(manifest.imports).length, 1);
});

test('watcher creation baselines existing source media without duplicating dataset files', async () => {
  const { dataset, source } = await makeWorkspace();
  await fs.mkdir(path.join(source, 'nested'), { recursive: true });
  await fs.mkdir(path.join(dataset, 'nested'), { recursive: true });
  await fs.writeFile(path.join(source, 'nested', 'a.jpg'), 'image-bytes');
  await fs.writeFile(path.join(dataset, 'nested', 'a.jpg'), 'image-bytes');
  await fs.writeFile(path.join(source, 'nested', 'a.txt'), 'caption that should not be copied');

  const watcher = await saveDatasetWatcher({
    datasetName: 'cats',
    sourcePath: source,
    includeSubfolders: true,
    preserveRelativePaths: true,
  });

  const first = await runStableImport(watcher);
  assert.equal(first.lastImportedCount, 0);
  await assert.rejects(() => fs.stat(path.join(dataset, 'nested', 'a_2.jpg')), /ENOENT/);

  const manifest = JSON.parse(await fs.readFile(path.join(dataset, DATASET_WATCHER_IMPORT_MANIFEST), 'utf-8'));
  assert.equal(Object.keys(manifest.imports).length, 1);
});

test('legacy watcher without a baseline records matching existing destinations instead of copying duplicates', async () => {
  const { dataset, source } = await makeWorkspace();
  await fs.writeFile(path.join(source, 'a.jpg'), 'image-bytes');
  await fs.writeFile(path.join(dataset, 'a.jpg'), 'image-bytes');

  const watcher = await saveDatasetWatcher({
    datasetName: 'cats',
    sourcePath: source,
  });
  await fs.rm(path.join(dataset, DATASET_WATCHER_IMPORT_MANIFEST), { force: true });

  const first = await runStableImport(watcher);
  assert.equal(first.lastImportedCount, 0);
  await assert.rejects(() => fs.stat(path.join(dataset, 'a_2.jpg')), /ENOENT/);

  const manifest = JSON.parse(await fs.readFile(path.join(dataset, DATASET_WATCHER_IMPORT_MANIFEST), 'utf-8'));
  assert.equal(Object.keys(manifest.imports).length, 1);
});

test('concurrent watcher runs against one dataset do not duplicate imports', async () => {
  const { dataset, source } = await makeWorkspace();

  const firstWatcher = await saveDatasetWatcher({
    datasetName: 'cats',
    sourcePath: source,
  });
  const secondWatcher = await saveDatasetWatcher({
    datasetName: 'cats',
    sourcePath: source,
  });
  await fs.writeFile(path.join(source, 'a.jpg'), 'image-bytes');

  await Promise.all([
    runDatasetWatcherOnce(firstWatcher, { now: 1_000, stableMs: 0 }),
    runDatasetWatcherOnce(secondWatcher, { now: 1_000, stableMs: 0 }),
  ]);
  const results = await Promise.all([
    runDatasetWatcherOnce(firstWatcher, { now: 1_001, stableMs: 0 }),
    runDatasetWatcherOnce(secondWatcher, { now: 1_001, stableMs: 0 }),
  ]);

  assert.equal(results.reduce((sum, result) => sum + result.lastImportedCount, 0), 1);
  assert.equal(await fs.readFile(path.join(dataset, 'a.jpg'), 'utf-8'), 'image-bytes');
  await assert.rejects(() => fs.stat(path.join(dataset, 'a_2.jpg')), /ENOENT/);

  const manifest = JSON.parse(await fs.readFile(path.join(dataset, DATASET_WATCHER_IMPORT_MANIFEST), 'utf-8'));
  assert.equal(Object.keys(manifest.imports).length, 1);
});

test('watcher ignores stale dataset lock left by interrupted scan', async () => {
  const { dataset, source } = await makeWorkspace();
  const watcher = await saveDatasetWatcher({
    datasetName: 'cats',
    sourcePath: source,
  });
  await fs.writeFile(path.join(source, 'a.jpg'), 'image-bytes');
  const lockPath = path.join(dataset, '.aitk_dataset_watch.lock');
  await fs.writeFile(lockPath, 'stale');
  await fs.utimes(lockPath, new Date(0), new Date(0));

  const result = await runStableImport(watcher, 3_000_000);
  assert.equal(result.lastImportedCount, 1);
  assert.equal(await fs.readFile(path.join(dataset, 'a.jpg'), 'utf-8'), 'image-bytes');
  await assert.rejects(() => fs.stat(lockPath), /ENOENT/);
});

test('project-space watchers resolve project slugs and import into project datasets', async () => {
  const { project, projectDataset, globalDataset, source } = await makeProjectWorkspace();

  const watcher = await saveDatasetWatcher({
    datasetName: 'cats',
    projectID: project.slug,
    sourcePath: source,
  });
  await fs.writeFile(path.join(source, 'a.jpg'), 'image-bytes');

  assert.equal(watcher.projectID, project.id);
  const listed = await listDatasetWatchers({ datasetName: 'cats', projectID: project.slug });
  assert.deepEqual(listed.map(item => item.id), [watcher.id]);

  const result = await runStableImport(watcher, 4_000);
  assert.equal(result.lastImportedCount, 1);
  assert.equal(await fs.readFile(path.join(projectDataset, 'a.jpg'), 'utf-8'), 'image-bytes');
  await assert.rejects(() => fs.stat(path.join(globalDataset, 'a.jpg')), /ENOENT/);
});

test('watcher auto-caption writes generated text sidecar for imported images', async () => {
  const { dataset, source } = await makeWorkspace();
  globalThis.fetch = openRouterFetchReturning('A small orange cat on a blue chair.');

  const watcher = await saveDatasetWatcher({
    datasetName: 'cats',
    sourcePath: source,
    autoCaption: {
      enabled: true,
      provider: 'openrouter',
      model: 'x-ai/grok-4.3',
      outputFormat: 'text',
      prompt: 'caption it',
      maxNewTokens: 64,
    },
  });
  await fs.writeFile(path.join(source, 'a.png'), 'image-bytes');

  const result = await runStableImport(watcher);
  assert.equal(result.lastImportedCount, 1);
  assert.equal(result.lastCaptionedCount, 1);
  assert.equal(await fs.readFile(path.join(dataset, 'a.txt'), 'utf-8'), 'A small orange cat on a blue chair.');
});

test('watcher auto-caption failure keeps media uncaptioned and records status', async () => {
  const { dataset, source } = await makeWorkspace();
  globalThis.fetch = openRouterFetchReturning('I cannot fulfill this request.');

  const watcher = await saveDatasetWatcher({
    datasetName: 'cats',
    sourcePath: source,
    autoCaption: {
      enabled: true,
      provider: 'openrouter',
      model: 'x-ai/grok-4.3',
      outputFormat: 'text',
      prompt: 'caption it',
      maxNewTokens: 64,
    },
  });
  await fs.writeFile(path.join(source, 'a.png'), 'image-bytes');

  const result = await runStableImport(watcher);
  assert.equal(result.lastImportedCount, 1);
  assert.equal(result.lastCaptionedCount, 0);
  assert.equal(result.state, 'error');
  assert.match(result.lastError, /refusal/i);
  assert.equal(await fs.readFile(path.join(dataset, 'a.png'), 'utf-8'), 'image-bytes');
  await assert.rejects(() => fs.stat(path.join(dataset, 'a.txt')), /ENOENT/);

  const statuses = await getDatasetWatcherStatuses([watcher.id]);
  assert.equal(statuses[watcher.id].state, 'error');
  assert.match(statuses[watcher.id].lastError, /refusal/i);
});
