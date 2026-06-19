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

function openRouterFetchAssertingSystemPrompt(expectedSystemPrompt, content) {
  return async (url, init) => {
    assert.equal(String(url), 'https://openrouter.ai/api/v1/chat/completions');
    const body = JSON.parse(String(init?.body || '{}'));
    assert.equal(body.messages?.[0]?.role, 'system');
    assert.equal(body.messages?.[0]?.content, expectedSystemPrompt);
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

function deferred() {
  let resolve;
  const promise = new Promise(done => {
    resolve = done;
  });
  return { promise, resolve };
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

  const statuses = await getDatasetWatcherStatuses([watcher.id]);
  assert.equal(statuses[watcher.id].lastImportedAt, new Date(1_001).toISOString());
  assert.equal(statuses[watcher.id].lastImportedCount, 0);

  const manifest = JSON.parse(await fs.readFile(path.join(dataset, DATASET_WATCHER_IMPORT_MANIFEST), 'utf-8'));
  assert.equal(Object.keys(manifest.imports).length, 1);
});

test('watcher flatten mode imports collisions into dataset root', async () => {
  const { dataset, source } = await makeWorkspace();

  const watcher = await saveDatasetWatcher({
    datasetName: 'cats',
    sourcePath: source,
    includeSubfolders: true,
    preserveRelativePaths: false,
  });
  await fs.mkdir(path.join(source, 'left'), { recursive: true });
  await fs.mkdir(path.join(source, 'right'), { recursive: true });
  await fs.writeFile(path.join(source, 'left', 'same.jpg'), 'left-image');
  await fs.writeFile(path.join(source, 'right', 'same.jpg'), 'right-image');

  const result = await runStableImport(watcher);
  assert.equal(result.lastImportedCount, 2);
  assert.equal(await fs.readFile(path.join(dataset, 'same.jpg'), 'utf-8'), 'left-image');
  assert.equal(await fs.readFile(path.join(dataset, 'same_2.jpg'), 'utf-8'), 'right-image');
  await assert.rejects(() => fs.stat(path.join(dataset, 'left', 'same.jpg')), /ENOENT/);
  await assert.rejects(() => fs.stat(path.join(dataset, 'right', 'same.jpg')), /ENOENT/);
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

test('multiple project-space watchers import source-relative files into the same dataset target', async () => {
  const { project, projectDataset, source } = await makeProjectWorkspace();
  const workspaceRoot = path.dirname(source);
  const sourceOne = path.join(workspaceRoot, 'watcher-one');
  const sourceTwo = path.join(workspaceRoot, 'watcher-two');
  await fs.mkdir(path.join(sourceOne, 'nested'), { recursive: true });
  await fs.mkdir(path.join(sourceTwo, 'nested'), { recursive: true });

  const firstWatcher = await saveDatasetWatcher({
    datasetName: 'cats',
    projectID: project.slug,
    sourcePath: sourceOne,
    includeSubfolders: true,
    preserveRelativePaths: true,
  });
  const secondWatcher = await saveDatasetWatcher({
    datasetName: 'cats',
    projectID: project.slug,
    sourcePath: sourceTwo,
    includeSubfolders: true,
    preserveRelativePaths: true,
  });

  await fs.writeFile(path.join(sourceOne, 'nested', 'item.jpg'), 'from-first-watcher');
  await fs.writeFile(path.join(sourceTwo, 'nested', 'item.jpg'), 'from-second-watcher');

  const firstResult = await runStableImport(firstWatcher, 5_000);
  const secondResult = await runStableImport(secondWatcher, 6_000);
  assert.equal(firstResult.lastImportedCount, 1);
  assert.equal(secondResult.lastImportedCount, 1);
  assert.equal(await fs.readFile(path.join(projectDataset, 'nested', 'item.jpg'), 'utf-8'), 'from-first-watcher');
  assert.equal(await fs.readFile(path.join(projectDataset, 'nested', 'item_2.jpg'), 'utf-8'), 'from-second-watcher');
  await assert.rejects(() => fs.stat(path.join(projectDataset, 'watcher-one', 'nested', 'item.jpg')), /ENOENT/);
  await assert.rejects(() => fs.stat(path.join(projectDataset, 'watcher-two', 'nested', 'item.jpg')), /ENOENT/);
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
  assert.equal(result.autoCaptionTotalCount, 1);
  assert.equal(result.autoCaptionPendingCount, 0);
  assert.equal(result.autoCaptionCompletedCount, 1);
  assert.equal(await fs.readFile(path.join(dataset, 'a.txt'), 'utf-8'), 'A small orange cat on a blue chair.');
});

test('watcher auto-caption status reports live pending progress', async () => {
  const { dataset, source } = await makeWorkspace();
  const firstStarted = deferred();
  const firstRelease = deferred();
  const secondStarted = deferred();
  const secondRelease = deferred();
  let fetchCount = 0;
  globalThis.fetch = async url => {
    assert.equal(String(url), 'https://openrouter.ai/api/v1/chat/completions');
    fetchCount += 1;
    if (fetchCount === 1) {
      firstStarted.resolve();
      await firstRelease.promise;
    } else {
      secondStarted.resolve();
      await secondRelease.promise;
    }
    return new Response(
      JSON.stringify({
        choices: [{ message: { content: `Caption ${fetchCount}` } }],
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  };

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
  await fs.writeFile(path.join(source, 'a.png'), 'a-bytes');
  await fs.writeFile(path.join(source, 'b.png'), 'b-bytes');

  await runDatasetWatcherOnce(watcher, { now: 7_000, stableMs: 0 });
  const runPromise = runDatasetWatcherOnce(watcher, { now: 7_001, stableMs: 0 });

  await firstStarted.promise;
  let statuses = await getDatasetWatcherStatuses([watcher.id]);
  assert.equal(statuses[watcher.id].state, 'captioning');
  assert.equal(statuses[watcher.id].autoCaptionTotalCount, 2);
  assert.equal(statuses[watcher.id].autoCaptionPendingCount, 2);
  assert.equal(statuses[watcher.id].autoCaptionCompletedCount, 0);
  assert.equal(statuses[watcher.id].autoCaptionActivePath, 'a.png');

  firstRelease.resolve();
  await secondStarted.promise;
  statuses = await getDatasetWatcherStatuses([watcher.id]);
  assert.equal(statuses[watcher.id].autoCaptionTotalCount, 2);
  assert.equal(statuses[watcher.id].autoCaptionPendingCount, 1);
  assert.equal(statuses[watcher.id].autoCaptionCompletedCount, 1);
  assert.equal(statuses[watcher.id].autoCaptionActivePath, 'b.png');

  secondRelease.resolve();
  const result = await runPromise;
  assert.equal(result.lastImportedCount, 2);
  assert.equal(result.lastCaptionedCount, 2);
  assert.equal(result.autoCaptionPendingCount, 0);
  assert.equal(result.autoCaptionCompletedCount, 2);
  assert.equal(result.autoCaptionActivePath, null);
  assert.equal(await fs.readFile(path.join(dataset, 'a.txt'), 'utf-8'), 'Caption 1');
  assert.equal(await fs.readFile(path.join(dataset, 'b.txt'), 'utf-8'), 'Caption 2');
});

test('watcher syncs ROOT_CAPTION.txt before auto-captioning imported images', async () => {
  const { dataset, source } = await makeWorkspace();
  globalThis.fetch = openRouterFetchAssertingSystemPrompt(
    'Use a concise dataset-wide system prompt.',
    'A concise generated caption.',
  );

  const watcher = await saveDatasetWatcher({
    datasetName: 'cats',
    sourcePath: source,
    includeSubfolders: true,
    autoCaption: {
      enabled: true,
      provider: 'openrouter',
      model: 'x-ai/grok-4.3',
      outputFormat: 'text',
      prompt: 'caption it',
      maxNewTokens: 64,
    },
  });
  await fs.mkdir(path.join(source, 'nested'), { recursive: true });
  await fs.writeFile(path.join(source, 'ROOT_CAPTION.txt'), 'Use a concise dataset-wide system prompt.');
  await fs.writeFile(path.join(source, 'nested', 'ROOT_CAPTION.txt'), 'nested root caption is just a sidecar');
  await fs.writeFile(path.join(source, 'a.png'), 'image-bytes');

  const result = await runStableImport(watcher);
  assert.equal(result.lastImportedCount, 2);
  assert.equal(result.lastCaptionedCount, 1);
  assert.equal(await fs.readFile(path.join(dataset, 'ROOT_CAPTION.txt'), 'utf-8'), 'Use a concise dataset-wide system prompt.');
  assert.equal(await fs.readFile(path.join(dataset, 'a.txt'), 'utf-8'), 'A concise generated caption.');
  await assert.rejects(() => fs.stat(path.join(dataset, 'nested', 'ROOT_CAPTION.txt')), /ENOENT/);
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
  assert.equal(result.autoCaptionTotalCount, 1);
  assert.equal(result.autoCaptionPendingCount, 0);
  assert.equal(result.autoCaptionCompletedCount, 1);
  assert.equal(result.state, 'error');
  assert.match(result.lastError, /refusal/i);
  assert.equal(await fs.readFile(path.join(dataset, 'a.png'), 'utf-8'), 'image-bytes');
  await assert.rejects(() => fs.stat(path.join(dataset, 'a.txt')), /ENOENT/);

  const statuses = await getDatasetWatcherStatuses([watcher.id]);
  assert.equal(statuses[watcher.id].state, 'error');
  assert.match(statuses[watcher.id].lastError, /refusal/i);
});
