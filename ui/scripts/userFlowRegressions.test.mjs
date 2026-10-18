import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { stripTypeScriptTypes } from 'node:module';

const source = name => fs.readFileSync(new URL('../src/' + name, import.meta.url), 'utf8');
function evaluate(code, dependencies = {}, names = []) {
  code = code.replace(/^import[\s\S]*?from ['"]([^'"]+)['"];\s*/gm, (whole, name) => {
    if (whole.startsWith('import type')) return '';
    const binding = whole.slice(6, whole.lastIndexOf('from')).trim();
    if (binding.startsWith('{')) return `const ${binding.replace(/\bas\b/g, ':')} = dependencies[${JSON.stringify(name)}];\n`;
    return `const ${binding} = dependencies[${JSON.stringify(name)}];\n`;
  }).replace(/export default /g, '').replace(/export (?=(?:async )?(?:function|class|const|let|type|interface)\b)/g, '');
  return new Function('dependencies', stripTypeScriptTypes(code) + `\nreturn { ${names.join(',')} };`)(dependencies);
}
const folder = evaluate(source('utils/folderImport.ts'), {}, ['FOLDER_IMPORT_MEDIA_EXTENSIONS', 'folderImportCaptionKey', 'folderImportExtension']);
const { planDatasetUploads } = evaluate(source('utils/datasetUploadPlan.ts'), { './folderImport': folder }, ['planDatasetUploads']);
const { isActiveJobStatus } = evaluate(source('utils/jobStatus.ts'), {}, ['isActiveJobStatus']);

test('startup polling covers all transient statuses and stops for terminal jobs', () => {
  for (const state of ['queued', 'starting', 'remote-starting', 'running', 'stopping']) assert.equal(isActiveJobStatus(state), true);
  for (const state of ['completed', 'stopped', 'failed', 'error']) assert.equal(isActiveJobStatus(state), false);
  assert.match(source('app/jobs/[jobID]/page.tsx'), /return isActiveJobStatus\(job.status\)/);
});

test('upload plans pair original stems, put media first, and reject ambiguous normalized names', () => {
  const plans = planDatasetUploads(['folder/photo.txt', 'folder/photo.png', 'folder/photo.json', 'other/photo.png'], crypto.randomUUID);
  assert.deepEqual(plans.map(group => group.indexes), [[1, 0, 2], [3]]);
  assert.notEqual(plans[0].id, plans[1].id);
  for (const names of [['a b.png', 'a_b.png', 'a_b.txt'], ['photo.png', 'photo.jpg'], ['Photo.png', 'photo.txt'], ['a.png', 'a.png'], ['../a.png']]) {
    assert.throws(() => planDatasetUploads(names, crypto.randomUUID));
  }
});

function reactHarness() {
  const slots = [], effects = [];
  let cursor = 0, pending = [];
  const react = {
    useState(value) { const id = cursor++; if (!(id in slots)) slots[id] = value; return [slots[id], next => { slots[id] = typeof next === 'function' ? next(slots[id]) : next; }]; },
    useRef(value) { const id = cursor++; return slots[id] ??= { current: value }; },
    useCallback(callback) { cursor++; return callback; },
    useEffect(callback, deps) { const id = cursor++; if (!effects[id] || deps.some((v, i) => v !== effects[id].deps[i])) pending.push(() => { effects[id]?.cleanup?.(); effects[id] = { deps, cleanup: callback() }; }); },
  };
  return { react, render(fn) { cursor = 0; const value = fn(); const run = pending; pending = []; run.forEach(f => f()); return value; }, cleanup() { effects.forEach(e => e?.cleanup?.()); } };
}
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
test('newer job refresh wins over delayed success/error and old navigation requests', async () => {
  const harness = reactHarness(), requests = [];
  const { useJob } = evaluate(source('hooks/useJob.tsx'), {
    react: harness.react, '@/utils/api': { apiClient: { get() { const d = deferred(); requests.push(d); return d.promise; } } }, './usePollLoop': () => {},
  }, ['useJob']);
  let scope = 'a';
  const render = () => harness.render(() => useJob(scope));
  let hook = render(); const older = hook.refreshJob(), newer = hook.refreshJob();
  requests[1].resolve({ data: { id: 'a', step: 20 } }); await newer;
  requests[0].resolve({ data: { id: 'a', step: 10 } }); await older;
  assert.equal(render().job.step, 20);
  hook = render(); const failure = hook.refreshJob(), success = hook.refreshJob();
  requests[3].resolve({ data: { id: 'a', step: 30 } }); await success;
  requests[2].reject(new Error('stale failure')); await failure;
  assert.equal(render().status, 'success');
  const previous = render().refreshJob(); scope = 'b'; render(); scope = 'a'; render();
  requests[4].resolve({ data: { id: 'a', step: 1 } }); await previous;
  assert.equal(render().job, null); harness.cleanup();
});

function callbackBody(text, start, end) { return text.slice(text.indexOf(start), text.indexOf(end, text.indexOf(start))); }
test('caption save success updates only the baseline when typing or navigating during save', async () => {
  const code = callbackBody(source('components/dataset-image-studio/DatasetImageStudio.tsx'), '  const saveCaptionForItem = useCallback(', '  const saveCaption = useCallback(');
  for (const navigate of [false, true]) {
    const pending = deferred(), cache = new Map(); let caption = 'first', saved = 'old';
    const selectedKeyRef = { current: 'image' }, latestCaptionRef = { current: caption };
    const args = { useCallback: fn => fn, isPlainTextCaptionItem: () => true, selectedKeyRef, latestCaptionRef,
      captionCacheRef: { current: cache }, apiClient: { post: () => pending.promise }, onPlainItemCaptioned: undefined,
      setCaptionText: value => { caption = value; }, setSavedCaption: value => { saved = value; },
      writeCaptionCache: (key, value) => cache.set(key, value), encryptedCaptionPaths: null, encryptedKey: null, onSaveEncryptedCaption: null };
    const run = new Function(...Object.keys(args), stripTypeScriptTypes(code) + '\nreturn saveCaptionForItem;')(...Object.values(args));
    const save = run({ kind: 'plain', path: '/image.png' }, 'image', 'first');
    latestCaptionRef.current = 'newer'; caption = 'newer'; cache.set('image', { caption: 'newer', saved: 'old', loaded: true });
    if (navigate) selectedKeyRef.current = 'other';
    pending.resolve({ data: {} }); await save;
    assert.equal(caption, 'newer'); assert.equal(cache.get('image').caption, 'newer'); assert.equal(cache.get('image').saved, 'first');
    if (!navigate) { assert.equal(saved, 'first'); assert.notEqual(caption, saved); }
  }
});

function uploadFixture() {
  const records = new Map(), held = new Set();
  const db = { runtime: { async get(key) { return records.get(key) ?? null; }, async compareAndSwap(key, version, value) {
    const row = records.get(key); if ((row?.version ?? null) !== version) return false;
    records.set(key, { version: (version ?? 0) + 1, value }); return true;
  } } };
  const withProcessLease = async (key, fn) => { if (held.has(key)) throw Object.assign(new Error('Busy'), { status: 409 }); held.add(key); try { return await fn(); } finally { held.delete(key); } };
  const containment = evaluate(source('server/pathContainment.ts'), { 'fs/promises': fsp, path }, ['isPathWithinRoot']);
  const reservations = evaluate(source('server/datasetUploadReservation.ts'), { 'fs/promises': fsp, path, crypto,
    './db': { db }, './processLease': { withProcessLease }, './pathContainment': containment }, ['publishDatasetUpload']);
  return reservations;
}
test('publication rejects existing media/caption stems across extensions and concurrent batches', async t => {
  const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-upload-regression-'));
  t.after(() => fsp.rm(root, { recursive: true, force: true }));
  const { publishDatasetUpload: publish } = uploadFixture();
  const write = (filename, token, contents = filename) => publish(root, path.join(root, filename), token,
    () => fsp.writeFile(path.join(root, filename), contents, { flag: 'wx' }));
  await fsp.writeFile(path.join(root, 'old.png'), 'original');
  await assert.rejects(write('old.png', 'new'), { status: 409 });
  await assert.rejects(write('old.txt', 'new'), { status: 409 });
  assert.equal(await fsp.readFile(path.join(root, 'old.png'), 'utf8'), 'original');
  await write('new.png', 'group'); await write('new.txt', 'group', 'correct caption');
  await assert.rejects(write('new.json', 'other'), { status: 409 });
  await assert.rejects(write('new.txt', 'group'), { status: 409 });
  const results = await Promise.allSettled([write('race.png', 'a'), write('race.txt', 'b')]);
  assert.equal(results.filter(result => result.status === 'fulfilled').length, 1);
  assert.equal((await fsp.readdir(root)).filter(name => name.startsWith('race.')).length, 1);
});

test('upload queue retains all-failed and mixed failures and closes only complete successes', async () => {
  const code = callbackBody(source('components/AddImagesModal.tsx'), '  const processQueue = useCallback(', '  const onDrop = useCallback(');
  for (const results of [['error'], ['done', 'error'], ['done']]) {
    let closed = false, cleared = false, completed = 0, uploading = false;
    const entries = results.map((_, id) => ({ id, file: { name: `image${id}.png` } }));
    const args = { useCallback: fn => fn, batchGenerationRef: { current: 0 }, abortRef: { current: false }, MAX_CONCURRENT: 3,
      planDatasetUploads, crypto, uploadSingleFile: async entry => results[entry.id], setIsUploading: value => { uploading = value; },
      modalInfoRef: { current: { onComplete: () => completed++ } }, setModalInfo: value => { closed = value === null; },
      resetState: () => { cleared = true; }, setErrorCount() {}, setFileEntries() {} };
    const process = new Function(...Object.keys(args), stripTypeScriptTypes(code) + '\nreturn processQueue;')(...Object.values(args));
    await process(entries);
    assert.equal(uploading, false); assert.equal(closed, results.every(value => value === 'done'));
    assert.equal(cleared, closed); assert.equal(completed, results.includes('done') ? 1 : 0);
  }
});

test('queue aggregates cover the corpus independently of view and pagination', async () => {
  const code = source('server/jobsApiList.ts').replace(/^export \{ ACTIVE_JOB_STATUSES \} from .*;\n/m, '');
  class CommandInputError extends Error {}
  const { listJobsForJobsApi } = evaluate(code, {
    '../utils/obsoleteWorkspaceGuard': { assertGlobalPayload: value => value, isLegacyScopedRecord: () => false },
    './db': { db: { jobs: {}, runtime: {} } }, './commandInput': { CommandInputError, isRecord: value => !!value && typeof value === 'object' },
    '../utils/jobStatus': { ACTIVE_JOB_STATUSES: ['queued', 'starting', 'remote-starting', 'running', 'stopping'] },
  }, ['listJobsForJobsApi']);
  const rows = ['running', 'completed', 'error'].map((status, index) => ({ id: String(index), status, created_at: new Date(), job_config: '{}' }));
  for (const view of ['active', 'history', 'failed', 'all']) {
    const page = await listJobsForJobsApi({ view, limit: '1' }, {
      listJobs: async options => rows.filter(row => !options.status || options.status.includes(row.status)).slice(0, options.limit),
      runtimeGet: async () => null, statusCounts: async options => {
        assert.equal('status' in options, false); assert.equal('before' in options, false);
        return { running: 1, completed: 1, error: 1 };
      },
    });
    assert.deepEqual(page.statusCounts, { running: 1, completed: 1, error: 1 }); assert.equal(page.jobs.length, 1);
  }
});

test('caption failure preserves drafts and a second save persists edits typed during the first', async () => {
  const code = callbackBody(source('components/dataset-image-studio/DatasetImageStudio.tsx'), '  const saveCaptionForItem = useCallback(', '  const saveCaption = useCallback(');
  const requests = [], cache = new Map(); let caption = 'first', saved = 'original';
  const latestCaptionRef = { current: caption }, selectedKeyRef = { current: 'a' };
  const args = { useCallback: fn => fn, isPlainTextCaptionItem: () => true, selectedKeyRef, latestCaptionRef,
    captionCacheRef: { current: cache }, apiClient: { post(_url, body) { const d = deferred(); requests.push({ ...d, body }); return d.promise; } },
    onPlainItemCaptioned: undefined, setCaptionText: value => { caption = value; }, setSavedCaption: value => { saved = value; },
    writeCaptionCache: (key, value) => cache.set(key, value), encryptedCaptionPaths: null, encryptedKey: null, onSaveEncryptedCaption: null };
  const save = new Function(...Object.keys(args), stripTypeScriptTypes(code) + '\nreturn saveCaptionForItem;')(...Object.values(args));
  const first = save({ kind: 'plain', path: '/a.png' }, 'a', caption);
  caption = latestCaptionRef.current = 'newer'; requests[0].resolve({ data: {} }); await first;
  const second = save({ kind: 'plain', path: '/a.png' }, 'a', caption);
  assert.equal(requests[1].body.caption, 'newer'); requests[1].reject(new Error('offline'));
  await assert.rejects(second, /offline/); assert.equal(caption, 'newer'); assert.equal(saved, 'first');
  const third = save({ kind: 'plain', path: '/a.png' }, 'a', caption);
  requests[2].resolve({ data: {} }); await third; assert.equal(saved, 'newer');
});

test('an older worker cannot silently rename media and then accept its original caption', async () => {
  const { uploadDatasetFile } = evaluate(source('utils/streamedUploads.ts'), {
    './api': { apiClient: { post: async () => ({ data: { files: ['photo_2.png'] } }) } },
  }, ['uploadDatasetFile']);
  await assert.rejects(uploadDatasetFile(new File(['bytes'], 'photo.png'), { datasetName: 'test', uploadGroup: crypto.randomUUID() }), /Update the worker/);
});
