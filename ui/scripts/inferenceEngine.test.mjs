import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import test from 'node:test';
import ts from 'typescript';

const require = createRequire(import.meta.url);
function loadSource(relative, overrides = {}) {
  const filename = new URL(relative, import.meta.url);
  const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const evaluatedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(
    name => (Object.hasOwn(overrides, name) ? overrides[name] : require(name)),
    evaluatedModule,
    evaluatedModule.exports,
  );
  return evaluatedModule.exports;
}
const { readEngineFrames, payloadToFloat32, latentToImage, parsePreview } = loadSource('../src/utils/engineStream.ts');
const { inferenceToken } = loadSource('../src/server/inferenceToken.ts');

test('Qwen Image 2.1 presets reach training and generation through the shared catalog', () => {
  const catalog = loadSource('../src/domain/modelOptions.ts', {
    './trainingDefaults': loadSource('../src/domain/trainingDefaults.ts'),
    '@/helpers/defaultSamples': loadSource('../src/helpers/defaultSamples.ts'),
    './modelCapabilities.json': require('../src/domain/modelCapabilities.json'),
  });
  const { getDefaultModelConfig } = loadSource('../src/domain/generationConfig.ts', {
    './modelOptions': catalog,
    '../utils/memoryProfiles': loadSource('../src/utils/memoryProfiles.ts'),
  });
  const choice = catalog.modelArchs.find(model => model.name === 'qwen_image_2');
  assert.ok(choice);
  assert.ok(choice.additionalSections.includes('datasets.multi_control_paths'));
  assert.ok(choice.additionalSections.includes('sample.multi_ctrl_imgs'));
  assert.equal(choice.defaults['config.process[0].sample.guidance_scale'][0], 3);
  assert.equal(choice.defaults['config.process[0].train.timestep_type'][0], 'shift');
  const model = getDefaultModelConfig(choice.name);
  assert.equal(model.name_or_path, 'Comfy-Org/Qwen-Image-2.1');
  assert.equal(model.qtype, 'convrot8');
  assert.equal(model.qtype_te, 'convrot8');
  assert.equal(model.low_vram, true);
});

test('generator defaults and controls follow installed model declarations', () => {
  const { getDefaultModelConfig, getDefaultSampler, archSupportsSection } = loadSource('../src/domain/generationConfig.ts', {
    './modelOptions': { modelArchs: [] },
    '../utils/memoryProfiles': { getLayerOffloadingMemoryProfile: () => ({ backend: 'layer', transformerPercent: 0.5, textEncoderPercent: 0.5 }) },
  });
  const installed = [{ name: 'custom_model', additionalSections: ['model.layer_offloading'], defaults: {
    'config.process[0].model.name_or_path': ['example/custom-model', ''],
    'config.process[0].model.quantize': [true, false],
    'config.process[0].model.qtype': ['convrot8', 'qfloat8'],
    'config.process[0].model.layer_offloading_transformer_percent': [0.75, 0.5],
    'config.process[0].sample.sampler': ['custom_sampler', 'flowmatch'],
  } }];
  const config = getDefaultModelConfig('custom_model', installed);
  assert.equal(config.name_or_path, 'example/custom-model');
  assert.equal(config.quantize, true);
  assert.equal(config.qtype, 'convrot8');
  assert.equal(config.layer_offloading_transformer_percent, 0.75);
  assert.equal(getDefaultSampler('custom_model', installed), 'custom_sampler');
  assert.equal(archSupportsSection('custom_model', 'model.layer_offloading', installed), true);
});

test('job notes round-trip atomically and reject escaping paths and oversized text', async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-job-notes-'));
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-job-notes-outside-'));
  t.after(() => { fs.rmSync(root, { recursive: true, force: true }); fs.rmSync(outside, { recursive: true, force: true }); });
  const { readJobNotes, writeJobNotes } = loadSource('../src/server/jobNotes.ts', {
    './jobFolder': { getSafeJobFolder: async () => root },
    './pathContainment': require('../dist/src/server/pathContainment.js'),
    './commandInput': require('../dist/src/server/commandInput.js'),
  });
  assert.equal(await readJobNotes({}), '');
  await writeJobNotes({}, 'First note');
  await writeJobNotes({}, 'Revised 音楽 note');
  assert.equal(await readJobNotes({}), 'Revised 音楽 note');
  assert.deepEqual(fs.readdirSync(root), ['notes.md']);
  await assert.rejects(writeJobNotes({}, 'a'.repeat(1024 * 1024 + 1)), /too large/);
  assert.equal(await readJobNotes({}), 'Revised 音楽 note');
  fs.unlinkSync(path.join(root, 'notes.md'));
  fs.symlinkSync(outside, path.join(root, 'notes.md'), process.platform === 'win32' ? 'junction' : 'dir');
  await assert.rejects(readJobNotes({}), /Invalid notes path/);
  await assert.rejects(writeJobNotes({}, 'escape'), /Invalid notes path/);
  assert.deepEqual(fs.readdirSync(outside), []);
});

test('inference proxy restricts routes and keeps internal credentials out of browser responses', async t => {
  let seen;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    seen = { url: String(url), options };
    return new Response('ok', { headers: { 'content-type': 'text/plain', 'x-engine-token': 'private', 'x-request-id': 'a123' } });
  });
  const { proxyInferenceRequest } = loadSource('../src/server/inferenceProxy.ts', {
    './commandInput': require('../dist/src/server/commandInput.js'),
    './inferenceEngine': { getInferenceEndpoint: async id => id === 'job-a' ? { url: 'http://127.0.0.1:12345', token: 'private' } : null },
  });
  const request = new Request('http://localhost/api/inference/health?job_id=job-a');
  for (const route of [['outputs', '..', 'secret'], ['assets'], ['health', 'extra'], ['http:', 'evil']]) {
    assert.equal((await proxyInferenceRequest(request, route)).status, 404);
  }
  assert.equal(seen, undefined);
  const response = await proxyInferenceRequest(request, ['health']);
  assert.equal(seen.url, 'http://127.0.0.1:12345/health');
  assert.equal(seen.options.headers.get('x-engine-token'), 'private');
  assert.equal(response.headers.get('x-engine-token'), null);
  assert.equal(response.headers.get('x-request-id'), 'a123');
  assert.equal(await response.text(), 'ok');
  assert.equal((await proxyInferenceRequest(new Request('http://localhost/?job_id=missing'), ['health'])).status, 503);
});

function frame(header, payload = new Uint8Array()) {
  const encoded = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(12 + encoded.length + payload.length);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, encoded.length, true);
  bytes.set(encoded, 4);
  view.setBigUint64(4 + encoded.length, BigInt(payload.length), true);
  bytes.set(payload, 12 + encoded.length);
  return bytes;
}
function response(chunks) {
  return new Response(
    new ReadableStream({
      start(c) {
        for (const chunk of chunks) c.enqueue(chunk);
        c.close();
      },
    }),
  );
}

test('engine frames survive arbitrary network fragmentation and preserve binary payloads', async () => {
  const chunks = [
    frame({ type: 'start', prompt: '音楽' }),
    frame({ type: 'latent' }, new Uint8Array([0, 255, 127])),
    frame({ type: 'end' }),
  ];
  const fragmented = chunks.flatMap(chunk => Array.from(chunk, byte => new Uint8Array([byte])));
  const actual = [];
  await readEngineFrames(response(fragmented), value => actual.push(value));
  assert.deepEqual(
    actual.map(value => value.header.type),
    ['start', 'latent', 'end'],
  );
  assert.equal(actual[0].header.prompt, '音楽');
  assert.deepEqual([...actual[1].payload], [0, 255, 127]);
});

test('engine frames reject oversized, malformed, and truncated data', async () => {
  const oversized = new Uint8Array(4);
  new DataView(oversized.buffer).setUint32(0, 2 ** 30, true);
  await assert.rejects(
    readEngineFrames(response([oversized]), () => {}),
    /too large/,
  );
  await assert.rejects(
    readEngineFrames(response([frame({ bad: true })]), () => {}),
    /Invalid engine frame/,
  );
  await assert.rejects(
    readEngineFrames(response([frame({ type: 'result' }).slice(0, -1)]), () => {}),
    /Truncated/,
  );
  const bigPayload = frame({ type: 'latent' });
  new DataView(bigPayload.buffer).setBigUint64(bigPayload.length - 8, 2n ** 40n, true);
  await assert.rejects(
    readEngineFrames(response([bigPayload]), () => {}),
    /too large/,
  );
});

test('half precision latent previews decode and project without accepting unsafe shapes', () => {
  const header = { dtype: 'float16', shape: [1, 1, 1, 3], layout: 'BCHW' };
  const data = payloadToFloat32(header, new Uint8Array([0, 188, 0, 0, 0, 60]));
  assert.deepEqual([...data], [-1, 0, 1]);
  const info = parsePreview({
    format: 'image',
    channels: 1,
    dims: 4,
    spatial: 8,
    temporal: 1,
    reshape: null,
    factors: [[1, 1, 1]],
    bias: null,
  });
  const image = latentToImage(header, data, info);
  assert.deepEqual([...image.frameData[0]], [0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
  assert.throws(() => payloadToFloat32({ ...header, shape: [1, -1, 3] }, new Uint8Array()), /shape/);
  assert.throws(() => payloadToFloat32({ ...header, shape: [1, 8192, 8192] }, new Uint8Array()), /too large/);
  assert.throws(() => payloadToFloat32(header, new Uint8Array(2)), /payload size/);
  assert.equal(parsePreview({ ...info, factors: [[NaN, 0, 0]] }), null);
});

test('engine credentials bind the managed secret, job, and execution attempt', t => {
  const original = process.env.AITK_INTERNAL_TOKEN;
  t.after(() => {
    if (original === undefined) delete process.env.AITK_INTERNAL_TOKEN;
    else process.env.AITK_INTERNAL_TOKEN = original;
  });
  process.env.AITK_INTERNAL_TOKEN = 'test-only-secret';
  const first = inferenceToken('job-a', 'attempt-a');
  assert.equal(first, inferenceToken('job-a', 'attempt-a'));
  assert.notEqual(first, inferenceToken('job-b', 'attempt-a'));
  assert.notEqual(first, inferenceToken('job-a', 'attempt-b'));
  process.env.AITK_INTERNAL_TOKEN = 'rotated-test-secret';
  assert.notEqual(first, inferenceToken('job-a', 'attempt-a'));
  delete process.env.AITK_INTERNAL_TOKEN;
  assert.throws(() => inferenceToken('job-a', 'attempt-a'), /managed Studio/);
});

test('endpoint discovery rejects stale attempts, unexpected processes, and external hosts', async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-engine-endpoint-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const job = {
    id: 'job-a',
    job_type: 'inference',
    worker_id: 'local',
    status: 'running',
    attempt_id: 'attempt-a',
    pid: 123,
  };
  const { getInferenceEndpoint } = loadSource('../src/server/inferenceEngine.ts', {
    './db': { db: { jobs: { findById: async () => job } } },
    './jobFolder': { getSafeJobFolder: async () => root },
    './pathContainment': require('../dist/src/server/pathContainment.js'),
    './commandInput': require('../dist/src/server/commandInput.js'),
    './inferenceToken': { inferenceToken: () => 'test-token' },
  });
  const endpoint = { job_id: job.id, attempt_id: job.attempt_id, pid: job.pid, host: '127.0.0.1', port: 12345 };
  const save = value => fs.writeFileSync(path.join(root, 'engine.json'), JSON.stringify(value));
  assert.equal(await getInferenceEndpoint(job.id), null);
  save(endpoint);
  assert.equal((await getInferenceEndpoint(job.id)).url, 'http://127.0.0.1:12345');
  for (const altered of [
    { attempt_id: 'old' },
    { pid: 321 },
    { job_id: 'different' },
    { host: 'example.com' },
    { port: 65536 },
    { port: 1.5 },
  ]) {
    save({ ...endpoint, ...altered });
    assert.equal(await getInferenceEndpoint(job.id), null);
  }
  save(endpoint);
  job.status = 'stopped';
  assert.equal(await getInferenceEndpoint(job.id), null);
});

test('installed UI extensions compile TSX, tolerate absent roots, and isolate compile errors', async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-extension-ui-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const good = path.join(root, 'extensions', 'example'),
    bad = path.join(root, 'extensions', 'broken');
  fs.mkdirSync(good, { recursive: true });
  fs.mkdirSync(bad);
  fs.writeFileSync(
    path.join(good, 'ui.tsx'),
    'export const AI_TOOLKIT_UI_MODELS = [{ name: "test", label: "Test", group: "image", modelNotes: <p>Example</p> }];',
  );
  fs.writeFileSync(path.join(bad, 'ui.tsx'), 'export const invalid = <');
  const { listExtensionUiModules } = loadSource('../src/server/extensionUi.ts', {
    '@/paths': { TOOLKIT_ROOT: root },
    './pathContainment': require('../dist/src/server/pathContainment.js'),
    module: { createRequire: () => require },
  });
  const result = await listExtensionUiModules();
  assert.equal(result.modules.length, 1);
  assert.equal(result.modules[0].id, 'extensions/example');
  const compiled = { exports: {} };
  new Function('require', 'module', 'exports', result.modules[0].code)(require, compiled, compiled.exports);
  assert.equal(compiled.exports.AI_TOOLKIT_UI_MODELS[0].modelNotes.props.children, 'Example');
  assert.equal(result.errors.length, 1);
  assert.match(result.errors[0], /extensions\/broken/);
});


test('Qwen references preserve order, legacy aliases, and explicit clearing', () => {
  const { qwenSampleReferences, transparentQwenPrompt, QWEN_IMAGE_PRESETS } = loadSource('../src/domain/qwenImage.ts');
  assert.deepEqual(qwenSampleReferences({ ctrl_img: 'one', ctrl_img_1: 'one', ctrl_img_2: 'two' }), ['one', 'two']);
  const ten = Array.from({ length: 10 }, (_, i) => `ref-${i}`);
  assert.deepEqual(qwenSampleReferences({ ctrl_imgs: ten, ctrl_img: 'stale' }), ten);
  assert.deepEqual(qwenSampleReferences({ ctrl_imgs: [], ctrl_img: 'stale' }), []);
  const prompt = transparentQwenPrompt('A dragon sticker.');
  assert.equal(transparentQwenPrompt(prompt), prompt);
  assert.ok(prompt.includes('A dragon sticker.'));
  assert.equal(QWEN_IMAGE_PRESETS.filter(preset => preset.steps === 40).length, 7);
  for (const preset of QWEN_IMAGE_PRESETS) {
    assert.equal(preset.width % 32, 0); assert.equal(preset.height % 32, 0);
  }
});
