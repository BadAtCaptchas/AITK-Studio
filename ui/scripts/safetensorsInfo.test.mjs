import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import ts from 'typescript';

function load(relative, deps) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(deps, name), name); return deps[name];
  }, module, module.exports);
  return module.exports;
}
const reader = load('../src/server/safetensorsInfo.ts', { 'node:fs/promises': fsp });
const makeFile = (header, bytes = 8) => {
  const json = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(json.length));
  return Buffer.concat([prefix, json, Buffer.alloc(bytes)]);
};
const tensor = { dtype: 'F32', shape: [2], data_offsets: [0, 8] };

test('reads tensor names, shapes, exact byte counts and metadata without loading weights', async t => {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-info-')); t.after(() => fsp.rm(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'model.safetensors');
  await fsp.writeFile(file, makeFile({ weight: tensor, __metadata__: { description: 'test' } }));
  const info = await reader.readSafetensorsInfo(file);
  assert.equal(info.totalParams, 2); assert.equal(info.totalBytes, 8); assert.equal(info.tensorCount, 1);
  assert.deepEqual(info.tensors, [{ name: 'weight', dtype: 'F32', shape: [2], numel: 2, bytes: 8 }]);
  assert.equal(info.metadata.description, 'test'); assert.equal(info.dtypeCounts.F32.count, 1);
  // Prototype-shaped tensor keys must remain ordinary data.
  await fsp.writeFile(file, makeFile(JSON.parse('{"__proto__":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}')));
  assert.equal((await reader.readSafetensorsInfo(file)).tensors[0].name, '__proto__');
});

test('rejects oversized, truncated, malformed and inconsistent headers', async t => {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-info-')); t.after(() => fsp.rm(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'model.safetensors');
  const huge = Buffer.alloc(10); huge.writeBigUInt64LE(BigInt(reader.MAX_HEADER_BYTES + 1));
  const inputs = [huge, Buffer.from('short'), makeFile(null), makeFile([]), makeFile({ weight: null }),
    makeFile({ weight: { ...tensor, shape: [-1] } }), makeFile({ weight: { ...tensor, shape: [Number.MAX_SAFE_INTEGER, 2] } }),
    makeFile({ weight: { ...tensor, data_offsets: [0, 9] } }), makeFile({ weight: { ...tensor, data_offsets: [1, 8] } }),
    makeFile({ weight: tensor, overlap: tensor }), makeFile({ weight: tensor, __metadata__: { bad: {} } }),
    makeFile({ weight: tensor }).subarray(0, 12)];
  for (const input of inputs) {
    await fsp.writeFile(file, input);
    await assert.rejects(reader.readSafetensorsInfo(file), reader.SafetensorsInfoError);
  }
});

test('route enforces authentication and file resolver denials before inspecting data', async () => {
  let authenticated = true, inspected = 0, resolved = 0;
  let resolution = { kind: 'response', status: 403, body: 'Access denied' };
  const command = load('../src/server/commandInput.ts', {
    '../utils/obsoleteWorkspaceGuard': load('../src/utils/obsoleteWorkspaceGuard.ts', {}),
  });
  const { POST } = load('../src/app/api/files/info/route.ts', {
    '@/server/commandInput': command,
    '@/server/fileServing': {
      isAcceleratedApiRequestAuthenticated: async () => authenticated,
      resolveDownloadFileRequest: async (filePath, headers) => { resolved++; assert.equal(filePath, '/data/100%25.safetensors'); assert.equal([...headers].length, 0); return resolution; },
    },
    '@/server/safetensorsInfo': { ...reader, readSafetensorsInfo: async () => { inspected++; return { tensors: [] }; } },
  });
  const request = (body = { filePath: '/data/100%25.safetensors' }) => new Request('http://localhost/api/files/info', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  });
  authenticated = false; assert.equal((await POST(request())).status, 401); assert.equal(resolved, 0);
  authenticated = true;
  for (const status of [403, 404, 400]) {
    resolution = { kind: 'response', status, body: 'Denied' };
    assert.equal((await POST(request())).status, status); assert.equal(inspected, 0);
  }
  assert.equal((await POST(request({ filePath: 'remote://worker/model.safetensors' }))).status, 400);
  assert.equal((await POST(request({ filePath: '/data/file.pt' }))).status, 400);
  const beforeObsolete = resolved;
  assert.equal((await POST(request({ project_id: 'retired', filePath: '/data/100%25.safetensors' }))).status, 400);
  assert.equal(resolved, beforeObsolete);
  resolution = { kind: 'response', status: 200, body: null, file: { path: '/canonical/model.safetensors' } };
  const response = await POST(request()); assert.equal(response.status, 200); assert.equal(inspected, 1);
  assert.equal(response.headers.get('cache-control'), 'no-store');
});

test('inspection resolver rejects traversal, sibling roots and symlink escapes', async t => {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-info-path-'));
  t.after(() => fsp.rm(dir, { recursive: true, force: true }));
  const root = path.join(dir, 'output'), datasets = path.join(dir, 'datasets'), outside = path.join(dir, 'output-other');
  await Promise.all([root, datasets, outside].map(folder => fsp.mkdir(folder)));
  const file = path.join(root, '100%25.safetensors');
  const escaped = path.join(outside, 'private.safetensors');
  await fsp.writeFile(file, makeFile({ weight: tensor }));
  await fsp.writeFile(escaped, makeFile({ weight: tensor }));
  const containment = load('../src/server/pathContainment.ts', { 'fs/promises': fsp, path });
  let encrypted = false;
  const serving = load('../src/server/fileServing.ts', {
    fs, 'fs/promises': fsp, path,
    './pathContainment': containment,
    './encryptedDatasets': { findEncryptedDatasetRoot: () => encrypted ? datasets : null },
    './settings': { flushCache() {}, getDatasetsRoot: async () => datasets, getTrainingFolder: async () => root },
    '../utils/authSession': { isRequestAuthenticated: () => true },
    '../utils/remoteDatasetRefs': {},
  });
  assert.equal((await serving.resolveDownloadFileRequest(file, new Headers())).status, 200);
  for (const target of [escaped, path.join(root, '..', 'output-other', 'private.safetensors')])
    assert.equal((await serving.resolveDownloadFileRequest(target, new Headers())).status, 403);
  const link = path.join(root, 'escape');
  await fsp.symlink(outside, link, process.platform === 'win32' ? 'junction' : 'dir');
  assert.equal((await serving.resolveDownloadFileRequest(path.join(link, 'private.safetensors'), new Headers())).status, 403);
  encrypted = true;
  assert.equal((await serving.resolveDownloadFileRequest(file, new Headers())).status, 403);
});
