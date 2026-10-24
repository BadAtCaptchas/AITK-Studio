import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { createRequire } from 'node:module';
import test from 'node:test';
import ts from 'typescript';

const require = createRequire(import.meta.url);


// Preserve Node 22's destroyed-source behavior even when tests run on newer Node.
const node22Readable = {
  toWeb(stream) {
    if (stream.destroyed) {
      const body = new ReadableStream();
      void body.cancel();
      return body;
    }
    return Readable.toWeb(stream);
  },
};

function evaluate(code, dependencies) {
  const output = ts.transpileModule(code, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loaded = { exports: {} };
  new Function('require', 'module', 'exports', output)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
      return dependencies[name];
    },
    loaded,
    loaded.exports,
  );
  return loaded.exports;
}

function fixture(t, kind, { onResolve, onOpen } = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'file-response-'));
  const filename = path.join(root, 'sample.txt');
  fs.writeFileSync(filename, 'sample bytes');
  const streams = [];
  t.after(async () => {
    await Promise.all(
      streams.map(async stream => {
        if (!stream.closed) {
          const closed = new Promise(resolve => stream.once('close', resolve));
          stream.destroy();
          await closed;
        }
      }),
    );
    fs.rmSync(root, { recursive: true, force: true });
  });
  const files = {
    ...fs,
    createReadStream(...args) {
      const stream = fs.createReadStream(...args);
      streams.push(stream);
      onOpen?.();
      return stream;
    },
  };
  const dependencies = {
    fs: files,
    stream: { Readable: node22Readable },
  };
  let serve;
  if (kind === 'samples') {
    const source = fs.readFileSync(
      new URL('../src/app/api/jobs/[jobID]/samples/[...samplePath]/route.ts', import.meta.url),
      'utf8',
    );
    const { GET } = evaluate(source, {
      ...dependencies,
      path,
      '@/server/db': { db: { jobs: { findById: async () => ({ id: 'test-job', worker_id: 'local' }) } } },
      '@/server/jobSamples': {
        resolveJobSampleFile: async () => {
          onResolve?.();
          return { path: filename, stat: fs.statSync(filename), contentType: 'text/plain' };
        },
      },
      '@/utils/authSession': { isRequestAuthenticated: async () => true },
      '@/server/remoteClient': {},
      '@/server/pathContainment': {},
      '@/server/sampleThumbnails': {},
      '@/server/audioArtwork': {},
      '@/server/sampleLayers': {},
      '@/server/sampleMetadata': {},
    });
    serve = request => GET(request, { params: Promise.resolve({ jobID: 'test-job', samplePath: ['sample.txt'] }) });
  } else {
    const relative = '../src/server/http/fileResponse.ts';
    const source = ts.createSourceFile(
      'route.ts',
      fs.readFileSync(new URL(relative, import.meta.url), 'utf8'),
      ts.ScriptTarget.Latest,
      true,
    );
    const handler = source.statements.find(
      node => ts.isFunctionDeclaration(node) && node.name?.text === 'fileResponse',
    );
    assert.ok(handler);
    const { fileResponse } = evaluate(
      `import fs from 'fs'; import { Readable } from 'stream'; const FILE_STREAM_BUFFER_BYTES = 65536; ${handler.getText(source)}`,
      dependencies,
    );
    serve = request =>
      fileResponse(request, { kind: 'response', status: 200, body: null, headers: {}, file: { path: filename } });
  }
  return { streams, serve };
}

for (const kind of ['samples', 'img', 'files']) {
  test(`${kind}: cancelled requests return without creating a disturbed response body`, async t => {
    const state = fixture(t, kind);
    const controller = new AbortController();
    controller.abort();
    const response = await state.serve(new Request('http://localhost/sample', { signal: controller.signal }));
    assert.equal(response.status, 499);
    assert.equal(response.body, null);
    assert.equal(state.streams.length, 0);
  });

  test(`${kind}: cancellation during stream setup cannot disturb the Response body`, async t => {
    const controller = new AbortController();
    const state = fixture(t, kind, { onOpen: () => controller.abort() });
    const response = await state.serve(new Request('http://localhost/sample', { signal: controller.signal }));
    assert.equal(response.status, 200);
    assert.equal(response.bodyUsed, false);
    await assert.rejects(response.text());
    assert.equal(state.streams[0].destroyed, true);
  });

  test(`${kind}: healthy responses stream the file and cancellation closes it`, async t => {
    const state = fixture(t, kind);
    const response = await state.serve(new Request('http://localhost/sample'));
    assert.equal(response.status, 200);
    assert.equal(await response.text(), 'sample bytes');
    const cancelled = await state.serve(new Request('http://localhost/sample'));
    const stream = state.streams.at(-1);
    const closed = new Promise(resolve => stream.once('close', resolve));
    await cancelled.body.cancel();
    await closed;
    assert.equal(stream.closed, true);
  });
}

test('samples: cancellation during asynchronous sample lookup does not open the file', async t => {
  const controller = new AbortController();
  const state = fixture(t, 'samples', { onResolve: () => controller.abort() });
  const response = await state.serve(new Request('http://localhost/sample', { signal: controller.signal }));
  assert.equal(response.status, 499);
  assert.equal(state.streams.length, 0);
});

test('samples: byte ranges and conditional responses still work', async t => {
  const state = fixture(t, 'samples');
  const response = await state.serve(new Request('http://localhost/sample', { headers: { range: 'bytes=0-5' } }));
  assert.equal(response.status, 206);
  assert.equal(response.headers.get('content-range'), 'bytes 0-5/12');
  assert.equal(await response.text(), 'sample');
  const cached = await state.serve(
    new Request('http://localhost/sample', { headers: { 'if-none-match': response.headers.get('etag') } }),
  );
  assert.equal(cached.status, 304);
  assert.equal(cached.body, null);
  const invalid = await state.serve(new Request('http://localhost/sample', { headers: { range: 'bytes=50-' } }));
  assert.equal(invalid.status, 416);
  assert.equal(state.streams.length, 1);
});
