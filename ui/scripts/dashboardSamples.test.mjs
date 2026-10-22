import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import * as React from 'react';
import * as jsxRuntime from 'react/jsx-runtime';
import { renderToStaticMarkup } from 'react-dom/server';
import * as icons from 'lucide-react';

function loadSource(relative, dependencies = {}) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    fileName: relative,
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  const evaluatedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    if (name === 'react/jsx-runtime') return jsxRuntime;
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
    return dependencies[name];
  }, evaluatedModule, evaluatedModule.exports);
  return evaluatedModule.exports;
}

const basic = loadSource('../src/utils/basic.ts');
const media = loadSource('../src/utils/media.ts', { './remoteDatasetRefs': loadSource('../src/utils/remoteDatasetRefs.ts') });
const jobStatus = loadSource('../src/utils/jobStatus.ts');

function fixture({ jobStatus: status = 'running', samples = [], sampleStatus = 'success' } = {}) {
  const requests = [];
  const job = { id: 'local-mirror', name: 'Example run', worker_id: 'remote-worker', remote_job_id: 'worker-job', status, step: 6250, created_at: '2026-01-01T00:00:00Z' };
  const preview = loadSource('../src/components/JobSamplePreview.tsx', {
    'lucide-react': icons, '@/utils/basic': basic, '@/utils/jobStatus': jobStatus, '@/utils/media': media,
    '@/hooks/useSampleImages': (id, interval) => { requests.push({ id, interval }); return { sampleImages: samples, status: sampleStatus }; },
  });
  // The operational Overview now mounts the full Samples workspace on demand.
  // Keep this preview component's routing/polling contract covered independently;
  // premiumShell.spec.ts covers the new Overview's actual Samples tab.
  return { requests, render: () => renderToStaticMarkup(React.createElement(preview.default, { job })) };
}

const remoteSample = name => `remote://local-mirror/img/${encodeURIComponent(`/api/jobs/worker-job/samples/${name}`)}/${encodeURIComponent(name)}`;

test('the compact run preview displays the latest local and remote samples through thumbnail routes', () => {
  for (const sample of [name => `/api/jobs/local-mirror/samples/${name}`, remoteSample]) {
    const samples = ['sample_10.png', 'sample_2.png', 'sample_11.png', 'sample_1.png', 'sample_12.txt', 'sample_12.mp4'].map(sample);
    const f = fixture({ samples });
    const html = f.render();
    assert.deepEqual(f.requests, [{ id: 'local-mirror', interval: 5000 }]);
    assert.match(html, /Latest run samples/);
    const sources = [...html.matchAll(/<img [^>]*src="([^"]+)"/g)].map(match => match[1].replaceAll('&amp;', '&'));
    assert.deepEqual(sources, ['sample_11.png', 'sample_10.png', 'sample_2.png'].map(name => media.getSampleThumbnailUrl(sample(name))));
    assert.equal(samples.length, 6, 'preview selection must not mutate the shared sample list');
  }
});

test('stopped runs show samples without continued polling and empty/error states remain explicit', () => {
  const stopped = fixture({ jobStatus: 'stopped', samples: [remoteSample('sample_10.png')] });
  assert.match(stopped.render(), /<img /);
  assert.equal(stopped.requests[0].interval, null);
  assert.match(fixture().render(), /No sample images yet/);
  assert.match(fixture({ sampleStatus: 'error' }).render(), /Samples unavailable/);
  assert.match(fixture({ sampleStatus: 'loading' }).render(), /Loading samples/);
});
