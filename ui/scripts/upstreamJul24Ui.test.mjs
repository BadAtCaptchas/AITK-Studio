import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const uiRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const toolkitRoot = path.resolve(uiRoot, '..');
const readUi = relativePath => fs.readFileSync(path.join(uiRoot, relativePath), 'utf8');
const readRoot = relativePath => fs.readFileSync(path.join(toolkitRoot, relativePath), 'utf8');

test('gated model metadata and Studio guidance cover the upstream architectures', () => {
  const options = readUi('src/app/jobs/new/options.ts');
  const simpleJob = readUi('src/app/jobs/new/SimpleJob.tsx');
  const mappings = new Map([
    ['flux', 'https://huggingface.co/black-forest-labs/FLUX.1-dev'],
    ['flux_kontext', 'https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev'],
    ['flux2', 'https://huggingface.co/black-forest-labs/FLUX.2-dev'],
    ['flux2_klein_9b', 'https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B'],
    ['ideogram4:fp8', 'https://huggingface.co/ideogram-ai/ideogram-4-fp8'],
    ['krea2', 'https://huggingface.co/krea/Krea-2-Raw'],
    ['krea2:turbo', 'https://huggingface.co/krea/Krea-2-Turbo'],
    ['krea2:o_edit', 'https://huggingface.co/krea/Krea-2-Raw'],
    ['krea2:o_edit_turbo', 'https://huggingface.co/krea/Krea-2-Turbo'],
  ]);

  assert.match(options, /gateUrl\?: string/);
  for (const [arch, gateUrl] of mappings) {
    const escapedArch = arch.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const escapedUrl = gateUrl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    assert.match(
      options,
      new RegExp(`name: '${escapedArch}'[\\s\\S]*?gateUrl: '${escapedUrl}'`),
      `${arch} should link to its gated model page`,
    );
  }

  assert.match(simpleJob, /modelArch\?\.gateUrl/);
  assert.match(simpleJob, /openDoc\(\{/);
  assert.match(simpleJob, /https:\/\/huggingface\.co\/settings\/tokens/);
  assert.match(simpleJob, /<Link href="\/settings"/);
  assert.match(simpleJob, /rel="noopener noreferrer"/);
});

test('Studio supported-model documentation contains the July 24 additions', () => {
  const readme = readRoot('README.md');

  assert.match(readme, /zhen-nan\/L2P/);
  assert.match(readme, /krea\/Krea-2-Turbo.*Krea 2 Turbo/);
  assert.match(readme, /krea\/Krea-2-Raw.*Krea 2 Edit Training/);
  assert.match(readme, /krea\/Krea-2-Turbo.*Krea 2 Turbo Edit Training/);
  assert.match(readme, /# AITK Studio/);
});

test('Next dev indicators are disabled without weakening Studio build checks', () => {
  const nextConfig = readUi('next.config.ts');

  assert.match(nextConfig, /devIndicators:\s*false/);
  assert.doesNotMatch(nextConfig, /ignoreBuildErrors/);
  assert.match(nextConfig, /bodySizeLimit:\s*'5gb'/);
  assert.match(nextConfig, /middlewareClientMaxBodySize:\s*'5gb'/);
});

test('the upstream sqlite3 package upgrade remains incorporated without dependency churn', () => {
  const packageJson = JSON.parse(readUi('package.json'));
  const packageLock = JSON.parse(readUi('package-lock.json'));

  assert.equal(packageJson.dependencies.sqlite3, '^6.0.1');
  assert.equal(packageLock.packages['node_modules/sqlite3'].version, '6.0.1');
});
