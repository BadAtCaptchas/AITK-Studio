import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

const repoRoot = process.cwd();
const read = relativePath => fs.readFileSync(path.join(repoRoot, relativePath), 'utf8');

const jobConfig = read('src/app/jobs/new/jobConfig.ts');
const simpleJob = read('src/app/jobs/new/SimpleJob.tsx');
const page = read('src/app/jobs/new/page.tsx');
const checkerPage = read('src/app/watermark/page.tsx');
const checkerRoute = read('src/app/api/watermark/check/route.ts');
const sidebar = read('src/components/Sidebar.tsx');
const codecUtils = read('src/utils/authenloraCodecs.ts');
const types = read('src/types.ts');

test('watermark config is typed, defaulted, and migrated', () => {
  assert.match(types, /export interface WatermarkConfig/);
  assert.match(types, /watermark\?: WatermarkConfig/);
  assert.match(jobConfig, /export const defaultWatermarkConfig: WatermarkConfig/);
  assert.match(jobConfig, /method: 'authenlora'/);
  assert.match(jobConfig, /codec_path: 'builtin:authenlora_48bits'/);
  assert.match(codecUtils, /AUTHENLORA_BUILTIN_CODEC_BITS/);
  assert.match(codecUtils, /builtin:authenlora_100bits/);
  assert.match(jobConfig, /jobConfig\.config\.process\[0\]\.watermark = \{/);
  assert.match(jobConfig, /\.\.\.defaultWatermarkConfig/);
});

test('simple job serializes AuthenLoRA controls into process watermark config', () => {
  assert.match(simpleJob, /Enable watermarking/);
  assert.match(simpleJob, /AUTHENLORA_CODEC_OPTIONS/);
  assert.match(codecUtils, /Built-in 48-bit/);
  assert.match(codecUtils, /builtin:authenlora_100bits/);
  assert.match(simpleJob, /config\.process\[0\]\.watermark\.enabled/);
  assert.match(simpleJob, /config\.process\[0\]\.watermark\.codec_path/);
  assert.match(simpleJob, /config\.process\[0\]\.watermark\.msg_bits/);
  assert.match(simpleJob, /config\.process\[0\]\.watermark\.watermark_loss_weight/);
  assert.match(simpleJob, /config\.process\[0\]\.watermark\.bake_on_save/);
});

test('save validation blocks invalid AuthenLoRA configs', () => {
  assert.match(page, /AuthenLoRA watermarking requires an image LoRA job/);
  assert.match(page, /AuthenLoRA watermarking requires a local codec path/);
  assert.match(page, /AuthenLoRA message bits must be greater than 0/);
  assert.match(page, /AUTHENLORA_BUILTIN_CODEC_BITS/);
  assert.match(page, /AuthenLoRA secret bits must be binary and match Message bits/);
  assert.match(page, /\['lora', 'locon', 'lycoris', 'lokr'\]/);
});

test('watermark checker page and API route are wired', () => {
  assert.match(sidebar, /Watermark/);
  assert.match(sidebar, /href: '\/watermark'/);
  assert.match(checkerPage, /Watermark Checker/);
  assert.match(checkerPage, /\/api\/watermark\/check/);
  assert.match(checkerPage, /decoded_bits/);
  assert.match(checkerPage, /watermark_status/);
  assert.match(checkerPage, /Watermark not detected/);
  assert.match(checkerPage, /All-zero clean message/);
  assert.match(checkerRoute, /check_authenlora_watermark\.py/);
  assert.match(checkerRoute, /formData\.get\('image'\)/);
  assert.match(checkerRoute, /MAX_IMAGE_BYTES/);
});
