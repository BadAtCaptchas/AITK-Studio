import { test, expect } from '@playwright/test';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import sharp from 'sharp';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';
import { installLocalMonaco } from './localMonaco';

test.setTimeout(180_000);

function adapter() {
  const stem = 'lora_unet_blocks_0_attn_wq';
  const json = JSON.stringify({
    [stem + '.lora_down.weight']: { dtype: 'F32', shape: [1, 6144], data_offsets: [0, 24576] },
    [stem + '.lora_up.weight']: { dtype: 'F32', shape: [6144, 1], data_offsets: [24576, 49152] },
    [stem + '.alpha']: { dtype: 'F32', shape: [], data_offsets: [49152, 49156] },
  });
  const header = Buffer.from(json.padEnd(Math.ceil(json.length / 8) * 8));
  const size = Buffer.alloc(8); size.writeBigUInt64LE(BigInt(header.length));
  const weights = Buffer.alloc(49156); weights.writeFloatLE(1, 49152);
  return Buffer.concat([size, header, weights]);
}

test('trainer uploads and inspects a local LoRA, saves independent initialization, and targets remote uploads', async ({ page }) => {
  await installLocalMonaco(page);
  await page.route('**/api/generate/loras?worker_id=local', route => route.fulfill({ json: { loras: [{ path: '/local-only.safetensors', label: 'Local-only adapter' }] } }));
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/workers', route => route.fulfill({ json: { workers: [{ id: 'worker-A', name: 'Test remote worker', enabled: true }] } }));
  await page.route('**/api/gpu?*', route => route.fulfill({ json: { hasNvidiaSmi: true, isMac: false, gpus: [
    { index: 0, name: 'Test GPU', utilization: { gpu: 0, memory: 0 }, memory: { total: 24576, free: 24576, used: 0 },
      power: { draw: 0, limit: 350 }, clocks: { graphics: 0, memory: 0 }, fan: { speed: 0 } },
  ] } }));
  await page.goto('/');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password' }).click();
  await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
  await page.goto('/jobs/new');
  await expect(page.getByRole('button', { name: 'Upload LoRA locally', exact: true })).toHaveCount(0);
  await expect(page.getByLabel('Further train this LoRA', { exact: true })).toHaveCount(0);
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Continue LoRA', exact: true }).click();
  await expect(page).toHaveURL(/\/loras\/continue$/);
  await expect(page.getByRole('heading', { name: 'Continue LoRA', exact: true })).toBeVisible();
  await expect(page.getByLabel('Model path', { exact: true })).toHaveValue('krea/Krea-2-Raw');
  await page.getByRole('combobox', { name: 'Continuation training base', exact: true }).selectOption('turbo');
  await expect(page.getByLabel('Model path', { exact: true })).toHaveValue('krea/Krea-2-Turbo');
  await page.getByLabel('Model path', { exact: true }).fill('/models/custom-turbo.safetensors');
  await expect(page.getByLabel('Model path', { exact: true })).toHaveValue('/models/custom-turbo.safetensors');
  const config = structuredClone(defaultJobConfig);
  config.config.name = 'continuation-browser-' + randomUUID().slice(0, 8);
  const p = config.config.process[0];
  p.model.arch = 'krea2'; p.model.name_or_path = 'krea/Krea-2-Raw';
  p.datasets[0].folder_path = '/fixture/new-examples';
  const chooser = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Import config', exact: true }).click();
  await (await chooser).setFiles({ name: 'training.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(config)) });
  const uploadChooser = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Upload LoRA locally', exact: true }).click();
  await (await uploadChooser).setFiles({ name: 'fixture.safetensors', mimeType: 'application/octet-stream', buffer: adapter() });
  await expect(page.getByLabel('Continuation source path')).toHaveValue(/fixture.*\.safetensors/, { timeout: 30_000 });
  await page.getByRole('button', { name: 'Inspect LoRA', exact: true }).click();
  await expect(page.getByText(/1 targets verified/)).toBeVisible({ timeout: 120_000 });
  await page.screenshot({ path: '.review-build/continuation-upload-local.png', fullPage: true });
  const saved = page.waitForResponse(r => r.url().endsWith('/api/jobs') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Create Job', exact: true }).first().click();
  const response = await saved;
  expect(response.ok(), await response.text()).toBe(true);
  const job = await response.json();
  const stored = JSON.parse(job.job_config).config.process[0];
  expect(stored.network.continuation.source_sha256).toMatch(/^[a-f0-9]{64}$/);
  expect(stored.network.continuation.training_base).toBe('raw');
  expect(job.step).toBe(0);
  expect(stored.train.source_preservation.mode).toBe('none');

  await page.goto(`/jobs/new?id=${job.id}`);
  await expect(page).toHaveURL(new RegExp(`/loras/continue\\?id=${job.id}$`));
  await expect(page.getByRole('heading', { name: 'Edit LoRA Continuation', exact: true })).toBeVisible();
  await page.getByLabel('Trigger word (optional)', { exact: true }).fill('unsaved_trigger');
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Runs', exact: true }).click();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await page.goto(`/jobs/new?id=${job.id}`);
  await expect(page).toHaveURL(new RegExp(`/loras/continue\\?id=${job.id}$`));
  await page.getByRole('button', { name: 'Restore draft', exact: true }).click();
  await expect(page.getByLabel('Trigger word (optional)', { exact: true })).toHaveValue('unsaved_trigger');
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Runs', exact: true }).click();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await page.goto(`/jobs/new?cloneId=${job.id}`);
  await expect(page).toHaveURL(new RegExp(`/loras/continue\\?cloneId=${job.id}$`));
  await expect(page.getByRole('button', { name: 'Upload LoRA locally', exact: true })).toBeEnabled();
  await page.goto('/loras/continue');
  const again = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Import config', exact: true }).click();
  await (await again).setFiles({ name: 'training.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(config)) });
  await expect(page.getByRole('option', { name: 'Local-only adapter', exact: true })).toHaveCount(1);
  let finishLibrary!: () => void;
  const delayedLibrary = new Promise<void>(resolve => { finishLibrary = resolve; });
  await page.route('**/api/generate/loras?worker_id=worker-A', async route => {
    await delayedLibrary;
    await route.fulfill({ json: { loras: [{ path: 'aitk-lora://remote/worker-A/remote.safetensors', label: 'Remote-only adapter' }] } });
  });
  const libraryRequest = page.waitForRequest('**/api/generate/loras?worker_id=worker-A');
  await page.getByRole('combobox', { name: 'Compute', exact: true }).first().fill('Test remote worker');
  await page.getByRole('option', { name: 'Test remote worker', exact: true }).click();
  await libraryRequest;
  await expect(page.getByRole('option', { name: 'Local-only adapter', exact: true })).toHaveCount(0);
  finishLibrary();
  await expect(page.getByRole('option', { name: 'Remote-only adapter', exact: true })).toHaveCount(1);
  let receivedWorker: string | undefined;
  await page.route('**/api/generate/loras/upload', async route => {
    receivedWorker = route.request().headers()['x-aitk-worker-id'];
    await route.fulfill({ json: { lora: { path: 'aitk-lora://remote/worker-A/%2Fworker%2Ffixture.safetensors' }, sha256: 'a'.repeat(64) } });
  });
  const remoteChooser = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Upload LoRA to worker', exact: true }).click();
  await (await remoteChooser).setFiles({ name: 'fixture.safetensors', mimeType: 'application/octet-stream', buffer: adapter() });
  await expect(page.getByLabel('Continuation source path')).toHaveValue(/^aitk-lora:\/\/remote\/worker-A\//);
  expect(receivedWorker).toBe('worker-A');
  await page.screenshot({ path: '.review-build/continuation-upload-remote.png', fullPage: true });

  // Importing another config while a request is in flight must invalidate it.
  let finishInspection!: () => void;
  const delayedInspection = new Promise<void>(resolve => { finishInspection = resolve; });
  await page.route('**/api/training/lora/inspect', async route => {
    await delayedInspection;
    await route.fulfill({ json: { version: 1, status: 'compatible', sha256: 'b'.repeat(64), format: 'aitk_comfy_native',
      tensor_count: 2, ranks: [9], architecture: 'krea2', topology_verified: true, size_bytes: 100,
      modules: [{ name: 'blocks.0.attn.wq', rank: 9, input: 6144, output: 6144, source_scale: 1 }], declared_training_base: null } });
  });
  const inspecting = page.waitForRequest('**/api/training/lora/inspect');
  await page.getByRole('button', { name: 'Inspect LoRA', exact: true }).click();
  await inspecting;
  const replacement = structuredClone(config);
  replacement.config.process[0].network!.continuation = { version: 1, source_sha256: '', training_base: 'raw', deployment_base: 'turbo' };
  replacement.config.process[0].network!.pretrained_lora_path = '/new-source.safetensors';
  const replacing = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Import config', exact: true }).click();
  await (await replacing).setFiles({ name: 'replacement.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(replacement)) });
  await expect(page.getByLabel('Continuation source path')).toHaveValue('/new-source.safetensors');
  const staleResponse = page.waitForResponse('**/api/training/lora/inspect');
  finishInspection();
  await staleResponse;
  await expect(page.getByText('Inspection is required.', { exact: true })).toBeVisible();
  await expect(page.getByText(/targets verified/)).toHaveCount(0);

  let finishUpload!: () => void;
  const delayedUpload = new Promise<void>(resolve => { finishUpload = resolve; });
  await page.unroute('**/api/generate/loras/upload');
  await page.route('**/api/generate/loras/upload', async route => {
    await delayedUpload;
    await route.fulfill({ json: { lora: { path: '/stale-upload.safetensors' }, sha256: 'c'.repeat(64) } });
  });
  const uploading = page.waitForRequest('**/api/generate/loras/upload');
  const staleChooser = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: /^Upload LoRA (locally|to worker)$/ }).click();
  await (await staleChooser).setFiles({ name: 'stale.safetensors', mimeType: 'application/octet-stream', buffer: adapter() });
  await uploading;
  replacement.config.process[0].network!.pretrained_lora_path = '/latest-source.safetensors';
  const latestConfig = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Import config', exact: true }).click();
  await (await latestConfig).setFiles({ name: 'latest.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(replacement)) });
  await expect(page.getByLabel('Continuation source path')).toHaveValue('/latest-source.safetensors');
  const staleUploadResponse = page.waitForResponse('**/api/generate/loras/upload');
  finishUpload();
  await staleUploadResponse;
  await expect(page.getByRole('button', { name: 'Inspect LoRA', exact: true })).toBeEnabled();
  await expect(page.getByLabel('Continuation source path')).toHaveValue('/latest-source.safetensors');
  const sourceInput = page.getByLabel('Continuation source path');
  await sourceInput.fill('');
  await sourceInput.pressSequentially('/typed-source.safetensors');
  await expect(sourceInput).toHaveValue('/typed-source.safetensors');
  await expect(sourceInput).toBeFocused();
});

test('continuation keeps separate recoverable drafts and imports into its dedicated workspace', async ({ page }) => {
  await installLocalMonaco(page);
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/workers', route => route.fulfill({ json: { workers: [] } }));
  page.on('dialog', dialog => dialog.accept());
  await page.goto('/');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password' }).click();
  await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
  await page.goto('/jobs/new');
  await page.getByLabel('Training name', { exact: true }).fill('ordinary_draft');
  const continuationLink = page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Continue LoRA', exact: true });
  await continuationLink.click();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('continued_lora');
  await page.getByLabel('Training name', { exact: true }).fill('continuation_draft');
  await page.getByRole('link', { name: 'New training', exact: true }).click();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('ordinary_draft');
  await continuationLink.click();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('continuation_draft');
  await page.reload();
  await page.getByRole('button', { name: 'Restore draft', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('continuation_draft');

  await page.getByRole('link', { name: 'New training', exact: true }).click();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await page.getByRole('button', { name: 'Restore draft', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('ordinary_draft');
  const imported = structuredClone(defaultJobConfig);
  imported.config.name = 'imported_continuation';
  const process = imported.config.process[0];
  process.model.arch = 'krea2'; process.model.name_or_path = 'krea/Krea-2-Raw';
  process.network!.continuation = { version: 1, source_sha256: '', training_base: 'raw', deployment_base: 'turbo' };
  process.network!.pretrained_lora_path = '/models/imported-source.safetensors';
  const chooser = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Import config', exact: true }).click();
  await (await chooser).setFiles({ name: 'continuation.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(imported)) });
  await expect(page).toHaveURL(/\/loras\/continue$/);
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('imported_continuation');
  await page.reload();
  await page.getByRole('button', { name: 'Restore draft', exact: true }).click();
  await expect(page.getByLabel('Continuation source path')).toHaveValue('/models/imported-source.safetensors');
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('imported_continuation');
  // A draft written before the dedicated workspace existed migrates once,
  // keeps its recovery baseline, and no longer occupies regular New training.
  await page.evaluate(() => {
    const key = 'aitk:training-draft:v1:continuation:new';
    const stored = sessionStorage.getItem(key);
    if (!stored) throw new Error('Expected a recoverable continuation draft');
    sessionStorage.setItem('aitk:training-draft:v1:new', stored);
    sessionStorage.removeItem(key);
  });
  // Leaving the live editor would autosave its old key; clear that copy after
  // the full-page navigation, before restoring the legacy draft.
  await page.goto('/jobs/new');
  await page.evaluate(() => sessionStorage.removeItem('aitk:training-draft:v1:continuation:new'));
  await page.getByRole('button', { name: 'Restore draft', exact: true }).click();
  await expect(page).toHaveURL(/\/loras\/continue$/);
  await page.reload();
  await page.getByRole('button', { name: 'Restore draft', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('imported_continuation');
  await page.getByRole('link', { name: 'New training', exact: true }).click();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'New Training Job', exact: true })).toBeVisible();
  await continuationLink.click();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('imported_continuation');
  await page.screenshot({ path: '.review-build/continuation-workspace-dark.png' });
  await page.getByRole('button', { name: 'More', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Switch to light mode', exact: true }).click();
  await page.screenshot({ path: '.review-build/continuation-workspace-light.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('heading', { name: 'Continue LoRA', exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: '.review-build/continuation-workspace-mobile.png' });
});

test('reviewed source samples become a dataset with clean captions and an immutable manifest', async ({ page }) => {
  await page.route('**/api/monitor', route => route.abort());
  await page.goto('/');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password' }).click();
  await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
  const name = 'preservation-review-' + randomUUID().slice(0, 8);
  const config = { job: 'generate', config: { name, process: [{ type: 'to_folder', model: { arch: 'krea2:turbo', name_or_path: 'krea/Krea-2-Turbo' },
    generate: { prompts: ['candid source photo'], width: 512, height: 512, sample_steps: 8, guidance_scale: 1 },
    sample: { samples: [{ prompt: 'candid source photo' }], sample_steps: 8, guidance_scale: 1 } }] } };
  const response = await page.request.post('/api/jobs', { headers: { Origin: 'http://127.0.0.1:15875' },
    data: { name, worker_id: 'local', gpu_ids: '0', job_type: 'generate', job_config: config } });
  expect(response.ok(), await response.text()).toBe(true);
  const job = await response.json();
  const settings = await (await page.request.get('/api/settings')).json();
  const root = path.resolve(settings.TRAINING_FOLDER);
  expect(path.dirname(path.dirname(root))).toBe(path.resolve(os.tmpdir()));
  expect(path.basename(path.dirname(root))).toMatch(/^aitk-browser-fixture-/);
  const folder = path.join(root, job.storage_key, 'samples');
  await mkdir(folder, { recursive: true });
  const filename = '1700000000000__000000000_0.png';
  const sampleBytes = await sharp({ create: { width: 32, height: 32, channels: 3, background: '#997744' } }).png().toBuffer();
  await writeFile(path.join(folder, filename), sampleBytes);
  await writeFile(path.join(folder, filename.replace('.png', '.txt')), 'candid source photo --seed 97 --steps 8');
  await writeFile(path.join(folder, filename + '.metadata.json'), JSON.stringify({ version: 1,
    prompt: 'candid source photo', negative_prompt: '', seed: 97, width: 512, height: 512,
    guidance_scale: 1, guidance_rescale: 0, sample_steps: 8, network_multiplier: 1, num_frames: 1, fps: 15,
    step: 0, prompt_index: 0, sampler: 'flowmatch', iteration_id: 'source-test', source_lora_sha256: 'a'.repeat(64), source_base: 'krea/Krea-2-Turbo',
    media_sha256: createHash('sha256').update(sampleBytes).digest('hex') }));
  await page.goto(`/jobs/${job.id}`);
  await page.getByRole('tab', { name: 'Samples', exact: true }).click();
  const toolbar = page.getByRole('toolbar', { name: 'Sample actions' });
  await toolbar.getByRole('button', { name: 'Select samples', exact: true }).click();
  await page.getByRole('checkbox', { name: 'Select sample 1', exact: true }).check();
  await toolbar.getByRole('button', { name: 'Use selected samples for preservation', exact: true }).click();
  await page.getByLabel('Preservation dataset name', { exact: true }).fill(name);
  const publication = page.waitForResponse(r => r.url().endsWith('/api/training/lora/preservation'));
  await page.getByRole('button', { name: 'Create dataset', exact: true }).click();
  const published = await publication;
  expect(published.ok(), await published.text()).toBe(true);
  const dataset = await published.json();
  expect(path.dirname(dataset.path)).toBe(path.resolve(settings.DATASETS_FOLDER));
  expect(await readFile(path.join(dataset.path, '0000.txt'), 'utf8')).toBe('candid source photo');
  const manifest = JSON.parse(await readFile(path.join(dataset.path, '.aitk-preservation.json'), 'utf8'));
  expect(manifest.source_sha256).toBe('a'.repeat(64));
  expect(manifest.examples[0].image_sha256).toMatch(/^[a-f0-9]{64}$/);
  await expect(page.getByText(`Created ${name}.`, { exact: false })).toBeVisible();
  await page.screenshot({ path: '.review-build/continuation-preservation-review.png' });
});
