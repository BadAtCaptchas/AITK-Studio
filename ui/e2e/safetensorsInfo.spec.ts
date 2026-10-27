import { test, expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { randomUUID } from 'node:crypto';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';

const origin = 'http://127.0.0.1:15875';
test('inspect a real checkpoint, filter tensors and read metadata', async ({ page }) => {
  await page.route('**/api/monitor', route => route.abort());
  await page.goto('/');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password' }).click();
  await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
  const config = structuredClone(defaultJobConfig);
  const name = `info-${randomUUID().slice(0, 8)}`;
  config.config.name = name;
  const response = await page.request.post('/api/jobs', {
    headers: { Origin: origin }, data: { name, gpu_ids: '0', worker_id: 'local', job_config: config },
  });
  expect(response.ok()).toBe(true);
  const job: unknown = await response.json();
  if (!job || typeof job !== 'object' || !('id' in job) || typeof job.id !== 'string'
    || !('storage_key' in job) || typeof job.storage_key !== 'string') throw new Error('Invalid fixture job');
  const settings: unknown = await (await page.request.get('/api/settings')).json();
  if (!settings || typeof settings !== 'object' || !('TRAINING_FOLDER' in settings)
    || typeof settings.TRAINING_FOLDER !== 'string') throw new Error('Invalid fixture settings');
  const root = path.resolve(settings.TRAINING_FOLDER);
  expect(path.dirname(path.dirname(root))).toBe(path.resolve(os.tmpdir()));
  expect(path.basename(path.dirname(root))).toMatch(/^aitk-browser-fixture-/);
  const folder = path.join(root, job.storage_key);
  expect(path.relative(root, folder)).not.toMatch(/^\.\./);
  await mkdir(folder, { recursive: true });
  const header = Buffer.from(JSON.stringify({
    'transformer.weight': { dtype: 'F32', shape: [2], data_offsets: [0, 8] },
    __metadata__: { description: 'Inspector browser regression' },
  }));
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(header.length));
  await writeFile(path.join(folder, 'inspect.safetensors'), Buffer.concat([prefix, header, Buffer.alloc(8)]));
  await page.goto(`/jobs/${job.id}`);
  await page.getByRole('button', { name: 'Inspect inspect.safetensors', exact: true }).click();
  const dialog = page.getByRole('dialog');
  await expect(dialog.getByText('transformer.weight', { exact: true })).toBeVisible();
  await expect(dialog.getByText('[2]', { exact: true })).toBeVisible();
  await dialog.getByRole('textbox', { name: 'Filter tensors' }).fill('missing');
  await expect(dialog.getByText('No tensors match')).toBeVisible();
  await dialog.getByRole('textbox', { name: 'Filter tensors' }).fill('F32');
  await expect(dialog.getByText('transformer.weight', { exact: true })).toBeVisible();
  await dialog.getByRole('button', { name: 'Metadata' }).click();
  await expect(dialog.getByText('Inspector browser regression', { exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await page.getByRole('button', { name: 'Inspect inspect.safetensors', exact: true }).click();
  await expect(page.getByRole('dialog').getByText('transformer.weight', { exact: true })).toBeVisible();
});
