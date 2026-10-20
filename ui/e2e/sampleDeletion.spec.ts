import { test, expect, type Page } from '@playwright/test';
import { mkdir, writeFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { randomUUID } from 'node:crypto';
import sharp from 'sharp';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';

const origin = 'http://127.0.0.1:15875';
test.setTimeout(120_000);
test.use({ viewport: { width: 1400, height: 1000 }, navigationTimeout: 60_000 });

async function openJob(page: Page) {
  await page.route('**/api/monitor', route => route.abort());
  await page.goto('/');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password' }).click();
  await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
  const config = structuredClone(defaultJobConfig);
  const name = `sample-deletion-${randomUUID().slice(0, 8)}`;
  config.config.name = name;
  config.config.process[0].sample.samples = ['one', 'two', 'three'].map(prompt => ({ prompt }));
  const response = await page.request.post('/api/jobs', {
    headers: { Origin: origin },
    data: { name, gpu_ids: '0', worker_id: 'local', job_config: config },
  });
  expect(response.ok()).toBe(true);
  const job: unknown = await response.json();
  if (!job || typeof job !== 'object' || !('id' in job) || typeof job.id !== 'string' ||
      !('storage_key' in job) || typeof job.storage_key !== 'string') throw new Error('Invalid fixture job');
  return { id: job.id, storageKey: job.storage_key };
}

test('delete selected and all local samples with confirmation and associated file cleanup', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const job = await openJob(page);
  const settings: unknown = await (await page.request.get('/api/settings')).json();
  if (!settings || typeof settings !== 'object' || !('TRAINING_FOLDER' in settings) ||
      typeof settings.TRAINING_FOLDER !== 'string') throw new Error('Missing fixture training folder');
  const trainingRoot = path.resolve(settings.TRAINING_FOLDER);
  // Refuse to create or delete samples unless this is the disposable browser-test workspace.
  expect(path.dirname(path.dirname(trainingRoot))).toBe(path.resolve(os.tmpdir()));
  expect(path.basename(path.dirname(trainingRoot))).toMatch(/^aitk-browser-fixture-/);
  const folder = path.join(trainingRoot, job.storageKey, 'samples');
  expect(path.relative(trainingRoot, folder)).not.toMatch(/^\.\./);
  await mkdir(path.join(folder, '.thumbs'), { recursive: true });
  const png = await sharp({ create: { width: 32, height: 32, channels: 3, background: '#4477aa' } }).png().toBuffer();
  const names = [0, 1, 2].map(i => `1700000000000__000000010_${i}.png`);
  for (const name of names) {
    await writeFile(path.join(folder, name), png);
    await writeFile(path.join(folder, name.replace('.png', '.txt')), 'fixture caption');
    await writeFile(path.join(folder, '.thumbs', `${name}.png`), png);
  }
  await writeFile(path.join(trainingRoot, job.storageKey, 'keep.safetensors'), 'checkpoint fixture');
  await page.goto(`/jobs/${job.id}`);
  await page.getByRole('button', { name: 'Samples', exact: true }).click();
  const toolbar = page.getByRole('toolbar', { name: 'Sample actions' });
  await expect(toolbar.getByRole('status')).toHaveText('3 samples');
  await toolbar.getByRole('button', { name: 'Select samples', exact: true }).click();
  await expect(toolbar.getByRole('button', { name: 'Delete selected', exact: true })).toBeDisabled();
  await toolbar.getByRole('button', { name: 'Select all', exact: true }).click();
  await expect(toolbar.getByRole('status')).toHaveText('3 of 3 selected');
  await toolbar.getByRole('button', { name: 'Clear selection', exact: true }).click();
  await page.getByRole('checkbox', { name: 'Select sample 1', exact: true }).check();
  await page.getByRole('checkbox', { name: 'Select sample 3', exact: true }).check();
  await expect(toolbar.getByRole('status')).toHaveText('2 of 3 selected');
  await expect(page.locator('img[src^="blob:"]')).toHaveCount(3);
  await page.screenshot({ path: '.review-build/sample-selection.png' });
  await page.getByRole('checkbox', { name: 'Select sample 3', exact: true }).press('Delete');
  const selectedDialog = page.getByRole('dialog', { name: 'Delete Selected Samples', exact: true });
  await expect(selectedDialog).toContainText('2 selected samples');
  await selectedDialog.getByRole('button', { name: 'Delete selected', exact: true }).click();
  await expect(selectedDialog).toBeHidden();
  await expect(toolbar.getByRole('status')).toHaveText('1 sample');
  expect((await readdir(folder)).sort()).toEqual(['.thumbs', names[1], names[1].replace('.png', '.txt')].sort());
  expect(await readdir(path.join(folder, '.thumbs'))).toEqual([`${names[1]}.png`]);
  await toolbar.getByRole('button', { name: 'Delete all', exact: true }).click();
  const allDialog = page.getByRole('dialog', { name: 'Delete All Samples', exact: true });
  await allDialog.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(allDialog).toBeHidden();
  expect(await readdir(folder)).toContain(names[1]);
  await toolbar.getByRole('button', { name: 'Delete all', exact: true }).click();
  // Samples generated after confirmation opens must not silently join the deletion.
  const newName = '1700000000001__000000020_0.png';
  await writeFile(path.join(folder, newName), png);
  await allDialog.getByRole('button', { name: 'Delete all', exact: true }).click();
  await expect(allDialog).toBeHidden();
  await expect(toolbar.getByRole('status')).toHaveText('1 sample');
  expect((await readdir(folder)).sort()).toEqual(['.thumbs', newName].sort());
  await toolbar.getByRole('button', { name: 'Delete all', exact: true }).click();
  await allDialog.getByRole('button', { name: 'Delete all', exact: true }).click();
  await expect(page.getByText('No Samples Found', { exact: true })).toBeVisible();
  expect(await readdir(folder)).toEqual(['.thumbs']);
  expect(await readdir(path.join(folder, '.thumbs'))).toEqual([]);
  expect(await readdir(path.join(trainingRoot, job.storageKey))).toContain('keep.safetensors');
  expect(errors).toEqual([]);
});

test('remote sample references use the mirror job and failed deletions remain selected for retry', async ({ page }) => {
  const job = await openJob(page);
  const names = ['1700000000000__000000010_0.png', '1700000000000__000000010_1.png'];
  let remaining = [...names];
  let fail = true;
  const deleted: string[] = [];
  await page.route(`**/api/jobs/${job.id}/samples`, route => route.fulfill({
    json: { samples: remaining.map(name => `remote://${job.id}/img/${encodeURIComponent(`/api/jobs/remote-job/samples/${name}`)}/${name}`) },
  }));
  await page.route('**/api/remote-assets?*', route => route.fulfill({ status: 404 }));
  await page.route(`**/api/jobs/${job.id}/samples/*`, route => {
    expect(route.request().method()).toBe('DELETE');
    const name = decodeURIComponent(new URL(route.request().url()).pathname.split('/').at(-1)!);
    if (fail && name === names[1]) return route.fulfill({ status: 500, json: { error: 'Fixture failure' } });
    remaining = remaining.filter(item => item !== name);
    deleted.push(name);
    return route.fulfill({ json: { deleted: true } });
  });
  await page.goto(`/jobs/${job.id}`);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Samples', exact: true }).click();
  const toolbar = page.getByRole('toolbar', { name: 'Sample actions' });
  await toolbar.getByRole('button', { name: 'Delete all', exact: true }).click();
  await page.getByRole('dialog', { name: 'Delete All Samples', exact: true })
    .getByRole('button', { name: 'Delete all', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Delete All Samples', exact: true })).toBeHidden();
  await expect(toolbar.getByRole('status')).toHaveText('1 of 1 selected');
  await expect(toolbar.getByRole('alert')).toContainText('Could not delete 1 sample.');
  await expect(page.getByRole('checkbox', { name: 'Select sample 1', exact: true })).toBeChecked();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: '.review-build/sample-selection-mobile.png' });
  expect(deleted).toEqual([names[0]]);
  fail = false;
  await toolbar.getByRole('button', { name: 'Delete selected', exact: true }).click();
  await page.getByRole('dialog', { name: 'Delete Selected Samples', exact: true })
    .getByRole('button', { name: 'Delete selected', exact: true }).click();
  await expect(page.getByText('No Samples Found', { exact: true })).toBeVisible();
  expect(deleted).toEqual(names);
});
