import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { captionFixture, installCaptionFixtures } from './captionFixture';

test.beforeEach(async ({ page }) => {
  await installCaptionFixtures(page);
});

async function launchFixture(page: Page, create = false) {
  const fixture = await captionFixture(page);
  await page.route('**/api/gpu?*', route =>
    route.fulfill({
      json: {
        hasNvidiaSmi: true,
        isMac: false,
        gpus: [
          {
            index: 0,
            name: 'Fixture GPU',
            driverVersion: 'fixture',
            temperature: 40,
            utilization: { gpu: 0, memory: 0 },
            memory: { total: 24576, free: 24576, used: 0 },
            power: { draw: 0, limit: 300 },
            clocks: { graphics: 0, memory: 0 },
            fan: { speed: 0 },
          },
        ],
      },
    }),
  );
  const state = { ...fixture.state, saveSucceeds: true, failStart: true, failQueue: false, failSave: false, copies: 0 };
  await page.route('**/api/jobs', route => {
    if (route.request().method() !== 'POST') return route.continue();
    const data: unknown = route.request().postDataJSON();
    state.saves.push(data);
    if (state.failSave) return route.fulfill({ status: 503, json: { error: 'Fixture save failure' } });
    return route.fulfill({
      json: { ...fixture.job, name: data && typeof data === 'object' && 'name' in data ? data.name : fixture.job.name },
    });
  });
  await page.route('**/api/jobs/caption-recovery-fixture/start', route => {
    state.starts++;
    return state.failStart
      ? route.fulfill({ status: 503, json: { error: 'Fixture start failure' } })
      : route.fulfill({ json: {} });
  });
  await page.route('**/api/queue/*/start?*', route => {
    state.queueStarts++;
    return state.failQueue
      ? route.fulfill({ status: 503, json: { error: 'Fixture queue failure' } })
      : route.fulfill({ json: {} });
  });
  await page.route('**/api/datasets/copy', route => {
    state.copies++;
    return route.fulfill({ json: { path: 'fixture/copied-caption-data' } });
  });
  if (create) {
    const dataset = `launch-caption-${randomUUID().slice(0, 8)}`;
    const upload = await page.request.post('/api/datasets/upload', {
      headers: {
        Origin: 'http://127.0.0.1:15875',
        'Content-Type': 'application/octet-stream',
        'X-AITK-Dataset-Name': encodeURIComponent(dataset),
        'X-AITK-File-Name': 'fixture.webp',
        'X-AITK-Upload-Group': randomUUID(),
      },
      data: await readFile('public/images/graphite-studio-hero.webp'),
    });
    expect(upload.status()).toBe(200);
    await page.goto(`/datasets/${dataset}`);
    await page.getByRole('button', { name: 'Generate captions', exact: true }).click();
  } else await page.getByLabel('Edit captions', { exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'Caption Dataset', exact: true });
  const submit = modal.locator('button[type="submit"]');
  await expect(submit).toBeEnabled();
  return { ...fixture, state, modal, submit };
}

test('caption start retry reuses the saved new job without saving a duplicate', async ({ page }) => {
  const { state, modal, submit } = await launchFixture(page, true);
  await submit.click();
  await expect.poll(() => state.starts).toBe(1);
  await expect(submit).toBeEnabled();
  state.failStart = false;
  await submit.click();
  await expect(modal).toHaveCount(0);
  await expect(page.getByRole('alert').filter({ hasText: 'captioning could not start' })).toHaveCount(0);
  expect(state.saves).toHaveLength(1);
  expect(state.saves[0]).toMatchObject({ gpu_ids: '0' });
  expect(state.starts).toBe(2);
  expect(state.queueStarts).toBe(1);
});

test('caption queue retry does not resave or restart a job that already started', async ({ page }) => {
  const { state, modal, submit } = await launchFixture(page);
  state.failStart = false;
  state.failQueue = true;
  await submit.click();
  await expect.poll(() => state.queueStarts).toBe(1);
  await expect(submit).toBeEnabled();
  await expect(modal.getByLabel('Caption Prompt', { exact: true })).toBeDisabled();
  await expect(modal.getByRole('alert').filter({ hasText: 'queue could not start' })).toBeVisible();
  state.failQueue = false;
  await submit.click();
  await expect(modal).toHaveCount(0);
  expect(state.saves).toHaveLength(1);
  expect(state.starts).toBe(1);
  expect(state.queueStarts).toBe(2);
  await expect(page.getByRole('alert').filter({ hasText: 'queue could not start' })).toHaveCount(0);
});

test('caption save conflicts keep actionable server guidance and do not start a job', async ({ page }) => {
  const { state, modal, submit } = await launchFixture(page);
  await page.route('**/api/jobs', route =>
    route.fulfill({
      status: 409,
      json: {
        error: 'Stop the active job before changing its configuration',
        code: 'JOB_ACTIVE',
      },
    }),
  );
  await submit.click();
  await expect(
    modal.getByRole('alert').filter({ hasText: 'Stop the active job before changing its configuration' }),
  ).toBeVisible();
  await expect(submit).toBeEnabled();
  expect(state.starts).toBe(0);
  expect(state.queueStarts).toBe(0);
});

test('caption retry saves newer edits to the same already-created job', async ({ page }) => {
  const { state, modal, submit, job } = await launchFixture(page, true);
  await submit.click();
  await expect.poll(() => state.starts).toBe(1);
  await expect(submit).toBeEnabled();
  await modal.getByLabel('Caption Prompt', { exact: true }).fill('Newer retry prompt');
  state.failStart = false;
  await submit.click();
  await expect(modal).toHaveCount(0);
  expect(state.saves).toHaveLength(2);
  expect(state.saves[1]).toMatchObject({
    id: job.id,
    job_config: { config: { process: [{ caption: { caption_prompt: 'Newer retry prompt' } }] } },
  });
});

test('caption save retry reuses the completed dataset copy', async ({ page }) => {
  const { state, modal, submit } = await launchFixture(page, true);
  // The existing JSON preset creates a dataset copy before saving a new job.
  await modal.getByRole('combobox', { name: 'Caption Output', exact: true }).click();
  await page.getByRole('option', { name: 'Convert to JSON', exact: true }).click();
  await modal.getByRole('combobox', { name: 'Destination', exact: true }).click();
  await page.getByRole('option', { name: 'Make dataset copy', exact: true }).click();
  state.failSave = true;
  await submit.click();
  await expect.poll(() => state.saves.length).toBe(1);
  await expect(submit).toBeEnabled();
  state.failSave = false;
  state.failStart = false;
  await submit.click();
  await expect(modal).toHaveCount(0);
  expect(state.copies).toBe(1);
  expect(state.saves[1]).toMatchObject({
    job_ref: 'fixture/copied-caption-data',
    job_config: { config: { process: [{ caption: { path_to_caption: 'fixture/copied-caption-data' } }] } },
  });
});
