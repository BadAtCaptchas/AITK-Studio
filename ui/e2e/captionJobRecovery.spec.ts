import { test, expect } from '@playwright/test';
import { captionFixture, installCaptionFixtures } from './captionFixture';

test.beforeEach(async ({ page }) => {
  await installCaptionFixtures(page);
});

test('failed caption source load blocks saving defaults and can retry the stored job', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const { state, config, job } = await captionFixture(page);
  state.fail = true;
  const failed = page.waitForResponse(
    response => response.url().includes('/api/jobs?id=') && response.status() === 503,
  );
  await page.getByLabel('Edit captions', { exact: true }).click();
  await failed;
  const modal = page.getByRole('dialog', { name: 'Caption Dataset', exact: true });
  const save = modal.getByRole('button', { name: 'Save & start captioning', exact: true });
  await expect(save).toBeDisabled();
  await expect(modal.getByText('Could not load caption job.', { exact: true })).toBeVisible();
  await save.evaluate(button =>
    button.closest('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })),
  );
  expect(state.saves).toEqual([]);
  state.fail = false;
  await modal.getByRole('button', { name: 'Retry loading caption job', exact: true }).click();
  await expect(save).toBeEnabled();
  await save.click();
  await expect.poll(() => state.saves.length).toBe(1);
  expect(state.saves[0]).toMatchObject({ id: job.id, name: job.name, job_config: config });
  await expect(modal.getByRole('alert').filter({ hasText: 'Failed to save job' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('malformed caption source is recoverable without rendering or saving it', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const { state, config } = await captionFixture(page);
  state.malformed = true;
  await page.getByLabel('Edit captions', { exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'Caption Dataset', exact: true });
  await expect(modal.getByText('Could not load caption job.', { exact: true })).toBeVisible();
  const save = modal.getByRole('button', { name: 'Save & start captioning', exact: true });
  await expect(save).toBeDisabled();
  expect(state.saves).toEqual([]);
  state.malformed = false;
  await modal.getByRole('button', { name: 'Retry loading caption job', exact: true }).click();
  await expect(save).toBeEnabled();
  await save.click();
  await expect.poll(() => state.saves.length).toBe(1);
  expect(state.saves[0]).toMatchObject({ job_config: config });
  expect(errors).toEqual([]);
});

test('closing a pending caption source prevents its late response from changing a reopened editor', async ({
  page,
}) => {
  const { state, config, job } = await captionFixture(page);
  let release!: () => void;
  const held = new Promise<void>(resolve => {
    release = resolve;
  });
  let started!: () => void;
  const sourceStarted = new Promise<void>(resolve => {
    started = resolve;
  });
  let finished!: () => void;
  const sourceFinished = new Promise<void>(resolve => {
    finished = resolve;
  });
  let first = true;
  await page.route('**/api/jobs?*', async route => {
    if (!first) return route.fulfill({ json: job });
    first = false;
    started();
    await held;
    const staleConfig = structuredClone(config);
    staleConfig.config.name = 'Stale closed editor';
    staleConfig.config.process[0].caption.caption_prompt = 'Stale closed prompt';
    await route.fulfill({ json: { ...job, job_config: JSON.stringify(staleConfig) } });
    finished();
  });
  await page.getByLabel('Edit captions', { exact: true }).click();
  await sourceStarted;
  const modal = page.getByRole('dialog', { name: 'Caption Dataset', exact: true });
  await modal.getByRole('button', { name: 'Close modal', exact: true }).click();
  await expect(modal).toHaveCount(0);
  await page.getByLabel('Edit captions', { exact: true }).click();
  const save = modal.getByRole('button', { name: 'Save & start captioning', exact: true });
  await expect(save).toBeEnabled();
  release();
  await sourceFinished;
  await save.click();
  await expect.poll(() => state.saves.length).toBe(1);
  expect(state.saves[0]).toMatchObject({ id: job.id, name: job.name, job_config: config });
});
