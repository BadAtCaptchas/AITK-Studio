import { test, expect } from '@playwright/test';
import { randomUUID } from 'node:crypto';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';

const headers = { Origin: 'http://127.0.0.1:15875' };
test.setTimeout(120_000);
test.use({ navigationTimeout: 60_000 });

test('migrate a training job, review requirements, and create an independent run at step zero', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/gpu?*', route =>
    route.fulfill({
      json: {
        hasNvidiaSmi: true,
        isMac: false,
        gpus: [
          {
            index: 0,
            name: 'Migration test GPU',
            utilization: { gpu: 0, memory: 0 },
            memory: { total: 24576, free: 24576, used: 0 },
            power: { draw: 0, limit: 350 },
            clocks: { graphics: 0, memory: 0 },
            fan: { speed: 0 },
          },
        ],
      },
    }),
  );
  await page.goto('/');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password' }).click();
  await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);

  const name = `migration-source-${randomUUID().slice(0, 8)}`;
  const config = structuredClone(defaultJobConfig);
  config.config.name = name;
  const p = config.config.process[0];
  p.model.arch = 'flux';
  p.model.name_or_path = 'black-forest-labs/FLUX.1-dev';
  p.train.lr = 0.00037;
  p.train.steps = 4200;
  p.train.batch_size = 3;
  p.train.gradient_accumulation = 5;
  p.network!.linear = 64;
  p.datasets[0].folder_path = '/fixture/migration-dataset';
  p.datasets[0].resolution = [640, 960];
  p.sample.samples = [{ prompt: 'my custom trigger in a garden' }];
  const created = await page.request.post('/api/jobs', {
    headers,
    data: {
      name,
      gpu_ids: '0',
      worker_id: 'local',
      job_config: config,
    },
  });
  expect(created.ok()).toBe(true);
  const source = await created.json();
  await page.goto(`/jobs/${source.id}`);
  await page.getByRole('button', { name: 'More job actions' }).click();
  await page.getByRole('menuitem', { name: 'Migrate to another model' }).click();
  await expect(page.getByRole('heading', { name: 'Migrate Training Job', exact: true })).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue(`${name}_migrated`);
  await expect(page.getByText('Training starts at step 0.', { exact: false }).first()).toBeVisible();

  const architecture = page.getByRole('combobox', { name: 'Model architecture', exact: true });
  const form = page.getByRole('tabpanel', { name: 'Form', exact: true });
  await architecture.fill('Qwen-Image-Edit');
  await page.getByRole('option', { name: 'Qwen-Image-Edit', exact: true }).click();
  await page.getByRole('button', { name: /\d+ configuration errors/ }).click();
  await expect(page.getByText(/Dataset 1 needs a control dataset/).first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Create Job', exact: true }).first()).toBeDisabled();
  await page.getByRole('button', { name: 'Close problems panel', exact: true }).click();
  await architecture.fill('Z-Image');
  await page.getByRole('option', { name: 'Z-Image', exact: true }).click();
  await form.getByText(/^Automatic adjustments/).click();
  await expect(
    form.getByText('Use destination model components and supported memory settings.', { exact: false }).first(),
  ).toBeVisible();
  await expect(page.getByLabel('Model path', { exact: true })).toHaveValue('Tongyi-MAI/Z-Image');
  await expect(page.getByLabel('Learning rate', { exact: true })).toHaveValue('0.00037');
  await form.getByText('Migrate to another model', { exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: '.review-build/model-migration.png' });

  // Leaving/restoring a migration draft must preserve its source and its independent save mode.
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Train', exact: true }).click();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs$/);
  await page.goBack();
  await expect(page.getByRole('heading', { name: 'Migrate Training Job', exact: true })).toBeVisible();
  await expect(page.getByLabel('Model path', { exact: true })).toHaveValue('Tongyi-MAI/Z-Image');
  const savedResponse = page.waitForResponse(
    response => response.url().endsWith('/api/jobs') && response.request().method() === 'POST',
  );
  await page.getByRole('button', { name: 'Create Job', exact: true }).first().click();
  const saved = await savedResponse;
  expect(saved.ok()).toBe(true);
  const target = await saved.json();
  expect(target.id).not.toBe(source.id);
  expect(target.storage_key).not.toBe(source.storage_key);
  expect(target.step).toBe(0);
  expect(target.status).toBe('stopped');
  expect(target.remote_job_id).toBeNull();
  const migrated = JSON.parse(target.job_config).config.process[0];
  expect(migrated.train.lr).toBe(0.00037);
  expect(migrated.train.steps).toBe(4200);
  expect(migrated.train.batch_size).toBe(3);
  expect(migrated.train.gradient_accumulation).toBe(5);
  expect(migrated.network.linear).toBe(64);
  expect(migrated.datasets[0].resolution).toEqual([640, 960]);
  expect(migrated.sample.samples).toEqual(p.sample.samples);
  expect(await (await page.request.get(`/api/jobs?id=${source.id}`)).json()).toMatchObject({
    job_config: source.job_config,
    storage_key: source.storage_key,
    step: 0,
    status: 'stopped',
  });
  expect(errors).toEqual([]);
});
