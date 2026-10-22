import { test, expect, type Page } from '@playwright/test';
import { randomUUID } from 'node:crypto';

test.setTimeout(120_000);
test.use({ navigationTimeout: 60_000 });

async function prepareRemoteForm(page: Page) {
  const errors: string[] = [];
  const imports: string[] = [];
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
            name: 'Test GPU',
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
  await page.route('**/api/datasets/import-remote', route => {
    imports.push(route.request().url());
    return route.abort();
  });
  await page.goto('/');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password' }).click();
  await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
  const headers = { Origin: 'http://127.0.0.1:15875' };
  const workerResponse = await page.request.post('/api/workers', {
    headers,
    data: {
      name: `Save test ${randomUUID()}`,
      base_url: 'http://127.0.0.1:1',
      api_token: 'fixture-only',
      enabled: true,
    },
  });
  expect(workerResponse.ok()).toBe(true);
  const worker = await workerResponse.json();
  const datasetRef = `aitk-dataset://remote/${encodeURIComponent(worker.id)}/cats`;
  await page.route('**/api/datasets/list*', route =>
    route.fulfill({
      json: {
        datasets: [
          {
            name: 'cats',
            encrypted: false,
            source: 'remote',
            worker_id: worker.id,
            worker_name: worker.name,
            ref: datasetRef,
            itemCount: 5,
            missingCaptionCount: 0,
          },
        ],
        errors: [],
      },
    }),
  );
  await page.goto('/jobs/new');
  const name = `remote-save-${randomUUID().slice(0, 8)}`;
  await page.getByLabel('Training name', { exact: true }).fill(name);
  await page.getByLabel('Model path', { exact: true }).fill('fixture/not-downloaded-yet');
  await page.getByRole('combobox', { name: 'Compute', exact: true }).fill(worker.name);
  await page.getByRole('option', { name: worker.name, exact: true }).click();
  const save = page.getByRole('button', { name: 'Create Job', exact: true }).first();
  await expect(save).toBeEnabled();
  return { save, worker, datasetRef, name, imports, errors };
}

test('a new remote job saves before its model is downloaded, without importing its same-worker dataset', async ({
  page,
}) => {
  const { save, worker, datasetRef, imports, errors } = await prepareRemoteForm(page);
  const savedResponse = page.waitForResponse(
    response => response.url().endsWith('/api/jobs') && response.request().method() === 'POST',
  );
  await save.click();
  const response = await savedResponse;
  expect(response.ok()).toBe(true);
  const saved = await response.json();
  expect(saved.worker_id).toBe(worker.id);
  expect(JSON.parse(saved.job_config).config.process[0].model.name_or_path).toBe('fixture/not-downloaded-yet');
  expect(JSON.parse(saved.job_config).config.process[0].datasets[0].folder_path).toBe(datasetRef);
  await expect(page).toHaveURL(new RegExp(`/jobs/${saved.id}$`));
  expect(imports).toEqual([]);
  expect(errors).toEqual([]);
});

test('a stalled save releases the button, keeps the form, and allows an explicit retry', async ({ page }) => {
  const { save, name, errors } = await prepareRemoteForm(page);
  await page.clock.install();
  let holdSave = true;
  let posts = 0;
  await page.route('**/api/jobs', route => {
    if (route.request().method() === 'POST') {
      posts += 1;
      if (holdSave) return;
    }
    return route.continue();
  });
  const sent = page.waitForRequest(request => request.url().endsWith('/api/jobs') && request.method() === 'POST');
  await save.click();
  await sent;
  await expect(save).toBeDisabled();
  await expect(save).toHaveText('Saving...');
  await page.clock.fastForward(30_001);
  await expect(
    page.getByRole('alert').filter({ hasText: 'The server did not confirm the save in time.' }),
  ).toBeVisible();
  await expect(save).toBeEnabled();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue(name);
  expect(posts).toBe(1);
  holdSave = false;
  const saved = page.waitForResponse(
    response => response.url().endsWith('/api/jobs') && response.request().method() === 'POST',
  );
  await save.click();
  const response = await saved;
  expect(response.ok()).toBe(true);
  const job = await response.json();
  await expect(page).toHaveURL(new RegExp(`/jobs/${job.id}$`));
  expect(posts).toBe(2);
  expect(errors).toEqual([]);
});
