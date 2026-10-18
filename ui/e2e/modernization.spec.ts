import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';

const origin = 'http://127.0.0.1:15875';
const headers = { Origin: origin };
const screenshotDir = '.review-build/modernization';
const pageErrors = new WeakMap<Page, string[]>();

async function capture(page: Page, name: string) {
  await page.screenshot({ path: `${screenshotDir}/${name}.png`, fullPage: true });
}
async function noPageOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
}
test.beforeEach(async ({ page }) => {
  const errors: string[] = [];
  pageErrors.set(page, errors);
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({ width: 1487, height: 1058 });
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
            driverVersion: 'fixture',
            temperature: 48,
            utilization: { gpu: 0, memory: 0 },
            memory: { total: 24576, free: 22528, used: 2048 },
            power: { draw: 35, limit: 350 },
            clocks: { graphics: 1000, memory: 7000 },
            fan: { speed: 30 },
          },
        ],
      },
    }),
  );
  await page.route('**/api/cpu?*', route =>
    route.fulfill({
      json: {
        name: 'Test CPU',
        cores: 16,
        temperature: 42,
        totalMemory: 65536,
        freeMemory: 49152,
        availableMemory: 49152,
        currentLoad: 12,
      },
    }),
  );
  await page.goto('/');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password' }).click();
  await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
});

test.afterEach(async ({ page }) => {
  expect(pageErrors.get(page)).toEqual([]);
});

test('continuous training, queue and detail retain navigation and responsive controls', async ({ page }) => {
  const name = `visual-run-${randomUUID().slice(0, 8)}`;
  const created = await page.request.post('/api/jobs', {
    headers,
    data: {
      name,
      job_type: 'train',
      gpu_ids: '0',
      worker_id: 'local',
      job_config: {
        job: 'extension',
        config: {
          name,
          process: [
            {
              type: 'diffusion_trainer',
              model: { arch: 'flux', name_or_path: 'fixture/no-weights' },
              train: { steps: 3000 },
              datasets: [],
            },
          ],
        },
      },
    },
  });
  expect(created.ok()).toBe(true);
  const job = await created.json();
  await page.goto('/jobs');
  await expect(page.getByRole('heading', { name: 'Queue', exact: true })).toBeVisible();
  await page.getByRole('button', { name: /^History/ }).click();
  await page.getByLabel('Search jobs').fill(name);
  await expect(page.getByText(name, { exact: true }).first()).toBeVisible();
  await capture(page, 'queue-history');
  await page.getByLabel('Search jobs').fill('no matching visual fixture');
  await expect(page.getByText(name, { exact: true })).toHaveCount(0);
  await page.goto(`/jobs/${job.id}`);
  await expect(page.getByText(name, { exact: true }).first()).toBeVisible();
  await expect(page.getByText('Loading log...', { exact: true })).toHaveCount(0);
  await expect(page.getByText('No log output yet.', { exact: true })).toBeVisible();
  await capture(page, 'job-overview-empty');
  // Render controlled live telemetry without starting a worker or GPU job.
  const liveJob = { ...job, status: 'running', step: 1560, info: 'Training in progress', speed_string: '2.4 s/step' };
  await page.route('**/api/jobs?*', route => {
    const query = new URL(route.request().url()).searchParams;
    return route.fulfill({
      json: query.has('id')
        ? liveJob
        : {
            jobs: [
              liveJob,
              { ...job, id: 'visual-queued', name: 'Queued fixture', status: 'queued', queue_position: 1 },
            ],
            nextCursor: null,
            statusCounts: { running: 1, queued: 1 },
          },
    });
  });
  await page.route('**/api/queue', route =>
    route.fulfill({
      json: {
        queues: [{ id: 'visual-queue', gpu_ids: '0', worker_id: 'local', is_running: true }],
      },
    }),
  );
  await page.route(`**/api/jobs/${job.id}/log*`, route =>
    route.fulfill({
      json: {
        log: 'Dataset loaded: 84 images\nTraining started\nCheckpoint saved at step 1500\nStep 1560 / 3000 — loss 0.081\nTraining...',
        offset: null,
        reset: true,
      },
    }),
  );
  await page.goto('/jobs');
  await expect(page.getByText('Queue running', { exact: true })).toBeVisible();
  await expect(page.getByText('Queued fixture', { exact: true })).toBeVisible();
  const stepsHeading = await page.locator('[data-queue-steps-heading]').first().boundingBox();
  const stepsCell = await page.locator('[data-queue-steps]').first().boundingBox();
  expect(stepsHeading).not.toBeNull();
  expect(stepsCell).not.toBeNull();
  expect(Math.abs(stepsHeading!.x - stepsCell!.x)).toBeLessThan(3);
  await capture(page, 'queue');
  await page.goto(`/jobs/${job.id}`);
  await expect(page.getByRole('region', { name: 'Training log' })).toContainText('Checkpoint saved at step 1500');
  await capture(page, 'job-overview');
  await page.unroute('**/api/jobs?*');
  await noPageOverflow(page);
  await page.goto('/jobs/new');
  await expect(page.getByLabel('Training name', { exact: true })).toBeVisible();
  await expect(page.locator('#job-training')).toBeVisible();
  await expect(page.locator('#job-dataset')).toBeVisible();
  await expect(page.locator('#training-step-navigation')).toHaveCount(0);
  await capture(page, 'new-training');
  await page.getByLabel('Training name', { exact: true }).fill('unsaved continuous form');
  await page.getByRole('link', { name: 'Queue', exact: true }).click();
  await expect(page.getByText('Leave training setup?', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('unsaved continuous form');
  await expect(page.getByText('Leave training setup?', { exact: true })).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await noPageOverflow(page);
  await expect(page.getByRole('heading', { name: 'New Training Job', exact: true })).toBeVisible();
  await expect(page.locator('.training-topbar').getByRole('button', { name: 'Create Job', exact: true })).toBeVisible();
  await capture(page, 'new-training-mobile');
  await page.setViewportSize({ width: 1024, height: 768 });
  await noPageOverflow(page);
  await expect(page.locator('.training-topbar').getByRole('button', { name: 'Create Job', exact: true })).toBeVisible();
});

test('caption gallery keeps dirty edits, reports save failures, and saves once before advancing', async ({ page }) => {
  const dataset = `visual-captions-${randomUUID().slice(0, 8)}`;
  const photo = await readFile('public/images/graphite-studio-hero.webp');
  for (let i = 1; i <= 8; i++) {
    const group = randomUUID();
    for (const [filename, data] of [
      [`asset-${i}.webp`, photo],
      [`asset-${i}.txt`, Buffer.from(`Studio image ${i}`)],
    ] as const) {
      const response = await page.request.post('/api/datasets/upload', {
        headers: {
          ...headers,
          'Content-Type': 'application/octet-stream',
          'X-AITK-Dataset-Name': encodeURIComponent(dataset),
          'X-AITK-File-Name': filename,
          'X-AITK-Upload-Group': group,
        },
        data,
      });
      expect(response.status()).toBe(200);
    }
  }
  await page.goto(`/datasets/${dataset}`);
  const caption = page.getByRole('textbox', { name: 'Image caption', exact: true });
  await expect(caption).toBeEnabled();
  await expect(page.getByRole('navigation', { name: 'Main navigation' })).toBeVisible();
  await capture(page, 'caption-studio');
  await caption.fill('Unsaved caption to preserve');
  page.once('dialog', dialog => dialog.dismiss());
  await page.getByRole('link', { name: 'Queue', exact: true }).click();
  await expect(caption).toHaveValue('Unsaved caption to preserve');
  await page.route('**/api/img/caption', route =>
    route.fulfill({ status: 500, json: { error: 'Fixture save failure' } }),
  );
  const failure = page.waitForEvent('dialog').then(async dialog => {
    expect(dialog.message()).toContain('Failed to save caption');
    await dialog.accept();
  });
  await page.getByRole('button', { name: 'Save & Next', exact: true }).click();
  await failure;
  await expect(caption).toHaveValue('Unsaved caption to preserve');
  await expect(page.getByRole('button', { name: 'Save & Next', exact: true })).toBeEnabled();
  await capture(page, 'caption-save-failure');
  await page.unroute('**/api/img/caption');
  let saves = 0;
  await page.route('**/api/img/caption', async route => {
    saves++;
    await route.fulfill({ response: await route.fetch() });
  });
  await page.getByRole('button', { name: 'Save & Next', exact: true }).evaluate(button => {
    (button as HTMLButtonElement).click();
    (button as HTMLButtonElement).click();
  });
  await expect(caption).not.toHaveValue('Unsaved caption to preserve');
  expect(saves).toBe(1);
  await page.getByRole('button', { name: 'Generate captions', exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'Caption Dataset' });
  await expect(modal.getByRole('heading', { name: 'Caption Dataset' })).toBeVisible();
  await expect(modal.getByRole('button', { name: 'Save & start captioning' })).toBeVisible();
  await capture(page, 'caption-configuration');
  await modal.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(modal).toHaveCount(0);
  await page.getByRole('textbox', { name: 'Search photos and captions' }).fill('asset-8');
  await expect(page.getByRole('button', { name: /asset-8.webp/ }).first()).toBeVisible();
  await page.getByRole('textbox', { name: 'Search photos and captions' }).fill('');
  await page.setViewportSize({ width: 390, height: 844 });
  await noPageOverflow(page);
  await caption.scrollIntoViewIfNeeded();
  await expect(caption).toBeVisible();
  await capture(page, 'caption-studio-mobile');
  const saveNext = page.getByRole('button', { name: 'Save & Next', exact: true });
  await saveNext.scrollIntoViewIfNeeded();
  await expect(saveNext).toBeVisible();
  await saveNext.focus();
  await expect(saveNext).toBeFocused();
  await capture(page, 'caption-actions-mobile');
  await page.setViewportSize({ width: 1024, height: 768 });
  await noPageOverflow(page);
});

test('Ideogram regions retain selection, descriptions, zoom, tabs and disconnected readiness', async ({ page }) => {
  await page.goto('/workflows/ideogram');
  await expect(page.getByRole('heading', { name: 'Ideogram Workflow Builder' })).toBeVisible();
  await page.getByLabel('Global Prompt').fill('A studio composition with a headline and product.');
  const canvas = page.locator('[data-composition-canvas]');
  await expect(canvas).toBeVisible();
  const elements = page.getByRole('heading', { name: 'Elements', exact: true }).locator('..');
  const elementButtons = elements.getByRole('button').filter({ hasText: /^(obj|text):/ });
  const originalCount = await elementButtons.count();
  await elements.getByRole('button', { name: 'Object', exact: true }).click();
  await expect(elementButtons).toHaveCount(originalCount + 1);
  await expect(page.getByRole('heading', { name: 'Selected Element' })).toBeVisible();
  await page.getByLabel('Description', { exact: false }).last().fill('Product region');
  await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
  await page.getByRole('button', { name: 'Zoom out', exact: true }).click();
  await capture(page, 'ideogram-regions');
  await page.getByRole('button', { name: 'json', exact: true }).click();
  await expect(page.getByText('Workflow JSON (ComfyUI API)', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'comfy', exact: true }).click();
  await expect(page.getByText('External ComfyUI', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'preview', exact: true }).click();
  await page.getByRole('button', { name: 'Delete selected', exact: true }).click();
  await expect(elementButtons).toHaveCount(originalCount);
  await page.setViewportSize({ width: 390, height: 844 });
  await noPageOverflow(page);
  await capture(page, 'ideogram-regions-mobile');
  await page.setViewportSize({ width: 1024, height: 768 });
  await noPageOverflow(page);
  await page.getByText('Run details & actions', { exact: true }).click();
  await expect(page.getByRole('button', { name: 'Download JSON', exact: true })).toBeVisible();
});

test('Ideogram exposes a queue failure and allows cancellation with a controlled Comfy connection', async ({
  page,
}) => {
  await page.routeWebSocket('ws://127.0.0.1:8188/**', () => {});
  await page.route('**/api/comfy/external/preflight', route =>
    route.fulfill({
      json: {
        ok: true,
        connected: true,
        serverUrl: 'http://127.0.0.1:8188',
        nodes: [],
        models: [],
        error: null,
      },
    }),
  );
  await page.route('**/api/comfy/external/loras?*', route => route.fulfill({ json: { loras: [] } }));
  await page.route('**/api/comfy/external/prompt', route =>
    route.fulfill({
      status: 502,
      json: { error: 'Controlled queue failure' },
    }),
  );
  await page.goto('/workflows/ideogram');
  await expect(page.getByRole('button', { name: 'Comfy ready', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'comfy', exact: true }).click();
  await page.getByRole('button', { name: 'Generate', exact: true }).click();
  await expect(page.getByText('Controlled queue failure', { exact: true }).first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate', exact: true })).toBeEnabled();
  await page.unroute('**/api/comfy/external/prompt');
  await page.route('**/api/comfy/external/prompt', route =>
    route.fulfill({ json: { promptId: 'controlled-prompt', queueNumber: 1 } }),
  );
  await page.route('**/api/comfy/external/import', route => route.fulfill({ json: { status: 'running', images: [] } }));
  let interrupted = 0;
  await page.route('**/api/comfy/external/interrupt', route => {
    interrupted++;
    return route.fulfill({ json: { ok: true } });
  });
  await page.getByRole('button', { name: 'Generate', exact: true }).click();
  await expect(page.getByText('Queued in ComfyUI', { exact: true })).toBeVisible();
  await page.locator('.ideogram-topbar').getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Generate', exact: true })).toBeEnabled();
  expect(interrupted).toBe(1);
});
