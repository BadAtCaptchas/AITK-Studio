import { test, expect, type BrowserContext, type Page } from '@playwright/test';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

let sessionCookies: Awaited<ReturnType<BrowserContext['cookies']>> = [];
const stamp = '2026-10-22T10:00:00.000Z';
function job(id: string, status = 'queued', worker = 'local', position = 1000) {
  return {
    id,
    name: id,
    worker_id: worker,
    gpu_ids: '0',
    job_config: JSON.stringify(defaultJobConfig),
    created_at: stamp,
    updated_at: stamp,
    status,
    stop: false,
    return_to_queue: false,
    step: status === 'running' ? 1560 : 0,
    info: status === 'running' ? 'Training' : '',
    speed_string: status === 'running' ? '2.4s/step' : '',
    queue_position: position,
    pid: null,
    job_type: 'train',
    job_ref: null,
    save_now: false,
    remote_job_id: worker === 'local' ? null : `remote-${id}`,
    remote_error: null,
    remote_sync_at: stamp,
  };
}

async function mockQueue(page: Page) {
  const jobs = [
    job('portrait_flux_v3', 'running', 'local', 0),
    job('style_flux_v2'),
    job('material_study', 'queued', 'local', 2000),
    job('remote_run', 'queued', 'lab-01'),
  ];
  jobs.push(...Array.from({ length: 70 }, (_, index) => job(`history_${String(index).padStart(2, '0')}`, 'stopped')));
  const reorders: unknown[] = [];
  const failedListCursors = new Set<string>();
  const failedJobDetails = new Set<string>();
  const listState: { queuedCount?: number } = {};
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/gpu?*', route => route.fulfill({ json: { hasNvidiaSmi: true, isMac: false, gpus: [] } }));
  await page.route('**/api/workers', route => route.fulfill({ json: { workers: [] } }));
  await page.route('**/api/queue', route =>
    route.fulfill({
      json: {
        queues: [
          { id: 1, worker_id: 'local', gpu_ids: '0', is_running: true },
          { id: 2, worker_id: 'lab-01', gpu_ids: '0', is_running: false },
        ],
        freshness: [],
      },
    }),
  );
  await page.route('**/api/queue/*/reorder', async route => {
    const body: unknown = route.request().postDataJSON();
    reorders.push(body);
    if (body && typeof body === 'object' && 'job_ids' in body && Array.isArray(body.job_ids)) {
      body.job_ids.forEach((id: unknown, index: number) => {
        const item = jobs.find(job => job.id === id);
        if (item) item.queue_position = (index + 1) * 1000;
      });
    }
    await route.fulfill({ json: { ok: true } });
  });
  await page.route('**/api/jobs?*', route => {
    const params = new URL(route.request().url()).searchParams;
    if (params.has('id')) {
      if (failedJobDetails.has(params.get('id') || ''))
        return route.fulfill({ status: 503, json: { error: 'Fixture detail unavailable' } });
      return route.fulfill({ json: jobs.find(job => job.id === params.get('id')) ?? null });
    }
    const view = params.get('view');
    const list = jobs.filter(
      job =>
        view === 'all' ||
        (view === 'history'
          ? job.status === 'stopped'
          : view === 'failed'
            ? false
            : ['running', 'queued'].includes(job.status)),
    );
    const cursor = Number(params.get('cursor') || 0);
    if (failedListCursors.has(String(cursor)))
      return route.fulfill({ status: 503, json: { error: 'Fixture page unavailable' } });
    const statusCounts = jobs.reduce<Record<string, number>>((counts, item) => {
      counts[item.status] = (counts[item.status] || 0) + 1;
      return counts;
    }, {});
    if (listState.queuedCount !== undefined) statusCounts.queued = listState.queuedCount;
    return route.fulfill({
      json: {
        jobs: list.slice(cursor, cursor + 50),
        nextCursor: cursor + 50 < list.length ? String(cursor + 50) : null,
        statusCounts,
        freshness: { updatedAt: new Date().toISOString(), failed: 0 },
      },
    });
  });
  await page.route('**/api/jobs/*/**', route => {
    const pathname = new URL(route.request().url()).pathname;
    const selectedJob = jobs.find(job => job.id === pathname.split('/')[3]);
    if (pathname.endsWith('/notes')) return route.fulfill({ json: { notes: '' } });
    if (pathname.endsWith('/log'))
      return route.fulfill({
        json: { log: selectedJob?.step ? `step ${selectedJob.step} · loss 0.081\n` : '', reset: true, offset: null },
      });
    if (pathname.endsWith('/metrics')) {
      if (!selectedJob?.step) return route.fulfill({ json: { keys: [], keyInfo: [], series: {} } });
      const points = Array.from({ length: 40 }, (_, index) => ({
        step: index * 40,
        value: 0.08 + 0.4 * Math.exp(-index / 7),
      }));
      return route.fulfill({
        json: {
          keys: ['loss/loss'],
          keyInfo: [{ key: 'loss/loss', first_seen_step: 0, last_seen_step: 1560 }],
          series: {
            'loss/loss': {
              key: 'loss/loss',
              totalCount: 40,
              firstStep: 0,
              lastStep: 1560,
              latest: points.at(-1),
              downsampled: false,
              points,
            },
          },
        },
      });
    }
    return route.fulfill({ json: {} });
  });
  return { jobs, reorders, failedListCursors, failedJobDetails, listState };
}

test.beforeEach(async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1024 });
  await page.route('https://cdn.jsdelivr.net/npm/monaco-editor@*/min/vs/**', async route => {
    const relative = new URL(route.request().url()).pathname.split('/min/vs/')[1];
    if (!relative || relative.split('/').includes('..')) return route.abort();
    const filename = path.join(process.cwd(), 'node_modules/monaco-editor/min/vs', relative);
    await route.fulfill({
      body: await readFile(filename),
      contentType: relative.endsWith('.css') ? 'text/css' : 'application/javascript',
    });
  });
  if (sessionCookies.length) await page.context().addCookies(sessionCookies);
  await page.goto('/');
  if (!sessionCookies.length) {
    await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
    await page.getByRole('button', { name: 'Check Password' }).click();
    await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
    sessionCookies = (await page.context().cookies()).filter(cookie => cookie.name === 'aitk_session');
  }
});

test('queue rows support keyboard inspection, bounded resizing and worker-safe Apply order', async ({ page }) => {
  const { reorders } = await mockQueue(page);
  await page.goto('/jobs');
  const style = page.getByRole('button', { name: 'Inspect style_flux_v2', exact: true });
  await style.focus();
  await page.keyboard.press('ArrowDown');
  const material = page.getByRole('button', { name: 'Inspect material_study', exact: true });
  await expect(material).toBeFocused();
  await expect(material).toHaveAttribute('aria-pressed', 'true');
  const separator = page.getByRole('separator', { name: 'Resize job inspector' });
  await separator.focus();
  await separator.press('ArrowLeft');
  await expect(separator).toHaveAttribute('aria-valuenow', '380');
  await page.getByRole('button', { name: 'Move up', exact: true }).click();
  await expect(page.getByText('Unsaved order', { exact: true })).toBeVisible();
  expect(reorders).toHaveLength(0);
  await page.getByRole('button', { name: 'Apply order', exact: true }).click();
  await expect.poll(() => reorders.length).toBe(1);
  expect(reorders[0]).toEqual({
    worker_id: 'local',
    job_ids: ['material_study', 'style_flux_v2'],
    expected_job_ids: ['style_flux_v2', 'material_study'],
  });
  await page.screenshot({ path: '.review-build/premium-queue-desktop.png' });
  await page.getByLabel('Search jobs', { exact: true }).fill('style');
  await expect(page.locator('[data-queue-row="style_flux_v2"]')).toHaveAttribute('draggable', 'false');
});

test('server reorder conflicts refresh the lane and discard the obsolete draft', async ({ page }) => {
  const { jobs } = await mockQueue(page);
  jobs.push(job('lighting_study', 'queued', 'local', 3000));
  let payload: unknown;
  await page.route('**/api/queue/*/reorder', async route => {
    payload = route.request().postDataJSON();
    jobs.find(job => job.id === 'lighting_study')!.queue_position = 2000;
    jobs.find(job => job.id === 'material_study')!.queue_position = 3000;
    await route.fulfill({
      status: 409,
      json: {
        error: 'Queue changed while you were arranging jobs. Review the current order and try again.',
        code: 'QUEUE_CHANGED',
      },
    });
  });
  await page.goto('/jobs');
  await page.getByRole('button', { name: 'Inspect material_study', exact: true }).click();
  await page.getByRole('button', { name: 'Move up', exact: true }).click();
  await page.getByRole('button', { name: 'Apply order', exact: true }).click();
  await expect(page.getByText('Queue order was not saved', { exact: true })).toBeVisible();
  expect(payload).toEqual({
    worker_id: 'local',
    job_ids: ['material_study', 'style_flux_v2', 'lighting_study'],
    expected_job_ids: ['style_flux_v2', 'material_study', 'lighting_study'],
  });
  await expect(page.getByText('Unsaved order', { exact: true })).toHaveCount(0);
  await expect
    .poll(() =>
      page
        .locator('[data-queue-row]')
        .evaluateAll(rows =>
          rows
            .map(row => row.getAttribute('data-queue-row'))
            .filter(id => ['style_flux_v2', 'material_study', 'lighting_study'].includes(id || '')),
        ),
    )
    .toEqual(['style_flux_v2', 'lighting_study', 'material_study']);
  await expect(page.getByRole('button', { name: 'Move up', exact: true })).toBeEnabled();
});

test('queue return restores filter, sort, selected row, loaded pages and scroll without browser storage', async ({
  page,
}) => {
  await mockQueue(page);
  await page.goto('/jobs');
  await page.getByRole('button', { name: /^History/ }).click();
  await page.getByLabel('Search jobs', { exact: true }).fill('history_');
  await page.getByLabel('Sort history').selectOption('name');
  await page.getByRole('button', { name: 'Load more jobs', exact: true }).click();
  const row = page.getByRole('button', { name: 'Inspect history_59', exact: true });
  await row.click();
  const before = await page.locator('.studio-queue-page').evaluate(element => element.scrollTop);
  expect(before).toBeGreaterThan(500);
  await page.locator('#queue-job-inspector').getByRole('link', { name: 'Open job', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs\/history_59$/);
  await page.getByRole('link', { name: 'Back to queue', exact: true }).click();
  await expect(page.getByLabel('Search jobs', { exact: true })).toHaveValue('history_');
  await expect(page.getByLabel('Sort history')).toHaveValue('name');
  await expect(row).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('[data-queue-row]')).toHaveCount(70);
  await expect
    .poll(() => page.locator('.studio-queue-page').evaluate(element => element.scrollTop))
    .toBeGreaterThan(before - 120);
  const stored = await page.evaluate(() =>
    JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } }),
  );
  expect(stored).not.toContain('history_59');
  expect(stored).not.toContain('job_config');
});

test('queue restoration retries a failed later page before restoring selection and scroll', async ({ page }) => {
  const { failedListCursors } = await mockQueue(page);
  await page.goto('/jobs');
  await page.getByRole('button', { name: /^History/ }).click();
  await page.getByLabel('Sort history').selectOption('name');
  await page.getByRole('button', { name: 'Load more jobs', exact: true }).click();
  const selected = page.getByRole('button', { name: 'Inspect history_59', exact: true });
  await selected.click();
  const before = await page.locator('.studio-queue-page').evaluate(element => element.scrollTop);
  await page.locator('#queue-job-inspector').getByRole('link', { name: 'Open job', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs\/history_59$/);
  failedListCursors.add('50');
  await page.getByRole('link', { name: 'Back to queue', exact: true }).click();
  await expect(page.getByText('Queue data is incomplete', { exact: true })).toBeVisible();
  await expect(page.locator('[data-queue-row]')).toHaveCount(50);
  failedListCursors.clear();
  await page.getByRole('button', { name: 'Retry', exact: true }).click();
  await expect(page.locator('[data-queue-row]')).toHaveCount(70);
  await expect(selected).toHaveAttribute('aria-pressed', 'true');
  await expect
    .poll(() => page.locator('.studio-queue-page').evaluate(element => element.scrollTop))
    .toBeGreaterThan(before - 120);
});

test('staged queue order is invalidated when another client changes the same lane order', async ({ page }) => {
  const { jobs, reorders } = await mockQueue(page);
  jobs.push(job('contrast_fixture', 'queued', 'local', 3000));
  await page.goto('/jobs');
  await page.getByRole('button', { name: 'Inspect style_flux_v2', exact: true }).click();
  await page.getByRole('button', { name: 'Move down', exact: true }).click();
  await expect(page.getByText('Unsaved order', { exact: true })).toBeVisible();
  // A normal refresh preserves the draft; a changed baseline must not be overwritten by Apply.
  const refresh = async () => {
    await Promise.all([
      page.waitForResponse(response => {
        const url = new URL(response.url());
        return url.pathname === '/api/jobs' && !url.searchParams.has('id');
      }),
      page.getByRole('button', { name: 'Refresh queue', exact: true }).click(),
    ]);
    await expect(page.getByRole('button', { name: 'Refresh queue', exact: true }).locator('svg')).not.toHaveClass(
      /animate-spin/,
    );
  };
  await refresh();
  await expect(page.getByText('Unsaved order', { exact: true })).toBeVisible();
  jobs.find(item => item.id === 'contrast_fixture')!.queue_position = 1000;
  jobs.find(item => item.id === 'style_flux_v2')!.queue_position = 3000;
  await refresh();
  await expect(
    page.getByText('Queue changed while you were arranging jobs. Review the current order and try again.', {
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.getByRole('button', { name: 'Apply order', exact: true })).toHaveCount(0);
  expect(reorders).toHaveLength(0);
});

test('monitor reports a failed refresh while retaining the last job details and supports retry', async ({ page }) => {
  const { failedJobDetails } = await mockQueue(page);
  await page.goto('/jobs/portrait_flux_v3');
  await expect(page.getByRole('heading', { name: 'Job: portrait_flux_v3', exact: true })).toBeVisible();
  failedJobDetails.add('portrait_flux_v3');
  await expect(page.getByText('Job details could not be refreshed', { exact: true })).toBeVisible({ timeout: 12000 });
  await expect(page.getByRole('heading', { name: 'Job: portrait_flux_v3', exact: true })).toBeVisible();
  failedJobDetails.clear();
  await page.getByRole('button', { name: 'Retry job refresh', exact: true }).click();
  await expect(page.getByText('Job details could not be refreshed', { exact: true })).toHaveCount(0);
});

test('notes load can retry without leaving the retained monitor panel', async ({ page }) => {
  await mockQueue(page);
  let failLoad = true;
  let savedNotes = '';
  await page.route('**/api/jobs/portrait_flux_v3/notes', route => {
    if (route.request().method() === 'POST') {
      const body: unknown = route.request().postDataJSON();
      if (body && typeof body === 'object' && 'notes' in body && typeof body.notes === 'string')
        savedNotes = body.notes;
      return route.fulfill({ json: { ok: true } });
    }
    return failLoad
      ? route.fulfill({ status: 503, json: { error: 'Fixture notes unavailable' } })
      : route.fulfill({ json: { notes: savedNotes } });
  });
  await page.goto('/jobs/portrait_flux_v3?tab=notes');
  await expect(page.getByText('Could not load notes.', { exact: true })).toBeVisible();
  failLoad = false;
  await page.getByRole('button', { name: 'Retry notes load', exact: true }).click();
  const notes = page.getByLabel('Job notes', { exact: true });
  await expect(notes).toBeEnabled();
  await notes.fill('Recovered note fixture');
  await page.getByRole('button', { name: 'Save notes', exact: true }).click();
  await expect.poll(() => savedNotes).toBe('Recovered note fixture');
  await expect(page.getByText('Saved', { exact: true })).toBeVisible();
  await page.getByRole('tab', { name: 'Metrics', exact: true }).click();
  await page.getByRole('tab', { name: 'Notes', exact: true }).click();
  await expect(notes).toHaveValue('Recovered note fixture');
});

for (const surface of ['compact inspector', 'full monitor']) {
  test(`metrics failure stays visible with cached points in ${surface}`, async ({ page }) => {
    await mockQueue(page);
    if (surface === 'compact inspector') {
      await page.goto('/jobs');
      await page.getByRole('button', { name: 'Inspect portrait_flux_v3', exact: true }).click();
    } else {
      await page.goto('/jobs/portrait_flux_v3?tab=metrics');
    }
    const scope =
      surface === 'compact inspector'
        ? page.locator('#queue-job-inspector')
        : page.getByRole('tabpanel', { name: 'Metrics', exact: true });
    await expect(scope.locator('.uplot')).toBeVisible();
    const pattern = '**/api/jobs/portrait_flux_v3/metrics?*';
    await page.route(pattern, route => route.fulfill({ status: 503, json: { error: 'Fixture metrics unavailable' } }));
    await scope.getByRole('button', { name: 'Refresh', exact: true }).click();
    await expect(
      scope.getByText('Metrics could not be refreshed. Showing the last loaded data.', { exact: true }),
    ).toBeVisible();
    await expect(scope.locator('.uplot')).toBeVisible();
    await page.unroute(pattern);
    await scope.getByRole('button', { name: 'Refresh', exact: true }).click();
    await expect(
      scope.getByText('Metrics could not be refreshed. Showing the last loaded data.', { exact: true }),
    ).toHaveCount(0);
  });
}

test('queue commands cannot reorder an incomplete snapshot with no remaining cursor', async ({ page }) => {
  const { listState, reorders } = await mockQueue(page);
  listState.queuedCount = 4;
  await page.goto('/jobs');
  await page.getByRole('button', { name: 'Inspect style_flux_v2', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Load more jobs', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Move down', exact: true })).toBeDisabled();
  await expect(page.locator('[data-queue-row="style_flux_v2"]')).toHaveAttribute('draggable', 'false');
  await expect(
    page.getByText('Queue data is still synchronizing. Refresh before reordering.', { exact: true }),
  ).toBeVisible();
  listState.queuedCount = 3;
  await page.getByRole('button', { name: 'Refresh queue', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Move down', exact: true })).toBeEnabled();
  expect(reorders).toHaveLength(0);
});

test('polling an applied order before its request resolves does not report a conflict', async ({ page }) => {
  const { jobs } = await mockQueue(page);
  let releaseReply: () => void = () => {};
  const reply = new Promise<void>(resolve => {
    releaseReply = resolve;
  });
  let applied = false;
  await page.route('**/api/queue/*/reorder', async route => {
    const body: unknown = route.request().postDataJSON();
    if (body && typeof body === 'object' && 'job_ids' in body && Array.isArray(body.job_ids)) {
      body.job_ids.forEach((id: unknown, index: number) => {
        const item = jobs.find(job => job.id === id);
        if (item) item.queue_position = (index + 1) * 1000;
      });
    }
    applied = true;
    await reply;
    await route.fulfill({ json: { ok: true } });
  });
  try {
    await page.goto('/jobs');
    await page.getByRole('button', { name: 'Inspect material_study', exact: true }).click();
    await page.getByRole('button', { name: 'Move up', exact: true }).click();
    await page.getByRole('button', { name: 'Apply order', exact: true }).click();
    await expect.poll(() => applied).toBe(true);
    await Promise.all([
      page.waitForResponse(response => {
        const url = new URL(response.url());
        return url.pathname === '/api/jobs' && !url.searchParams.has('id');
      }),
      page.getByRole('button', { name: 'Refresh queue', exact: true }).click(),
    ]);
    await expect(page.getByRole('button', { name: 'Refresh queue', exact: true }).locator('svg')).not.toHaveClass(
      /animate-spin/,
    );
  } finally {
    releaseReply();
  }
  await expect(page.getByRole('button', { name: 'Saving…', exact: true })).toHaveCount(0);
  await expect(page.getByText('Queue order was not saved', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Unsaved order', { exact: true })).toHaveCount(0);
});

for (const status of ['starting', 'remote-starting']) {
  test(`monitor continues log polling while a job is ${status}`, async ({ page }) => {
    const { jobs } = await mockQueue(page);
    jobs[0].status = status;
    let logRequests = 0;
    await page.route('**/api/jobs/portrait_flux_v3/log', route => {
      logRequests++;
      return route.fulfill({ json: { log: 'Preparing training fixture\n', offset: null, reset: true } });
    });
    await page.clock.install();
    await page.goto('/jobs/portrait_flux_v3');
    await expect.poll(() => logRequests).toBe(1);
    await page.clock.fastForward(6000);
    await expect.poll(() => logRequests).toBeGreaterThan(1);
  });
}

test('narrow inspector dismisses to its row, and monitor deep links retain unfinished notes', async ({ page }) => {
  await mockQueue(page);
  await page.setViewportSize({ width: 1024, height: 900 });
  await page.goto('/jobs');
  const row = page.getByRole('button', { name: 'Inspect style_flux_v2', exact: true });
  await row.click();
  const drawer = page.getByRole('dialog', { name: 'Job inspector', exact: true });
  await expect(drawer).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(drawer).toBeHidden();
  await expect(row).toBeFocused();
  await page.setViewportSize({ width: 390, height: 844 });
  await row.click();
  await expect(drawer).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: '.review-build/premium-queue-mobile-inspector.png' });
  await page.keyboard.press('Escape');
  await page.goto('/jobs/portrait_flux_v3?tab=metrics');
  await expect(page.getByRole('tab', { name: 'Metrics', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.getByRole('tab', { name: 'Notes', exact: true }).click();
  await expect(page).toHaveURL(/\?tab=notes$/);
  await expect(page.getByRole('tabpanel', { name: 'Notes', exact: true })).toBeVisible();
  const notes = page.getByLabel('Job notes', { exact: true });
  await notes.fill('unfinished note fixture');
  await page.getByRole('tab', { name: 'Metrics', exact: true }).click();
  await page.getByRole('tab', { name: 'Notes', exact: true }).click();
  await expect(notes).toHaveValue('unfinished note fixture');
  await page.getByRole('tab', { name: 'Notes', exact: true }).press('Home');
  await expect(page.getByRole('tab', { name: 'Overview', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('tab', { name: 'Overview', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.getByRole('tab', { name: 'Configuration', exact: true }).click();
  const editor = page.locator('#job-panel-config .monaco-editor');
  await expect(editor).toBeVisible();
  expect((await editor.boundingBox())?.height).toBeGreaterThan(300);
});

test('partial queue failure keeps jobs readable and disables queue commands', async ({ page }) => {
  await mockQueue(page);
  await page.route('**/api/queue', route =>
    route.fulfill({ status: 503, json: { error: 'Fixture worker unavailable' } }),
  );
  await page.goto('/jobs');
  await expect(page.getByRole('button', { name: 'Inspect portrait_flux_v3', exact: true })).toBeVisible();
  await expect(page.getByText('Queue data is incomplete', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start queue', exact: true }).first()).toBeDisabled();
});

async function expectInspectorChartFits(page: Page) {
  await expect(page.locator('#queue-job-inspector .uplot')).toBeVisible();
  const geometry = () =>
    page.locator('#queue-job-inspector .uplot').evaluate(plot => {
      const host = plot.parentElement?.parentElement;
      const wrap = plot.querySelector('.u-wrap');
      const canvas = plot.querySelector('canvas');
      if (!host || !wrap || !canvas) return { fits: false, reason: 'Missing chart elements' };
      const bounds = (element: Element) => {
        const rect = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return {
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height,
          cssWidth: style.width,
          cssHeight: style.height,
          position: style.position,
          overflow: style.overflow,
        };
      };
      const hostBounds = bounds(host);
      const plotBounds = bounds(plot);
      const wrapBounds = bounds(wrap);
      const canvasBounds = bounds(canvas);
      const fitsHost = (element: ReturnType<typeof bounds>) =>
        element.x >= hostBounds.x - 3 &&
        element.y >= hostBounds.y - 3 &&
        element.x + element.width <= hostBounds.x + hostBounds.width + 3 &&
        element.y + element.height <= hostBounds.y + hostBounds.height + 3;
      return {
        fits: fitsHost(plotBounds) && fitsHost(wrapBounds) && fitsHost(canvasBounds),
        host: hostBounds,
        plot: plotBounds,
        wrap: wrapBounds,
        canvas: { ...canvasBounds, pixelWidth: canvas.width, pixelHeight: canvas.height },
        devicePixelRatio: window.devicePixelRatio,
      };
    });
  await expect.poll(geometry).toMatchObject({ fits: true });
  // uPlot applies the measured size in an animation frame. Wait for its canvas paint as well.
  await page.evaluate(
    () => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))),
  );
}

test('queue remains usable in both themes across desktop, tablet, mobile and reflow widths', async ({
  page,
  browser,
  baseURL,
}) => {
  await mockQueue(page);
  await page.goto('/jobs');
  for (const theme of ['dark', 'light'] as const) {
    if (theme === 'light') {
      await page.getByRole('button', { name: 'More tools', exact: true }).click();
      await page.getByRole('menuitem', { name: 'Switch to light mode', exact: true }).click();
      await page.keyboard.press('Escape');
    }
    for (const width of [1487, 1440, 1280, 1024, 720, 390]) {
      await page.setViewportSize({ width, height: width === 1487 ? 1058 : 900 });
      const row = page.getByRole('button', { name: 'Inspect portrait_flux_v3', exact: true });
      await row.click();
      await expect(page.locator('#queue-job-inspector')).toBeVisible();
      await expectInspectorChartFits(page);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: `.review-build/premium-queue-${theme}-${width}.png` });
      await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    }
  }

  // 720 CSS pixels at 2x scale matches the reflow footprint of a 1440px desktop at 200% zoom.
  for (const theme of ['dark', 'light'] as const) {
    const context = await browser.newContext({
      baseURL,
      viewport: { width: 720, height: 512 },
      deviceScaleFactor: 2,
    });
    try {
      await context.addCookies(await page.context().cookies());
      await context.addInitScript(selectedTheme => localStorage.setItem('theme', selectedTheme), theme);
      const zoomPage = await context.newPage();
      await mockQueue(zoomPage);
      await zoomPage.goto('/jobs');
      await expect(zoomPage.getByLabel('Search jobs', { exact: true })).toBeInViewport();
      await expect(zoomPage.getByRole('button', { name: /^Queue\s/ })).toBeInViewport();
      await expect(zoomPage.getByRole('button', { name: 'Start queue', exact: true }).first()).toBeInViewport();
      const row = zoomPage.getByRole('button', { name: 'Inspect portrait_flux_v3', exact: true });
      await row.click();
      await expect(zoomPage.getByRole('dialog', { name: 'Job inspector', exact: true })).toBeVisible();
      await expect(zoomPage.getByRole('button', { name: 'Close inspector', exact: true })).toBeInViewport();
      await expectInspectorChartFits(zoomPage);
      expect(await zoomPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await zoomPage.screenshot({ path: `.review-build/premium-queue-${theme}-200-percent-reflow.png` });
      await zoomPage.keyboard.press('Escape');
      await expect(row).toBeFocused();
    } finally {
      await context.close();
    }
  }
});
