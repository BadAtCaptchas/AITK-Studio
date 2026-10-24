import { expect, test, type BrowserContext, type Page } from '@playwright/test';
import { getSampleThumbnailUrl } from '../src/utils/media';

let sessionCookies: Awaited<ReturnType<BrowserContext['cookies']>> = [];
async function login(page: Page) {
  if (sessionCookies.length) {
    await page.context().addCookies(sessionCookies);
    return;
  }
  await page.goto('/');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password' }).click();
  await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
  sessionCookies = (await page.context().cookies()).filter(cookie => cookie.name === 'aitk_session');
}

for (const storageFailure of ['write', 'read-and-write'] as const) {
  test(`theme remains usable when browser storage blocks ${storageFailure}`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/monitor', route => route.abort());
    await login(page);
    await page.goto('/models');
    await page.evaluate(failure => {
      const originalGet = Storage.prototype.getItem;
      const originalSet = Storage.prototype.setItem;
      Storage.prototype.setItem = function (key, value) {
        if (key === 'theme') throw new DOMException('Fixture storage denied', 'QuotaExceededError');
        return originalSet.call(this, key, value);
      };
      if (failure === 'read-and-write') Storage.prototype.getItem = function (key) {
        if (key === 'theme') throw new DOMException('Fixture storage denied', 'SecurityError');
        return originalGet.call(this, key);
      };
    }, storageFailure);
    await page.getByRole('button', { name: 'More', exact: true }).click();
    await page.getByRole('menuitem', { name: 'Switch to light mode', exact: true }).click();
    await expect(page.locator('html')).not.toHaveClass(/dark/);
    await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Datasets', exact: true }).click();
    await expect(page).toHaveURL(/\/datasets$/);
    await page.getByRole('button', { name: 'More', exact: true }).click();
    await page.getByRole('menuitem', { name: 'Switch to dark mode', exact: true }).click();
    await expect(page.locator('html')).toHaveClass(/dark/);
    expect(errors).toEqual([]);
  });
}

test('header utility menu exposes theme, updates and logout to keyboard users', async ({ page }) => {
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/updater', route =>
    route.fulfill({ json: { state: 'disabled', message: 'Fixture updater disabled' } }),
  );
  await login(page);
  await page.goto('/models');
  const trigger = page.getByRole('button', { name: 'More', exact: true });
  await trigger.focus();
  await trigger.press('Enter');
  const menu = page.getByRole('menu');
  await expect(menu).toBeVisible();
  await menu.press('End');
  const logout = menu.getByRole('menuitem', { name: 'Log out', exact: true });
  await expect(logout).toBeVisible();
  await expect(menu).toHaveAttribute('aria-activedescendant', (await logout.getAttribute('id')) ?? '');
  await menu.press('ArrowUp');
  const updates = menu.getByRole('menuitem', { name: 'Updates off', exact: true });
  await expect(menu).toHaveAttribute('aria-activedescendant', (await updates.getAttribute('id')) ?? '');
  await menu.press('ArrowUp');
  const theme = menu.getByRole('menuitem', { name: 'Switch to light mode', exact: true });
  await expect(menu).toHaveAttribute('aria-activedescendant', (await theme.getAttribute('id')) ?? '');
  await menu.press('Enter');
  await expect(page.locator('html')).not.toHaveClass(/dark/);
  await expect(menu).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await trigger.press('Enter');
  await menu.press('End');
  await menu.press('ArrowUp');
  await menu.press('Enter');
  await expect(page).toHaveURL(/\/updates$/);
  await page.getByRole('button', { name: 'Search commands', exact: true }).click();
  const search = page
    .getByRole('dialog', { name: 'Search commands' })
    .getByRole('textbox', { name: 'Search commands' });
  await search.fill('Models');
  await search.dispatchEvent('keydown', { key: 'Enter', code: 'Enter', isComposing: true });
  await expect(search).toBeVisible();
  await expect(page).toHaveURL(/\/updates$/);
  await search.press('Enter');
  await expect(page).toHaveURL(/\/models$/);
});

test('Overview output resets for a new attempt and stops polling completed runs', async ({ page }) => {
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/gpu?*', route => route.fulfill({ json: { gpus: [], hasNvidiaSmi: false, isMac: false } }));
  await page.route('**/api/jobs/*/metrics*', route => route.fulfill({ json: { keys: [], series: {} } }));
  const job = {
    id: 'output-attempt-fixture', name: 'Output attempt fixture', status: 'running', attempt_id: 'attempt-old',
    job_type: 'train', worker_id: 'local', gpu_ids: '0', step: 100, stop: false, queue_position: 0,
    info: '', speed_string: '', created_at: '2026-10-23T10:00:00Z', updated_at: '2026-10-23T10:00:00Z',
    job_config: JSON.stringify({ config: { process: [{ train: { steps: 1000 }, model: { arch: 'flux' } }] } }),
  };
  const logRequests: Array<{ attempt: string; offset: string | null }> = [];
  await page.route('**/api/jobs?*', route => {
    const params = new URL(route.request().url()).searchParams;
    return route.fulfill({ json: params.has('id') ? job : {
      jobs: params.get('view') === 'active' ? [job] : [], nextCursor: null, statusCounts: { [job.status]: 1 },
    } });
  });
  await page.route('**/api/jobs/output-attempt-fixture/log*', route => {
    const offset = new URL(route.request().url()).searchParams.get('offset');
    logRequests.push({ attempt: job.attempt_id, offset });
    return route.fulfill({ json: {
      log: offset === null ? `${job.attempt_id} beginning\n` : '', offset: 100, reset: offset === null,
    } });
  });
  await login(page);
  await page.clock.install();
  await page.goto('/dashboard');
  await expect(page.getByLabel('Training output', { exact: true })).toContainText('attempt-old beginning');
  job.attempt_id = 'attempt-new';
  job.step = 0;
  await page.clock.fastForward(5500);
  await expect(page.getByLabel('Training output', { exact: true })).toContainText('attempt-new beginning');
  await expect(page.getByLabel('Training output', { exact: true })).not.toContainText('attempt-old');
  expect(logRequests.some(request => request.attempt === 'attempt-new' && request.offset === null)).toBe(true);
  job.status = 'completed';
  await page.clock.fastForward(5500);
  await expect(page.getByRole('complementary', { name: 'Run inspector' })).toContainText('completed');
  // Let the final one-shot request settle before measuring inactive polling.
  await expect.poll(() => page.getByText('Training log', { exact: true }).count()).toBe(1);
  const settled = logRequests.length;
  await page.clock.fastForward(12000);
  await expect(page.getByLabel('Training output', { exact: true })).toContainText('attempt-new beginning');
  expect(logRequests).toHaveLength(settled);
});

test('Overview keeps starting jobs active and exposes independent history freshness failures', async ({ page }) => {
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/gpu?*', route => route.fulfill({ json: { gpus: [], hasNvidiaSmi: false, isMac: false } }));
  await page.route('**/api/jobs/*/metrics*', route => route.fulfill({ json: { keys: [], series: {} } }));
  await page.route('**/api/jobs/*/log*', route => route.fulfill({ json: { log: '', offset: null, reset: true } }));
  const jobs = ['starting', 'remote-starting', 'completed'].map((status, index) => ({
    id: `shell-status-${index}`,
    name: `Fixture ${status}`,
    status,
    job_type: 'train',
    worker_id: 'local',
    gpu_ids: '0',
    step: 0,
    stop: false,
    queue_position: 0,
    info: '',
    speed_string: '',
    created_at: '2026-10-22T10:00:00Z',
    updated_at: '2026-10-22T10:00:00Z',
    job_config: JSON.stringify({
      job: 'extension',
      config: {
        name: `Fixture ${status}`,
        process: [{ type: 'diffusion_trainer', train: { steps: 100 }, model: { arch: 'flux' }, datasets: [] }],
      },
    }),
  }));
  await page.route('**/api/jobs?*', route => {
    const query = new URL(route.request().url()).searchParams;
    const id = query.get('id');
    const history = query.get('view') === 'history';
    return route.fulfill({
      json: id
        ? jobs.find(job => job.id === id)
        : {
            jobs: jobs.filter(job => (history ? job.status === 'completed' : job.status !== 'completed')),
            nextCursor: null,
            statusCounts: { starting: 1, 'remote-starting': 1, completed: 1 },
            freshness: { updatedAt: '2026-10-22T10:00:00Z', failed: history ? 1 : 0 },
          },
    });
  });
  await login(page);
  await page.goto('/dashboard');
  await expect(page.getByText(/Some worker updates failed/)).toBeVisible();
  await page.getByRole('button', { name: /^Active/ }).click();
  const rows = page.locator('tbody');
  await expect(rows.getByRole('button', { name: 'Fixture starting', exact: true })).toBeVisible();
  await expect(rows.getByRole('button', { name: 'Fixture remote-starting', exact: true })).toBeVisible();
  await expect(rows.getByRole('button', { name: 'Fixture completed', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: /^Recent/ }).click();
  await expect(rows.getByRole('button', { name: 'Fixture completed', exact: true })).toBeVisible();
  await expect(rows.getByRole('button', { name: /^Fixture .*starting$/ })).toHaveCount(0);
});

test('Overview Samples uses local and remote thumbnail routes and exposes empty and failed loads', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1024 });
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/gpu?*', route => route.fulfill({ json: { gpus: [], hasNvidiaSmi: false, isMac: false } }));
  await page.route('**/api/jobs/*/metrics*', route => route.fulfill({ json: { keys: [], series: {} } }));
  await page.route('**/api/jobs/*/log*', route => route.fulfill({ json: { log: '', offset: null, reset: true } }));
  let remote = false;
  let sampleState: 'ready' | 'empty' | 'error' = 'ready';
  const localSample = '/api/jobs/sample-fixture/samples/100__100_0.png';
  const remoteSample = `remote://sample-fixture/img/${encodeURIComponent('/api/jobs/worker-job/samples/100__100_0.png')}/100__100_0.png`;
  const thumbnailRequests = new Set<string>();
  const png = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZtsAAAAASUVORK5CYII=',
    'base64',
  );
  for (const pattern of ['**/api/jobs/*/samples/*.png*', '**/api/remote-assets?*']) {
    await page.route(pattern, route => {
      const url = new URL(route.request().url());
      thumbnailRequests.add(`${url.pathname}${url.search}`);
      return route.fulfill({ body: png, contentType: 'image/png' });
    });
  }
  await page.route('**/api/jobs/sample-fixture/samples?*', route =>
    route.fulfill({
      status: sampleState === 'error' ? 503 : 200,
      json:
        sampleState === 'error'
          ? { error: 'Fixture samples unavailable' }
          : { samples: sampleState === 'empty' ? [] : [remote ? remoteSample : localSample] },
    }),
  );
  await page.route('**/api/jobs?*', route => {
    const job = {
      id: 'sample-fixture',
      name: 'Sample fixture',
      status: 'stopped',
      job_type: 'train',
      worker_id: remote ? 'remote-worker' : 'local',
      remote_job_id: remote ? 'worker-job' : null,
      gpu_ids: '0',
      step: 100,
      stop: false,
      queue_position: 0,
      info: '',
      speed_string: '',
      created_at: '2026-10-22T10:00:00Z',
      updated_at: '2026-10-22T10:00:00Z',
      job_config: JSON.stringify({
        job: 'extension',
        config: {
          name: 'Sample fixture',
          process: [
            {
              type: 'diffusion_trainer',
              model: { arch: 'flux' },
              train: { steps: 100 },
              sample: { samples: [{ prompt: 'fixture' }] },
              datasets: [],
            },
          ],
        },
      }),
    };
    const query = new URL(route.request().url()).searchParams;
    return route.fulfill({
      json: query.has('id')
        ? job
        : { jobs: query.get('view') === 'active' ? [] : [job], nextCursor: null, statusCounts: { stopped: 1 } },
    });
  });
  await login(page);
  for (const useRemote of [false, true]) {
    remote = useRemote;
    await page.goto('/dashboard');
    await page.getByRole('tab', { name: 'Samples', exact: true }).click();
    await expect(page.getByRole('toolbar', { name: 'Sample actions' })).toBeVisible();
    await expect
      .poll(() => thumbnailRequests.has(getSampleThumbnailUrl(remote ? remoteSample : localSample)))
      .toBe(true);
    await expect(page.getByAltText('Sample Image', { exact: true })).toBeVisible();
    await page.goto('/models');
  }
  for (const state of ['empty', 'error'] as const) {
    sampleState = state;
    await page.goto('/dashboard');
    await page.getByRole('tab', { name: 'Samples', exact: true }).click();
    await expect(
      page.getByText(state === 'empty' ? 'No Samples Found' : 'Error Loading Samples', { exact: true }),
    ).toBeVisible();
    await page.goto('/models');
  }
});
