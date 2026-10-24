import { test, expect, type BrowserContext, type Page } from '@playwright/test';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';

let cookies: Awaited<ReturnType<BrowserContext['cookies']>> = [];
const errors = new WeakMap<Page, string[]>();

test.beforeEach(async ({ page }) => {
  const pageErrors: string[] = [];
  errors.set(page, pageErrors);
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/gpu?*', route => route.fulfill({ json: { hasNvidiaSmi: true, isMac: false, gpus: [] } }));
  if (cookies.length) await page.context().addCookies(cookies);
  else {
    await page.goto('/');
    await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
    await page.getByRole('button', { name: 'Check Password', exact: true }).click();
    await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
    cookies = (await page.context().cookies()).filter(cookie => cookie.name === 'aitk_session');
  }
});

test.afterEach(async ({ page }) => { expect(errors.get(page)).toEqual([]); });

async function notesFixture(page: Page) {
  const state = { notes: 'Saved fixture notes', saveFails: false, submissions: [] as unknown[] };
  const job = {
    id: 'notes-fixture', name: 'notes-fixture', worker_id: 'local', gpu_ids: '0',
    status: 'queued', job_type: 'train', job_config: JSON.stringify(defaultJobConfig),
    created_at: '2026-10-23T00:00:00Z', updated_at: '2026-10-23T00:00:00Z',
    step: 0, stop: false, return_to_queue: false, pid: null, info: '', speed_string: '', queue_position: 1000,
  };
  await page.route('**/api/jobs?*', route => route.fulfill({ json:
    new URL(route.request().url()).searchParams.has('id') ? job : {
      jobs: [job], nextCursor: null, statusCounts: { queued: 1 }, freshness: { updatedAt: new Date().toISOString(), failed: 0 },
    },
  }));
  await page.route('**/api/queue', route => route.fulfill({ json: { queues: [], freshness: [] } }));
  await page.route('**/api/workers', route => route.fulfill({ json: { workers: [] } }));
  await page.route('**/api/jobs/notes-fixture/**', async route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname.endsWith('/notes')) {
      if (route.request().method() === 'POST') {
        const body: unknown = route.request().postDataJSON();
        state.submissions.push(body);
        if (state.saveFails) return route.fulfill({ status: 503, json: { error: 'Fixture save failed' } });
        if (body && typeof body === 'object' && 'notes' in body && typeof body.notes === 'string') state.notes = body.notes;
      }
      return route.fulfill({ json: { notes: state.notes } });
    }
    if (pathname.endsWith('/metrics')) return route.fulfill({ json: { keys: [], keyInfo: [], series: {} } });
    if (pathname.endsWith('/log')) return route.fulfill({ json: { log: '', reset: true, offset: null } });
    return route.fulfill({ json: {} });
  });
  await page.goto('/jobs');
  await page.getByRole('link', { name: 'Open notes-fixture', exact: true }).click();
  await page.getByRole('tab', { name: 'Notes', exact: true }).click();
  await expect(page.getByLabel('Job notes', { exact: true })).toHaveValue(state.notes);
  return state;
}

async function modelsLink(page: Page) {
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Models', exact: true }).click();
}

test('unsaved Notes remain protected while their retained monitor tab is hidden', async ({ page }) => {
  const state = await notesFixture(page);
  await page.getByLabel('Job notes', { exact: true }).fill('Unsaved retained notes');
  await page.getByRole('tab', { name: 'Metrics', exact: true }).click();
  await expect(page).toHaveURL(/tab=loss_log/);
  await modelsLink(page);
  await expect(page.getByRole('heading', { name: 'Leave notes?', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs\/notes-fixture/);
  await page.getByRole('tab', { name: 'Notes', exact: true }).click();
  await expect(page.getByLabel('Job notes', { exact: true })).toHaveValue('Unsaved retained notes');
  await modelsLink(page);
  await page.getByRole('button', { name: 'Discard changes', exact: true }).click();
  await expect(page).toHaveURL(/\/models$/);
  expect(state.submissions).toEqual([]);
});

test('Notes guard covers browser Back and failed saves, and clears after revert or successful save', async ({ page }) => {
  const state = await notesFixture(page);
  const notes = page.getByLabel('Job notes', { exact: true });
  await notes.fill('Back-protected notes');
  await page.evaluate(() => window.history.back());
  await expect(page.getByRole('heading', { name: 'Leave notes?', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Leave notes?', exact: true })).not.toBeVisible();
  await expect(notes).toHaveValue('Back-protected notes');
  await notes.fill(state.notes);
  await modelsLink(page);
  await expect(page).toHaveURL(/\/models$/);
  await page.goto('/jobs/notes-fixture?tab=notes');
  await expect(notes).toHaveValue(state.notes);
  await notes.fill('New saved notes');
  state.saveFails = true;
  await page.getByRole('button', { name: 'Save notes', exact: true }).click();
  await expect(page.getByText('Could not save notes.', { exact: true })).toBeVisible();
  await modelsLink(page);
  await expect(page.getByRole('heading', { name: 'Leave notes?', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  state.saveFails = false;
  await page.getByRole('button', { name: 'Save notes', exact: true }).click();
  await expect(page.getByText('Saved', { exact: true })).toBeVisible();
  await modelsLink(page);
  await expect(page).toHaveURL(/\/models$/);
  expect(state.notes).toBe('New saved notes');
  expect(state.submissions).toEqual([{ notes: 'New saved notes' }, { notes: 'New saved notes' }]);
});

test('fallback Back cancellation restores the current retained monitor tab', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(window, 'navigation', { value: undefined, configurable: true }));
  await notesFixture(page);
  await page.getByLabel('Job notes', { exact: true }).fill('Fallback-protected notes');
  await page.getByRole('tab', { name: 'Metrics', exact: true }).click();
  await expect(page).toHaveURL(/tab=loss_log/);
  await page.evaluate(() => window.history.back());
  await expect(page.getByRole('heading', { name: 'Leave notes?', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page).toHaveURL(/tab=loss_log/);
  await expect(page.getByRole('tab', { name: 'Metrics', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.getByRole('tab', { name: 'Notes', exact: true }).click();
  await expect(page.getByLabel('Job notes', { exact: true })).toHaveValue('Fallback-protected notes');
  await page.evaluate(() => window.history.back());
  await expect(page.getByRole('heading', { name: 'Leave notes?', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Discard changes', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs$/);
  await page.evaluate(() => window.history.forward());
  await expect(page).toHaveURL(/\/jobs\/notes-fixture\?tab=notes$/);
  await expect(page.getByLabel('Job notes', { exact: true })).toHaveValue('Saved fixture notes');
});
