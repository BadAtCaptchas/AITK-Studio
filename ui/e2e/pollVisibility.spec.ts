import { test, expect } from '@playwright/test';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';
import { installCaptionFixtures } from './captionFixture';

test('dashboard polling pauses while hidden and collapsed output resumes from its offset', async ({ page }) => {
  await installCaptionFixtures(page);
  const counts = { jobs: 0, queues: 0, metrics: 0, logs: 0 };
  const offsets: Array<string | null> = [];
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const job = {
    id: 'visible-poll-fixture', name: 'Visibility fixture', status: 'running', attempt_id: 'one',
    job_type: 'train', worker_id: 'local', gpu_ids: '0', step: 1, stop: false, queue_position: 0,
    info: '', speed_string: '', created_at: '2026-10-31T00:00:00Z', updated_at: '2026-10-31T00:00:00Z',
    job_config: JSON.stringify(defaultJobConfig),
  };
  await page.route('**/api/jobs?*', route => {
    counts.jobs++;
    const params = new URL(route.request().url()).searchParams;
    return route.fulfill({ json: params.has('id') ? job : {
      jobs: params.get('view') === 'active' ? [job] : [], nextCursor: null, statusCounts: { running: 1 },
    } });
  });
  await page.route('**/api/queue*', route => {
    counts.queues++;
    return route.fulfill({ json: { queues: [], freshness: [] } });
  });
  await page.route('**/api/jobs/visible-poll-fixture/metrics*', route => {
    counts.metrics++;
    return route.fulfill({ json: { keys: [], series: {} } });
  });
  await page.route('**/api/jobs/visible-poll-fixture/log*', route => {
    counts.logs++;
    const offset = new URL(route.request().url()).searchParams.get('offset');
    offsets.push(offset);
    return route.fulfill({ json: { log: offset === null ? 'visible output\n' : '', offset: 15, reset: offset === null } });
  });
  await page.clock.install();
  await page.goto('/dashboard');
  await expect(page.getByLabel('Training output', { exact: true })).toContainText('visible output');
  await expect.poll(() => counts.metrics).toBeGreaterThan(0);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'hidden' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  const before = { ...counts };
  await page.clock.fastForward(60_000);
  expect(counts).toEqual(before);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'visible' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  for (const key of ['jobs', 'queues', 'metrics', 'logs'] as const) {
    await expect.poll(() => counts[key]).toBeGreaterThan(before[key]);
  }
  const output = page.locator('details.studio-output');
  await output.locator('summary').click();
  await expect(output).not.toHaveAttribute('open', '');
  const collapsedCount = counts.logs;
  await page.clock.fastForward(15_000);
  expect(counts.logs).toBe(collapsedCount);
  await output.locator('summary').click();
  await expect.poll(() => counts.logs).toBeGreaterThan(collapsedCount);
  expect(offsets.at(-1)).toBe('15');
  await expect(page.getByLabel('Training output', { exact: true })).toContainText('visible output');
  expect(errors).toEqual([]);
});
