import { test, expect } from '@playwright/test';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';
import { installCaptionFixtures } from './captionFixture';

test('hidden job metrics stop requesting and resume immediately', async ({ page }) => {
  await installCaptionFixtures(page);
  await page.clock.install();
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  let jobs = 0, metrics = 0;
  const job = {
    id: 'poll-performance', name: 'poll-performance', worker_id: 'local', gpu_ids: '0',
    status: 'running', job_type: 'train', job_config: JSON.stringify(defaultJobConfig),
    created_at: '2026-10-31T00:00:00Z', updated_at: '2026-10-31T00:00:00Z',
    step: 100, stop: false, return_to_queue: false, pid: null, info: '', speed_string: '', queue_position: 0,
  };
  await page.route('**/api/jobs?*', route => { jobs++; return route.fulfill({ json: job }); });
  await page.route('**/api/jobs/poll-performance/**', route => {
    if (new URL(route.request().url()).pathname.endsWith('/metrics')) {
      metrics++;
      const point = { step: 100, value: 0.25 };
      return route.fulfill({ json: { keys: ['loss/loss'], series: { 'loss/loss': { points: [point], latest: point } } } });
    }
    return route.fulfill({ json: { log: '', reset: true, offset: null, progress: null } });
  });
  await page.goto('/jobs/poll-performance?tab=loss_log');
  await expect(page.getByRole('tab', { name: 'Metrics', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect.poll(() => metrics).toBeGreaterThan(0);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  const hiddenCounts = { jobs, metrics };
  await page.clock.fastForward(30_000);
  expect({ jobs, metrics }).toEqual(hiddenCounts);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect.poll(() => metrics).toBeGreaterThan(hiddenCounts.metrics);
  await expect.poll(() => jobs).toBeGreaterThan(hiddenCounts.jobs);
  expect(errors).toEqual([]);
});
