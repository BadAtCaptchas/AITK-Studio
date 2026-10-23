import { test, expect } from '@playwright/test';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';
import { installCaptionFixtures } from './captionFixture';

test.beforeEach(async ({ page }) => {
  await installCaptionFixtures(page);
});

for (const [action, label, diagnostic] of [
  ['save_now', 'Save Next Step', 'Error requesting checkpoint save:'],
  ['sample_now', 'Sample Next Step', 'Error requesting sample:'],
] as const) {
  test(`${label} exposes actionable failure without logging its transport or job configuration`, async ({ page }) => {
    const logs: { text: string; arguments: number }[] = [];
    const errors: string[] = [];
    page.on('console', message => logs.push({ text: message.text(), arguments: message.args().length }));
    page.on('pageerror', error => errors.push(error.message));
    const job = {
      id: 'action-log-fixture',
      name: 'action-log-fixture',
      worker_id: 'local',
      gpu_ids: '0',
      status: 'running',
      job_type: 'train',
      job_config: JSON.stringify(defaultJobConfig),
      created_at: '2026-10-23T00:00:00Z',
      updated_at: '2026-10-23T00:00:00Z',
      step: 1,
      stop: false,
      return_to_queue: false,
      pid: null,
      info: '',
      speed_string: '',
      queue_position: 0,
    };
    await page.route('**/api/jobs?*', route => route.fulfill({ json: job }));
    await page.route('**/api/jobs/action-log-fixture/**', route => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname.endsWith(`/${action}`))
        return route.fulfill({
          status: 503,
          json: {
            error: 'Fixture action temporarily unavailable',
            job_config: 'synthetic-private-job-config',
          },
        });
      if (pathname.endsWith('/log')) return route.fulfill({ json: { log: '', reset: true, offset: null } });
      if (pathname.endsWith('/metrics')) return route.fulfill({ json: { keys: [], keyInfo: [], series: {} } });
      if (pathname.endsWith('/samples')) return route.fulfill({ json: { samples: [] } });
      return route.fulfill({ json: {} });
    });
    await page.goto('/jobs/action-log-fixture');
    await page.getByRole('button', { name: 'More job actions', exact: true }).click();
    await page.getByRole('menuitem', { name: label, exact: true }).click();
    await expect(page.getByRole('alert').filter({ hasText: 'Fixture action temporarily unavailable' })).toBeVisible();
    expect(logs.find(log => log.text.startsWith(diagnostic))).toEqual({ text: diagnostic, arguments: 1 });
    expect(logs.filter(log => log.text.startsWith('Error setting job to save')).every(log => log.arguments === 1)).toBe(
      true,
    );
    expect(logs.map(log => log.text).join('\n')).not.toContain('synthetic-private-job-config');
    expect(errors).toEqual([]);
  });
}
