import { test, expect } from '@playwright/test';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';
import { installCaptionFixtures } from './captionFixture';

test.beforeEach(async ({ page }) => {
  await installCaptionFixtures(page);
});

for (const kind of ['download', 'install'] as const) {
  test(`${kind} progress keeps the fresh job snapshot when an earlier poll finishes`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    const makeProgress = (second: number) => ({
      version: 1,
      status: kind === 'download' ? 'downloading' : 'installing',
      message: `Fixture ${kind} snapshot ${second}`,
      percent: second,
      error: null,
      updatedAt: `2026-10-23T12:00:${String(second).padStart(2, '0')}.000Z`,
      ...(kind === 'download'
        ? { fileName: 'fixture.bin', activeCount: 1, bytesDownloaded: second, bytesTotal: 100, downloads: [] }
        : { step: 'install', root: null, startedAt: '2026-10-23T12:00:00.000Z' }),
    });
    let snapshot = makeProgress(1),
      jobRequests = 0,
      progressRequests = 0;
    let release: () => void = () => {};
    const gate = new Promise<void>(resolve => {
      release = resolve;
    });
    const endpoint = kind === 'download' ? '/hf-download-progress' : '/comfy-install-progress';
    const job = {
      id: 'progress-fixture',
      name: 'progress-fixture',
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
    await page.route('**/api/jobs?*', route => {
      jobRequests++;
      return route.fulfill({
        json: { ...job, [kind === 'download' ? 'hf_download_progress' : 'comfy_install_progress']: snapshot },
      });
    });
    await page.route('**/api/jobs/progress-fixture/**', async route => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname.endsWith(endpoint)) {
        progressRequests++;
        if (progressRequests === 1) {
          await gate;
          await route.fulfill({ json: { progress: makeProgress(2) } }).catch(() => {});
          return;
        }
        return route.fulfill({ json: { progress: snapshot } });
      }
      if (pathname.endsWith('/log')) return route.fulfill({ json: { log: '', reset: true, offset: null } });
      if (pathname.endsWith('/metrics')) return route.fulfill({ json: { keys: [], keyInfo: [], series: {} } });
      return route.fulfill({ json: { progress: null } });
    });
    try {
      await page.goto('/jobs/progress-fixture');
      await expect(page.getByText(`Fixture ${kind} snapshot 1`, { exact: true })).toBeVisible();
      await expect.poll(() => progressRequests).toBe(1);
      snapshot = makeProgress(5);
      await expect.poll(() => jobRequests, { timeout: 10000 }).toBeGreaterThan(1);
      await expect(page.getByText(`Fixture ${kind} snapshot 5`, { exact: true })).toBeVisible();
      release();
      // Observe the retained display throughout the old response's completion.
      await expect
        .poll(
          async () => {
            expect(await page.getByText(`Fixture ${kind} snapshot 2`, { exact: true }).count()).toBe(0);
            return progressRequests;
          },
          { timeout: 5000 },
        )
        .toBeGreaterThan(1);
      await expect(page.getByText(`Fixture ${kind} snapshot 5`, { exact: true })).toBeVisible();
      expect(errors).toEqual([]);
    } finally {
      release();
    }
  });
}
