import { expect, type BrowserContext, type Page } from '@playwright/test';
import { defaultCaptionJobConfig } from '../src/helpers/captionJobConfig';
import { installLocalMonaco } from './localMonaco';

let cookies: Awaited<ReturnType<BrowserContext['cookies']>> = [];

export async function installCaptionFixtures(page: Page): Promise<void> {
  await installLocalMonaco(page);
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/gpu?*', route => route.fulfill({ json: { hasNvidiaSmi: true, isMac: false, gpus: [] } }));
  await page.route('**/api/workers', route => route.fulfill({ json: { workers: [] } }));
  await page.route('**/api/ollama-workers', route => route.fulfill({ json: { workers: [] } }));
  if (cookies.length) await page.context().addCookies(cookies);
  else {
    await page.goto('/');
    await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
    await page.getByRole('button', { name: 'Check Password', exact: true }).click();
    await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
    cookies = (await page.context().cookies()).filter(cookie => cookie.name === 'aitk_session');
  }
}

export async function captionFixture(page: Page) {
  const state = {
    fail: false,
    malformed: false,
    saveSucceeds: false,
    saves: [] as unknown[],
    starts: 0,
    queueStarts: 0,
  };
  const config = structuredClone(defaultCaptionJobConfig);
  config.config.name = 'Stored caption fixture';
  config.config.process[0].caption.path_to_caption = 'fixture/caption-data';
  config.config.process[0].caption.caption_prompt = 'Preserve this stored caption prompt';
  const job = {
    id: 'caption-recovery-fixture',
    name: 'caption-recovery-fixture',
    worker_id: 'local',
    gpu_ids: '0',
    status: 'stopped',
    job_type: 'caption',
    job_config: JSON.stringify(config),
    job_ref: 'fixture/caption-data',
    created_at: '2026-10-23T00:00:00Z',
    updated_at: '2026-10-23T00:00:00Z',
    step: 0,
    stop: false,
    return_to_queue: false,
    pid: null,
    info: '',
    speed_string: '',
    queue_position: 0,
  };
  await page.route('**/api/jobs?*', route =>
    state.fail
      ? route.fulfill({ status: 503, json: { error: 'Fixture caption source unavailable' } })
      : route.fulfill({ json: state.malformed ? { ...job, job_config: '{"config":{"process":[]}}' } : job }),
  );
  await page.route('**/api/jobs', route => {
    if (route.request().method() !== 'POST') return route.continue();
    state.saves.push(route.request().postDataJSON());
    return state.saveSucceeds
      ? route.fulfill({ json: job })
      : route.fulfill({ status: 503, json: { error: 'Fixture caption save unavailable' } });
  });
  await page.route('**/api/jobs/caption-recovery-fixture/**', route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname.endsWith('/log')) return route.fulfill({ json: { log: '', reset: true, offset: null } });
    if (pathname.endsWith('/metrics')) return route.fulfill({ json: { keys: [], series: {} } });
    if (pathname.endsWith('/start')) state.starts += 1;
    return route.fulfill({ json: {} });
  });
  await page.route('**/api/queue/*/start?*', route => {
    state.queueStarts += 1;
    return route.fulfill({ json: {} });
  });
  await page.goto('/jobs/caption-recovery-fixture');
  await expect(page.getByLabel('Edit captions', { exact: true })).toBeVisible();
  return { state, config, job };
}
