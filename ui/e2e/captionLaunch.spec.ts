import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { captionFixture, installCaptionFixtures } from './captionFixture';

test.beforeEach(async ({ page }) => {
  await installCaptionFixtures(page);
});

async function launchFixture(page: Page, create = false) {
  const fixture = await captionFixture(page);
  await page.route('**/api/gpu?*', route =>
    route.fulfill({
      json: {
        hasNvidiaSmi: true,
        isMac: false,
        gpus: [
          {
            index: 0,
            name: 'Fixture GPU',
            driverVersion: 'fixture',
            temperature: 40,
            utilization: { gpu: 0, memory: 0 },
            memory: { total: 24576, free: 24576, used: 0 },
            power: { draw: 0, limit: 300 },
            clocks: { graphics: 0, memory: 0 },
            fan: { speed: 0 },
          },
        ],
      },
    }),
  );
  const state = {
    ...fixture.state,
    saveSucceeds: true,
    failStart: true,
    failQueue: false,
    failSave: false,
    copies: 0,
    startKeys: [] as string[],
  };
  await page.route('**/api/jobs', route => {
    if (route.request().method() !== 'POST') return route.continue();
    const data: unknown = route.request().postDataJSON();
    state.saves.push(data);
    if (state.failSave) return route.fulfill({ status: 503, json: { error: 'Fixture save failure' } });
    return route.fulfill({
      json: { ...fixture.job, name: data && typeof data === 'object' && 'name' in data ? data.name : fixture.job.name },
    });
  });
  await page.route('**/api/jobs/caption-recovery-fixture/start', route => {
    state.starts++;
    const body: unknown = route.request().postDataJSON();
    if (body && typeof body === 'object' && 'idempotencyKey' in body && typeof body.idempotencyKey === 'string')
      state.startKeys.push(body.idempotencyKey);
    return state.failStart
      ? route.fulfill({ status: 422, json: { error: 'Fixture start preflight rejected' } })
      : route.fulfill({ json: { ...fixture.job, status: 'queued' } });
  });
  await page.route('**/api/queue/*/start?*', route => {
    state.queueStarts++;
    return state.failQueue
      ? route.fulfill({ status: 503, json: { error: 'Fixture queue failure' } })
      : route.fulfill({ json: {} });
  });
  await page.route('**/api/datasets/copy', route => {
    state.copies++;
    return route.fulfill({ json: { path: 'fixture/copied-caption-data' } });
  });
  if (create) {
    const dataset = `launch-caption-${randomUUID().slice(0, 8)}`;
    const upload = await page.request.post('/api/datasets/upload', {
      headers: {
        Origin: 'http://127.0.0.1:15875',
        'Content-Type': 'application/octet-stream',
        'X-AITK-Dataset-Name': encodeURIComponent(dataset),
        'X-AITK-File-Name': 'fixture.webp',
        'X-AITK-Upload-Group': randomUUID(),
      },
      data: await readFile('public/images/graphite-studio-hero.webp'),
    });
    expect(upload.status()).toBe(200);
    await page.goto(`/datasets/${dataset}`);
    await page.getByRole('button', { name: 'Generate captions', exact: true }).click();
  } else await page.getByLabel('Edit captions', { exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'Caption Dataset', exact: true });
  const submit = modal.locator('button[type="submit"]');
  await expect(submit).toBeEnabled();
  return { ...fixture, state, modal, submit };
}

test('caption start retry reuses the saved new job without saving a duplicate', async ({ page }) => {
  const { state, modal, submit } = await launchFixture(page, true);
  await submit.click();
  await expect.poll(() => state.starts).toBe(1);
  await expect(submit).toBeEnabled();
  state.failStart = false;
  await submit.click();
  await expect(modal).toHaveCount(0);
  await expect(page.getByRole('alert').filter({ hasText: 'captioning could not start' })).toHaveCount(0);
  expect(state.saves).toHaveLength(1);
  expect(state.saves[0]).toMatchObject({ gpu_ids: '0' });
  expect(state.starts).toBe(2);
  expect(state.startKeys).toHaveLength(2);
  expect(state.startKeys[1]).not.toBe(state.startKeys[0]);
  expect(state.queueStarts).toBe(1);
});

test('caption queue retry does not resave or restart a job that already started', async ({ page }) => {
  const { state, modal, submit } = await launchFixture(page);
  state.failStart = false;
  state.failQueue = true;
  await submit.click();
  await expect.poll(() => state.queueStarts).toBe(1);
  await expect(submit).toBeEnabled();
  await expect(modal.getByLabel('Caption Prompt', { exact: true })).toBeDisabled();
  await expect(modal.getByRole('alert').filter({ hasText: 'queue could not start' })).toBeVisible();
  state.failQueue = false;
  await submit.click();
  await expect(modal).toHaveCount(0);
  expect(state.saves).toHaveLength(1);
  expect(state.starts).toBe(1);
  expect(state.queueStarts).toBe(2);
  await expect(page.getByRole('alert').filter({ hasText: 'queue could not start' })).toHaveCount(0);
});

test('caption save conflicts keep actionable server guidance and do not start a job', async ({ page }) => {
  const { state, modal, submit } = await launchFixture(page);
  await page.route('**/api/jobs', route =>
    route.fulfill({
      status: 409,
      json: {
        error: 'Stop the active job before changing its configuration',
        code: 'JOB_ACTIVE',
      },
    }),
  );
  await submit.click();
  await expect(
    modal.getByRole('alert').filter({ hasText: 'Stop the active job before changing its configuration' }),
  ).toBeVisible();
  await expect(submit).toBeEnabled();
  expect(state.starts).toBe(0);
  expect(state.queueStarts).toBe(0);
});

test('caption retry saves newer edits to the same already-created job', async ({ page }) => {
  const { state, modal, submit, job } = await launchFixture(page, true);
  await submit.click();
  await expect.poll(() => state.starts).toBe(1);
  await expect(submit).toBeEnabled();
  await modal.getByLabel('Caption Prompt', { exact: true }).fill('Newer retry prompt');
  state.failStart = false;
  await submit.click();
  await expect(modal).toHaveCount(0);
  expect(state.saves).toHaveLength(2);
  expect(state.saves[1]).toMatchObject({
    id: job.id,
    job_config: { config: { process: [{ caption: { caption_prompt: 'Newer retry prompt' } }] } },
  });
  expect(state.startKeys).toHaveLength(2);
  expect(state.startKeys[1]).not.toBe(state.startKeys[0]);
});

test('caption save retry reuses the completed dataset copy', async ({ page }) => {
  const { state, modal, submit } = await launchFixture(page, true);
  // The existing JSON preset creates a dataset copy before saving a new job.
  await modal.getByRole('combobox', { name: 'Caption Output', exact: true }).click();
  await page.getByRole('option', { name: 'Convert to JSON', exact: true }).click();
  await modal.getByRole('combobox', { name: 'Destination', exact: true }).click();
  await page.getByRole('option', { name: 'Make dataset copy', exact: true }).click();
  state.failSave = true;
  await submit.click();
  await expect.poll(() => state.saves.length).toBe(1);
  await expect(submit).toBeEnabled();
  state.failSave = false;
  state.failStart = false;
  await submit.click();
  await expect(modal).toHaveCount(0);
  expect(state.copies).toBe(1);
  expect(state.saves[1]).toMatchObject({
    job_ref: 'fixture/copied-caption-data',
    job_config: { config: { process: [{ caption: { path_to_caption: 'fixture/copied-caption-data' } }] } },
  });
});

for (const acceptedStatus of ['queued', 'completed'] as const) {
  test(`caption retries recover a lost ${acceptedStatus} start acknowledgement without starting twice`, async ({
    page,
  }) => {
    const { state, modal, submit, job } = await launchFixture(page, true);
    const keys: string[] = [];
    const accepted = new Set<string>();
    let enqueues = 0;
    await page.route('**/api/jobs/caption-recovery-fixture/start', async route => {
      const body: unknown = route.request().postDataJSON();
      const key = body && typeof body === 'object' && 'idempotencyKey' in body ? body.idempotencyKey : null;
      expect(typeof key).toBe('string');
      if (typeof key !== 'string') return route.abort();
      keys.push(key);
      if (!accepted.has(key)) {
        accepted.add(key);
        enqueues++;
      }
      if (keys.length === 1) return route.abort('failed');
      return route.fulfill({ json: { ...job, status: acceptedStatus } });
    });
    await submit.click();
    await expect(submit).toBeEnabled();
    await expect(modal.getByRole('alert').filter({ hasText: 'Captioning may already have started' })).toBeVisible();
    await expect(modal.getByLabel('Caption Prompt', { exact: true })).toBeDisabled();
    await submit.click();
    await expect(modal).toHaveCount(0);
    expect(keys).toHaveLength(2);
    expect(keys[1]).toBe(keys[0]);
    expect(enqueues).toBe(1);
    expect(state.saves).toHaveLength(1);
    expect(state.queueStarts).toBe(1);
    await expect(page.getByRole('alert').filter({ hasText: 'Captioning may already have started' })).toHaveCount(0);
  });
}

test('a definitively failed start operation can retry unchanged configuration as a new command', async ({ page }) => {
  const { state, modal, submit, job } = await launchFixture(page);
  const keys: unknown[] = [];
  await page.route('**/api/jobs/caption-recovery-fixture/start', route => {
    const body: unknown = route.request().postDataJSON();
    keys.push(body && typeof body === 'object' && 'idempotencyKey' in body ? body.idempotencyKey : null);
    if (keys.length > 1) return route.fulfill({ json: { ...job, status: 'queued' } });
    return route.fulfill({
      json: {
        startID: 'failed-start-fixture',
        progress: {
          startID: 'failed-start-fixture',
          jobID: job.id,
          status: 'failed',
          error: 'Fixture worker unavailable',
        },
      },
    });
  });
  await submit.click();
  await expect(submit).toBeEnabled();
  await expect(modal.getByRole('alert').filter({ hasText: 'Fixture worker unavailable' })).toBeVisible();
  await expect(modal.getByLabel('Caption Prompt', { exact: true })).toBeEnabled();
  await submit.click();
  await expect(modal).toHaveCount(0);
  expect(keys).toHaveLength(2);
  expect(typeof keys[0]).toBe('string');
  expect(keys[1]).not.toBe(keys[0]);
  expect(state.saves).toHaveLength(1);
  expect(state.queueStarts).toBe(1);
});

for (const failure of ['lost', 'busy', 'invalid'] as const) {
  test(`caption creation recovers an ${failure} acknowledgement with the same saved input`, async ({ page }) => {
    const { state, modal, submit, job } = await launchFixture(page, true);
    const bodies: Record<string, unknown>[] = [];
    const createdKeys = new Set<unknown>();
    await page.route('**/api/jobs', route => {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      bodies.push(body);
      createdKeys.add(body.idempotencyKey);
      if (bodies.length === 1) {
        if (failure === 'lost') return route.abort('failed');
        if (failure === 'busy') return route.fulfill({ status: 409, json: { code: 'CREATE_BUSY', error: 'Creation still in progress' } });
        return route.fulfill({ json: { accepted: true } });
      }
      if (failure === 'lost' && bodies.length === 2) return route.fulfill({ status: 401, json: { error: 'Fixture session expired' } });
      return route.fulfill({ json: { ...job, name: body.name } });
    });
    state.failStart = false;
    await submit.click();
    await expect(submit).toBeEnabled();
    await expect(modal.getByRole('alert').filter({ hasText: 'Job may already have been saved' })).toBeVisible();
    await expect(modal.getByLabel('Caption Prompt', { exact: true })).toBeDisabled();
    expect(state.starts).toBe(0);
    await submit.click();
    if (failure === 'lost') {
      await expect(submit).toBeEnabled();
      await expect(modal.getByRole('alert').filter({ hasText: 'Fixture session expired' })).toBeVisible();
      await expect(modal.getByLabel('Caption Prompt', { exact: true })).toBeDisabled();
      await submit.click();
    }
    await expect(modal).toHaveCount(0);
    expect(bodies).toHaveLength(failure === 'lost' ? 3 : 2);
    expect(bodies[0].idempotencyKey).toEqual(expect.any(String));
    expect(bodies[1]).toEqual(bodies[0]);
    if (failure === 'lost') expect(bodies[2]).toEqual(bodies[0]);
    expect(createdKeys.size).toBe(1);
    expect(state.starts).toBe(1);
    expect(state.queueStarts).toBe(1);
  });
}

test('a rejected caption creation permits edits and uses a fresh command for the retry', async ({ page }) => {
  const { state, modal, submit, job } = await launchFixture(page, true);
  const bodies: Record<string, unknown>[] = [];
  await page.route('**/api/jobs', route => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    bodies.push(body);
    if (bodies.length === 1) return route.fulfill({ status: 400, json: { code: 'INVALID_COMMAND', error: 'Fixture input rejected' } });
    return route.fulfill({ json: { ...job, name: body.name } });
  });
  state.failStart = false;
  await submit.click();
  await expect(submit).toBeEnabled();
  await expect(modal.getByLabel('Caption Prompt', { exact: true })).toBeEnabled();
  await modal.getByLabel('Caption Prompt', { exact: true }).fill('Corrected caption prompt');
  await submit.click();
  await expect(modal).toHaveCount(0);
  expect(bodies).toHaveLength(2);
  expect(bodies[0].idempotencyKey).toEqual(expect.any(String));
  expect(bodies[1].idempotencyKey).not.toBe(bodies[0].idempotencyKey);
  expect(bodies[1]).toMatchObject({ job_config: { config: { process: [{ caption: { caption_prompt: 'Corrected caption prompt' } }] } } });
});

test('a recovered caption job changed elsewhere is never started or replaced automatically', async ({ page }) => {
  const { state, modal, submit } = await launchFixture(page, true);
  const bodies: unknown[] = [];
  await page.route('**/api/jobs', route => {
    bodies.push(route.request().postDataJSON());
    if (bodies.length === 1) return route.abort('failed');
    return route.fulfill({ status: 409, json: { code: 'CREATE_RESULT_CHANGED', error: 'The saved job changed after this creation command. Review it before starting.' } });
  });
  await submit.click();
  await expect(submit).toBeEnabled();
  await submit.click();
  await expect(submit).toBeEnabled();
  await expect(modal.getByRole('alert').filter({ hasText: 'Review it before starting' })).toBeVisible();
  await expect(modal.getByLabel('Caption Prompt', { exact: true })).toBeDisabled();
  await submit.click();
  await expect.poll(() => bodies.length).toBe(3);
  expect(bodies[1]).toEqual(bodies[0]);
  expect(bodies[2]).toEqual(bodies[0]);
  expect(state.starts).toBe(0);
  expect(state.queueStarts).toBe(0);
});

test('caption copy retry reuses its command after a lost publication acknowledgement', async ({ page }) => {
  const { state, modal, submit } = await launchFixture(page, true);
  const bodies: Record<string, unknown>[] = [];
  await page.route('**/api/datasets/copy', route => {
    bodies.push(route.request().postDataJSON() as Record<string, unknown>);
    if (bodies.length === 1) return route.abort('failed');
    return route.fulfill({ json: { path: 'fixture/recovered-copy' } });
  });
  await modal.getByRole('combobox', { name: 'Caption Output', exact: true }).click();
  await page.getByRole('option', { name: 'Convert to JSON', exact: true }).click();
  await modal.getByRole('combobox', { name: 'Destination', exact: true }).click();
  await page.getByRole('option', { name: 'Make dataset copy', exact: true }).click();
  await submit.click();
  await expect(submit).toBeEnabled();
  await expect(modal.getByRole('alert').filter({ hasText: 'Failed to prepare caption dataset' })).toBeVisible();
  state.failStart = false;
  await submit.click();
  await expect(modal).toHaveCount(0);
  expect(bodies).toHaveLength(2);
  expect(bodies[0].idempotencyKey).toEqual(expect.any(String));
  expect(bodies[1]).toEqual(bodies[0]);
  expect(state.saves).toHaveLength(1);
  expect(state.saves[0]).toMatchObject({ job_ref: 'fixture/recovered-copy' });
  expect(state.starts).toBe(1);
});
