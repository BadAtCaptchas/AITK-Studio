import { test, expect } from '@playwright/test';
import archiver from 'archiver';
import { randomUUID } from 'crypto';
const origin = 'http://127.0.0.1:15875';
const png = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZtsAAAAASUVORK5CYII=',
  'base64',
);
test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password' }).click();
  await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
});
test('fresh production stack authenticates, protects commands, and revokes the cookie', async ({ page, context }) => {
  const cookies = await context.cookies();
  const session = cookies.find(cookie => cookie.name === 'aitk_session');
  expect(session?.httpOnly).toBe(true);
  expect((await page.request.get('/api/jobs?limit=20')).ok()).toBe(true);
  expect((await page.request.get('/api/jobs/missing/start')).status()).toBe(405);
  expect(
    (await page.request.post('/api/jobs', { headers: { Origin: 'https://untrusted.invalid' }, data: {} })).status(),
  ).toBe(403);
  expect((await page.request.get('/api/jobs', { headers: { Host: 'rebind.invalid' } })).status()).toBe(403);
  expect(
    (
      await page.request.post('/api/jobs', {
        headers: { Origin: origin, 'Content-Type': 'application/json' },
        data: '{',
      })
    ).status(),
  ).toBe(400);
  expect(
    (
      await page.request.post('/api/jobs', { headers: { Origin: origin, 'Content-Type': 'text/plain' }, data: '{}' })
    ).status(),
  ).toBe(415);
  expect(
    (
      await page.request.post('/api/jobs', {
        headers: { Origin: origin },
        data: { oversized: 'x'.repeat(2 * 1024 * 1024) },
      })
    ).status(),
  ).toBe(413);
  expect((await page.request.delete('/api/auth', { headers: { Origin: origin } })).status()).toBe(200);
  expect((await page.request.get('/api/jobs')).status()).toBe(401);
});

async function datasetArchive(encrypted: boolean): Promise<Buffer> {
  const archive = archiver('zip');
  const chunks: Buffer[] = [];
  const finished = new Promise<Buffer>((resolve, reject) => {
    archive.on('data', (chunk: Buffer) => chunks.push(chunk));
    archive.on('end', () => resolve(Buffer.concat(chunks)));
    archive.on('error', reject);
  });
  archive.append(
    JSON.stringify({
      format: 'ai-toolkit-dataset-export',
      version: 1,
      exportedAt: new Date().toISOString(),
      source: { app: 'ai-toolkit', datasetName: 'browser-archive' },
      dataset: { name: 'browser-archive', archivePath: 'dataset', encrypted },
    }),
    { name: 'manifest.json' },
  );
  archive.append(png, { name: 'dataset/sample.png' });
  archive.append('browser caption fixture', { name: 'dataset/caption.txt' });
  if (encrypted)
    archive.append(JSON.stringify({ format: 'aitk-encrypted-dataset', version: 1 }), {
      name: 'dataset/.aitk_encrypted_dataset.json',
    });
  await archive.finalize();
  return finished;
}

test('archive acceptance survives client detach, replays once, and rejects encrypted imports', async ({ page }) => {
  for (const encrypted of [false, true]) {
    const data = await datasetArchive(encrypted),
      uploadID = randomUUID();
    const endpoint = '/api/datasets/import-archive';
    const query = `uploadID=${uploadID}&chunksTotal=1&fileBytes=${data.length}`;
    const chunk = await page.request.post(`${endpoint}?aitk_upload=chunk&chunkIndex=0&${query}`, {
      headers: { Origin: origin, 'Content-Type': 'application/octet-stream' },
      data,
    });
    expect(chunk.status()).toBe(200);
    const accepted = await page.request.post(`${endpoint}?aitk_upload=complete&${query}`, {
      headers: { Origin: origin },
    });
    expect(accepted.status()).toBe(202);
    const operationID = (await accepted.json()).operationID;
    await page.goto('/settings'); // Operation no longer depends on the accepting page.
    await expect
      .poll(
        async () =>
          (await (await page.request.get(`${endpoint}?aitk_upload=status&uploadID=${uploadID}`)).json()).state,
        { timeout: 45000 },
      )
      .toBe(encrypted ? 'failed' : 'completed');
    const result = await (await page.request.get(`${endpoint}?aitk_upload=status&uploadID=${uploadID}`)).json();
    if (encrypted) {
      expect(result.error).toBe(
        'Encrypted datasets have been retired. Decrypt this dataset to a plain copy before using it.',
      );
      expect(result.result).toBeFalsy();
      continue;
    }
    const replay = await page.request.post(`${endpoint}?aitk_upload=complete&${query}`, {
      headers: { Origin: origin },
    });
    expect(replay.status()).toBe(200);
    expect((await replay.json()).operationID).toBe(operationID);
    const file = '/api/files/' + encodeURIComponent(result.result.path + '/sample.png');
    const download = await page.request.get(file);
    expect(download.status()).toBe(200);
    expect(await download.body()).toEqual(png);
    const etag = download.headers().etag;
    expect((await page.request.get(file, { headers: { 'If-None-Match': etag } })).status()).toBe(304);
    const range = await page.request.get(file, { headers: { Range: 'bytes=0-6' } });
    expect(range.status()).toBe(206);
    expect(await range.body()).toEqual(png.subarray(0, 7));
    expect((await page.request.get(file, { headers: { Range: 'bytes=9999-' } })).status()).toBe(416);
  }
});
test('job creation validates configuration and returns paged summaries', async ({ page }) => {
  const config = {
    job: 'extension',
    config: {
      name: 'browser-fixture-run',
      process: [
        {
          type: 'diffusion_trainer',
          model: { arch: 'flux', name_or_path: 'fixture/no-weights' },
          train: { steps: 10 },
          datasets: [],
        },
      ],
    },
  };
  const body = { name: config.config.name, job_type: 'train', gpu_ids: '0', worker_id: 'local', job_config: config };
  const invalid = structuredClone(body);
  invalid.job_config.config.process[0].train.steps = -1;
  expect((await page.request.post('/api/jobs', { headers: { Origin: origin }, data: invalid })).status()).toBe(400);
  const created = await page.request.post('/api/jobs', { headers: { Origin: origin }, data: body });
  expect(created.ok()).toBe(true);
  const job = await created.json();
  expect(job.storage_key).toBe('job-' + job.id);
  const pageResponse = await (await page.request.get('/api/jobs?limit=1')).json();
  expect(pageResponse.jobs).toHaveLength(1);
  expect(pageResponse.jobs[0].is_summary).toBe(true);
  expect((await page.request.get('/api/jobs?view=active')).ok()).toBe(true);
  expect((await page.request.post('/api/jobs/' + job.id + '/delete', { headers: { Origin: origin } })).ok()).toBe(true);
});
test('training drafts ignore unchanged interactions and guard edited navigation', async ({ page }) => {
  await page.goto('/jobs/new');
  const name = page.getByLabel('Training name', { exact: true });
  await expect(name).toBeVisible();
  await name.click();
  await page.getByRole('link', { name: 'Settings', exact: true }).first().click();
  await expect(page.getByText('Leave training setup?', { exact: true })).toHaveCount(0);
  await page.goto('/jobs/new');
  await name.fill('unsaved browser draft');
  await page.getByRole('link', { name: 'Settings', exact: true }).first().click();
  await expect(page.getByText('Leave training setup?', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(name).toHaveValue('unsaved browser draft');
  const canceled = page.waitForEvent('dialog').then(async dialog => {
    expect(dialog.type()).toBe('beforeunload');
    await dialog.dismiss();
  });
  await page.evaluate(() => history.back());
  await canceled;
  await expect(name).toHaveValue('unsaved browser draft');
});

test('streamed media and captions stay paired and collisions leave existing content untouched', async ({ page }) => {
  const datasetName = `paired-${randomUUID()}`;
  const upload = (filename: string, data: Buffer, group: string) => page.request.post('/api/datasets/upload', {
    headers: { Origin: origin, 'Content-Type': 'application/octet-stream',
      'X-AITK-Dataset-Name': encodeURIComponent(datasetName), 'X-AITK-File-Name': encodeURIComponent(filename),
      'X-AITK-Upload-Group': group }, data,
  });
  const original = randomUUID();
  expect((await upload('photo.png', png, original)).status()).toBe(200);
  expect((await upload('photo.txt', Buffer.from('original caption'), original)).status()).toBe(200);
  const conflicting = randomUUID();
  expect((await upload('photo.png', Buffer.from('different image'), conflicting)).status()).toBe(409);
  expect((await upload('photo.txt', Buffer.from('wrong caption'), conflicting)).status()).toBe(409);
  expect((await upload('photo.json', Buffer.from('wrong caption'), conflicting)).status()).toBe(409);
  const listing = await (await page.request.get('/api/datasets/list')).json();
  const dataset = listing.find((item: { name: string }) => item.name === datasetName);
  expect(dataset).toBeTruthy();
  const imagePath = dataset.path + '/photo.png';
  const caption = await page.request.post('/api/caption/get', { headers: { Origin: origin }, data: { imgPath: imagePath } });
  expect(caption.ok()).toBe(true);
  expect(await caption.text()).toBe('original caption');
  const image = await page.request.get('/api/files/' + encodeURIComponent(imagePath));
  expect(await image.body()).toEqual(png);
  const pair = randomUUID();
  expect((await upload('second.png', png, pair)).status()).toBe(200);
  expect((await upload('second.txt', Buffer.from('second caption'), pair)).status()).toBe(200);
});

test('queue counts stay independent of the requested view and page size', async ({ page }) => {
  const all = await (await page.request.get('/api/jobs?view=all&limit=1')).json();
  expect(all.statusCounts).toBeTruthy();
  for (const view of ['active', 'history', 'failed']) {
    const filtered = await (await page.request.get(`/api/jobs?view=${view}&limit=1`)).json();
    expect(filtered.statusCounts).toEqual(all.statusCounts);
  }
});
