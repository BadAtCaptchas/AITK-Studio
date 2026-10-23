import { test, expect, type BrowserContext, type Page } from '@playwright/test';
import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

const origin = 'http://127.0.0.1:15875';
const image = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZtsAAAAASUVORK5CYII=',
  'base64',
);
let cookies: Awaited<ReturnType<BrowserContext['cookies']>> = [];
const errors = new WeakMap<Page, string[]>();

test.beforeEach(async ({ page }) => {
  const pageErrors: string[] = [];
  errors.set(page, pageErrors);
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.route('**/api/monitor', route => route.abort());
  await page.route('https://cdn.jsdelivr.net/npm/monaco-editor@*/min/vs/**', async route => {
    const relative = new URL(route.request().url()).pathname.split('/min/vs/')[1];
    if (!relative || relative.split('/').includes('..')) return route.abort();
    await route.fulfill({
      body: await readFile(path.join(process.cwd(), 'node_modules/monaco-editor/min/vs', relative)),
      contentType: relative.endsWith('.css') ? 'text/css' : 'application/javascript',
    });
  });
  if (cookies.length) await page.context().addCookies(cookies);
  else {
    await page.goto('/');
    await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
    await page.getByRole('button', { name: 'Check Password', exact: true }).click();
    await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
    cookies = (await page.context().cookies()).filter(cookie => cookie.name === 'aitk_session');
  }
});

test.afterEach(async ({ page }) => {
  expect(errors.get(page)).toEqual([]);
});

async function modelsLink(page: Page) {
  await page
    .getByRole('navigation', { name: 'Main navigation' })
    .getByRole('link', { name: 'Models', exact: true })
    .click();
}

test('Settings protects unsaved edits through links and browser Back, then permits saved navigation', async ({
  page,
}) => {
  await page.goto('/models');
  await page.getByRole('link', { name: 'Settings', exact: true }).click();
  const folder = page.getByLabel('Training folder', { exact: true });
  await expect(folder).toBeEnabled();
  const initial = await folder.inputValue();
  await folder.fill(`${initial}-draft`);
  await modelsLink(page);
  await expect(page.getByRole('heading', { name: 'Leave Settings?', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page).toHaveURL(/\/settings$/);
  await expect(folder).toHaveValue(`${initial}-draft`);
  await page.evaluate(() => window.history.back());
  await expect(page.getByRole('heading', { name: 'Leave Settings?', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page).toHaveURL(/\/settings$/);
  await expect(folder).toHaveValue(`${initial}-draft`);
  await page.route('**/api/settings', route =>
    route.request().method() === 'POST' ? route.fulfill({ json: { success: true } }) : route.continue(),
  );
  await page.getByRole('button', { name: 'Save settings', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeDisabled();
  await modelsLink(page);
  await expect(page).toHaveURL(/\/models$/);
});

type ImportFixtureWindow = Window & { finishPromptRead?: () => void; promptReadPending?: boolean };

async function delayPromptRead(page: Page) {
  await page.addInitScript(() => {
    const original = File.prototype.text;
    File.prototype.text = function () {
      if (this.name !== 'slow-prompts.txt') return original.call(this);
      const fixtureWindow = window as ImportFixtureWindow;
      fixtureWindow.promptReadPending = true;
      return new Promise<string>(resolve => {
        fixtureWindow.finishPromptRead = () => resolve('Older imported prompt');
      });
    };
  });
  await page.goto('/generate');
  await expect(page.getByLabel('Prompt text', { exact: true })).toBeVisible();
  await page
    .locator('input[type="file"][accept*=".txt"]')
    .setInputFiles({ name: 'slow-prompts.txt', mimeType: 'text/plain', buffer: Buffer.from('Older imported prompt') });
  await expect.poll(() => page.evaluate(() => Boolean((window as ImportFixtureWindow).promptReadPending))).toBe(true);
}

test('a delayed Generate prompt import cannot overwrite newer typing', async ({ page }) => {
  await delayPromptRead(page);
  const prompt = page.getByLabel('Prompt text', { exact: true });
  await prompt.fill('Newer manually edited prompt');
  await page.evaluate(() => (window as ImportFixtureWindow).finishPromptRead?.());
  await page.waitForTimeout(100);
  await expect(prompt).toHaveValue('Newer manually edited prompt');
});

test('a delayed Generate prompt import cannot overwrite a newer imported file', async ({ page }) => {
  await delayPromptRead(page);
  await page
    .locator('input[type="file"][accept*=".txt"]')
    .setInputFiles({ name: 'new-prompts.txt', mimeType: 'text/plain', buffer: Buffer.from('Newer imported prompt') });
  const prompt = page.getByLabel('Prompt text', { exact: true });
  await expect(prompt).toHaveValue('Newer imported prompt');
  await page.evaluate(() => (window as ImportFixtureWindow).finishPromptRead?.());
  await page.waitForTimeout(100);
  await expect(prompt).toHaveValue('Newer imported prompt');
});

test('late LoRA upload keeps a newer Base model choice', async ({ page }) => {
  const uploaded = {
    id: 'fixture-lora',
    label: 'Uploaded fixture',
    path: 'fixture/uploaded.safetensors',
    filename: 'uploaded.safetensors',
    source: 'uploaded',
    updatedAt: '2026-10-23T00:00:00Z',
    sizeBytes: 128,
    model: { arch: 'flux', name_or_path: 'fixture/upload-base' },
  };
  await page.route('**/api/generate/loras', route => route.fulfill({ json: { loras: [] } }));
  let finish: (() => Promise<void>) | undefined;
  await page.route('**/api/generate/loras/upload', async route => {
    await new Promise<void>(resolve => {
      finish = async () => {
        await route.fulfill({ json: { lora: uploaded } });
        resolve();
      };
    });
  });
  await page.goto('/generate');
  await expect(page.getByLabel('Base Model', { exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'LoRA', exact: true }).click();
  await page.locator('input[type="file"][accept=".safetensors"]').setInputFiles({
    name: 'uploaded.safetensors',
    mimeType: 'application/octet-stream',
    buffer: Buffer.from('intercepted fixture upload'),
  });
  await expect.poll(() => Boolean(finish)).toBe(true);
  await page.getByRole('button', { name: 'Base model', exact: true }).click();
  await page.getByLabel('Base Model', { exact: true }).fill('fixture/newer-model-choice');
  const response = page.waitForResponse('**/api/generate/loras/upload');
  await finish!();
  await response;
  await page.waitForTimeout(100);
  await expect(page.getByRole('button', { name: 'Base model', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByLabel('Base Model', { exact: true })).toHaveValue('fixture/newer-model-choice');
});

async function twoImageDataset(page: Page) {
  const dataset = `goal-workspace-${randomUUID().slice(0, 8)}`;
  const uploadGroup = randomUUID();
  for (const [name, data] of [
    ['a.png', image],
    ['a.txt', Buffer.from('First caption')],
    ['b.png', image],
    ['b.txt', Buffer.from('Second caption')],
  ] as const) {
    const response = await page.request.post('/api/datasets/upload', {
      headers: {
        Origin: origin,
        'Content-Type': 'application/octet-stream',
        'X-AITK-Dataset-Name': encodeURIComponent(dataset),
        'X-AITK-File-Name': name,
        'X-AITK-Upload-Group': uploadGroup,
      },
      data,
    });
    expect(response.status()).toBe(200);
  }
  return dataset;
}

test('Dataset Save and Next stays on edits made while the caption save is pending', async ({ page }) => {
  const dataset = await twoImageDataset(page);
  await page.goto(`/datasets/${dataset}`);
  const caption = page.getByRole('textbox', { name: 'Image caption', exact: true });
  await expect(caption).toHaveValue('First caption');
  let finish: (() => Promise<void>) | undefined;
  let requests = 0;
  await page.route('**/api/img/caption', async route => {
    if (route.request().method() !== 'POST' || ++requests > 1) return route.continue();
    const response = await route.fetch();
    await new Promise<void>(resolve => {
      finish = async () => {
        await route.fulfill({ response });
        resolve();
      };
    });
  });
  await caption.fill('Submitted caption');
  const advance = page.getByRole('button', { name: 'Save & Next', exact: true });
  await advance.click();
  await expect.poll(() => Boolean(finish)).toBe(true);
  await caption.fill('Newer unsaved caption');
  const response = page.waitForResponse(
    response => response.url().endsWith('/api/img/caption') && response.request().method() === 'POST',
  );
  await finish!();
  await response;
  await expect(advance).toBeEnabled();
  await expect(caption).toHaveValue('Newer unsaved caption');
  await advance.click();
  await expect(caption).toHaveValue('Second caption');
  await page.reload();
  await expect(caption).toHaveValue('Newer unsaved caption');
});

test('Dataset bulk caption actions stop when the current caption cannot be saved', async ({ page }) => {
  const dataset = await twoImageDataset(page);
  await page.goto(`/datasets/${dataset}`);
  const caption = page.getByRole('textbox', { name: 'Image caption', exact: true });
  await expect(caption).toHaveValue('First caption');
  await caption.fill('First caption with unsaved edits');
  let bulkRequests = 0;
  await page.route('**/api/img/caption', route =>
    route.request().method() === 'POST'
      ? route.fulfill({ status: 503, json: { error: 'Fixture caption save failed' } })
      : route.continue(),
  );
  await page.route('**/api/datasets/caption-bulk', route => {
    bulkRequests++;
    return route.fulfill({ json: { found: 1, affected: 1 } });
  });
  page.on('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'Batch actions', exact: true }).click();
  await page.getByPlaceholder('Find caption keywords', { exact: true }).fill('First');
  const remove = page.getByRole('button', { name: 'Remove words', exact: true });
  await expect(remove).toBeEnabled();
  const failedSave = page.waitForResponse(
    response => response.url().endsWith('/api/img/caption') && response.request().method() === 'POST',
  );
  await remove.click();
  expect((await failedSave).status()).toBe(503);
  await expect(remove).toBeEnabled();
  expect(bulkRequests).toBe(0);
  await expect(caption).toHaveValue('First caption with unsaved edits');
  let deletes = 0;
  await page.route('**/api/img/delete-bulk', route => {
    deletes++;
    return route.fulfill({ json: { requested: 2, deleted: 0, failed: 2 } });
  });
  await page.getByRole('button', { name: 'Select shown', exact: true }).click();
  const failedDeleteSave = page.waitForResponse(
    response => response.url().endsWith('/api/img/caption') && response.request().method() === 'POST',
  );
  const deleteSelected = page.getByRole('button', { name: 'Delete selected', exact: true });
  await deleteSelected.click();
  expect((await failedDeleteSave).status()).toBe(503);
  await expect(deleteSelected).toBeEnabled();
  expect(deletes).toBe(0);
  await expect(caption).toHaveValue('First caption with unsaved edits');
});
