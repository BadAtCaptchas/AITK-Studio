import { test, expect, type BrowserContext, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

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
