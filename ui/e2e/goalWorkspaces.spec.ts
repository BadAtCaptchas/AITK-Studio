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
