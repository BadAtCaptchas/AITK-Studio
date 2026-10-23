import { test, expect, type BrowserContext, type Page } from '@playwright/test';
import { randomUUID } from 'node:crypto';

const origin = 'http://127.0.0.1:15875';
const image = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZtsAAAAASUVORK5CYII=',
  'base64',
);
const errors = new WeakMap<Page, string[]>();
let sessionCookies: Awaited<ReturnType<BrowserContext['cookies']>> = [];

test.beforeEach(async ({ page }) => {
  const pageErrors: string[] = [];
  errors.set(page, pageErrors);
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/gpu?*', route =>
    route.fulfill({
      json: {
        hasNvidiaSmi: true,
        isMac: false,
        gpus: [
          {
            index: 0,
            name: 'Fixture GPU',
            temperature: 40,
            utilization: { gpu: 0, memory: 0 },
            memory: { total: 24576, used: 2048, free: 22528 },
            power: { draw: 30, limit: 300 },
            clocks: { graphics: 1000, memory: 7000 },
            fan: { speed: 30 },
          },
        ],
      },
    }),
  );
  if (sessionCookies.length) await page.context().addCookies(sessionCookies);
  else {
    await page.goto('/');
    await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
    await page.getByRole('button', { name: 'Check Password', exact: true }).click();
    await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
    sessionCookies = (await page.context().cookies()).filter(cookie => cookie.name === 'aitk_session');
  }
});

test.afterEach(async ({ page }) => {
  expect(errors.get(page)).toEqual([]);
});

async function fixtureDataset(page: Page) {
  const dataset = `deep-workspace-${randomUUID().slice(0, 8)}`;
  const uploadGroup = randomUUID();
  for (const [name, data] of [
    ['asset.png', image],
    ['asset.txt', Buffer.from('Original fixture caption')],
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

async function blockWorkspaceStorage(page: Page) {
  await page.addInitScript(() => {
    for (const method of ['getItem', 'setItem', 'removeItem'] as const) {
      const original = Storage.prototype[method];
      Object.defineProperty(Storage.prototype, method, {
        configurable: true,
        value: function (this: Storage, key: string, value?: string) {
          if (/recaption|AITK_CLOUDFLARED_AUTO_DOWNLOAD/.test(key))
            throw new DOMException('Fixture storage blocked', 'SecurityError');
          return Reflect.apply(original, this, method === 'setItem' ? [key, value] : [key]);
        },
      });
    }
  });
}

test('Settings remains usable when optional preference storage is blocked', async ({ page }) => {
  await blockWorkspaceStorage(page);
  await page.goto('/settings');
  await expect(page.getByLabel('Training folder', { exact: true })).toBeEnabled();
  await page
    .getByRole('navigation', { name: 'Settings sections' })
    .getByRole('button', { name: 'Advanced', exact: true })
    .click();
  const preference = page.getByRole('switch', { name: 'Auto-download Cloudflared', exact: true });
  await expect(preference).toHaveAttribute('aria-checked', 'true');
  await preference.click();
  await expect(preference).toHaveAttribute('aria-checked', 'false');
});

test('Dataset caption editing and saving survives blocked recaption storage', async ({ page }) => {
  const dataset = await fixtureDataset(page);
  await blockWorkspaceStorage(page);
  await page.goto(`/datasets/${dataset}`);
  const caption = page.getByRole('textbox', { name: 'Image caption', exact: true });
  await expect(caption).toHaveValue('Original fixture caption');
  await caption.fill('Updated fixture caption');
  const saved = page.waitForResponse(
    response => response.url().endsWith('/api/img/caption') && response.request().method() === 'POST',
  );
  await caption.press('Control+s');
  expect((await saved).ok()).toBe(true);
  await page.reload();
  await expect(caption).toHaveValue('Updated fixture caption');
});
