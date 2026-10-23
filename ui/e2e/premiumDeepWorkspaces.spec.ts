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

test('late Settings save cannot overwrite edits in a newly opened Settings editor', async ({ page }) => {
  await page.goto('/settings');
  const folder = page.getByLabel('Training folder', { exact: true });
  await expect(folder).toBeEnabled();
  const initial = await folder.inputValue();
  let finish: (() => Promise<void>) | undefined;
  await page.route('**/api/settings', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    await new Promise<void>(resolve => {
      finish = async () => {
        await route.fulfill({ json: { success: true } });
        resolve();
      };
    });
  });
  await folder.fill(`${initial}-submitted`);
  await page.getByRole('button', { name: 'Save settings', exact: true }).click();
  await expect.poll(() => Boolean(finish)).toBe(true);
  await page
    .getByRole('navigation', { name: 'Main navigation' })
    .getByRole('link', { name: 'Models', exact: true })
    .click();
  await page.getByRole('button', { name: 'Discard changes', exact: true }).click();
  await expect(page).toHaveURL(/\/models$/);
  await page.getByRole('link', { name: 'Settings', exact: true }).click();
  await expect(folder).toBeEnabled();
  const newer = `${initial}-new-editor`;
  await folder.fill(newer);
  const response = page.waitForResponse(
    response => response.url().endsWith('/api/settings') && response.request().method() === 'POST',
  );
  await finish!();
  await response;
  // Give the old component's asynchronous completion time to publish its event.
  await page.waitForTimeout(150);
  await expect(folder).toHaveValue(newer);
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeEnabled();
});

test('late Generate job creation cannot redirect away from a later workspace', async ({ page }) => {
  await page.goto('/generate');
  await expect(page.getByRole('combobox', { name: 'Architecture', exact: true })).toBeEnabled();
  await page.getByLabel('Base Model', { exact: true }).fill('fixture/base-model');
  await page.getByLabel('Prompt text', { exact: true }).fill('First fixture prompt\nSecond fixture prompt');
  await page.locator('summary').filter({ hasText: 'Run settings' }).click();
  const startImmediately = page.getByRole('switch', { name: 'Start job now', exact: true });
  await startImmediately.click();
  await expect(startImmediately).toHaveAttribute('aria-checked', 'false');
  let finish: (() => Promise<void>) | undefined;
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    await new Promise<void>(resolve => {
      finish = async () => {
        await route.fulfill({ json: { id: 'late-generation-fixture' } });
        resolve();
      };
    });
  });
  await page.getByRole('button', { name: 'Create Job', exact: true }).click();
  await expect.poll(() => Boolean(finish)).toBe(true);
  await page
    .getByRole('navigation', { name: 'Main navigation' })
    .getByRole('link', { name: 'Models', exact: true })
    .click();
  await expect(page).toHaveURL(/\/models$/);
  const response = page.waitForResponse(
    response => response.url().endsWith('/api/jobs') && response.request().method() === 'POST',
  );
  await finish!();
  await response;
  await page.waitForTimeout(150);
  await expect(page).toHaveURL(/\/models$/);
});

test('late Settings service status cannot restart polling after navigation', async ({ page }) => {
  await page.clock.install();
  let calls = 0;
  let finish: (() => Promise<void>) | undefined;
  await page.route('**/api/comfy/install', async route => {
    calls++;
    const json = { installed: false, installing: true, root: '', logPath: '' };
    if (calls > 1) return route.fulfill({ json });
    await new Promise<void>(resolve => {
      finish = async () => {
        await route.fulfill({ json });
        resolve();
      };
    });
  });
  await page.goto('/settings');
  await expect.poll(() => Boolean(finish)).toBe(true);
  await page
    .getByRole('navigation', { name: 'Main navigation' })
    .getByRole('link', { name: 'Models', exact: true })
    .click();
  await expect(page).toHaveURL(/\/models$/);
  const response = page.waitForResponse('**/api/comfy/install');
  await finish!();
  await response;
  await page.waitForTimeout(100);
  await page.clock.fastForward(6000);
  await page.waitForTimeout(100);
  expect(calls).toBe(1);
});

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
