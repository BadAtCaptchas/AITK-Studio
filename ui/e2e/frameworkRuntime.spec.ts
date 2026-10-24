import { test, expect } from '@playwright/test';

test('API access, method handling and shell fallback stay separate', async ({ request }) => {
  const protectedResponse = await request.get('/api/settings');
  expect(protectedResponse.status()).toBe(401);
  const login = await request.post('/api/auth', { data: { token: 'browser-fixture-local-only' } });
  expect(login.status()).toBe(200);
  expect(login.headers()['set-cookie']).toMatch(/HttpOnly/i);
  expect(login.headers()['set-cookie']).toMatch(/SameSite=Strict/i);
  const unsupported = await request.patch('/api/auth', { data: {}, headers: { Origin: 'http://127.0.0.1:15875' } });
  expect(unsupported.status()).toBe(405);
  expect(unsupported.headers().allow).toContain('POST');
  const missing = await request.get('/api/does-not-exist', { headers: { Accept: 'text/html' } });
  expect(missing.status()).toBe(404);
  expect(missing.headers()['content-type']).toContain('application/json');
  const source = await request.get('/src/server/db.ts');
  expect(source.status()).toBe(404);
  const config = await request.get('/app-config.js');
  expect(config.headers()['cache-control']).toBe('no-store');
  expect(await config.text()).not.toContain('browser-fixture-local-only');
  await request.delete('/api/auth', { headers: { Origin: 'http://127.0.0.1:15875' } });
  expect((await request.get('/api/settings')).status()).toBe(401);
});

test('deep links, browser navigation and standalone window metadata work', async ({ page }) => {
  await page.goto('/settings');
  await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
  await page.getByRole('button', { name: 'Check Password', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();
  const initial = await page.evaluate(() => performance.timeOrigin);
  await page.getByRole('link', { name: 'Monitor', exact: true }).click();
  await expect(page).toHaveURL(/\/dashboard$/);
  expect(await page.evaluate(() => performance.timeOrigin)).toBe(initial);
  await page.goBack();
  await expect(page).toHaveURL(/\/settings$/);
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();
  expect(await page.locator('link[rel="manifest"]').getAttribute('href')).toBe('/manifest.webmanifest');
  await page.goto('/missing-page');
  await expect(page.getByRole('heading', { name: 'Page not found', exact: true })).toBeVisible();
});
