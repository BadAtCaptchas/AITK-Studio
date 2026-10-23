import { test, expect, type BrowserContext, type Page } from '@playwright/test';

let cookies: Awaited<ReturnType<BrowserContext['cookies']>> = [];
const errors = new WeakMap<Page, string[]>();

test.beforeEach(async ({ page }) => {
  const pageErrors: string[] = [];
  errors.set(page, pageErrors);
  page.on('pageerror', error => pageErrors.push(error.message));
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
});

test.afterEach(async ({ page }) => {
  expect(errors.get(page)).toEqual([]);
});

for (const kind of [
  {
    label: 'training worker',
    endpoint: '/api/workers',
    section: 'Compute',
    namePlaceholder: 'Worker name',
    urlPlaceholder: 'https://worker.example.com',
    tokenPlaceholder: 'Leave blank to keep existing API token',
    newTokenPlaceholder: 'Worker AI_TOOLKIT_AUTH token',
    tokenField: 'api_token',
    submit: 'Update worker',
    add: 'Add worker',
    error: 'Failed to save worker.',
  },
  {
    label: 'Ollama endpoint',
    endpoint: '/api/ollama-workers',
    section: 'Advanced',
    namePlaceholder: 'Endpoint name',
    urlPlaceholder: 'http://ollama-host:11434',
    tokenPlaceholder: 'Leave blank to keep existing bearer token',
    newTokenPlaceholder: 'Optional bearer token',
    tokenField: 'auth_token',
    submit: 'Update endpoint',
    add: 'Add endpoint',
    error: 'Failed to save Remote Ollama endpoint.',
  },
] as const) {
  for (const historyMode of kind.label === 'training worker' ? ['native', 'fallback'] : ['native']) {
    test(`unsaved ${kind.label} draft is protected through links and ${historyMode} Back`, async ({ page }) => {
      if (historyMode === 'fallback') {
        await page.addInitScript(() =>
          Object.defineProperty(window, 'navigation', { value: undefined, configurable: true }),
        );
      }
      await page.goto('/models');
      await page.getByRole('link', { name: 'Settings', exact: true }).click();
      await expect(page.getByLabel('Training folder', { exact: true })).toBeEnabled();
      const section = page
        .getByRole('navigation', { name: 'Settings sections' })
        .getByRole('button', { name: kind.section, exact: true });
      await section.click();
      const name = page.getByPlaceholder(kind.namePlaceholder, { exact: true });
      const url = page.getByPlaceholder(kind.urlPlaceholder, { exact: true });
      const token = page.getByPlaceholder(kind.newTokenPlaceholder, { exact: true });
      await name.fill('Unsaved navigation fixture');
      await url.fill('https://navigation-fixture.invalid');
      await token.fill('fixture-token-navigation');
      expect(await page.evaluate(() => !window.dispatchEvent(new Event('beforeunload', { cancelable: true })))).toBe(
        true,
      );
      const models = page
        .getByRole('navigation', { name: 'Main navigation' })
        .getByRole('link', { name: 'Models', exact: true });
      await models.click();
      const leave = page.getByRole('dialog', { name: 'Leave Settings?', exact: true });
      await expect(leave.getByRole('heading', { name: 'Leave Settings?', exact: true })).toBeVisible();
      await leave.getByRole('button', { name: 'Cancel', exact: true }).click();
      await expect(page).toHaveURL(/\/settings$/);
      await expect(name).toHaveValue('Unsaved navigation fixture');
      await page.evaluate(() => window.history.back());
      await expect(leave.getByRole('heading', { name: 'Leave Settings?', exact: true })).toBeVisible();
      await leave.getByRole('button', { name: 'Cancel', exact: true }).click();
      await expect(page).toHaveURL(/\/settings$/);
      await expect(token).toHaveValue('fixture-token-navigation');
      await page
        .getByRole('navigation', { name: 'Settings sections' })
        .getByRole('button', { name: 'Appearance', exact: true })
        .click();
      await expect(leave).toHaveCount(0);
      await section.click();
      await expect(name).toHaveValue('Unsaved navigation fixture');
      await name.clear();
      await url.clear();
      await token.clear();
      expect(await page.evaluate(() => !window.dispatchEvent(new Event('beforeunload', { cancelable: true })))).toBe(
        false,
      );
      await models.click();
      await expect(page).toHaveURL(/\/models$/);
      expect(await page.evaluate(() => JSON.stringify(localStorage))).not.toContain('fixture-token-navigation');
    });
  }

  test(`delayed ${kind.label} save preserves newer local name, URL and token edits`, async ({ page }) => {
    const worker = {
      id: 'fixture-worker',
      name: `Original ${kind.label}`,
      base_url: 'https://fixture-worker.invalid',
      enabled: false,
      offline_bypass_enabled: false,
      last_status: 'unknown',
      last_error: null,
      last_checked_at: null,
      capabilities: '{}',
      gpus: '[]',
      models: '[]',
      model_count: 0,
      created_at: '2026-10-23T00:00:00Z',
      updated_at: '2026-10-23T00:00:00Z',
    };
    const submissions: unknown[] = [];
    let finish: (() => Promise<void>) | undefined;
    await page.route(`**${kind.endpoint}`, async route => {
      if (route.request().method() !== 'POST') return route.fulfill({ json: { workers: [worker] } });
      const submitted: unknown = route.request().postDataJSON();
      submissions.push(submitted);
      if (submissions.length > 1) return route.fulfill({ json: worker });
      await new Promise<void>(resolve => {
        finish = async () => {
          await route.fulfill({ json: worker });
          resolve();
        };
      });
    });
    await page.goto('/settings');
    await expect(page.getByLabel('Training folder', { exact: true })).toBeEnabled();
    await page
      .getByRole('navigation', { name: 'Settings sections' })
      .getByRole('button', { name: kind.section, exact: true })
      .click();
    // Both worker rows group the name/details and their Edit control in this parent.
    await page
      .getByText(worker.name, { exact: true })
      .locator('..')
      .locator('..')
      .getByRole('button', { name: 'Edit', exact: true })
      .click();
    const name = page.getByPlaceholder(kind.namePlaceholder, { exact: true });
    const url = page.getByPlaceholder(kind.urlPlaceholder, { exact: true });
    const token = page.getByPlaceholder(kind.tokenPlaceholder, { exact: true });
    await expect(name).toHaveValue(worker.name);
    await name.fill('First submitted fixture name');
    await url.fill('https://first-fixture.invalid');
    await token.fill('fixture-token-submitted');
    await page.getByRole('button', { name: kind.submit, exact: true }).click();
    await expect.poll(() => Boolean(finish)).toBe(true);
    const models = page
      .getByRole('navigation', { name: 'Main navigation' })
      .getByRole('link', { name: 'Models', exact: true });
    const leave = page.getByRole('dialog', { name: 'Leave Settings?', exact: true });
    await models.click();
    await expect(leave.getByText(/Leaving will not cancel that request/)).toBeVisible();
    await leave.getByRole('button', { name: 'Cancel', exact: true }).click();
    await name.fill('Newer unsaved fixture name');
    await url.fill('https://newer-fixture.invalid');
    await token.fill('fixture-token-newer');
    const response = page.waitForResponse(
      response => response.url().endsWith(kind.endpoint) && response.request().method() === 'POST',
    );
    await finish!();
    await response;
    await page.waitForTimeout(100);
    await expect(name).toHaveValue('Newer unsaved fixture name');
    await expect(url).toHaveValue('https://newer-fixture.invalid');
    await expect(token).toHaveValue('fixture-token-newer');
    await models.click();
    await expect(leave.getByRole('heading', { name: 'Leave Settings?', exact: true })).toBeVisible();
    await leave.getByRole('button', { name: 'Cancel', exact: true }).click();
    expect(submissions[0]).toMatchObject({
      id: worker.id,
      name: 'First submitted fixture name',
      base_url: 'https://first-fixture.invalid',
      [kind.tokenField]: 'fixture-token-submitted',
    });
    await page.getByRole('button', { name: kind.submit, exact: true }).click();
    await expect(name).toHaveValue('');
    expect(submissions[1]).toMatchObject({
      id: worker.id,
      name: 'Newer unsaved fixture name',
      base_url: 'https://newer-fixture.invalid',
      [kind.tokenField]: 'fixture-token-newer',
    });
    expect(await page.evaluate(() => JSON.stringify(localStorage))).not.toContain('fixture-token');
    await models.click();
    await expect(page).toHaveURL(/\/models$/);
  });

  test(`${kind.label} clean baseline survives failed saves and keeps main Settings edits independent`, async ({
    page,
  }) => {
    const worker = {
      id: 'fixture-baseline-worker',
      name: 'Baseline fixture',
      base_url: 'https://baseline-fixture.invalid',
      enabled: false,
      offline_bypass_enabled: false,
      last_status: 'unknown',
      last_error: null,
      last_checked_at: null,
      capabilities: '{}',
      gpus: '[]',
      models: '[]',
      model_count: 0,
      created_at: '2026-10-23T00:00:00Z',
      updated_at: '2026-10-23T00:00:00Z',
    };
    let saves = 0;
    await page.route(`**${kind.endpoint}`, async route => {
      if (route.request().method() !== 'POST') return route.fulfill({ json: { workers: [worker] } });
      saves += 1;
      return saves === 1
        ? route.fulfill({ status: 503, json: { error: 'Fixture save unavailable' } })
        : route.fulfill({ json: worker });
    });
    const models = page
      .getByRole('navigation', { name: 'Main navigation' })
      .getByRole('link', { name: 'Models', exact: true });
    const name = page.getByPlaceholder(kind.namePlaceholder, { exact: true });
    const token = page.getByPlaceholder(kind.tokenPlaceholder, { exact: true });
    const leave = page.getByRole('dialog', { name: 'Leave Settings?', exact: true });
    const edit = async () => {
      await page.getByRole('link', { name: 'Settings', exact: true }).click();
      await expect(page.getByLabel('Training folder', { exact: true })).toBeEnabled();
      await page
        .getByRole('navigation', { name: 'Settings sections' })
        .getByRole('button', { name: kind.section, exact: true })
        .click();
      await page
        .getByText(worker.name, { exact: true })
        .locator('..')
        .locator('..')
        .getByRole('button', { name: 'Edit', exact: true })
        .click();
      await expect(name).toHaveValue(worker.name);
    };
    await page.goto('/models');
    await edit();
    // Merely opening an existing worker is not an unsaved change.
    await models.click();
    await expect(page).toHaveURL(/\/models$/);
    await edit();
    await name.fill('Failed baseline fixture');
    await token.fill('fixture-token-failed-baseline');
    await page.getByRole('button', { name: kind.submit, exact: true }).click();
    await expect(page.getByText(kind.error, { exact: true })).toBeVisible();
    await models.click();
    await expect(leave.getByRole('heading', { name: 'Leave Settings?', exact: true })).toBeVisible();
    await leave.getByRole('button', { name: 'Cancel', exact: true }).click();
    await name.fill(worker.name);
    await token.clear();
    await models.click();
    await expect(page).toHaveURL(/\/models$/);
    await edit();
    const folder = page.getByLabel('Training folder', { exact: true });
    const initialFolder = await folder.inputValue();
    await folder.fill(`${initialFolder}-unsaved-worker-fixture`);
    await name.fill('Successfully saved fixture');
    await page.getByRole('button', { name: kind.submit, exact: true }).click();
    await expect(name).toHaveValue('');
    await models.click();
    await expect(leave.getByRole('heading', { name: 'Leave Settings?', exact: true })).toBeVisible();
    await leave.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(folder).toHaveValue(`${initialFolder}-unsaved-worker-fixture`);
    await folder.fill(initialFolder);
    await models.click();
    await expect(page).toHaveURL(/\/models$/);
  });

  test(`new ${kind.label} retains its saved identity and prevents duplicate pending submissions`, async ({ page }) => {
    const submissions: unknown[] = [];
    let finish: (() => Promise<void>) | undefined;
    await page.route(`**${kind.endpoint}`, async route => {
      if (route.request().method() !== 'POST') return route.fulfill({ json: { workers: [] } });
      submissions.push(route.request().postDataJSON());
      if (submissions.length > 1) return route.fulfill({ json: { id: 'fixture-created-worker' } });
      await new Promise<void>(resolve => {
        finish = async () => {
          await route.fulfill({ json: { id: 'fixture-created-worker' } });
          resolve();
        };
      });
    });
    await page.goto('/settings');
    await expect(page.getByLabel('Training folder', { exact: true })).toBeEnabled();
    await page
      .getByRole('navigation', { name: 'Settings sections' })
      .getByRole('button', { name: kind.section, exact: true })
      .click();
    const name = page.getByPlaceholder(kind.namePlaceholder, { exact: true });
    await name.fill('Submitted new fixture');
    await page.getByPlaceholder(kind.urlPlaceholder, { exact: true }).fill('https://created-fixture.invalid');
    await page.getByPlaceholder(kind.newTokenPlaceholder, { exact: true }).fill('fixture-token-created');
    // Two submit events in one task exercise the guard before disabled state can commit.
    await name.evaluate(input => {
      const form = input.closest('form')!;
      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    });
    await expect.poll(() => Boolean(finish)).toBe(true);
    await expect(page.getByRole('button', { name: kind.add, exact: true })).toBeDisabled();
    await name.fill('Newer created fixture');
    await page.getByPlaceholder(kind.newTokenPlaceholder, { exact: true }).fill('fixture-token-created-newer');
    await finish!();
    await expect(page.getByRole('button', { name: kind.submit, exact: true })).toBeEnabled();
    await expect(name).toHaveValue('Newer created fixture');
    expect(submissions).toHaveLength(1);
    expect(submissions[0]).toMatchObject({ id: '', name: 'Submitted new fixture' });
    await expect(page.getByPlaceholder(kind.tokenPlaceholder, { exact: true })).toHaveValue(
      'fixture-token-created-newer',
    );
    // The clean baseline is now the created worker, not the original empty form.
    await name.fill('Submitted new fixture');
    await page.getByPlaceholder(kind.tokenPlaceholder, { exact: true }).clear();
    expect(await page.evaluate(() => !window.dispatchEvent(new Event('beforeunload', { cancelable: true })))).toBe(
      false,
    );
    await name.fill('Newer created fixture');
    await page.getByPlaceholder(kind.tokenPlaceholder, { exact: true }).fill('fixture-token-created-newer');
    await page.getByRole('button', { name: kind.submit, exact: true }).click();
    await expect(name).toHaveValue('');
    expect(submissions).toHaveLength(2);
    expect(submissions[1]).toMatchObject({
      id: 'fixture-created-worker',
      name: 'Newer created fixture',
      [kind.tokenField]: 'fixture-token-created-newer',
    });
  });

  for (const nextDraft of ['another worker', 'cancelled worker'] as const) {
    test(`delayed ${kind.label} save leaves the ${nextDraft} draft untouched`, async ({ page }) => {
      const workers = ['first', 'second'].map(id => ({
        id: `fixture-${id}`,
        name: `${id} fixture`,
        base_url: `https://${id}-fixture.invalid`,
        enabled: false,
        offline_bypass_enabled: false,
        last_status: 'unknown',
        last_error: null,
        last_checked_at: null,
        capabilities: '{}',
        gpus: '[]',
        models: '[]',
        model_count: 0,
        created_at: '2026-10-23T00:00:00Z',
        updated_at: '2026-10-23T00:00:00Z',
      }));
      const submissions: unknown[] = [];
      let finish: (() => Promise<void>) | undefined;
      await page.route(`**${kind.endpoint}`, async route => {
        if (route.request().method() !== 'POST') return route.fulfill({ json: { workers } });
        submissions.push(route.request().postDataJSON());
        if (submissions.length > 1) return route.fulfill({ json: workers[1] });
        await new Promise<void>(resolve => {
          finish = async () => {
            await route.fulfill({ json: workers[0] });
            resolve();
          };
        });
      });
      await page.goto('/settings');
      await expect(page.getByLabel('Training folder', { exact: true })).toBeEnabled();
      await page
        .getByRole('navigation', { name: 'Settings sections' })
        .getByRole('button', { name: kind.section, exact: true })
        .click();
      await page
        .getByText(workers[0].name, { exact: true })
        .locator('..')
        .locator('..')
        .getByRole('button', { name: 'Edit', exact: true })
        .click();
      await page.getByRole('button', { name: kind.submit, exact: true }).click();
      await expect.poll(() => Boolean(finish)).toBe(true);
      if (nextDraft === 'another worker') {
        await page
          .getByText(workers[1].name, { exact: true })
          .locator('..')
          .locator('..')
          .getByRole('button', { name: 'Edit', exact: true })
          .click();
      } else {
        await page.getByRole('button', { name: 'Cancel', exact: true }).click();
      }
      const name = page.getByPlaceholder(kind.namePlaceholder, { exact: true });
      await name.fill('Unrelated unsaved fixture');
      await page.getByPlaceholder(kind.urlPlaceholder, { exact: true }).fill('https://unrelated-fixture.invalid');
      const token = page.getByPlaceholder(
        nextDraft === 'another worker' ? kind.tokenPlaceholder : kind.newTokenPlaceholder,
        { exact: true },
      );
      await token.fill('fixture-token-unrelated');
      await finish!();
      const submit = page.getByRole('button', {
        name: nextDraft === 'another worker' ? kind.submit : kind.add,
        exact: true,
      });
      await expect(submit).toBeEnabled();
      await expect(name).toHaveValue('Unrelated unsaved fixture');
      await expect(token).toHaveValue('fixture-token-unrelated');
      await submit.click();
      await expect(name).toHaveValue('');
      expect(submissions[1]).toMatchObject({
        id: nextDraft === 'another worker' ? workers[1].id : '',
        name: 'Unrelated unsaved fixture',
        base_url: 'https://unrelated-fixture.invalid',
        [kind.tokenField]: 'fixture-token-unrelated',
      });
    });
  }

  test(`failed ${kind.label} save preserves its draft and permits retry`, async ({ page }) => {
    const submissions: unknown[] = [];
    const browserErrors: string[] = [];
    page.on('console', message => {
      if (message.type() === 'error') browserErrors.push(message.text());
    });
    await page.route(`**${kind.endpoint}`, async route => {
      if (route.request().method() !== 'POST') return route.fulfill({ json: { workers: [] } });
      submissions.push(route.request().postDataJSON());
      return submissions.length === 1
        ? route.fulfill({ status: 503, json: { error: 'Fixture save unavailable' } })
        : route.fulfill({ json: { id: 'fixture-retried-worker' } });
    });
    await page.goto('/settings');
    await expect(page.getByLabel('Training folder', { exact: true })).toBeEnabled();
    await page
      .getByRole('navigation', { name: 'Settings sections' })
      .getByRole('button', { name: kind.section, exact: true })
      .click();
    const name = page.getByPlaceholder(kind.namePlaceholder, { exact: true });
    const token = page.getByPlaceholder(kind.newTokenPlaceholder, { exact: true });
    await name.fill('Retry fixture');
    await page.getByPlaceholder(kind.urlPlaceholder, { exact: true }).fill('https://retry-fixture.invalid');
    await token.fill('fixture-token-retry');
    const submit = page.getByRole('button', { name: kind.add, exact: true });
    await submit.click();
    await expect(page.getByText(kind.error, { exact: true })).toBeVisible();
    await expect(name).toHaveValue('Retry fixture');
    await expect(token).toHaveValue('fixture-token-retry');
    await expect(submit).toBeEnabled();
    await name.fill('Retry newer fixture');
    await submit.click();
    await expect(name).toHaveValue('');
    await expect(page.getByText(kind.error, { exact: true })).toHaveCount(0);
    expect(submissions[1]).toMatchObject({
      id: '',
      name: 'Retry newer fixture',
      [kind.tokenField]: 'fixture-token-retry',
    });
    expect(browserErrors.join('\n')).not.toContain('fixture-token');
    expect(browserErrors.join('\n')).not.toContain('AxiosError');
    expect(await page.evaluate(() => JSON.stringify(localStorage))).not.toContain('fixture-token');
  });
}
