import { test, expect, type Browser, type BrowserContext, type Page } from '@playwright/test';
import { mkdir, readFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import type { DatasetSummary, Job } from '../src/types';

const origin = 'http://127.0.0.1:15875';
const headers = { Origin: origin };
const captures = '.review-build/premium-workspaces';
const image = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZtsAAAAASUVORK5CYII=',
  'base64',
);
let cookies: Awaited<ReturnType<BrowserContext['cookies']>> = [];
const pageErrors = new WeakMap<Page, string[]>();

test.setTimeout(120_000);

async function capture(page: Page, name: string) {
  await page.screenshot({ path: `${captures}/${name}.png`, fullPage: true, animations: 'disabled' });
}

async function noPageOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
}

async function telemetry(page: Page) {
  // Exercise the real editor without depending on a public asset CDN.
  await page.route('https://cdn.jsdelivr.net/npm/monaco-editor@*/min/vs/**', async route => {
    const relative = new URL(route.request().url()).pathname.split('/min/vs/')[1];
    if (!relative || relative.split('/').includes('..')) return route.abort();
    await route.fulfill({
      body: await readFile(path.join(process.cwd(), 'node_modules/monaco-editor/min/vs', relative)),
      contentType: relative.endsWith('.css') ? 'text/css' : 'application/javascript',
    });
  });
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/gpu?*', route =>
    route.fulfill({
      json: {
        hasNvidiaSmi: true,
        isMac: false,
        gpus: [
          {
            index: 0,
            name: 'NVIDIA RTX 5090',
            driverVersion: 'fixture',
            temperature: 62,
            utilization: { gpu: 98, memory: 81 },
            memory: { total: 32768, used: 26419, free: 6349 },
            power: { draw: 312, limit: 575 },
            clocks: { graphics: 2100, memory: 14000 },
            fan: { speed: 48 },
          },
        ],
      },
    }),
  );
  await page.route('**/api/cpu?*', route =>
    route.fulfill({
      json: {
        name: 'Workstation CPU',
        cores: 16,
        temperature: 48,
        totalMemory: 65536,
        freeMemory: 26419,
        availableMemory: 26419,
        currentLoad: 18,
      },
    }),
  );
}

async function captureAdvancedLayouts(browser: Browser, dataset: string) {
  for (const theme of ['dark', 'light'] as const) {
    for (const size of [
      { width: 320, height: 844, scale: 1, label: '320' },
      { width: 390, height: 844, scale: 1, label: '390' },
      { width: 720, height: 512, scale: 2, label: '200-percent-reflow' },
    ]) {
      const context = await browser.newContext({
        baseURL: origin,
        viewport: { width: 1440, height: 1024 },
        deviceScaleFactor: size.scale,
      });
      try {
        await context.addCookies(cookies);
        await context.addInitScript(value => localStorage.setItem('theme', value), theme);
        const page = await context.newPage();
        const errors: string[] = [];
        page.on('pageerror', error => errors.push(error.message));
        await telemetry(page);
        await page.goto(`/datasets/${dataset}`);
        await expect(page.getByRole('textbox', { name: 'Image caption', exact: true })).toBeEnabled();
        await page.getByRole('button', { name: 'Layers', exact: true }).click();
        const editor = page.getByRole('region', { name: 'Layers editor', exact: true });
        await editor.getByRole('button', { name: /^Fixture object Object/ }).click();
        await page.setViewportSize({ width: size.width, height: size.height });
        await noPageOverflow(page);
        await expect(page.getByRole('button', { name: 'Layered documents', exact: true })).toBeInViewport({ ratio: 1 });
        await expect(page.getByRole('button', { name: 'Dataset settings', exact: true })).toBeInViewport({ ratio: 1 });
        const canvas = editor.getByRole('main', { name: 'Annotation canvas' });
        const canvasBounds = await canvas.boundingBox();
        expect(canvasBounds?.width).toBeGreaterThanOrEqual(240);
        expect(canvasBounds?.height).toBeGreaterThanOrEqual(300);
        await capture(page, `dataset-editor-${theme}-${size.label}-canvas`);
        const inspector = editor.getByRole('complementary', { name: 'Annotation inspector' });
        await inspector.scrollIntoViewIfNeeded();
        expect(await inspector.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
        await capture(page, `dataset-editor-${theme}-${size.label}-inspector`);
        await editor.getByRole('button', { name: 'Inspector', exact: true }).click();
        await expect(inspector).toBeHidden();
        await editor.getByRole('button', { name: 'Caption and actions', exact: true }).click();
        const caption = editor.getByRole('textbox', { name: 'Image caption', exact: true });
        await caption.scrollIntoViewIfNeeded();
        await expect(caption).toBeInViewport({ ratio: 1 });
        await capture(page, `dataset-editor-${theme}-${size.label}-caption`);
        const save = editor.getByRole('button', { name: 'Save & Next', exact: true });
        await save.scrollIntoViewIfNeeded();
        await expect(save).toBeEnabled();
        await expect(save).toBeInViewport({ ratio: 1 });
        await expect(editor.getByRole('button', { name: 'Next', exact: true })).toBeInViewport({ ratio: 1 });
        await capture(page, `dataset-editor-${theme}-${size.label}-actions`);
        if (size.width === 390) {
          let currentEditor = editor;
          for (const mode of [
            { label: 'Boxes', region: 'Boxes editor' },
            { label: 'Palettes', region: 'Image palette editor' },
            { label: 'JSON', region: 'Caption JSON editor' },
          ]) {
            await currentEditor.getByRole('button', { name: 'Tools', exact: true }).click();
            await page
              .getByRole('menu', { name: 'Workspace tools' })
              .getByRole('menuitem', { name: mode.label, exact: true })
              .click();
            currentEditor = page.getByRole('region', { name: mode.region, exact: true });
            const modeCanvas = await currentEditor.getByRole('main', { name: 'Annotation canvas' }).boundingBox();
            expect(modeCanvas?.width).toBeGreaterThanOrEqual(240);
            expect(modeCanvas?.height).toBeGreaterThanOrEqual(300);
            await currentEditor.getByRole('button', { name: 'Inspector', exact: true }).click();
            const modeInspector = currentEditor.getByRole('complementary', { name: 'Annotation inspector' });
            await modeInspector.scrollIntoViewIfNeeded();
            await expect(modeInspector).toBeVisible();
            if (mode.label === 'JSON') {
              await expect(modeInspector.locator('.monaco-editor')).toBeVisible();
              await expect(modeInspector.getByRole('button', { name: 'Apply JSON', exact: true })).toBeInViewport({
                ratio: 1,
              });
              await expect(modeInspector.getByRole('button', { name: 'Format JSON', exact: true })).toBeInViewport({
                ratio: 1,
              });
              await modeInspector
                .getByRole('navigation', { name: 'JSON sections' })
                .getByRole('button', { name: /Fixture object/ })
                .scrollIntoViewIfNeeded();
            }
            await capture(page, `dataset-editor-${theme}-${size.label}-${mode.label.toLowerCase()}`);
            await currentEditor.getByRole('button', { name: 'Inspector', exact: true }).click();
            await currentEditor.getByRole('button', { name: 'Caption and actions', exact: true }).click();
            const modeSave = currentEditor.getByRole('button', { name: 'Save & Next', exact: true });
            await modeSave.scrollIntoViewIfNeeded();
            await expect(modeSave).toBeEnabled();
            await expect(modeSave).toBeInViewport({ ratio: 1 });
            await capture(page, `dataset-editor-${theme}-${size.label}-${mode.label.toLowerCase()}-actions`);
          }
        }
        expect(errors).toEqual([]);
      } finally {
        await context.close();
      }
    }
  }
}

test.beforeAll(async ({ browser }) => {
  await mkdir(captures, { recursive: true });
  const context = await browser.newContext({ baseURL: origin });
  try {
    const page = await context.newPage();
    await telemetry(page);
    await page.goto('/');
    await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
    await page.getByRole('button', { name: 'Check Password', exact: true }).click();
    await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
    cookies = await context.cookies();
  } finally {
    await context.close();
  }
});

test.beforeEach(async ({ page, context }) => {
  await context.addCookies(cookies);
  await page.setViewportSize({ width: 1440, height: 1024 });
  await page.addInitScript(() => localStorage.setItem('theme', 'dark'));
  await telemetry(page);
  const errors: string[] = [];
  pageErrors.set(page, errors);
  page.on('pageerror', error => errors.push(error.message));
});

test.afterEach(async ({ page }) => {
  expect(pageErrors.get(page)).toEqual([]);
});

async function modelFixtures(page: Page) {
  await page.route('**/api/generate/loras', route =>
    route.fulfill({
      json: {
        loras: [
          {
            id: 'model-a',
            label: 'Aster',
            filename: 'aster.safetensors',
            sizeBytes: 300 * 1024 ** 2,
            updatedAt: '2026-10-20T10:00:00Z',
            triggerWords: ['portrait'],
            jobId: 'aster-job',
          },
          {
            id: 'model-b',
            label: 'Birch',
            filename: 'birch.safetensors',
            sizeBytes: 80 * 1024 ** 2,
            updatedAt: '2026-10-22T10:00:00Z',
            triggerWords: ['product'],
          },
          {
            id: 'model-c',
            label: 'Cedar',
            filename: 'cedar.safetensors',
            sizeBytes: 1536 * 1024 ** 2,
            updatedAt: '2026-10-21T10:00:00Z',
            triggerWords: ['material'],
          },
        ],
      },
    }),
  );
}

test('model library sorts numeric size and update dates, filters triggers, and keeps direct destinations', async ({
  page,
}) => {
  await modelFixtures(page);
  await page.goto('/models');
  const names = page.locator('tbody th[scope="row"]');
  await expect(names).toHaveText(['Birch', 'Cedar', 'Aster']);
  await page.getByRole('button', { name: 'Size', exact: true }).click();
  await expect(names).toHaveText(['Cedar', 'Aster', 'Birch']);
  await page.getByRole('button', { name: 'Size', exact: true }).click();
  await expect(names).toHaveText(['Birch', 'Aster', 'Cedar']);
  await page.getByRole('button', { name: 'Name', exact: true }).click();
  await expect(names).toHaveText(['Aster', 'Birch', 'Cedar']);
  await page.getByLabel('Search models').fill('portrait');
  await expect(names).toHaveText(['Aster']);
  await expect(page.getByRole('link', { name: 'Training run', exact: true })).toHaveAttribute(
    'href',
    '/jobs/aster-job',
  );
  await expect(page.locator('tbody').getByRole('link', { name: 'Generate', exact: true })).toHaveAttribute(
    'href',
    '/generate?model_ref=model-a',
  );
  await page.getByLabel('Search models').fill('');
  await capture(page, 'models-dark-desktop');
  await page.setViewportSize({ width: 390, height: 844 });
  await noPageOverflow(page);
  await expect(page.getByLabel('Search models')).toBeVisible();
  const mobileModel = page.getByRole('article', { name: 'Birch', exact: true });
  await expect(mobileModel).toBeVisible();
  await expect(mobileModel.getByRole('link', { name: 'Generate', exact: true })).toBeInViewport({ ratio: 1 });
  await expect(page.locator('table')).toBeHidden();
  await capture(page, 'models-dark-mobile');
});

test('generation settings resize by keyboard, collapse without losing inputs, and restore layout preferences', async ({
  page,
}) => {
  await page.goto('/generate');
  const separator = page.getByRole('separator', { name: 'Resize generation settings' });
  await expect(separator).toHaveAttribute('aria-valuenow', '352');
  await separator.focus();
  await page.keyboard.press('End');
  await expect(separator).toHaveAttribute('aria-valuenow', '480');
  await page.keyboard.press('ArrowRight');
  await expect(separator).toHaveAttribute('aria-valuenow', '464');
  const prompt = page.getByLabel('Prompt text', { exact: true });
  await prompt.fill('A preserved prompt for the workbench test');
  const settingsToggle = page.getByRole('button', { name: 'Settings', exact: true });
  await settingsToggle.click();
  await expect(page.getByRole('complementary', { name: 'Generation settings' })).toBeHidden();
  await expect(prompt).toHaveValue('A preserved prompt for the workbench test');
  await page.reload();
  await expect(settingsToggle).toHaveAttribute('aria-expanded', 'false');
  await settingsToggle.click();
  await expect(separator).toHaveAttribute('aria-valuenow', '464');
  await capture(page, 'generate-dark-desktop');
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(separator).toBeHidden();
  await expect(settingsToggle).toBeInViewport();
  await expect(page.getByRole('complementary', { name: 'Generation settings' })).toBeVisible();
  await noPageOverflow(page);
  await capture(page, 'generate-dark-mobile');
});

for (const width of [1440, 390]) {
  test(`dataset shortcut help traps focus and isolates editor shortcuts at ${width}px`, async ({ page, browser }) => {
    const dataset = `premium-shortcuts-${randomUUID().slice(0, 8)}`;
    const caption = JSON.stringify({
      high_level_description: 'A fixture composition.',
      compositional_deconstruction: { elements: [{ type: 'obj', bbox: [100, 100, 600, 700], desc: 'Fixture object' }] },
    });
    for (let index = 1; index <= 2; index++) {
      const group = randomUUID();
      for (const [filename, data] of [
        [`asset-${index}.png`, image],
        [`asset-${index}.txt`, Buffer.from(caption)],
      ] as const) {
        const response = await page.request.post('/api/datasets/upload', {
          headers: {
            ...headers,
            'Content-Type': 'application/octet-stream',
            'X-AITK-Dataset-Name': encodeURIComponent(dataset),
            'X-AITK-File-Name': filename,
            'X-AITK-Upload-Group': group,
          },
          data,
        });
        expect(response.status()).toBe(200);
      }
    }
    await page.goto(`/datasets/${dataset}`);
    await expect(page.getByRole('textbox', { name: 'Image caption', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: 'Layers', exact: true }).click();
    const editor = page.getByRole('region', { name: 'Layers editor', exact: true });
    await expect(editor).toBeVisible();
    const layer = editor.getByRole('button', { name: /^Fixture object Object/ });
    await layer.click();
    const filmstrip = editor.getByRole('navigation', { name: 'Dataset filmstrip' });
    const selectedAsset = await filmstrip.locator('[aria-current="true"]').getAttribute('aria-label');
    let saves = 0;
    await page.route('**/api/img/caption', route => {
      if (route.request().method() === 'POST') saves++;
      return route.continue();
    });
    await page.setViewportSize({ width, height: width === 390 ? 844 : 1024 });
    const inspector = editor.getByRole('complementary', { name: 'Annotation inspector' });
    const originalY = await inspector.getByRole('spinbutton', { name: 'y', exact: true }).inputValue();
    await inspector.focus();
    for (const key of ['ArrowDown', 'ArrowRight', 'Delete', 'Control+d']) await page.keyboard.press(key);
    await expect(layer).toHaveCount(1);
    await expect(inspector.getByRole('spinbutton', { name: 'y', exact: true })).toHaveValue(originalY);
    await expect(inspector.getByRole('spinbutton', { name: 'x', exact: true })).toHaveValue('100');
    await capture(page, `dataset-editor-dark-${width}`);
    const shortcutButton = editor.getByRole('button', { name: 'Shortcuts', exact: true });
    await shortcutButton.click();
    const modal = page.getByRole('dialog', { name: 'Dataset shortcuts', exact: true });
    await expect(modal.getByRole('heading', { name: 'Dataset shortcuts', exact: true })).toBeVisible();
    const headingBounds = await modal.getByRole('heading', { name: 'Dataset shortcuts', exact: true }).boundingBox();
    expect(headingBounds?.y).toBeGreaterThanOrEqual(0);
    await expect(modal.getByText('Save and advance to the next image', { exact: true })).toBeVisible();
    // The header close button is outside the help body: global shortcuts must still ignore it.
    await modal.getByRole('button', { name: 'Close modal', exact: true }).focus();
    for (const key of ['Control+s', 'Control+Enter', 'Delete', 'ArrowRight', 'Control+d'])
      await page.keyboard.press(key);
    expect(saves).toBe(0);
    await page.keyboard.press('Tab');
    expect(await modal.evaluate(dialog => dialog.contains(document.activeElement))).toBe(true);
    await capture(page, `dataset-shortcuts-${width}`);
    await page.keyboard.press('Escape');
    await expect(modal).toHaveCount(0);
    await expect(shortcutButton).toBeFocused();
    // Headless UI correctly removes the background from the accessibility tree while open.
    // Verify the untouched editor after closing instead of querying hidden background roles.
    await expect(filmstrip.locator('[aria-current="true"]')).toHaveAttribute('aria-label', selectedAsset!);
    await expect(layer).toHaveCount(1);
    await noPageOverflow(page);
    if (width === 1440) await captureAdvancedLayouts(browser, dataset);
  });
}

test('settings recover from initial load failure before enabling edits', async ({ page }) => {
  let unavailable = true;
  await page.route('**/api/settings', route => {
    if (route.request().method() === 'GET' && unavailable) {
      return route.fulfill({ status: 503, json: { error: 'Fixture temporarily unavailable' } });
    }
    return route.continue();
  });
  await page.goto('/settings');
  const failure = page.getByRole('alert').filter({ hasText: 'Settings could not be loaded' });
  await expect(failure).toBeVisible();
  await expect(page.getByLabel('Training folder', { exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeDisabled();
  unavailable = false;
  await failure.getByRole('button', { name: 'Retry', exact: true }).click();
  await expect(failure).toHaveCount(0);
  await expect(page.getByLabel('Training folder', { exact: true })).toBeEnabled();
  await expect(page.getByLabel('Training folder', { exact: true })).not.toHaveValue('');
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeDisabled();
});

test('settings keep failed edits visible for retry and expose responsive save and section controls', async ({
  page,
}) => {
  await page.goto('/settings');
  const trainingFolder = page.getByLabel('Training folder', { exact: true });
  await expect(trainingFolder).not.toHaveValue('');
  const editedFolder = `${await trainingFolder.inputValue()}-unsaved-fixture`;
  await trainingFolder.fill(editedFolder);
  let saves = 0;
  await page.route('**/api/settings', route => {
    if (route.request().method() !== 'POST') return route.continue();
    saves++;
    return route.fulfill(
      saves === 1 ? { status: 500, json: { error: 'Fixture save failure' } } : { json: { success: true } },
    );
  });
  await page.getByRole('button', { name: 'Save settings', exact: true }).click();
  const failure = page.getByRole('alert').filter({ hasText: 'Settings could not be saved' });
  await expect(failure).toBeVisible();
  // Regression: the previous finally timer erased failure feedback after two seconds.
  await page.waitForTimeout(2200);
  await expect(failure).toBeVisible();
  await expect(trainingFolder).toHaveValue(editedFolder);
  await failure.getByRole('button', { name: 'Retry', exact: true }).click();
  await expect(failure).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeDisabled();
  expect(saves).toBe(2);
  await page
    .getByRole('navigation', { name: 'Settings sections' })
    .getByRole('button', { name: 'Advanced', exact: true })
    .click();
  await expect(page.getByRole('switch', { name: 'Offline mode', exact: true })).toBeInViewport();
  await capture(page, 'settings-dark-desktop');
  await page.setViewportSize({ width: 390, height: 844 });
  await noPageOverflow(page);
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeVisible();
  await expect(
    page.getByRole('navigation', { name: 'Settings sections' }).getByRole('button', { name: 'Storage', exact: true }),
  ).toBeInViewport();
  await capture(page, 'settings-dark-mobile');
});

test('settings preserve newer local edits when an earlier save completes', async ({ page }) => {
  await page.goto('/settings');
  const folder = page.getByLabel('Training folder', { exact: true });
  await expect(folder).not.toHaveValue('');
  const original = await folder.inputValue();
  const submitted = `${original}-first-edit`;
  const newer = `${original}-newer-edit`;
  const token = page.getByLabel('Hugging Face token', { exact: true });
  const otherToken = page.getByLabel('OpenRouter API key', { exact: true });
  let finishSave: (() => Promise<void>) | undefined;
  const requests: unknown[] = [];
  await page.route('**/api/settings', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    requests.push(route.request().postDataJSON());
    if (requests.length > 1) return route.fulfill({ json: { success: true } });
    await new Promise<void>(resolve => {
      finishSave = async () => {
        await route.fulfill({ json: { success: true } });
        resolve();
      };
    });
  });
  await folder.fill(submitted);
  await token.fill('fixture-token-first');
  await otherToken.fill('fixture-openrouter-submitted');
  await page.getByRole('button', { name: 'Save settings', exact: true }).click();
  await expect.poll(() => Boolean(finishSave)).toBe(true);
  await folder.fill(newer);
  await token.fill('fixture-token-newer');
  await finishSave!();
  await expect(folder).toHaveValue(newer);
  await expect(token).toHaveValue('fixture-token-newer');
  await expect(otherToken).toHaveValue('');
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeEnabled();
  expect(requests[0]).toMatchObject({ TRAINING_FOLDER: submitted, HF_TOKEN: 'fixture-token-first' });
  await page.getByRole('button', { name: 'Save settings', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeDisabled();
  expect(requests[1]).toMatchObject({
    TRAINING_FOLDER: newer,
    HF_TOKEN: 'fixture-token-newer',
    OPENROUTER_API_KEY: '',
  });
  await expect(token).toHaveValue('');
  expect(await page.evaluate(() => JSON.stringify(localStorage))).not.toContain('fixture-token');
});

test('command search supports Ctrl+K, Escape focus restoration, and link-based navigation', async ({ page }) => {
  await page.goto('/models');
  const trigger = page.getByRole('button', { name: 'Search commands', exact: true });
  await trigger.focus();
  await page.keyboard.press('Control+k');
  const dialog = page.getByRole('dialog', { name: 'Search commands', exact: true });
  const search = dialog.getByRole('textbox', { name: 'Search commands', exact: true });
  await expect(search).toBeFocused();
  await search.fill('Datasets');
  await expect(dialog.getByRole('link', { name: /^Datasets/ })).toHaveAttribute('href', '/datasets');
  await page.setViewportSize({ width: 390, height: 600 });
  await search.fill('');
  for (let index = 0; index < 9; index++) await search.press('ArrowDown');
  await expect(search).toHaveAccessibleDescription('Selected: Watermark. Press Enter to open.');
  await expect(dialog.getByRole('link', { name: /^Watermark/ })).toBeInViewport();
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await page.keyboard.press('Control+k');
  await search.fill('Datasets');
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/datasets$/);
  await expect(page.getByRole('heading', { name: 'Datasets', exact: true })).toBeVisible();
  await expect(dialog).toHaveCount(0);
});

function runFixture(id: string, name: string, status: string, step: number, total: number, worker = 'local'): Job {
  return {
    id,
    name,
    status,
    step,
    worker_id: worker,
    remote_job_id: worker === 'local' ? null : id,
    remote_sync_at: null,
    remote_error: null,
    gpu_ids: '0',
    job_type: 'train',
    job_ref: null,
    stop: false,
    return_to_queue: false,
    pid: null,
    info: status === 'running' ? 'Training in progress' : '',
    speed_string: status === 'running' ? '2.4 s/step' : '',
    queue_position: status === 'queued' ? 1 : 0,
    save_now: false,
    sample_now: false,
    created_at: '2026-10-22T10:00:00Z',
    updated_at: '2026-10-22T10:21:12Z',
    job_config: JSON.stringify({
      job: 'extension',
      config: {
        name,
        process: [
          {
            type: 'diffusion_trainer',
            model: { arch: 'flux', name_or_path: 'fixture/no-weights' },
            train: { steps: total },
            save: { save_every: 500 },
            datasets: [],
            sample: { prompts: [] },
          },
        ],
      },
    }),
  };
}

async function monitorFixtures(page: Page) {
  const jobs = [
    runFixture('portrait', 'portrait_flux_v3', 'running', 1560, 3000),
    runFixture('product', 'product_qwen_v2', 'running', 820, 2000, 'lab-01'),
    runFixture('style', 'style_flux_v2', 'queued', 0, 2400),
    runFixture('material', 'material_study', 'queued', 0, 1800, 'lab-01'),
    runFixture('character', 'character_lora_v4', 'completed', 3000, 3000),
  ];
  await page.route('**/api/jobs?*', route => {
    const query = new URL(route.request().url()).searchParams;
    const id = query.get('id');
    const pageJobs =
      query.get('view') === 'active'
        ? jobs.filter(job => ['running', 'queued', 'stopping'].includes(job.status))
        : query.get('view') === 'history'
          ? jobs.filter(job => !['running', 'queued', 'stopping'].includes(job.status))
          : jobs;
    return route.fulfill({
      json: id
        ? (jobs.find(job => job.id === id) ?? jobs[0])
        : {
            jobs: pageJobs,
            nextCursor: null,
            statusCounts: { running: 2, queued: 2, completed: 1 },
            freshness: { updatedAt: '2026-10-22T10:21:12Z', failed: 0 },
          },
    });
  });
  await page.route('**/api/queue', route =>
    route.fulfill({
      json: {
        queues: [
          { id: 1, worker_id: 'local', gpu_ids: '0', is_running: true },
          { id: 2, worker_id: 'lab-01', gpu_ids: '0', is_running: true },
        ],
      },
    }),
  );
  await page.route('**/api/workers', route =>
    route.fulfill({
      json: {
        workers: [
          {
            id: 'lab-01',
            name: 'lab-01',
            base_url: 'http://lab-01.invalid',
            enabled: true,
            offline_bypass_enabled: false,
            last_status: 'online',
            last_error: null,
            last_checked_at: '2026-10-22T10:21:00Z',
            capabilities: '{}',
            gpus: '[]',
            created_at: '2026-10-22T10:00:00Z',
            updated_at: '2026-10-22T10:21:00Z',
          },
        ],
      },
    }),
  );
  const points = Array.from({ length: 157 }, (_, index) => ({
    step: index * 10,
    value: 0.08 + 0.39 * Math.exp(-index / 24) + 0.004 * Math.sin(index * 1.7),
    wall_time: 1792663200 + index * 24,
  }));
  await page.route('**/api/jobs/*/metrics*', route =>
    route.fulfill({
      json: {
        keys: ['loss/loss'],
        keyInfo: [{ key: 'loss/loss', first_seen_step: 0, last_seen_step: 1560 }],
        series: {
          'loss/loss': {
            key: 'loss/loss',
            totalCount: points.length,
            firstStep: 0,
            lastStep: 1560,
            latest: points.at(-1),
            downsampled: false,
            points,
          },
        },
      },
    }),
  );
  await page.route('**/api/jobs/*/log*', route =>
    route.fulfill({
      json: {
        log: '[06:21:04] step 1558 · loss 0.084\n[06:21:09] Checkpoint saved at step 1560\n[06:21:12] step 1560 · loss 0.081',
        offset: null,
        reset: true,
      },
    }),
  );
  await page.route('**/api/jobs/*/samples*', route => route.fulfill({ json: { images: [] } }));
}

test('monitor renders the selected workbench in both themes across desktop, tablet, mobile, and 200 percent reflow', async ({
  page,
  browser,
}) => {
  await monitorFixtures(page);
  await page.goto('/dashboard');
  await expect(page.getByRole('heading', { name: 'Runs', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'All runs: 5', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Active: 4', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Recent: 1', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Failed: 0', exact: true })).toBeVisible();
  await expect(page.getByRole('progressbar', { name: 'portrait_flux_v3 progress', exact: true })).toHaveAttribute(
    'aria-valuenow',
    '52',
  );
  await expect(page.getByText('portrait_flux_v3', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('NVIDIA RTX 5090', { exact: true }).first()).toBeVisible();
  for (const theme of ['dark', 'light'] as const) {
    if (theme === 'light') {
      await page.getByRole('button', { name: 'More', exact: true }).click();
      await page.getByTitle('Switch to light mode', { exact: true }).click();
      await page.keyboard.press('Escape');
      await expect(page.locator('html')).not.toHaveClass(/dark/);
    }
    for (const viewport of [
      { width: 1440, height: 1024 },
      { width: 1487, height: 1058 },
      { width: 1280, height: 900 },
      { width: 1024, height: 768 },
      { width: 390, height: 844 },
    ]) {
      await page.setViewportSize(viewport);
      await noPageOverflow(page);
      await expect(page.getByRole('navigation', { name: 'Main navigation' })).toBeVisible();
      await capture(page, `monitor-${theme}-${viewport.width}`);
    }
  }
  // 720 CSS pixels at 2x device scale is the reflow footprint of a 1440px desktop at 200% zoom.
  for (const theme of ['dark', 'light'] as const) {
    const zoomContext = await browser.newContext({
      baseURL: origin,
      viewport: { width: 720, height: 512 },
      deviceScaleFactor: 2,
    });
    try {
      await zoomContext.addCookies(cookies);
      await zoomContext.addInitScript(value => localStorage.setItem('theme', value), theme);
      const zoomPage = await zoomContext.newPage();
      const zoomErrors: string[] = [];
      zoomPage.on('pageerror', error => zoomErrors.push(error.message));
      await telemetry(zoomPage);
      await monitorFixtures(zoomPage);
      await zoomPage.goto('/dashboard');
      await noPageOverflow(zoomPage);
      await expect(zoomPage.getByRole('heading', { name: 'Runs', exact: true })).toBeVisible();
      await expect(zoomPage.getByText('portrait_flux_v3', { exact: true }).first()).toBeVisible();
      await capture(zoomPage, `monitor-${theme}-200-percent-reflow`);
      expect(zoomErrors).toEqual([]);
    } finally {
      await zoomContext.close();
    }
  }
  await page.setViewportSize({ width: 1440, height: 1024 });
  await page.route('**/api/gpu?*', route => route.fulfill({ status: 503, json: { error: 'Fixture GPU unavailable' } }));
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Runs', exact: true })).toBeVisible();
  await expect(page.getByText('portrait_flux_v3', { exact: true }).first()).toBeVisible();
  await capture(page, 'monitor-hardware-unavailable');
  // Active/history polls can overlap during a status transition; one ID must remain one visible run.
  const overlap = runFixture('portrait', 'portrait_flux_v3', 'running', 1560, 3000);
  await page.route('**/api/jobs?*', route => {
    const id = new URL(route.request().url()).searchParams.get('id');
    return route.fulfill({ json: id ? overlap : { jobs: [overlap], nextCursor: null, statusCounts: { running: 1 } } });
  });
  await page.reload();
  await expect(page.locator('tbody').getByRole('button', { name: 'portrait_flux_v3', exact: true })).toHaveCount(1);
});

test('library and utility workspaces retain usable controls in both themes and responsive sizes', async ({
  browser,
}) => {
  test.setTimeout(240_000);
  const datasets = [
    { name: 'Portrait studies', encrypted: false, itemCount: 48, captionedItemCount: 48, missingCaptionCount: 0 },
    { name: 'Product materials', encrypted: false, itemCount: 126, captionedItemCount: 119, missingCaptionCount: 7 },
    { name: 'Private collection', encrypted: true },
  ] satisfies DatasetSummary[];
  const surfaces = [
    { path: '/models', heading: 'Models', name: 'models' },
    { path: '/generate', heading: 'Generate', name: 'generate' },
    { path: '/settings', heading: 'Settings', name: 'settings' },
    { path: '/datasets', heading: 'Datasets', name: 'datasets' },
    { path: '/tools', heading: 'Tools', name: 'tools' },
    { path: '/workflows', heading: 'Workflows', name: 'workflows' },
  ] as const;
  for (const theme of ['dark', 'light'] as const) {
    for (const size of [
      { width: 1440, height: 1024, scale: 1, label: '1440' },
      { width: 1024, height: 768, scale: 1, label: '1024' },
      { width: 390, height: 844, scale: 1, label: '390' },
      { width: 720, height: 512, scale: 2, label: '200-percent-reflow' },
    ]) {
      const context = await browser.newContext({
        baseURL: origin,
        viewport: { width: size.width, height: size.height },
        deviceScaleFactor: size.scale,
      });
      try {
        await context.addCookies(cookies);
        await context.addInitScript(value => localStorage.setItem('theme', value), theme);
        const page = await context.newPage();
        const errors: string[] = [];
        page.on('pageerror', error => errors.push(error.message));
        await telemetry(page);
        await modelFixtures(page);
        await page.route('**/api/datasets/list*', route => route.fulfill({ json: { datasets, errors: [] } }));
        for (const surface of surfaces) {
          await page.goto(surface.path);
          await expect(page.getByRole('heading', { name: surface.heading, exact: true })).toBeVisible();
          if (theme === 'dark') await expect(page.locator('html')).toHaveClass(/dark/);
          else await expect(page.locator('html')).not.toHaveClass(/dark/);
          await noPageOverflow(page);
          switch (surface.name) {
            case 'models':
              await expect(page.locator('tbody th[scope="row"]')).toHaveText(['Birch', 'Cedar', 'Aster']);
              await expect(page.getByLabel('Search models')).toBeInViewport({ ratio: 1 });
              break;
            case 'generate':
              await expect(page.getByRole('combobox', { name: 'GPU', exact: true })).toBeEnabled();
              await expect(page.getByRole('combobox', { name: 'Architecture', exact: true })).toBeEnabled();
              await expect(page.locator('.studio-page-topbar .aitk-react-select__single-value')).toHaveText('GPU #0');
              expect(
                await page.locator('.studio-page-topbar .aitk-react-select__single-value').evaluate(element => {
                  const range = document.createRange();
                  range.selectNodeContents(element);
                  const text = range.getBoundingClientRect();
                  const label = element.getBoundingClientRect();
                  const container = element.parentElement!.getBoundingClientRect();
                  return (
                    text.left >= Math.max(label.left, container.left) - 1 &&
                    text.right <= Math.min(label.right, container.right) + 1
                  );
                }),
              ).toBe(true);
              await expect(page.getByLabel('Prompt text', { exact: true })).toBeVisible();
              await expect(page.getByRole('button', { name: 'Settings', exact: true })).toBeInViewport({ ratio: 1 });
              break;
            case 'settings':
              await expect(page.getByLabel('Training folder', { exact: true })).not.toHaveValue('');
              await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeInViewport({
                ratio: 1,
              });
              await expect(page.getByRole('navigation', { name: 'Settings sections' })).toBeVisible();
              {
                const navBounds = await page.getByRole('navigation', { name: 'Settings sections' }).boundingBox();
                const toolbarBounds = await page.locator('.studio-page-topbar').boundingBox();
                expect(navBounds!.y - (toolbarBounds!.y + toolbarBounds!.height)).toBeLessThanOrEqual(24);
                const diagnostics = page.locator('summary').filter({ hasText: 'Runtime diagnostics' });
                await expect(diagnostics).toBeInViewport({ ratio: 1 });
                expect(
                  await diagnostics.evaluate(element => {
                    const bounds = element.getBoundingClientRect();
                    const top = document.elementFromPoint(bounds.x + 10, bounds.y + bounds.height / 2);
                    return top === element || Boolean(top && element.contains(top));
                  }),
                ).toBe(true);
              }
              break;
            case 'datasets':
              await expect(page.getByLabel('Search datasets')).toBeInViewport({ ratio: 1 });
              await expect(page.getByRole('button', { name: 'Create dataset', exact: true })).toBeInViewport({
                ratio: 1,
              });
              await expect(page.getByRole('link', { name: 'Portrait studies', exact: true })).toBeVisible();
              if (size.width === 390) {
                const mobileDataset = page.getByRole('article', { name: 'Portrait studies', exact: true });
                await expect(mobileDataset).toBeVisible();
                await expect(mobileDataset.getByRole('link', { name: 'Train', exact: true })).toBeInViewport({
                  ratio: 1,
                });
              }
              if (theme === 'dark' && size.width === 1440) {
                await page.getByRole('button', { name: 'Icons', exact: true }).click();
                await page.reload();
                await expect(page.getByRole('button', { name: 'Icons', exact: true })).toHaveAttribute(
                  'aria-pressed',
                  'true',
                );
                await page.getByRole('button', { name: 'Details', exact: true }).click();
              }
              expect(
                await page.getByRole('link', { name: 'Decrypt to plain dataset', exact: true }).evaluate(element => {
                  const bounds = element.getBoundingClientRect();
                  const parent = element.parentElement!.getBoundingClientRect();
                  return bounds.left >= parent.left - 1 && bounds.right <= parent.right + 1;
                }),
              ).toBe(true);
              expect(
                await page.getByRole('link', { name: 'Decrypt to plain dataset', exact: true }).evaluate(element => {
                  const actions = Array.from(element.parentElement!.children).map(
                    child => child.getBoundingClientRect().top,
                  );
                  return Math.max(...actions) - Math.min(...actions) <= 1;
                }),
              ).toBe(true);
              break;
            case 'tools':
              await expect(page.getByRole('link', { name: /Workflows Build and run/ })).toHaveAttribute(
                'href',
                '/workflows',
              );
              await expect(page.getByRole('link', { name: /Watermark Manage and verify/ })).toBeInViewport({
                ratio: 1,
              });
              break;
            case 'workflows':
              await expect(page.getByRole('link', { name: /Ideogram Workflow Builder Ideogram 4/ })).toHaveAttribute(
                'href',
                '/workflows/ideogram',
              );
              break;
          }
          await capture(page, `${surface.name}-${theme}-${size.label}`);
          if (surface.name === 'models' && size.width < 800) {
            const modelList = size.width < 640 ? page.getByRole('article') : page.locator('tbody');
            const generate = modelList.getByRole('link', { name: 'Generate', exact: true }).first();
            await generate.focus();
            await expect(generate).toBeInViewport({ ratio: 1 });
            await capture(page, `models-${theme}-${size.label}-actions`);
          }
          if (surface.name === 'generate' && size.width < 1024) {
            await page.getByLabel('Prompt text', { exact: true }).scrollIntoViewIfNeeded();
            await capture(page, `generate-${theme}-${size.label}-prompt`);
            await page.getByRole('combobox', { name: 'Architecture', exact: true }).scrollIntoViewIfNeeded();
            await capture(page, `generate-${theme}-${size.label}-settings`);
          }
        }
        expect(errors).toEqual([]);
      } finally {
        await context.close();
      }
    }
  }
});
