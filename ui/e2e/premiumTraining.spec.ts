import { test, expect, type BrowserContext, type Page } from '@playwright/test';
import type { editor } from 'monaco-editor';
import { installLocalMonaco } from './localMonaco';
import YAML from 'yaml';
import { defaultJobConfig } from '../src/app/jobs/new/jobConfig';
import type { JobConfig } from '../src/types';

let sessionCookies: Awaited<ReturnType<BrowserContext['cookies']>> = [];

async function editorSnapshot(page: Page) {
  return page.evaluate(() => {
    const monaco = (window as Window & { monaco?: { editor: { getEditors(): editor.ICodeEditor[] } } }).monaco;
    const instance = monaco?.editor.getEditors()[0];
    return { text: instance?.getValue(), position: instance?.getPosition(), selection: instance?.getSelection() };
  });
}

async function pasteEditorText(page: Page, text: string) {
  const input = page.locator('[data-config-editor] textarea.inputarea');
  await expect(input).toBeVisible();
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.evaluate(value => navigator.clipboard.writeText(value), text);
  await input.focus();
  await page.keyboard.press('Control+a');
  await page.keyboard.press('Control+v');
}

async function expectReadableComputeValues(page: Page) {
  for (const [name, text] of [['Compute', 'Local worker'], ['Trainer', 'LoRA Trainer'], ['GPU', 'GPU #0']]) {
    await expect.poll(async () => page.getByRole('combobox', { name, exact: true }).evaluate(input => {
      const label = input.closest('.aitk-react-select__control')?.querySelector<HTMLElement>('.aitk-react-select__single-value');
      if (!label) return null;
      const range = document.createRange();
      range.selectNodeContents(label);
      const textBounds = range.getBoundingClientRect();
      const bounds = label.getBoundingClientRect();
      return { text: label.textContent, fits: textBounds.left >= bounds.left - 1 && textBounds.right <= bounds.right + 1 };
    })).toEqual({ text, fits: true });
  }
}

async function installTrainingFixtures(page: Page) {
  await page.route('**/api/datasets/list*', route => route.fulfill({ json: {
    datasets: [{ name: 'portraits_v2', encrypted: false }], errors: [],
  } }));
  await installLocalMonaco(page);
  await page.route('**/api/monitor', route => route.abort());
  await page.route('**/api/gpu?*', route => route.fulfill({ json: {
    hasNvidiaSmi: true, isMac: false, gpus: [{ index: 0, name: 'Fixture GPU', driverVersion: 'fixture', temperature: 48,
      utilization: { gpu: 0, memory: 0 }, memory: { total: 24576, free: 22528, used: 2048 },
      power: { draw: 35, limit: 350 }, clocks: { graphics: 1000, memory: 7000 }, fan: { speed: 30 } }],
  } }));
}

test.beforeEach(async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1024 });
  await installTrainingFixtures(page);
  if (sessionCookies.length) await page.context().addCookies(sessionCookies);
  await page.goto('/');
  if (!sessionCookies.length) {
    await page.getByLabel('Password', { exact: true }).fill('browser-fixture-local-only');
    await page.getByRole('button', { name: 'Check Password' }).click();
    await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0);
    sessionCookies = (await page.context().cookies()).filter(cookie => cookie.name === 'aitk_session');
  }
  await page.goto('/jobs/new');
  await expect(page.getByLabel('Training name', { exact: true })).toBeEnabled();
  // React Select is loaded separately with SSR disabled; a text field alone
  // does not mean the form is ready for interaction or a reference capture.
  await expect(page.getByRole('combobox', { name: 'Model architecture', exact: true })).toBeEnabled();
  await expect(page.getByRole('combobox', { name: 'Compute', exact: true })).toBeVisible();
});

for (const extension of ['json', 'jsonc']) {
  test(`${extension} imports preserve URLs and literal comment text`, async ({ page }) => {
    const config = structuredClone(defaultJobConfig);
    config.config.name = `import_${extension}_fixture`;
    config.config.process[0].sample.samples = [{ prompt: 'a sign saying /* KEEP THIS */ and \\"quoted\\"' }];
    if (extension === 'json') config.config.process[0].model.name_or_path = 'https://models.invalid/base-model';
    const text = `${extension === 'jsonc' ? '// configuration comment\n/* block comment */\n' : ''}${JSON.stringify(config, null, 2)}`;
    await page.locator('input[type="file"][accept=".yaml,.yml,.json,.jsonc"]').setInputFiles({
      name: `fixture.${extension}`, mimeType: 'application/json', buffer: Buffer.from(text),
    });
    await expect(page.getByLabel('Training name', { exact: true })).toHaveValue(config.config.name);
    await page.getByRole('tab', { name: 'Config', exact: true }).click();
    await expect(page.locator('[data-config-editor] textarea.inputarea')).toBeVisible();
    const imported = YAML.parse((await editorSnapshot(page)).text ?? '') as JobConfig;
    expect(imported.config.process[0].model.name_or_path).toEqual(config.config.process[0].model.name_or_path);
    expect(imported.config.process[0].sample.samples).toEqual(config.config.process[0].sample.samples);
  });
}

test('correcting a Config-only structure restores the Form editor', async ({ page }) => {
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  await expect(page.locator('[data-config-editor] textarea.inputarea')).toBeVisible();
  const original = (await editorSnapshot(page)).text ?? '';
  const config = YAML.parse(original) as JobConfig;
  Reflect.deleteProperty(config.config.process[0].train, 'optimizer_params');
  await pasteEditorText(page, YAML.stringify(config));
  await page.getByRole('tab', { name: 'Form', exact: true }).click();
  await expect(page.getByText('This configuration requires the Config editor.', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Open Config', exact: true }).click();
  await pasteEditorText(page, original);
  await page.getByRole('tab', { name: 'Form', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toBeEnabled();
  await expect(page.getByLabel('Weight decay', { exact: true })).toBeVisible();
});

for (const mode of ['Config', 'import']) {
test(`explicit Orbit configuration settings survive applying a complete ${mode} document`, async ({ page }) => {
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  await expect(page.locator('[data-config-editor] textarea.inputarea')).toBeVisible();
  const config = YAML.parse((await editorSnapshot(page)).text ?? '') as JobConfig;
  const process = config.config.process[0];
  process.model.arch = 'flux2';
  process.model.name_or_path = 'black-forest-labs/FLUX.2-dev';
  process.model.low_vram = true;
  process.model.quantize = true;
  process.model.qtype = 'orbit4';
  process.train.batch_size = 3;
  process.train.optimizer = 'adamw';
  process.train.cache_text_embeddings = false;
  process.train.unload_text_encoder = false;
  const rawText = YAML.stringify(config);
  if (mode === 'Config') await pasteEditorText(page, rawText);
  else await page.locator('input[type="file"][accept=".yaml,.yml,.json,.jsonc"]').setInputFiles({
    name: 'orbit.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(config)),
  });
  await page.getByRole('tab', { name: 'Form', exact: true }).click();
  await expect(page.getByLabel('Batch size', { exact: true })).toHaveValue('3');
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  // Config retains the user's exact document, including omitted additive
  // defaults. Verify the transformed configuration at its actual save boundary.
  if (mode === 'Config') expect((await editorSnapshot(page)).text).toEqual(rawText);
  await expect(page.getByRole('button', { name: 'Create Job', exact: true })).toBeEnabled();
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    return route.fulfill({ status: 503, json: { error: 'Fixture Orbit save unavailable' } });
  });
  const posted = page.waitForRequest(request => request.url().endsWith('/api/jobs') && request.method() === 'POST');
  await page.getByRole('button', { name: 'Create Job', exact: true }).click();
  const payload = (await posted).postDataJSON() as { job_config: JobConfig };
  const applied = payload.job_config.config.process[0];
  expect(applied.train.batch_size).toBe(3);
  expect(applied.train.optimizer).toBe('adamw');
  expect(applied.train.cache_text_embeddings).toBe(false);
  expect(applied.train.unload_text_encoder).toBe(false);
  expect(applied.model.layer_offloading).toBe(true);
  expect(applied.model.layer_offloading_transformer_percent).toBe(0.7);
  expect(applied.model.layer_offloading_text_encoder_percent).toBe(0.5);
  await expect(page.getByRole('alert').filter({ hasText: 'Fixture Orbit save unavailable' })).toBeVisible();
});
}

test('cyclic YAML imports report an error and preserve the current editable draft', async ({ page }) => {
  await page.getByLabel('Training name', { exact: true }).fill('retained_after_bad_import');
  const config = structuredClone(defaultJobConfig);
  const text = `${YAML.stringify(config)}\nextra: &recursive\n  nested: *recursive\n`;
  await page.locator('input[type="file"][accept=".yaml,.yml,.json,.jsonc"]').setInputFiles({
    name: 'recursive.yaml', mimeType: 'text/yaml', buffer: Buffer.from(text),
  });
  await expect(page.getByText('Configuration cannot be serialized. Remove circular references or excessively nested values.', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('retained_after_bad_import');
  await expect(page.getByRole('button', { name: 'Create Job', exact: true })).toBeEnabled();
});

test('query-only training navigation renews the dirty guard while retaining the draft', async ({ page }) => {
  await page.goto('/jobs/new?dataset=portraits_v2');
  await expect(page.getByLabel('Training name', { exact: true })).toBeEnabled();
  await page.getByLabel('Training name', { exact: true }).fill('query_draft_before');
  await page.getByRole('link', { name: 'New training', exact: true }).click();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs\/new$/);
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('query_draft_before');
  await page.getByLabel('Training name', { exact: true }).fill('query_draft_after');
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Train', exact: true }).click();
  await expect(page.getByText('Leave training setup?', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs\/new$/);
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('query_draft_after');
});

test('reverting a draft clears stale recovery and does not resurrect canceled edits on return', async ({ page }) => {
  const originalName = await page.getByLabel('Training name', { exact: true }).inputValue();
  await page.getByLabel('Training name', { exact: true }).fill('reverted_draft_fixture');
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Train', exact: true }).click();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  expect(await page.evaluate(() => sessionStorage.getItem('aitk:training-draft:v1:new') !== null)).toBe(true);
  await page.getByLabel('Training name', { exact: true }).fill(originalName);
  await expect.poll(() => page.evaluate(() => sessionStorage.getItem('aitk:training-draft:v1:new') !== null)).toBe(false);
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Train', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs$/);
  await page.goBack();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue(originalName);
  await expect(page.getByRole('region', { name: 'Recover training draft', exact: true })).toHaveCount(0);
});

test('saving clears parsed and raw drafts before opening another new training job', async ({ page }) => {
  const originalName = await page.getByLabel('Training name', { exact: true }).inputValue();
  await page.getByLabel('Training name', { exact: true }).fill('saved_lifecycle_fixture');
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  const input = page.locator('[data-config-editor] textarea.inputarea');
  await expect(input).toBeVisible();
  await input.focus();
  await page.keyboard.press('Control+End');
  await page.keyboard.insertText('\n# saved-memory-only-fixture');
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    return route.fulfill({ json: { id: 'saved-lifecycle-fixture' } });
  });
  await page.getByRole('button', { name: 'Create Job', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs\/saved-lifecycle-fixture$/);
  expect(await page.evaluate(() => sessionStorage.getItem('aitk:training-draft:v1:new') !== null)).toBe(false);
  await page.getByRole('link', { name: 'New training', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toHaveValue(originalName);
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  await expect(input).toBeVisible();
  expect((await editorSnapshot(page)).text).not.toContain('saved-memory-only-fixture');
  expect((await editorSnapshot(page)).text).not.toContain('saved_lifecycle_fixture');
  await expect(page.getByRole('button', { name: 'Create Job', exact: true })).toBeEnabled();
});

test('source jobs load before editing and a failed clone load can retry without default-state saves', async ({ page }) => {
  const source = structuredClone(defaultJobConfig);
  source.config.name = 'loaded_source';
  source.config.process[0].train.optimizer_params = { weight_decay: 0.017, betas: [0.8, 0.95] };
  source.config.process[0].datasets[0].folder_path = 'fixture/source-dataset';
  let release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  let requests = 0;
  await page.route('**/api/jobs?*', async route => {
    if (new URL(route.request().url()).searchParams.get('id') !== 'delayed-source') return route.continue();
    requests += 1;
    if (requests === 1) {
      await pending;
      return route.fulfill({ status: 503, json: { error: 'Fixture source temporarily unavailable' } });
    }
    return route.fulfill({ json: { job_type: 'train', job_config: JSON.stringify(source), gpu_ids: '0', worker_id: 'local' } });
  });
  try {
    await page.goto('/jobs/new?cloneId=delayed-source');
    await expect(page.getByText('Loading source training job', { exact: true })).toBeVisible();
    await expect(page.getByLabel('Training name', { exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Create Job', exact: true })).toHaveCount(0);
    release();
    await page.getByRole('button', { name: 'Retry source job', exact: true }).click();
    await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('loaded_source_copy');
    await page.getByRole('tab', { name: 'Config', exact: true }).click();
    await expect(page.locator('[data-config-editor] textarea.inputarea')).toBeVisible();
    const loaded = YAML.parse((await editorSnapshot(page)).text ?? '') as JobConfig;
    expect(loaded.config.process[0].train.optimizer_params).toEqual(source.config.process[0].train.optimizer_params);
    expect(requests).toBe(2);
  } finally {
    release();
  }
});

test('valid advanced Config values survive Form edits and a pending save locks both views until failure', async ({ page }) => {
  await page.getByRole('link', { name: 'Training', exact: true }).click();
  await expect.poll(() => page.locator('#training-form-view').evaluate(element => element.scrollTop)).toBeGreaterThan(0);
  const formScroll = await page.locator('#training-form-view').evaluate(element => element.scrollTop);
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  const input = page.locator('[data-config-editor] textarea.inputarea');
  await expect(input).toBeVisible();
  const config = YAML.parse((await editorSnapshot(page)).text ?? '') as JobConfig;
  config.config.process[0].train.optimizer_params = { weight_decay: 0.017, betas: [0.8, 0.95] };
  await input.focus();
  await page.keyboard.press('Control+a');
  // Multiline text must use the paste path: CDP insertText is treated as typing
  // by Monaco and applies auto-indent to every newline in the document.
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.evaluate(text => navigator.clipboard.writeText(text), YAML.stringify(config));
  await page.keyboard.press('Control+v');
  expect((await editorSnapshot(page)).text).toEqual(YAML.stringify(config));
  await page.getByRole('tab', { name: 'Form', exact: true }).click();
  await expect.poll(() => page.locator('#training-form-view').evaluate(element => element.scrollTop)).toBeCloseTo(formScroll, 0);
  await expect(page).toHaveURL(/#job-training$/);
  await page.getByLabel('Training name', { exact: true }).fill('advanced_save_fixture');
  await page.getByLabel('Learning rate', { exact: true }).fill('0.00042');
  let release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  let savedPayload: { job_config: JobConfig } | undefined;
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    savedPayload = route.request().postDataJSON() as { job_config: JobConfig };
    await pending;
    return route.fulfill({ status: 503, json: { error: 'Fixture save unavailable' } });
  });
  try {
    const posted = page.waitForRequest(request => request.url().endsWith('/api/jobs') && request.method() === 'POST');
    await page.getByRole('button', { name: 'Create Job', exact: true }).click();
    await posted;
    await expect(page.getByLabel('Training name', { exact: true })).toBeDisabled();
    await expect(page.locator('#job-model')).toHaveAttribute('inert', '');
    // React Select hides its disabled input while leaving the selected value visible.
    const computeControl = page.locator('.aitk-react-select__control:visible')
      .filter({ has: page.locator('input[aria-label="Compute"]') });
    await expect(computeControl).toHaveCount(1);
    await expect(computeControl.locator('input')).toBeDisabled();
    await computeControl.click({ force: true });
    await expect(page.getByRole('listbox')).toHaveCount(0);
    await computeControl.locator('input').evaluate(element => element.focus());
    await expect(computeControl.locator('input')).not.toBeFocused();
    await page.getByRole('tab', { name: 'Config', exact: true }).click();
    const before = (await editorSnapshot(page)).text;
    await input.focus();
    await page.keyboard.insertText('\nshould_not_be_inserted: true');
    expect((await editorSnapshot(page)).text).toEqual(before);
    expect(savedPayload?.job_config.config.process[0].train.optimizer_params).toEqual(config.config.process[0].train.optimizer_params);
    expect(savedPayload?.job_config.config.process[0].train.lr).toBe(0.00042);
    expect(savedPayload?.job_config.config.name).toBe('advanced_save_fixture');
    release();
    await expect(page.getByRole('alert').filter({ hasText: 'Fixture save unavailable' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Create Job', exact: true })).toBeEnabled();
    await page.getByRole('tab', { name: 'Form', exact: true }).click();
    await expect(page.getByLabel('Training name', { exact: true })).toBeEnabled();
    await expect(page.getByLabel('Training name', { exact: true })).toHaveValue('advanced_save_fixture');
  } finally {
    release();
  }
});

test('a save response cannot navigate away from a workspace entered while the save was pending', async ({ page }) => {
  await page.getByLabel('Training name', { exact: true }).fill('leave_pending_save');
  let release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    await pending;
    return route.fulfill({ json: { id: 'late-save-result' } });
  });
  try {
    const posted = page.waitForRequest(request => request.url().endsWith('/api/jobs') && request.method() === 'POST');
    await page.getByRole('button', { name: 'Create Job', exact: true }).click();
    await posted;
    await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Train', exact: true }).click();
    await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
    await expect(page).toHaveURL(/\/jobs$/);
    const completed = page.waitForResponse(response => response.url().endsWith('/api/jobs') && response.request().method() === 'POST');
    release();
    await (await completed).finished();
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    await expect(page).toHaveURL(/\/jobs$/);
  } finally {
    release();
  }
});

test('continuous sections, parameter navigation and responsive Create control', async ({ page, browser, baseURL }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await expect(page.locator('#job-training')).toBeVisible();
  await expect(page.locator('#job-dataset')).toBeVisible();
  await expect(page.locator('#training-step-navigation, #job-review')).toHaveCount(0);
  await page.setViewportSize({ width: 1487, height: 1058 });
  await expectReadableComputeValues(page);
  const baseLoraSummary = page.locator('summary').filter({ hasText: 'Continue from an existing LoRA' });
  await expect(page.getByLabel('Base LoRA path (optional)', { exact: true })).toBeHidden();
  await baseLoraSummary.click();
  await expect(page.getByLabel('Base LoRA path (optional)', { exact: true })).toBeVisible();
  await baseLoraSummary.click();
  await page.locator('#training-form-view').evaluate(element => { element.scrollTop = 0; });
  await expect(page.getByRole('button', { name: 'Create Job', exact: true })).toBeEnabled();
  await page.screenshot({ path: '.review-build/premium-training-mock-viewport-dark.png' });
  await page.setViewportSize({ width: 1440, height: 1024 });
  await page.getByRole('link', { name: 'Training', exact: true }).click();
  await expect(page).toHaveURL(/#job-training$/);
  await page.getByRole('searchbox', { name: 'Find parameter' }).fill('learning rate');
  await page.getByRole('button', { name: 'Learning rate Training', exact: true }).click();
  await expect(page.getByLabel('Learning rate', { exact: true })).toBeFocused();
  await page.getByRole('searchbox', { name: 'Find parameter' }).fill('');
  await page.getByLabel('Training name', { exact: true }).fill('portrait_flux_v4');
  await page.locator('#training-form-view').evaluate(element => { element.scrollTop = 0; });
  await page.screenshot({ path: '.review-build/premium-training-desktop-dark.png' });
  await page.getByRole('button', { name: 'More tools', exact: true }).click();
  await page.getByTitle('Switch to light mode', { exact: true }).click();
  await page.keyboard.press('Escape');
  await page.screenshot({ path: '.review-build/premium-training-desktop-light.png' });
  await page.getByRole('button', { name: 'More tools', exact: true }).click();
  await page.getByTitle('Switch to dark mode', { exact: true }).click();
  await page.keyboard.press('Escape');
  for (const width of [1440, 1024, 390]) {
    await page.setViewportSize({ width, height: 900 });
    const create = page.getByRole('button', { name: 'Create Job', exact: true });
    await expect(create).toBeInViewport();
    await expectReadableComputeValues(page);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: `.review-build/premium-training-${width}.png` });
  }
  await page.screenshot({ path: '.review-build/premium-training-mobile.png' });
  await page.getByRole('button', { name: 'More tools', exact: true }).click();
  await page.getByTitle('Switch to light mode', { exact: true }).click();
  await page.keyboard.press('Escape');
  for (const width of [1024, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await expect(page.getByRole('button', { name: 'Create Job', exact: true })).toBeInViewport();
    await expectReadableComputeValues(page);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: `.review-build/premium-training-light-${width}.png` });
  }
  expect(pageErrors).toEqual([]);

  // The 720 CSS-pixel viewport at 2x scale matches 1440px desktop reflow at 200% zoom.
  const zoomContext = await browser.newContext({ baseURL, viewport: { width: 720, height: 512 }, deviceScaleFactor: 2 });
  try {
    await zoomContext.addCookies(await page.context().cookies());
    await zoomContext.addInitScript(() => localStorage.setItem('theme', 'dark'));
    const zoomPage = await zoomContext.newPage();
    const zoomErrors: string[] = [];
    zoomPage.on('pageerror', error => zoomErrors.push(error.message));
    await installTrainingFixtures(zoomPage);
    await zoomPage.goto('/jobs/new');
    await expect(zoomPage.getByLabel('Training name', { exact: true })).toBeEnabled();
    await expect(zoomPage.getByRole('combobox', { name: 'Model architecture', exact: true })).toBeEnabled();
    await expect(zoomPage.getByRole('combobox', { name: 'Compute', exact: true })).toBeVisible();
    for (const theme of ['dark', 'light']) {
      if (theme === 'light') {
        await zoomPage.getByRole('button', { name: 'More tools', exact: true }).click();
        await zoomPage.getByTitle('Switch to light mode', { exact: true }).click();
        await zoomPage.keyboard.press('Escape');
      }
      await expect(zoomPage.getByRole('button', { name: 'Create Job', exact: true })).toBeInViewport();
      await expectReadableComputeValues(zoomPage);
      expect(await zoomPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await zoomPage.screenshot({ path: `.review-build/premium-training-${theme}-200-percent-reflow.png` });
    }
    expect(zoomErrors).toEqual([]);
  } finally {
    await zoomContext.close();
  }
});

for (const navigationMode of ['fallback', 'native'] as const) {
test(`${navigationMode} browser history protects and restores an unfinished Config draft`, async ({ page }) => {
  if (navigationMode === 'fallback') {
    await page.addInitScript(() => Object.defineProperty(window, 'navigation', { value: undefined, configurable: true }));
  }
  await page.goto('/jobs');
  expect(await page.evaluate(() => typeof (window as Window & { navigation?: unknown }).navigation !== 'undefined')).toBe(navigationMode === 'native');
  await page.getByRole('link', { name: 'New training', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toBeEnabled();
  await page.getByRole('link', { name: 'Training', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs\/new#job-training$/);
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  const input = page.locator('[data-config-editor] textarea.inputarea');
  await expect(input).toBeVisible();
  await input.focus();
  await page.keyboard.press('Control+End');
  await page.keyboard.insertText('\n# history-memory-only-fixture\nbroken: [');
  await expect(page.locator('[data-config-editor] [role="alert"]')).toBeVisible();
  const rawText = (await editorSnapshot(page)).text;

  // The section hash is its own history entry. Traverse past it to the queue.
  await page.evaluate(() => window.history.go(-2));
  await expect(page.getByText('Leave training setup?', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs\/new#job-training$/);
  await expect(page.getByRole('tab', { name: 'Config', exact: true })).toHaveAttribute('aria-selected', 'true');
  expect((await editorSnapshot(page)).text).toEqual(rawText);

  // Cancellation must not consume the queue entry or duplicate this draft's
  // entries: repeat the same traversal, then confirm its original destination.
  await page.evaluate(() => window.history.go(-2));
  await expect(page.getByText('Leave training setup?', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs$/);
  await page.evaluate(() => window.history.forward());
  await expect(page).toHaveURL(/\/jobs\/new$/);
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  await expect(page.locator('[data-config-editor] [role="alert"]')).toBeVisible();
  expect((await editorSnapshot(page)).text).toEqual(rawText);
  await page.evaluate(() => window.history.forward());
  await expect(page).toHaveURL(/\/jobs\/new#job-training$/);
  expect((await editorSnapshot(page)).text).toEqual(rawText);
  expect(await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage }))).not.toContain('history-memory-only-fixture');
});
}

test('invalid YAML, cursor and selection survive switching views and guarded navigation without browser persistence', async ({ page }) => {
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  const input = page.locator('[data-config-editor] textarea.inputarea');
  await expect(input).toBeVisible();
  await input.focus();
  await page.keyboard.press('Control+End');
  await page.keyboard.insertText('\n# memory-only-fixture\nbroken: [');
  await expect(page.locator('[data-config-editor] [role="alert"]')).toBeVisible();
  await page.keyboard.press('Shift+ArrowLeft');
  const before = await editorSnapshot(page);
  expect(before.text).toContain('broken: [');
  await page.getByRole('tab', { name: 'Form', exact: true }).click();
  await expect(page.getByLabel('Training name', { exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Create Job', exact: true })).toBeDisabled();
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  expect(await editorSnapshot(page)).toEqual(before);
  const stored = await page.evaluate(() => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } }));
  expect(stored).not.toContain('memory-only-fixture');
  await page.getByRole('tab', { name: 'Form', exact: true }).click();
  await page.keyboard.press('End');
  await expect(page.getByRole('tab', { name: 'Config', exact: true })).toBeFocused();
  await page.keyboard.press('Home');
  await expect(page.getByRole('tab', { name: 'Form', exact: true })).toBeFocused();
  await page.keyboard.press('Control+k');
  await page.getByRole('textbox', { name: 'Search commands', exact: true }).fill('Queue');
  await page.getByRole('navigation', { name: 'Command results' }).getByRole('link', { name: /Queue/ }).click();
  await expect(page.getByText('Leave training setup?', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.keyboard.press('Escape');
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Train', exact: true }).click();
  await expect(page.getByText('Leave training setup?', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  expect((await editorSnapshot(page)).text).toEqual(before.text);
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Train', exact: true }).click();
  await page.getByRole('button', { name: 'Leave setup', exact: true }).click();
  await expect(page).toHaveURL(/\/jobs$/);
  await page.goBack();
  await expect(page).toHaveURL(/\/jobs\/new$/);
  await page.getByRole('tab', { name: 'Config', exact: true }).click();
  await expect(page.locator('[data-config-editor] [role="alert"]')).toBeVisible();
  expect((await editorSnapshot(page)).text).toEqual(before.text);
  await page.screenshot({ path: '.review-build/premium-training-invalid-config.png' });
  for (const theme of ['dark', 'light']) {
    if (theme === 'light') {
      await page.getByRole('button', { name: 'More tools', exact: true }).click();
      await page.getByTitle('Switch to light mode', { exact: true }).click();
      await page.keyboard.press('Escape');
    }
    const colors = await page.locator('[data-config-editor] [role="alert"]').evaluate(element => {
      const style = getComputedStyle(element);
      const luminance = (color: string) => {
        const channels = (color.match(/[\d.]+/g) ?? []).slice(0, 3).map(Number).map(value => {
          const channel = value / 255;
          return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
        });
        return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
      };
      const foreground = luminance(style.color);
      const background = luminance(style.backgroundColor);
      return { color: style.color, background: style.backgroundColor,
        ratio: (Math.max(foreground, background) + 0.05) / (Math.min(foreground, background) + 0.05) };
    });
    await test.info().attach(`yaml-error-contrast-${theme}`, { body: JSON.stringify(colors), contentType: 'application/json' });
    expect(colors.ratio, `YAML error contrast in ${theme} theme`).toBeGreaterThanOrEqual(4.5);
  }
});
