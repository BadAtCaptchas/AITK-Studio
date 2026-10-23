import { test, expect, type Page } from '@playwright/test';
import type { editor } from 'monaco-editor';
import YAML from 'yaml';
import { captionFixture, installCaptionFixtures } from './captionFixture';

const pageErrors = new WeakMap<Page, string[]>();
const invalidYaml = '# Keep this unfinished caption draft exactly\njob: extension\nconfig: [unfinished\n';

test.beforeEach(async ({ page }) => {
  const errors: string[] = [];
  pageErrors.set(page, errors);
  page.on('pageerror', error => errors.push(error.message));
  await installCaptionFixtures(page);
});

test.afterEach(async ({ page }) => {
  expect(pageErrors.get(page)).toEqual([]);
});

async function editorText(page: Page): Promise<string | undefined> {
  return page.evaluate(() => {
    const monaco = (window as Window & { monaco?: { editor: { getEditors(): editor.ICodeEditor[] } } }).monaco;
    return monaco?.editor.getEditors()[0]?.getValue();
  });
}

async function pasteConfig(page: Page, text: string): Promise<void> {
  const input = page
    .getByRole('dialog', { name: 'Caption Dataset', exact: true })
    .locator('[data-config-editor] textarea.inputarea');
  await expect(input).toBeVisible();
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.evaluate(value => navigator.clipboard.writeText(value), text);
  await input.focus();
  await page.keyboard.press('Control+a');
  await page.keyboard.press('Control+v');
  await expect.poll(() => editorText(page)).toBe(text);
}

async function openAdvanced(page: Page) {
  const fixture = await captionFixture(page);
  await page.getByLabel('Edit captions', { exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'Caption Dataset', exact: true });
  const save = modal.getByRole('button', { name: 'Save & start captioning', exact: true });
  await expect(save).toBeEnabled();
  await modal.getByRole('button', { name: 'Advanced', exact: true }).click();
  await expect(modal.locator('[data-config-editor] textarea.inputarea')).toBeVisible();
  return { ...fixture, modal, save };
}

test('invalid caption YAML blocks normal and forced saves, then saves corrected values', async ({ page }) => {
  const { state, config, job, modal, save } = await openAdvanced(page);
  await pasteConfig(page, invalidYaml);
  await expect(save).toBeDisabled();
  await save.evaluate(button =>
    button.closest('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })),
  );
  await page.waitForTimeout(100);
  expect(state.saves).toEqual([]);
  expect(state.starts).toBe(0);
  expect(state.queueStarts).toBe(0);
  const corrected = structuredClone(config);
  corrected.config.process[0].caption.caption_prompt =
    'Corrected prompt: preserve literal # text and https://fixture.invalid';
  corrected.config.process[0].caption.max_new_tokens = 512;
  corrected.config.process[0].caption.max_res = 768;
  state.saveSucceeds = true;
  await pasteConfig(page, YAML.stringify(corrected));
  await expect(save).toBeEnabled();
  await save.click();
  await expect(modal).toHaveCount(0);
  expect(state.saves).toHaveLength(1);
  expect(state.saves[0]).toMatchObject({ id: job.id, name: job.name, job_config: corrected });
  expect(state.starts).toBe(1);
  expect(state.queueStarts).toBe(1);
});

test('switching caption views retains exact invalid YAML and cannot save the previous config', async ({ page }) => {
  const { state, modal, save } = await openAdvanced(page);
  await pasteConfig(page, invalidYaml);
  await modal.getByRole('button', { name: 'Simple', exact: true }).click();
  await expect(save).toBeDisabled();
  await save.evaluate(button =>
    button.closest('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })),
  );
  await page.waitForTimeout(100);
  expect(state.saves).toEqual([]);
  await modal.getByRole('button', { name: 'Advanced', exact: true }).click();
  await expect(modal.locator('[data-config-editor] textarea.inputarea')).toBeVisible();
  await expect.poll(() => editorText(page)).toBe(invalidYaml);
  await expect(save).toBeDisabled();
});

for (const issue of ['empty process list', 'non-string caption prompt'] as const) {
  test(`semantic caption config error (${issue}) is rejected safely and can be corrected`, async ({ page }) => {
    const { state, config, modal, save } = await openAdvanced(page);
    const invalid = structuredClone(config);
    if (issue === 'empty process list') invalid.config.process = [];
    else Reflect.set(invalid.config.process[0].caption, 'caption_prompt', ['invalid', 'array']);
    const text = YAML.stringify(invalid);
    await pasteConfig(page, text);
    await expect(save).toBeDisabled();
    await save.evaluate(button =>
      button.closest('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })),
    );
    await page.waitForTimeout(100);
    expect(state.saves).toEqual([]);
    await modal.getByRole('button', { name: 'Simple', exact: true }).click();
    await expect(save).toBeDisabled();
    await modal.getByRole('button', { name: 'Advanced', exact: true }).click();
    await expect.poll(() => editorText(page)).toBe(text);
    await pasteConfig(page, YAML.stringify(config));
    await expect(save).toBeEnabled();
    await modal.getByRole('button', { name: 'Simple', exact: true }).click();
    await expect(modal.getByLabel('Caption Prompt', { exact: true })).toHaveValue(
      config.config.process[0].caption.caption_prompt ?? '',
    );
    await save.click();
    await expect.poll(() => state.saves.length).toBe(1);
    expect(state.saves[0]).toMatchObject({ job_config: config });
  });
}
