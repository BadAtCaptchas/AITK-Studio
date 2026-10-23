import type { Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

/** Use the installed editor assets so browser fixtures do not require a CDN. */
export async function installLocalMonaco(page: Page): Promise<void> {
  await page.route('https://cdn.jsdelivr.net/npm/monaco-editor@*/min/vs/**', async route => {
    const relative = new URL(route.request().url()).pathname.split('/min/vs/')[1];
    if (!relative || relative.split('/').includes('..')) return route.abort();
    const filename = path.join(process.cwd(), 'node_modules/monaco-editor/min/vs', relative);
    await route.fulfill({ body: await readFile(filename), contentType: relative.endsWith('.css') ? 'text/css' : 'application/javascript' });
  });
}
