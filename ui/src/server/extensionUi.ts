import fs from 'fs/promises';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';
import { isPathWithinRoot } from './pathContainment';

export interface ExtensionUiModule {
  id: string;
  code: string;
}
const cache = new Map<string, { mtime: number; size: number; code: string }>();
export async function listExtensionUiModules(): Promise<{ modules: ExtensionUiModule[]; errors: string[] }> {
  const modules: ExtensionUiModule[] = [],
    errors: string[] = [];
  const nodeModule = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ 'module');
  const loadModule = nodeModule.createRequire(path.join(TOOLKIT_ROOT, 'ui', 'package.json'));
  const swc: { transform: (source: string, options: Record<string, unknown>) => Promise<{ code: string }> } =
    loadModule('next/dist/build/swc');
  for (const dir of ['extensions_built_in', 'extensions']) {
    const root = await fs.realpath(path.join(TOOLKIT_ROOT, dir)).catch((error: unknown) => {
      if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') return null;
      throw error;
    });
    if (!root) continue;
    const entries = await fs.readdir(root, { withFileTypes: true });
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (!entry.isDirectory() || /^[._]/.test(entry.name)) continue;
      const id = `${dir}/${entry.name}`;
      try {
        for (const name of ['ui.tsx', 'ui.ts', 'ui.jsx', 'ui.js']) {
          const file = await fs.realpath(path.join(root, entry.name, name)).catch(() => null);
          if (!file) continue;
          if (!isPathWithinRoot(root, file)) throw new Error('UI file escapes its extension root');
          const stat = await fs.stat(file);
          if (!stat.isFile() || stat.size > 2 * 1024 * 1024) throw new Error('Invalid extension UI file');
          let compiled = cache.get(file);
          if (!compiled || compiled.mtime !== stat.mtimeMs || compiled.size !== stat.size) {
            const result = await swc.transform(await fs.readFile(file, 'utf8'), {
              filename: file,
              jsc: {
                parser: /\.tsx?$/.test(name)
                  ? { syntax: 'typescript', tsx: name.endsWith('x') }
                  : { syntax: 'ecmascript', jsx: true },
                transform: { react: { runtime: 'automatic' } },
                target: 'es2020',
              },
              module: { type: 'commonjs' },
              sourceMaps: false,
              isModule: true,
            });
            compiled = { mtime: stat.mtimeMs, size: stat.size, code: result.code };
            cache.set(file, compiled);
          }
          modules.push({ id, code: compiled.code });
          break;
        }
      } catch (error) {
        errors.push(`${id}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  }
  return { modules, errors };
}
