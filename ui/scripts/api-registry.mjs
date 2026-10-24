import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'];

export async function apiRouteFiles(root) {
  const entries = await fs.readdir(root, { withFileTypes: true });
  const groups = await Promise.all(entries.map(entry => {
    const filename = path.join(root, entry.name);
    return entry.isDirectory() ? apiRouteFiles(filename) : /^route\.tsx?$/.test(entry.name) ? [filename] : [];
  }));
  return groups.flat().sort((a, b) => {
    // Literal routes precede parameters and catch-alls (e.g. /img/upload).
    const score = value => value.split(path.sep).map(part => part.startsWith('[...') ? '2' : part.startsWith('[') ? '1' : `0${part}`).join('/');
    return score(a).localeCompare(score(b));
  });
}

export async function generateApiRegistry(uiRoot) {
  const output = path.join(uiRoot, '.runtime');
  await fs.mkdir(output, { recursive: true });
  const lines = [
    "import type { Hono } from 'hono';",
    "import { registerRoute, routeParameter } from '../src/server/http/apiRoutes';",
    'export function registerApiRoutes(app: Hono): void {',
  ];
  for (const file of await apiRouteFiles(path.join(uiRoot, 'src/app/api'))) {
    const source = await fs.readFile(file, 'utf8');
    const exported = methods.filter(method => new RegExp(`export (?:async )?(?:function|const) ${method}\\b`).test(source));
    if (!exported.length) throw new Error(`No HTTP methods exported by ${file}`);
    const segments = path.relative(path.join(uiRoot, 'src/app'), path.dirname(file)).split(path.sep);
    const params = [];
    const route = '/' + segments.map(segment => {
      const match = /^\[(\.\.\.)?(\w+)\]$/.exec(segment);
      if (!match) return segment;
      const [, catchAll, name] = match;
      params.push(`${name}: routeParameter(c, ${JSON.stringify(name)})${catchAll ? ".split('/')" : ''}`);
      return `:${name}${catchAll ? '{.+}' : ''}`;
    }).join('/');
    const modulePath = '../' + path.relative(uiRoot, file).replaceAll(path.sep, '/').replace(/\.tsx?$/, '');
    lines.push(`  registerRoute(app, ${JSON.stringify(route)}, ${JSON.stringify(exported)}, () => import(${JSON.stringify(modulePath)}), c => ({ ${params.join(', ')} }));`);
  }
  lines.push('}', '');
  const filename = path.join(output, 'api-routes.ts');
  const contents = lines.join('\n');
  if (await fs.readFile(filename, 'utf8').catch(() => '') !== contents) await fs.writeFile(filename, contents);
  return filename;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await generateApiRegistry(path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'));
