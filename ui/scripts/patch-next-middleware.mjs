// Backport https://github.com/vercel/next.js/pull/85418 to the supported Next 15 runtime.
// Without awaiting finalize, a protected POST can reach its route with a disturbed body.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const root = path.dirname(require.resolve('next/package.json'));
const version = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).version;
if (!version.startsWith('15.'))
  throw new Error('Revalidate the Node middleware compatibility patch before changing Next major versions.');
for (const name of ['dist/server/next-server.js', 'dist/esm/server/next-server.js']) {
  const filename = path.join(root, name),
    source = fs.readFileSync(filename, 'utf8');
  if (source.includes('await requestData.body.finalize();')) continue;
  const before = 'requestData.body.finalize();';
  if (source.split(before).length !== 2)
    throw new Error('Next middleware finalization changed; review the upstream fix before continuing.');
  fs.writeFileSync(filename, source.replace(before, 'await requestData.body.finalize();'));
}
