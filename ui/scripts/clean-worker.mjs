import fs from 'fs';
import path from 'path';

for (const relativePath of ['dist/cron', 'dist/src']) {
  fs.rmSync(path.resolve(process.cwd(), relativePath), { recursive: true, force: true });
}
