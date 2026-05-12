import path from 'node:path';
import { defineConfig } from 'prisma/config';

const toolkitRoot = path.resolve(process.cwd(), '..');
const sqlitePath = path.resolve(process.env.AITK_SQLITE_PATH || path.join(toolkitRoot, 'aitk_db.db'));
const sqliteUrl = `file:${sqlitePath.replace(/\\/g, '/')}`;

export default defineConfig({
  schema: 'prisma/schema.prisma',
  datasource: {
    url: sqliteUrl,
  },
});
