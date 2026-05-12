import { execFileSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { MongoClient } from 'mongodb';

const provider = (process.env.AITK_DB_PROVIDER || 'sqlite').trim().toLowerCase();
const toolkitRoot = path.resolve(process.cwd(), '..');
const sqlitePath = path.resolve(process.env.AITK_SQLITE_PATH || path.join(toolkitRoot, 'aitk_db.db'));
const mongoUri = process.env.AITK_MONGODB_URI?.trim();
const mongoDbName = process.env.AITK_MONGODB_DB?.trim() || 'ai_toolkit';
const prismaCli = path.join(process.cwd(), 'node_modules', '.bin', process.platform === 'win32' ? 'prisma.cmd' : 'prisma');
const prismaExecOptions = { stdio: 'inherit', shell: process.platform === 'win32' };

if (!['sqlite', 'mongodb'].includes(provider)) {
  throw new Error(`Invalid AITK_DB_PROVIDER "${provider}". Expected "sqlite" or "mongodb".`);
}

process.env.DATABASE_URL = `file:${sqlitePath.replace(/\\/g, '/')}`;

console.log(`Generating Prisma client for SQLite fallback (${process.env.DATABASE_URL})...`);
execFileSync(prismaCli, ['generate'], prismaExecOptions);

if (provider === 'sqlite') {
  console.log('Preparing SQLite database...');
  fs.mkdirSync(path.dirname(sqlitePath), { recursive: true });
  fs.closeSync(fs.openSync(sqlitePath, 'a'));
  execFileSync(prismaCli, ['db', 'push'], prismaExecOptions);
  process.exit(0);
}

if (!mongoUri) {
  throw new Error('AITK_MONGODB_URI is required when AITK_DB_PROVIDER=mongodb.');
}

console.log(`Preparing MongoDB database "${mongoDbName}"...`);
const client = new MongoClient(mongoUri);
try {
  await client.connect();
  const db = client.db(mongoDbName);
  await Promise.all([
    db.collection('jobs').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { name: 1 }, unique: true },
      { key: { status: 1 } },
      { key: { gpu_ids: 1 } },
      { key: { job_type: 1 } },
      { key: { job_ref: 1 } },
      { key: { queue_position: 1 } },
    ]),
    db.collection('queues').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { gpu_ids: 1 }, unique: true },
    ]),
    db.collection('settings').createIndexes([{ key: { key: 1 }, unique: true }]),
    db.collection('metrics').createIndexes([
      { key: { job_id: 1, step: 1, key: 1 }, unique: true },
      { key: { job_id: 1, key: 1, step: 1 } },
    ]),
    db.collection('metric_keys').createIndexes([{ key: { job_id: 1, key: 1 }, unique: true }]),
  ]);
  console.log('MongoDB indexes are ready.');
} finally {
  await client.close();
}
