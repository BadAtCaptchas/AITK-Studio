import { execFileSync } from 'child_process';
import path from 'path';
import { MongoClient } from 'mongodb';

const provider = (process.env.AITK_DB_PROVIDER || 'sqlite').trim().toLowerCase();
const toolkitRoot = path.resolve(process.cwd(), '..');
const sqlitePath = path.resolve(process.env.AITK_SQLITE_PATH || path.join(toolkitRoot, 'aitk_db.db'));
const mongoUri = process.env.AITK_MONGODB_URI?.trim();
const mongoDbName = process.env.AITK_MONGODB_DB?.trim() || 'ai_toolkit';
const prismaCli = path.join(process.cwd(), 'node_modules', 'prisma', 'build', 'index.js');

if (!['sqlite', 'mongodb'].includes(provider)) {
  throw new Error(`Invalid AITK_DB_PROVIDER "${provider}". Expected "sqlite" or "mongodb".`);
}

process.env.DATABASE_URL = `file:${sqlitePath.replace(/\\/g, '/')}`;

console.log(`Generating Prisma client for SQLite fallback (${process.env.DATABASE_URL})...`);
execFileSync(process.execPath, [prismaCli, 'generate'], { stdio: 'inherit' });

if (provider === 'sqlite') {
  console.log('Preparing SQLite database...');
  execFileSync(process.execPath, [prismaCli, 'db', 'push'], { stdio: 'inherit' });
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
    db.collection('system_metric_samples').createIndexes([
      { key: { created_at: 1 } },
      { key: { scope: 1, device_id: 1, metric: 1, created_at: 1 } },
    ]),
    db.collection('model_artifacts').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { kind: 1 } },
      { key: { job_id: 1 } },
      { key: { source: 1 } },
      { key: { exists: 1 } },
    ]),
    db.collection('evaluation_runs').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { status: 1 } },
      { key: { created_at: -1 } },
    ]),
    db.collection('evaluation_items').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { run_id: 1 } },
      { key: { status: 1 } },
    ]),
    db.collection('alert_rules').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { enabled: 1 } },
    ]),
    db.collection('alert_events').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { fingerprint: 1 }, unique: true },
      { key: { status: 1 } },
      { key: { severity: 1 } },
      { key: { resource_type: 1, resource_id: 1 } },
      { key: { rule_id: 1 } },
    ]),
  ]);
  console.log('MongoDB indexes are ready.');
} finally {
  await client.close();
}
