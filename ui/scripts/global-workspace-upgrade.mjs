// The only database compatibility layer for retired project workspaces.
// Run with application workers stopped. Never touches files or storage folders.
const legacyTables = ['Project', 'ProjectReplica', 'ProjectSyncOperation'];
const legacyCollections = ['projects', 'project_replicas', 'project_sync_operations'];
const retiredSettings = ['PROJECTS_FOLDER', 'PROJECTS_ENABLED'];
const manualMigration = () =>
  new Error(
    'Global workspace upgrade refused: legacy project data or conflicting job names require manual migration. The database and storage folders have not been changed.',
  );
const all = (db, sql, params = []) =>
  new Promise((resolve, reject) => {
    db.all(sql, params, (error, rows) => (error ? reject(error) : resolve(rows)));
  });
const run = (db, sql) =>
  new Promise((resolve, reject) => {
    db.run(sql, error => (error ? reject(error) : resolve()));
  });
const quote = name => `"${name.replaceAll('"', '""')}"`;

function hasScopedWatcher(value) {
  if (!value) return false;
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch {
    return false;
  }
  const rows = Array.isArray(parsed) ? parsed : parsed?.watchers || [];
  return rows.some(
    row => row && ['project_id', 'projectID', 'projectId'].some(key => row[key] != null && row[key] !== ''),
  );
}

export function stripLegacyScopeFields(row) {
  const { project_id, remote_project_id, ...globalRow } = row;
  return globalRow;
}

export async function preflightSqliteGlobalUpgrade(db) {
  const tables = new Set((await all(db, "SELECT name FROM sqlite_master WHERE type='table'")).map(row => row.name));
  for (const table of legacyTables) {
    if (tables.has(table) && (await all(db, `SELECT 1 FROM ${quote(table)} LIMIT 1`)).length) throw manualMigration();
  }
  for (const [table, field] of [
    ['Job', 'project_id'],
    ['JobReplica', 'remote_project_id'],
  ]) {
    const columns = await all(db, `PRAGMA table_info(${quote(table)})`);
    if (
      columns.some(column => column.name === field) &&
      (
        await all(
          db,
          `SELECT 1 FROM ${quote(table)} WHERE ${quote(field)} IS NOT NULL AND trim(${quote(field)}) <> '' LIMIT 1`,
        )
      ).length
    ) {
      throw manualMigration();
    }
  }
  if (tables.has('Settings')) {
    const rows = await all(db, "SELECT value FROM Settings WHERE key = 'DATASET_WATCHERS_V1'");
    if (rows.some(row => hasScopedWatcher(row.value))) throw manualMigration();
  }
  if (tables.has('Job') && (await all(db, 'SELECT name FROM Job GROUP BY name HAVING count(*) > 1 LIMIT 1')).length)
    throw manualMigration();
}

export async function upgradeSqliteGlobalWorkspace(db) {
  await run(db, 'BEGIN IMMEDIATE');
  try {
    await preflightSqliteGlobalUpgrade(db);
    for (const [table, field] of [
      ['Job', 'project_id'],
      ['JobReplica', 'remote_project_id'],
    ]) {
      const columns = await all(db, `PRAGMA table_info(${quote(table)})`);
      if (!columns.some(column => column.name === field)) continue;
      const indexes = await all(db, `PRAGMA index_list(${quote(table)})`);
      for (const index of indexes) {
        const indexedColumns = await all(db, `PRAGMA index_info(${quote(index.name)})`);
        const definition = await all(db, 'SELECT sql FROM sqlite_master WHERE type = ? AND name = ?', [
          'index',
          index.name,
        ]);
        if (indexedColumns.some(column => column.name === field) || definition.some(row => row.sql?.includes(field))) {
          // Old releases used explicit indexes for scope. Refuse unknown table constraints.
          if (index.origin !== 'c') throw new Error('Legacy scope constraint requires manual migration.');
          await run(db, `DROP INDEX ${quote(index.name)}`);
        }
      }
      await run(db, `ALTER TABLE ${quote(table)} DROP COLUMN ${quote(field)}`);
    }
    for (const table of legacyTables) await run(db, `DROP TABLE IF EXISTS ${quote(table)}`);
    const tables = new Set((await all(db, "SELECT name FROM sqlite_master WHERE type='table'")).map(row => row.name));
    if (tables.has('Job')) await run(db, 'CREATE UNIQUE INDEX IF NOT EXISTS Job_name_key ON Job(name)');
    if (tables.has('Settings'))
      await run(db, `DELETE FROM Settings WHERE key IN (${retiredSettings.map(value => `'${value}'`).join(',')})`);
    await run(db, 'COMMIT');
  } catch (error) {
    await run(db, 'ROLLBACK');
    throw error;
  }
}

export async function preflightMongoGlobalUpgrade(db) {
  for (const collection of legacyCollections) {
    if (await db.collection(collection).findOne({})) throw manualMigration();
  }
  for (const [collection, field] of [
    ['jobs', 'project_id'],
    ['job_replicas', 'remote_project_id'],
  ]) {
    if (await db.collection(collection).findOne({ [field]: { $exists: true, $nin: [null, ''] } }))
      throw manualMigration();
  }
  const watcherSetting = await db.collection('settings').findOne({ key: 'DATASET_WATCHERS_V1' });
  if (hasScopedWatcher(watcherSetting?.value)) throw manualMigration();
  const duplicate = await db
    .collection('jobs')
    .aggregate([{ $group: { _id: '$name', count: { $sum: 1 } } }, { $match: { count: { $gt: 1 } } }, { $limit: 1 }])
    .toArray();
  if (duplicate.length) throw manualMigration();
}

export async function upgradeMongoGlobalWorkspace(db) {
  // All data checks precede writes, including index changes. Restart-safe on standalone MongoDB.
  await preflightMongoGlobalUpgrade(db);
  const jobs = db.collection('jobs');
  const indexes = await jobs.indexes().catch(error => {
    if (error.code === 26) return [];
    throw error;
  });
  for (const index of indexes) {
    if ('project_id' in index.key || ('name' in index.key && !index.unique)) await jobs.dropIndex(index.name);
  }
  await jobs.createIndex({ name: 1 }, { unique: true });
  await jobs.updateMany({}, { $unset: { project_id: '' } });
  await db.collection('job_replicas').updateMany({}, { $unset: { remote_project_id: '' } });
  for (const collection of legacyCollections) {
    await db
      .collection(collection)
      .drop()
      .catch(error => {
        if (error.code !== 26) throw error;
      });
  }
  await db.collection('settings').deleteMany({ key: { $in: retiredSettings } });
}
