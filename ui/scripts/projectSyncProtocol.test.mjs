import test from 'node:test';
import {
  createProjectAssetUrl,
  normalizeProjectAssetPath,
  verifyProjectAssetSignature,
} from '../dist/src/server/projectAssetUrls.js';
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  appendProjectSyncChunk,
  assertProjectSyncPathContained,
  buildProjectSyncManifest,
  commitProjectSyncPlan,
  collectCredentialConfigKeys,
  detectProjectSyncConflicts,
  deterministicKeepBothPath,
  diffProjectSyncManifests,
  getProjectSyncChunkReceipt,
  hashManifestEntries,
  hashPortableProjectJobs,
  isProjectSyncPathExcluded,
  makePortableProjectRef,
  parseHttpByteRange,
  parseProjectSyncManifest,
  parsePortableProjectJobSnapshot,
  parseReplicaExecutionAuthorization,
  parsePortableProjectRef,
  portableizeProjectConfig,
  PROJECT_SYNC_CHUNK_BYTES,
  PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES,
  PROJECT_SYNC_TRANSFER_BLOCKING_JOB_STATUSES,
  resolvePortableProjectConfig,
} from '../dist/src/server/projectSyncProtocol.js';

async function withProject(run) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-sync-'));
  try {
    for (const zone of ['datasets', 'configs', 'runs', 'outputs', 'models', 'assets', 'notes', 'cache']) {
      await fs.mkdir(path.join(root, zone), { recursive: true });
    }
    await run(root);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
}

test('project asset URLs are relative, signed, and expire without exposing host paths', () => {
  const assetUrl = createProjectAssetUrl('project-1', 'runs/example/samples/frame.jpg', 'inline', Date.now() + 60_000);
  assert.equal(assetUrl.includes('E%3A') || assetUrl.includes('%5C'), false);
  const parsed = new URL(assetUrl, 'http://localhost');
  const relativePath = normalizeProjectAssetPath(parsed.searchParams.get('path'));
  const expires = Number(parsed.searchParams.get('expires'));
  const signature = parsed.searchParams.get('sig');
  assert.equal(
    verifyProjectAssetSignature({
      projectID: 'project-1',
      relativePath,
      disposition: 'inline',
      expires,
      signature,
    }),
    true,
  );
  assert.equal(
    verifyProjectAssetSignature({
      projectID: 'project-1',
      relativePath: 'runs/example/samples/other.jpg',
      disposition: 'inline',
      expires,
      signature,
    }),
    false,
  );
  assert.throws(() => normalizeProjectAssetPath('../outside.jpg'), /Invalid project asset path/);
});

function entry(filePath, contents) {
  const bytes = Buffer.from(contents);
  return {
    path: filePath,
    size: bytes.length,
    sha256: createHash('sha256').update(bytes).digest('hex'),
    modified_at: '2026-01-01T00:00:00.000Z',
  };
}

function manifest(files) {
  return {
    protocol: 'project-sync-v1',
    project_id: 'project-1',
    profile: 'full',
    generated_at: '2026-01-01T00:00:00.000Z',
    hash: 'unused',
    files,
  };
}

test('portable refs round trip and reject traversal', () => {
  const ref = makePortableProjectRef('1c84b032-48c3-4b95-b6b2-e8fd89f9abf8', 'datasets/portraits/a b.png');
  assert.equal(ref, 'aitk-project://1c84b032-48c3-4b95-b6b2-e8fd89f9abf8/datasets/portraits/a%20b.png');
  assert.deepEqual(parsePortableProjectRef(ref), {
    project_id: '1c84b032-48c3-4b95-b6b2-e8fd89f9abf8',
    relative_path: 'datasets/portraits/a b.png',
  });
  assert.throws(() => makePortableProjectRef('p', '../../outside.txt'), /escapes the project root/);
});

test('replica execution authorization requires protocol, bearer token, project, and home identity', () => {
  assert.equal(parseReplicaExecutionAuthorization(new Headers(), 'worker-secret'), null);
  const validHeaders = new Headers({
    Authorization: 'Bearer worker-secret',
    'X-AITK-Project-Sync': 'project-sync-v1',
    'X-AITK-Project-ID': 'project-1',
    'X-AITK-Home-Instance': 'home-1',
  });
  assert.deepEqual(parseReplicaExecutionAuthorization(validHeaders, 'worker-secret'), {
    protocol: 'project-sync-v1',
    projectID: 'project-1',
    homeInstanceID: 'home-1',
  });
  assert.throws(
    () => parseReplicaExecutionAuthorization(validHeaders, 'wrong-secret'),
    error => error?.status === 401 && error?.code === 'PROJECT_SYNC_EXECUTION_UNAUTHORIZED',
  );
  validHeaders.delete('X-AITK-Project-ID');
  assert.throws(
    () => parseReplicaExecutionAuthorization(validHeaders, 'worker-secret'),
    error => error?.status === 403 && error?.code === 'PROJECT_SYNC_EXECUTION_UNAUTHORIZED',
  );
});

test('queued jobs block lifecycle operations but do not block their own launch transfer', () => {
  assert.equal(PROJECT_SYNC_TRANSFER_BLOCKING_JOB_STATUSES.includes('queued'), false);
  assert.equal(PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES.includes('queued'), true);
  for (const status of ['starting', 'running', 'stopping']) {
    assert.equal(PROJECT_SYNC_TRANSFER_BLOCKING_JOB_STATUSES.includes(status), true);
    assert.equal(PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES.includes(status), true);
  }
});

test('project configs replace host paths with portable refs and resolve only within the same project', async () => {
  await withProject(async root => {
    const config = {
      dataset: path.join(root, 'datasets', 'portraits'),
      output: path.join(root, 'outputs', 'latest'),
      external: path.join(path.dirname(root), 'external-model.safetensors'),
    };
    const portable = portableizeProjectConfig(config, root, 'project-1');
    assert.equal(portable.dataset, 'aitk-project://project-1/datasets/portraits');
    assert.equal(portable.output, 'aitk-project://project-1/outputs/latest');
    assert.equal(portable.external, config.external);
    assert.deepEqual(resolvePortableProjectConfig(portable, root, 'project-1'), config);
    assert.throws(
      () => resolvePortableProjectConfig({ dataset: 'aitk-project://other/datasets/x' }, root, 'project-1'),
      error => error?.code === 'PROJECT_SYNC_CROSS_PROJECT_REF',
    );
  });
});

test('profile exclusions omit cache, transient files, credentials, and secret key material', () => {
  assert.equal(isProjectSyncPathExcluded('cache/previews/x.png', 'full'), true);
  assert.equal(isProjectSyncPathExcluded('datasets/x/image.png.part', 'full'), true);
  assert.equal(isProjectSyncPathExcluded('configs/.env', 'full'), true);
  assert.equal(isProjectSyncPathExcluded('configs/credentials.json', 'full'), true);
  assert.equal(isProjectSyncPathExcluded('datasets/encrypted/objects/0001.bin', 'full'), false);
  assert.equal(isProjectSyncPathExcluded('outputs/sample.png', 'results'), false);
  assert.equal(isProjectSyncPathExcluded('datasets/sample.png', 'results'), true);
  assert.deepEqual(
    collectCredentialConfigKeys({ provider: { api_key: 'do-not-transfer' }, sampling: { max_new_tokens: 64 } }),
    ['api_key'],
  );
});

test('manifests are SHA-256 addressed, profile aware, and do not create folders', async () => {
  await withProject(async root => {
    await fs.writeFile(path.join(root, 'datasets', 'photo.png'), 'dataset bytes');
    await fs.writeFile(path.join(root, 'outputs', 'sample.png'), 'output bytes');
    await fs.writeFile(path.join(root, 'cache', 'ignore.bin'), 'cache bytes');
    await fs.writeFile(path.join(root, 'configs', '.env'), 'TOKEN=nope');
    const results = await buildProjectSyncManifest(root, 'project-1', 'results');
    assert.deepEqual(results.files.map(file => file.path), ['outputs/sample.png']);
    const full = await buildProjectSyncManifest(root, 'project-1', 'full');
    assert.deepEqual(full.files.map(file => file.path), ['datasets/photo.png', 'outputs/sample.png']);
    assert.match(full.hash, /^[a-f0-9]{64}$/);
  });
});

test('manifest creation refuses credential-bearing project config files without exposing the value', async () => {
  await withProject(async root => {
    await fs.writeFile(path.join(root, 'configs', 'train.json'), JSON.stringify({ api_token: 'sensitive-value' }));
    await assert.rejects(
      buildProjectSyncManifest(root, 'project-1', 'launch'),
      error => error?.code === 'PROJECT_SYNC_SECRET_BLOCKED' && !error.message.includes('sensitive-value'),
    );
  });
});

test('manifest diff and three-way conflicts cover modify and delete combinations', () => {
  const baseA = entry('configs/a.yaml', 'base');
  const homeA = entry('configs/a.yaml', 'home');
  const workerA = entry('configs/a.yaml', 'worker');
  const baseB = entry('configs/b.yaml', 'base-b');
  const workerB = entry('configs/b.yaml', 'worker-b');
  const baseC = entry('configs/c.yaml', 'base-c');
  const homeC = entry('configs/c.yaml', 'home-c');
  const conflicts = detectProjectSyncConflicts(
    manifest([baseA, baseB, baseC]),
    manifest([homeA, homeC]),
    manifest([workerA, workerB]),
  );
  assert.deepEqual(
    conflicts.map(conflict => [conflict.path, conflict.kind]),
    [
      ['configs/a.yaml', 'both-modified'],
      ['configs/b.yaml', 'home-deleted-worker-modified'],
      ['configs/c.yaml', 'worker-deleted-home-modified'],
    ],
  );
  const diff = diffProjectSyncManifests(manifest([homeA, homeC]), manifest([workerA, workerB]));
  assert.deepEqual(diff.add_or_update.map(file => file.path), ['configs/a.yaml', 'configs/c.yaml']);
  assert.deepEqual(diff.delete, ['configs/b.yaml']);
  assert.equal(deterministicKeepBothPath('configs/a.yaml', 'worker-1234-abcd'), 'configs/a.worker-worker12.yaml');
});

test('remote manifests are runtime validated, hash checked, and duplicate paths rejected', () => {
  const files = [entry('outputs/a.png', 'a')];
  const value = {
    ...manifest(files),
    profile: 'results',
    hash: hashManifestEntries(files),
  };
  assert.deepEqual(parseProjectSyncManifest(value, 'project-1', 'results').files, files);
  assert.throws(
    () => parseProjectSyncManifest({ ...value, hash: '0'.repeat(64) }, 'project-1', 'results'),
    error => error?.code === 'PROJECT_SYNC_HASH_MISMATCH',
  );
  const duplicateFiles = [files[0], files[0]];
  assert.throws(
    () =>
      parseProjectSyncManifest(
        { ...value, files: duplicateFiles, hash: hashManifestEntries(duplicateFiles) },
        'project-1',
        'results',
      ),
    error => error?.code === 'PROJECT_SYNC_MANIFEST_INVALID',
  );
});

test('portable project job snapshots preserve immutable IDs, metadata, dates, and configs', () => {
  const jobs = [
    {
      id: 'job-portable-1',
      name: 'Portable training job',
      source_worker_id: 'local',
      remote_job_id: null,
      remote_sync_at: null,
      remote_error: null,
      gpu_ids: '0',
      job_config: { job: 'extension', config: { name: 'portable', datasets: ['aitk-project://project-1/datasets/a'] } },
      status: 'queued',
      stop: false,
      return_to_queue: false,
      step: 42,
      info: 'Queued for later',
      speed_string: '',
      queue_position: 1000,
      job_type: 'train',
      job_ref: null,
      save_now: false,
      sample_now: false,
      created_at: '2026-01-01T00:00:00.000Z',
      updated_at: '2026-01-02T00:00:00.000Z',
    },
  ];
  const snapshot = {
    protocol: 'project-sync-v1',
    project_id: 'project-1',
    home_instance_id: 'home-instance-1',
    generated_at: '2026-01-03T00:00:00.000Z',
    hash: hashPortableProjectJobs(jobs),
    jobs,
  };
  assert.deepEqual(parsePortableProjectJobSnapshot(snapshot, 'project-1', 'home-instance-1'), snapshot);
  assert.throws(
    () => parsePortableProjectJobSnapshot({ ...snapshot, jobs: [{ ...jobs[0], id: 'changed' }] }, 'project-1', 'home-instance-1'),
    error => error?.code === 'PROJECT_SYNC_HASH_MISMATCH',
  );
});

test('resumable 8 MiB chunks enforce offsets, verify hashes, and atomically commit files', async () => {
  await withProject(async root => {
    const bytes = Buffer.alloc(PROJECT_SYNC_CHUNK_BYTES + 19, 7);
    const sha256 = createHash('sha256').update(bytes).digest('hex');
    const first = await appendProjectSyncChunk(root, sha256, bytes.length, 0, bytes.subarray(0, PROJECT_SYNC_CHUNK_BYTES));
    assert.equal(first.complete, false);
    assert.equal(first.received, PROJECT_SYNC_CHUNK_BYTES);
    await assert.rejects(
      appendProjectSyncChunk(root, sha256, bytes.length, 0, bytes.subarray(PROJECT_SYNC_CHUNK_BYTES)),
      error => error?.code === 'PROJECT_SYNC_OFFSET_MISMATCH',
    );
    const complete = await appendProjectSyncChunk(
      root,
      sha256,
      bytes.length,
      PROJECT_SYNC_CHUNK_BYTES,
      bytes.subarray(PROJECT_SYNC_CHUNK_BYTES),
    );
    assert.equal(complete.complete, true);
    assert.deepEqual(await getProjectSyncChunkReceipt(root, sha256, bytes.length), complete);
    const file = {
      path: 'outputs/samples/result.bin',
      size: bytes.length,
      sha256,
      modified_at: new Date().toISOString(),
    };
    await commitProjectSyncPlan(root, {
      project_id: randomUUID(),
      profile: 'results',
      operation_id: randomUUID(),
      files: [file],
    });
    assert.deepEqual(await fs.readFile(path.join(root, 'outputs', 'samples', 'result.bin')), bytes);
  });
});

test('multi-file commit rolls every path back after an injected mid-install failure', async () => {
  await withProject(async root => {
    const targets = [
      { path: 'outputs/a.bin', old: Buffer.from('old-a'), next: Buffer.from('new-a') },
      { path: 'outputs/b.bin', old: Buffer.from('old-b'), next: Buffer.from('new-b') },
    ];
    for (const target of targets) {
      const absolute = path.join(root, ...target.path.split('/'));
      await fs.mkdir(path.dirname(absolute), { recursive: true });
      await fs.writeFile(absolute, target.old);
      target.sha256 = createHash('sha256').update(target.next).digest('hex');
      await appendProjectSyncChunk(root, target.sha256, target.next.length, 0, target.next);
    }
    const plan = {
      project_id: 'project-1',
      profile: 'results',
      operation_id: randomUUID(),
      files: targets.map(target => ({
        path: target.path,
        size: target.next.length,
        sha256: target.sha256,
        modified_at: new Date().toISOString(),
      })),
    };
    await assert.rejects(
      commitProjectSyncPlan(root, plan, { failAfterInstalledFiles: 1 }),
      error => error?.code === 'PROJECT_SYNC_TEST_FAILURE',
    );
    for (const target of targets) {
      assert.deepEqual(await fs.readFile(path.join(root, ...target.path.split('/'))), target.old);
    }
    await commitProjectSyncPlan(root, plan);
    for (const target of targets) {
      assert.deepEqual(await fs.readFile(path.join(root, ...target.path.split('/'))), target.next);
    }
  });
});

test('hash mismatch removes poisoned partial data', async () => {
  await withProject(async root => {
    const wrongHash = createHash('sha256').update('expected').digest('hex');
    await assert.rejects(
      appendProjectSyncChunk(root, wrongHash, 6, 0, Buffer.from('actual')),
      error => error?.code === 'PROJECT_SYNC_HASH_MISMATCH',
    );
    const receipt = await getProjectSyncChunkReceipt(root, wrongHash, 6);
    assert.equal(receipt.received, 0);
    assert.equal(receipt.complete, false);
  });
});

test('range parsing supports full, bounded, open, and suffix reads', () => {
  assert.deepEqual(parseHttpByteRange(null, 100), { start: 0, end: 99, partial: false });
  assert.deepEqual(parseHttpByteRange('bytes=10-19', 100), { start: 10, end: 19, partial: true });
  assert.deepEqual(parseHttpByteRange('bytes=90-', 100), { start: 90, end: 99, partial: true });
  assert.deepEqual(parseHttpByteRange('bytes=-10', 100), { start: 90, end: 99, partial: true });
  assert.throws(() => parseHttpByteRange('bytes=100-101', 100), error => error?.status === 416);
});

test('sync containment rejects paths crossing a symlinked directory', async t => {
  await withProject(async root => {
    const outside = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-sync-outside-'));
    try {
      const link = path.join(root, 'outputs', 'escape');
      try {
        await fs.symlink(outside, link, process.platform === 'win32' ? 'junction' : 'dir');
      } catch (error) {
        if (error?.code === 'EPERM' || error?.code === 'EACCES') {
          t.skip('Creating a test symlink is not permitted on this host');
          return;
        }
        throw error;
      }
      await assert.rejects(
        assertProjectSyncPathContained(root, path.join(link, 'leak.bin')),
        error => error?.code === 'PROJECT_SYNC_PATH_INVALID',
      );
    } finally {
      await fs.rm(outside, { recursive: true, force: true });
    }
  });
});
