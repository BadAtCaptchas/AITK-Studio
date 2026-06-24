import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { cleanProjectSlug, getProjectRoots, isPathInside, PROJECT_FOLDERS } from '../dist/src/server/projects.js';
import { copyDatasetBetweenRoots } from '../dist/src/server/datasetCopy.js';
import { deleteDatasetFolder } from '../dist/src/server/datasetDelete.js';
import { transferProjectDatasetsToGlobal } from '../dist/src/server/datasetTransfer.js';
import {
  DatasetScopeError,
  isPathInside as isDatasetScopePathInside,
  rejectRemoteProjectScope,
  resolveDatasetScope,
} from '../dist/src/server/datasetScope.js';
import {
  areProjectsEnabled,
  assertProjectsEnabled,
  flushCache,
  getDatasetsRoot,
  getProjectsRoot,
  getTrainingFolder,
  PROJECT_SPACES_DISABLED_MESSAGE,
  PROJECTS_ENABLED_KEY,
} from '../dist/src/server/settings.js';
import { db } from '../dist/src/server/db.js';

async function withProjectsEnabledSetting(value, fn) {
  const previous = await db.settings.get(PROJECTS_ENABLED_KEY);
  try {
    if (value === null) {
      await db.settings.delete(PROJECTS_ENABLED_KEY);
    } else {
      await db.settings.upsert(PROJECTS_ENABLED_KEY, value);
    }
    flushCache();
    return await fn();
  } finally {
    if (previous) {
      await db.settings.upsert(PROJECTS_ENABLED_KEY, previous.value);
    } else {
      await db.settings.delete(PROJECTS_ENABLED_KEY);
    }
    flushCache();
  }
}

async function listDatasetSummariesForTest(datasetsRoot) {
  await fs.mkdir(datasetsRoot, { recursive: true });
  const entries = await fs.readdir(datasetsRoot, { withFileTypes: true });
  return entries
    .filter(entry => entry.isDirectory() && !entry.name.startsWith('.'))
    .map(entry => ({
      name: entry.name,
      encrypted: false,
      source: 'local',
      worker_id: 'local',
      worker_name: 'Local',
      path: path.join(datasetsRoot, entry.name),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

function createTransferDeps({ projectID = 'project-1', projectDatasetsRoot, globalDatasetsRoot, jobs = [], updateJobConfig }) {
  const jobStore = jobs.map(job => ({ ...job }));
  const deps = {
    resolveDatasetScope: async projectIdentifier =>
      projectIdentifier
        ? {
            project: { id: projectID },
            projectID,
            datasetsRoot: projectDatasetsRoot,
            trainingRoot: path.join(path.dirname(projectDatasetsRoot), 'runs'),
          }
        : {
            project: null,
            projectID: null,
            datasetsRoot: globalDatasetsRoot,
            trainingRoot: path.join(path.dirname(globalDatasetsRoot), 'runs'),
          },
    listDatasetSummaries: listDatasetSummariesForTest,
    copyDatasetBetweenRoots,
    deleteDatasetFolder,
    listProjectJobs: async () => jobStore,
    updateJobConfig:
      updateJobConfig ||
      (async (jobID, jobConfig) => {
        const job = jobStore.find(item => item.id === jobID);
        if (!job) throw new Error(`Job not found: ${jobID}`);
        job.job_config = jobConfig;
        return job;
      }),
  };
  return { deps, jobs: jobStore };
}

test('cleanProjectSlug creates stable URL-safe project slugs', () => {
  assert.equal(cleanProjectSlug('Flux Portrait Set'), 'flux-portrait-set');
  assert.equal(cleanProjectSlug('  Product_pack shots!!  '), 'product-pack-shots');
  assert.equal(cleanProjectSlug('../bad/project'), 'bad-project');
  assert.equal(cleanProjectSlug(''), '');
});

test('isPathInside accepts children and rejects traversal siblings', () => {
  const root = path.join(os.tmpdir(), 'aitk-project-root');
  assert.equal(isPathInside(root, root), true);
  assert.equal(isPathInside(root, path.join(root, 'datasets', 'portraits')), true);
  assert.equal(isPathInside(root, path.join(root, '..', 'other-project')), false);
});

test('getProjectRoots resolves sandbox folders and rejects roots outside PROJECTS_FOLDER', async () => {
  const projectsRoot = path.resolve(await getProjectsRoot());
  const project = {
    id: 'test-project',
    slug: 'test-project',
    name: 'Test Project',
    description: '',
    badge_asset: null,
    root_path: '',
    created_at: new Date(),
    updated_at: new Date(),
  };

  const roots = await getProjectRoots(project);
  assert.equal(roots.root, path.join(projectsRoot, project.slug));
  for (const folder of PROJECT_FOLDERS) {
    assert.equal(roots[folder], path.join(roots.root, folder));
  }

  await assert.rejects(
    () =>
      getProjectRoots({
        ...project,
        root_path: path.join(path.dirname(projectsRoot), 'outside-project-root'),
      }),
    /Project root must be inside PROJECTS_FOLDER/,
  );
});

test('resolveDatasetScope keeps omitted project_id on global roots', async () => {
  const scope = await resolveDatasetScope(null);
  assert.equal(scope.project, null);
  assert.equal(scope.projectID, null);
  assert.equal(scope.datasetsRoot, await getDatasetsRoot());
  assert.equal(scope.trainingRoot, await getTrainingFolder());
});

test('project spaces are disabled by default and can be enabled', async () => {
  await withProjectsEnabledSetting(null, async () => {
    assert.equal(await areProjectsEnabled(), false);
    await assert.rejects(
      () => assertProjectsEnabled(),
      error => error?.status === 403 && error.message === PROJECT_SPACES_DISABLED_MESSAGE,
    );
  });

  await withProjectsEnabledSetting('true', async () => {
    assert.equal(await areProjectsEnabled(), true);
    await assert.doesNotReject(() => assertProjectsEnabled());
  });
});

test('disabled project spaces reject project-scoped dataset resolution but keep global scope available', async () => {
  await withProjectsEnabledSetting('false', async () => {
    await assert.rejects(
      () => resolveDatasetScope('project-slug'),
      error => error instanceof DatasetScopeError && error.status === 403 && error.message === PROJECT_SPACES_DISABLED_MESSAGE,
    );

    const globalScope = await resolveDatasetScope(null);
    assert.equal(globalScope.projectID, null);
    assert.equal(globalScope.datasetsRoot, await getDatasetsRoot());
  });
});

test('dataset roots isolate same dataset names between global and project spaces', async () => {
  const project = {
    id: 'isolation-project',
    slug: 'isolation-project',
    name: 'Isolation Project',
    description: '',
    badge_asset: null,
    root_path: '',
    created_at: new Date(),
    updated_at: new Date(),
  };
  const projectRoots = await getProjectRoots(project);
  const datasetName = 'same-name';
  const globalDatasetPath = path.resolve(await getDatasetsRoot(), datasetName);
  const projectDatasetPath = path.resolve(projectRoots.datasets, datasetName);

  assert.notEqual(projectDatasetPath, globalDatasetPath);
  assert.equal(isDatasetScopePathInside(projectRoots.datasets, projectDatasetPath), true);
  assert.equal(isDatasetScopePathInside(projectRoots.datasets, path.join(projectRoots.datasets, '..', '..', datasetName)), false);
});

test('copyDatasetBetweenRoots imports a global dataset into a project dataset root', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-dataset-copy-'));
  const globalRoot = path.join(tempRoot, 'global-datasets');
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const sourceDataset = path.join(globalRoot, 'portrait-set');

  await fs.mkdir(sourceDataset, { recursive: true });
  await fs.writeFile(path.join(sourceDataset, 'sample.txt'), 'caption');

  const firstCopy = await copyDatasetBetweenRoots({
    datasetPath: sourceDataset,
    sourceDatasetsRoot: globalRoot,
    destinationDatasetsRoot: projectDatasetRoot,
    requestedName: 'portrait-set',
  });
  assert.equal(firstCopy.name, 'portrait-set');
  assert.equal(await fs.readFile(path.join(firstCopy.path, 'sample.txt'), 'utf8'), 'caption');

  const secondCopy = await copyDatasetBetweenRoots({
    datasetPath: sourceDataset,
    sourceDatasetsRoot: globalRoot,
    destinationDatasetsRoot: projectDatasetRoot,
    requestedName: 'portrait-set',
  });
  assert.equal(secondCopy.name, 'portrait-set_2');
});

test('copyDatasetBetweenRoots rejects source traversal outside the declared root', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-dataset-copy-reject-'));
  const globalRoot = path.join(tempRoot, 'global-datasets');
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const outsideDataset = path.join(tempRoot, 'outside-dataset');

  await fs.mkdir(globalRoot, { recursive: true });
  await fs.mkdir(outsideDataset, { recursive: true });

  await assert.rejects(
    () =>
      copyDatasetBetweenRoots({
        datasetPath: outsideDataset,
        sourceDatasetsRoot: globalRoot,
        destinationDatasetsRoot: projectDatasetRoot,
        requestedName: 'outside-dataset',
      }),
    /inside .*datasets folder/,
  );
});

test('deleteDatasetFolder removes only the project-scoped dataset', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-dataset-delete-'));
  const globalRoot = path.join(tempRoot, 'global-datasets');
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const datasetName = 'same-name';
  const globalDataset = path.join(globalRoot, datasetName);
  const projectDataset = path.join(projectDatasetRoot, datasetName);

  await fs.mkdir(globalDataset, { recursive: true });
  await fs.mkdir(projectDataset, { recursive: true });
  await fs.writeFile(path.join(globalDataset, 'sample.txt'), 'global');
  await fs.writeFile(path.join(projectDataset, 'sample.txt'), 'project');

  const result = await deleteDatasetFolder(projectDatasetRoot, datasetName);

  assert.equal(result.success, true);
  assert.equal(result.deleted, true);
  await assert.rejects(() => fs.access(projectDataset));
  assert.equal(await fs.readFile(path.join(globalDataset, 'sample.txt'), 'utf8'), 'global');
});

test('deleteDatasetFolder rejects traversal and ignores missing datasets', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-dataset-delete-invalid-'));
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const dataset = path.join(projectDatasetRoot, 'keep-me');

  await fs.mkdir(dataset, { recursive: true });
  await fs.writeFile(path.join(dataset, 'sample.txt'), 'project');

  await assert.rejects(
    () => deleteDatasetFolder(projectDatasetRoot, '../keep-me'),
    error => error?.status === 400 && /Invalid dataset path/.test(error.message),
  );
  assert.equal(await fs.readFile(path.join(dataset, 'sample.txt'), 'utf8'), 'project');

  const missing = await deleteDatasetFolder(projectDatasetRoot, 'missing');
  assert.equal(missing.success, true);
  assert.equal(missing.deleted, false);
});

test('transferProjectDatasetsToGlobal copies a single project dataset to global', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-transfer-copy-'));
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const globalRoot = path.join(tempRoot, 'global-datasets');
  const sourceDataset = path.join(projectDatasetRoot, 'portrait-set');

  await fs.mkdir(sourceDataset, { recursive: true });
  await fs.writeFile(path.join(sourceDataset, 'sample.txt'), 'project');

  const { deps } = createTransferDeps({ projectDatasetsRoot: projectDatasetRoot, globalDatasetsRoot: globalRoot });
  const result = await transferProjectDatasetsToGlobal(
    { sourceProjectID: 'project-1', operation: 'copy', datasetNames: ['portrait-set'] },
    deps,
  );

  assert.equal(result.copiedCount, 1);
  assert.equal(result.movedCount, 0);
  assert.equal(result.failedCount, 0);
  assert.equal(result.results[0].destinationName, 'portrait-set');
  assert.equal(await fs.readFile(path.join(globalRoot, 'portrait-set', 'sample.txt'), 'utf8'), 'project');
  assert.equal(await fs.readFile(path.join(sourceDataset, 'sample.txt'), 'utf8'), 'project');
});

test('transferProjectDatasetsToGlobal suffixes global destination conflicts', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-transfer-conflict-'));
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const globalRoot = path.join(tempRoot, 'global-datasets');

  await fs.mkdir(path.join(projectDatasetRoot, 'same-name'), { recursive: true });
  await fs.mkdir(path.join(globalRoot, 'same-name'), { recursive: true });
  await fs.writeFile(path.join(projectDatasetRoot, 'same-name', 'sample.txt'), 'project');
  await fs.writeFile(path.join(globalRoot, 'same-name', 'sample.txt'), 'global');

  const { deps } = createTransferDeps({ projectDatasetsRoot: projectDatasetRoot, globalDatasetsRoot: globalRoot });
  const result = await transferProjectDatasetsToGlobal(
    { sourceProjectID: 'project-1', operation: 'copy', datasetNames: ['same-name'] },
    deps,
  );

  assert.equal(result.results[0].destinationName, 'same-name_2');
  assert.equal(await fs.readFile(path.join(globalRoot, 'same-name', 'sample.txt'), 'utf8'), 'global');
  assert.equal(await fs.readFile(path.join(globalRoot, 'same-name_2', 'sample.txt'), 'utf8'), 'project');
});

test('transferProjectDatasetsToGlobal moves one dataset and rewrites project job refs', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-transfer-move-'));
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const globalRoot = path.join(tempRoot, 'global-datasets');
  const sourceDataset = path.join(projectDatasetRoot, 'portrait-set');
  const sourceControlPath = path.join(sourceDataset, 'controls');

  await fs.mkdir(sourceControlPath, { recursive: true });
  await fs.writeFile(path.join(sourceDataset, 'sample.txt'), 'project');

  const jobs = [
    {
      id: 'job-1',
      name: 'portrait-job',
      status: 'stopped',
      job_config: JSON.stringify({
        config: {
          process: [
            {
              datasets: [
                {
                  folder_path: sourceDataset,
                  control_path: [sourceControlPath],
                  mask_path: path.join(sourceDataset, 'masks'),
                },
              ],
              caption: { path_to_caption: sourceDataset },
            },
          ],
        },
      }),
    },
  ];
  const { deps, jobs: jobStore } = createTransferDeps({ projectDatasetsRoot: projectDatasetRoot, globalDatasetsRoot: globalRoot, jobs });

  const result = await transferProjectDatasetsToGlobal(
    { sourceProjectID: 'project-1', operation: 'move', datasetNames: ['portrait-set'] },
    deps,
  );
  const destinationPath = path.join(globalRoot, 'portrait-set');
  const rewrittenConfig = JSON.parse(jobStore[0].job_config);
  const rewrittenProcess = rewrittenConfig.config.process[0];

  assert.equal(result.movedCount, 1);
  assert.equal(result.failedCount, 0);
  assert.equal(result.results[0].rewrittenJobCount, 1);
  await assert.rejects(() => fs.access(sourceDataset));
  assert.equal(await fs.readFile(path.join(destinationPath, 'sample.txt'), 'utf8'), 'project');
  assert.equal(rewrittenProcess.datasets[0].folder_path, destinationPath);
  assert.equal(rewrittenProcess.datasets[0].control_path[0], path.join(destinationPath, 'controls'));
  assert.equal(rewrittenProcess.datasets[0].mask_path, path.join(destinationPath, 'masks'));
  assert.equal(rewrittenProcess.caption.path_to_caption, destinationPath);
});

test('transferProjectDatasetsToGlobal copies all project datasets without deleting sources', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-transfer-copy-all-'));
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const globalRoot = path.join(tempRoot, 'global-datasets');

  await fs.mkdir(path.join(projectDatasetRoot, 'a'), { recursive: true });
  await fs.mkdir(path.join(projectDatasetRoot, 'b'), { recursive: true });
  await fs.mkdir(path.join(projectDatasetRoot, '.hidden'), { recursive: true });
  await fs.writeFile(path.join(projectDatasetRoot, 'a', 'sample.txt'), 'a');
  await fs.writeFile(path.join(projectDatasetRoot, 'b', 'sample.txt'), 'b');

  const { deps } = createTransferDeps({ projectDatasetsRoot: projectDatasetRoot, globalDatasetsRoot: globalRoot });
  const result = await transferProjectDatasetsToGlobal({ sourceProjectID: 'project-1', operation: 'copy', all: true }, deps);

  assert.deepEqual(result.results.map(item => item.sourceName), ['a', 'b']);
  assert.equal(result.copiedCount, 2);
  assert.equal(result.movedCount, 0);
  assert.equal(await fs.readFile(path.join(projectDatasetRoot, 'a', 'sample.txt'), 'utf8'), 'a');
  assert.equal(await fs.readFile(path.join(projectDatasetRoot, 'b', 'sample.txt'), 'utf8'), 'b');
  assert.equal(await fs.readFile(path.join(globalRoot, 'a', 'sample.txt'), 'utf8'), 'a');
  assert.equal(await fs.readFile(path.join(globalRoot, 'b', 'sample.txt'), 'utf8'), 'b');
  await assert.rejects(() => fs.access(path.join(globalRoot, '.hidden')));
});

test('transferProjectDatasetsToGlobal moves all datasets and rewrites refs', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-transfer-move-all-'));
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const globalRoot = path.join(tempRoot, 'global-datasets');
  const sourceA = path.join(projectDatasetRoot, 'a');
  const sourceB = path.join(projectDatasetRoot, 'b');

  await fs.mkdir(sourceA, { recursive: true });
  await fs.mkdir(sourceB, { recursive: true });
  await fs.writeFile(path.join(sourceA, 'sample.txt'), 'a');
  await fs.writeFile(path.join(sourceB, 'sample.txt'), 'b');

  const jobs = [
    {
      id: 'job-1',
      name: 'bulk-job',
      status: 'stopped',
      job_config: JSON.stringify({
        config: {
          process: [
            {
              datasets: [{ folder_path: sourceA }, { folder_path: sourceB }],
            },
          ],
        },
      }),
    },
  ];
  const { deps, jobs: jobStore } = createTransferDeps({ projectDatasetsRoot: projectDatasetRoot, globalDatasetsRoot: globalRoot, jobs });
  const result = await transferProjectDatasetsToGlobal({ sourceProjectID: 'project-1', operation: 'move', all: true }, deps);
  const rewrittenDatasets = JSON.parse(jobStore[0].job_config).config.process[0].datasets;

  assert.equal(result.movedCount, 2);
  assert.equal(result.failedCount, 0);
  await assert.rejects(() => fs.access(sourceA));
  await assert.rejects(() => fs.access(sourceB));
  assert.equal(rewrittenDatasets[0].folder_path, path.join(globalRoot, 'a'));
  assert.equal(rewrittenDatasets[1].folder_path, path.join(globalRoot, 'b'));
});

test('transferProjectDatasetsToGlobal keeps source when move rewrite fails', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-transfer-rewrite-fail-'));
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const globalRoot = path.join(tempRoot, 'global-datasets');
  const sourceDataset = path.join(projectDatasetRoot, 'portrait-set');

  await fs.mkdir(sourceDataset, { recursive: true });
  await fs.writeFile(path.join(sourceDataset, 'sample.txt'), 'project');

  const jobs = [
    {
      id: 'job-1',
      name: 'portrait-job',
      status: 'stopped',
      job_config: JSON.stringify({ config: { process: [{ datasets: [{ folder_path: sourceDataset }] }] } }),
    },
  ];
  const { deps } = createTransferDeps({
    projectDatasetsRoot: projectDatasetRoot,
    globalDatasetsRoot: globalRoot,
    jobs,
    updateJobConfig: async () => {
      throw new Error('database unavailable');
    },
  });
  const result = await transferProjectDatasetsToGlobal(
    { sourceProjectID: 'project-1', operation: 'move', datasetNames: ['portrait-set'] },
    deps,
  );

  assert.equal(result.copiedCount, 1);
  assert.equal(result.movedCount, 0);
  assert.equal(result.failedCount, 1);
  assert.match(result.results[0].error, /kept the project dataset/i);
  assert.equal(await fs.readFile(path.join(sourceDataset, 'sample.txt'), 'utf8'), 'project');
  assert.equal(await fs.readFile(path.join(globalRoot, 'portrait-set', 'sample.txt'), 'utf8'), 'project');
});

test('transferProjectDatasetsToGlobal blocks move for running referenced jobs', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-transfer-running-'));
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const globalRoot = path.join(tempRoot, 'global-datasets');
  const sourceDataset = path.join(projectDatasetRoot, 'portrait-set');

  await fs.mkdir(sourceDataset, { recursive: true });
  await fs.writeFile(path.join(sourceDataset, 'sample.txt'), 'project');

  const jobs = [
    {
      id: 'job-1',
      name: 'running-job',
      status: 'running',
      job_config: JSON.stringify({ config: { process: [{ datasets: [{ folder_path: sourceDataset }] }] } }),
    },
  ];
  const { deps } = createTransferDeps({ projectDatasetsRoot: projectDatasetRoot, globalDatasetsRoot: globalRoot, jobs });

  await assert.rejects(
    () => transferProjectDatasetsToGlobal({ sourceProjectID: 'project-1', operation: 'move', datasetNames: ['portrait-set'] }, deps),
    error => error?.status === 409 && /running-job/.test(error.message),
  );
  assert.equal(await fs.readFile(path.join(sourceDataset, 'sample.txt'), 'utf8'), 'project');
  await assert.rejects(() => fs.access(path.join(globalRoot, 'portrait-set')));
});

test('transferProjectDatasetsToGlobal rejects traversal dataset names', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-transfer-invalid-'));
  const projectDatasetRoot = path.join(tempRoot, 'projects', 'portraits', 'datasets');
  const globalRoot = path.join(tempRoot, 'global-datasets');
  const { deps } = createTransferDeps({ projectDatasetsRoot: projectDatasetRoot, globalDatasetsRoot: globalRoot });

  await assert.rejects(
    () => transferProjectDatasetsToGlobal({ sourceProjectID: 'project-1', operation: 'copy', datasetNames: ['../bad'] }, deps),
    /path separators/,
  );
});

test('project dataset edits reject remote workers', () => {
  assert.throws(
    () => rejectRemoteProjectScope('worker-1', 'project-1'),
    error => error instanceof DatasetScopeError && error.status === 400,
  );
  assert.doesNotThrow(() => rejectRemoteProjectScope('local', 'project-1'));
  assert.doesNotThrow(() => rejectRemoteProjectScope('worker-1', null));
});
