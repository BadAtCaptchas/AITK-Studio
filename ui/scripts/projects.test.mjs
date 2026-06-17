import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { cleanProjectSlug, getProjectRoots, isPathInside, PROJECT_FOLDERS } from '../dist/src/server/projects.js';
import { copyDatasetBetweenRoots } from '../dist/src/server/datasetCopy.js';
import {
  DatasetScopeError,
  isPathInside as isDatasetScopePathInside,
  rejectRemoteProjectScope,
  resolveDatasetScope,
} from '../dist/src/server/datasetScope.js';
import { getDatasetsRoot, getProjectsRoot, getTrainingFolder } from '../dist/src/server/settings.js';

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

test('project dataset edits reject remote workers', () => {
  assert.throws(
    () => rejectRemoteProjectScope('worker-1', 'project-1'),
    error => error instanceof DatasetScopeError && error.status === 400,
  );
  assert.doesNotThrow(() => rejectRemoteProjectScope('local', 'project-1'));
  assert.doesNotThrow(() => rejectRemoteProjectScope('worker-1', null));
});
