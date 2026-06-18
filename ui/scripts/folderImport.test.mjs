import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  FOLDER_IMPORT_SUPPORTED_EXTENSIONS,
  ROOT_CAPTION_FILE_NAME,
  createFlattenedFileNameAllocator,
  flattenedFolderImportFileNames,
  folderImportCaptionKey,
  isRootCaptionPath,
  isFolderImportCaptionSidecarPath,
  stripFolderImportRoot,
} = require('../dist/src/utils/folderImport.js');

test('folder import caption keys pair by normalized relative path stem', () => {
  assert.equal(folderImportCaptionKey('cats/a.png'), 'cats/a');
  assert.equal(folderImportCaptionKey('cats\\a.txt'), 'cats/a');
  assert.equal(folderImportCaptionKey('cats\\a.json'), 'cats/a');
  assert.notEqual(folderImportCaptionKey('cats/a.png'), folderImportCaptionKey('dogs/a.txt'));
});

test('folder import treats JSON captions as supported sidecars', () => {
  assert.equal(FOLDER_IMPORT_SUPPORTED_EXTENSIONS.has('.json'), true);
  assert.equal(FOLDER_IMPORT_SUPPORTED_EXTENSIONS.has('.jxl'), true);
  assert.equal(isFolderImportCaptionSidecarPath('cats/a.json'), true);
  assert.equal(isFolderImportCaptionSidecarPath('cats/a.jxl'), false);
  assert.equal(isFolderImportCaptionSidecarPath('cats/a.png'), false);
});

test('folder import reserves root caption metadata', () => {
  assert.equal(ROOT_CAPTION_FILE_NAME, 'ROOT_CAPTION.txt');
  assert.equal(isRootCaptionPath('ROOT_CAPTION.txt'), true);
  assert.equal(isRootCaptionPath('root_caption.TXT'), true);
  assert.equal(isRootCaptionPath('cats/ROOT_CAPTION.txt'), false);
  assert.equal(isFolderImportCaptionSidecarPath('ROOT_CAPTION.txt'), false);
  assert.equal(isFolderImportCaptionSidecarPath('cats/ROOT_CAPTION.txt'), true);
});

test('folder import strips one top-level folder for separate imports', () => {
  assert.equal(stripFolderImportRoot('cats/a.png', 'a.png'), 'a.png');
  assert.equal(stripFolderImportRoot('cats/nested/a.png', 'a.png'), 'nested/a.png');
  assert.equal(stripFolderImportRoot('a.png', 'a.png'), 'a.png');
});

test('folder import flattened names are deterministic and case-insensitively unique', () => {
  assert.deepEqual(
    flattenedFolderImportFileNames([
      'cats/image.png',
      'dogs/image.png',
      'birds/IMAGE.PNG',
      'fish/image_2.png',
      'fish/image.png',
    ]),
    ['image.png', 'image_2.png', 'IMAGE_3.PNG', 'image_2_2.png', 'image_4.png'],
  );
});

test('folder import allocator respects existing names', () => {
  const allocate = createFlattenedFileNameAllocator(['image.png', 'image_2.png']);
  assert.equal(allocate('new/image.png'), 'image_3.png');
});
