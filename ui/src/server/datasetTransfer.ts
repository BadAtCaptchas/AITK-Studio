import { isRecord } from './commandInput';
import { isEncryptedDatasetFolder } from './encryptedDatasets';
import { listLayeredImages, validateLayeredImageAssets } from './layeredImages';
import archiver from 'archiver';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
export { extractZipSafely } from './safeArchive';
import {
  isPathInside as isArchivePathInside,
  listFilesRecursive,
  safeNameSegment,
  shouldIncludeDatasetExportPath,
  validateArchiveEntryName,
} from './trainingJobTransfer';

export const DATASET_EXPORT_FORMAT = 'ai-toolkit-dataset-export';
export const DATASET_EXPORT_VERSION = 1;

export type DatasetExportManifest = {
  format: typeof DATASET_EXPORT_FORMAT;
  version: typeof DATASET_EXPORT_VERSION;
  exportedAt: string;
  source: {
    app: 'ai-toolkit';
    datasetName: string;
  };
  dataset: {
    name: string;
    archivePath: 'dataset';
    encrypted: boolean;
  };
};

export function datasetExportFileName(datasetName: string) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  return `${safeNameSegment(datasetName, 'dataset')}_${timestamp}.aitk-dataset.zip`;
}

export async function createDatasetExportArchive(datasetName: string, datasetFolder: string, outputPath: string) {
  const realDatasetFolder = await fsp.realpath(datasetFolder).catch(() => path.resolve(datasetFolder));
  await listLayeredImages(realDatasetFolder);
  const files = await listFilesRecursive(realDatasetFolder, shouldIncludeDatasetExportPath);
  const manifest: DatasetExportManifest = {
    format: DATASET_EXPORT_FORMAT,
    version: DATASET_EXPORT_VERSION,
    exportedAt: new Date().toISOString(),
    source: {
      app: 'ai-toolkit',
      datasetName,
    },
    dataset: {
      name: datasetName,
      archivePath: 'dataset',
      encrypted: isEncryptedDatasetFolder(realDatasetFolder),
    },
  };

  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  await new Promise<void>((resolve, reject) => {
    const output = fs.createWriteStream(outputPath);
    const archive = archiver('zip', { zlib: { level: 9 } });

    output.on('close', resolve);
    output.on('error', reject);
    archive.on('error', reject);
    archive.pipe(output);
    archive.append(JSON.stringify(manifest, null, 2), { name: 'manifest.json' });
    if (files.length === 0) archive.append('', { name: 'dataset/.empty' });
    for (const relativePath of files) {
      archive.file(path.join(realDatasetFolder, relativePath), {
        name: path.posix.join('dataset', relativePath.replace(/\\/g, '/')),
      });
    }
    archive.finalize().catch(reject);
  });

  return manifest;
}


export async function readDatasetExportManifest(extractRoot: string): Promise<DatasetExportManifest> {
  const filename = path.join(extractRoot, 'manifest.json');
  if ((await fsp.stat(filename)).size > 2 * 1024 ** 2) throw new Error('Dataset manifest exceeds the JSON limit');
  const manifest: unknown = JSON.parse(await fsp.readFile(filename, 'utf-8'));
  if (!isRecord(manifest) || manifest.format !== DATASET_EXPORT_FORMAT || manifest.version !== DATASET_EXPORT_VERSION ||
      !isRecord(manifest.dataset) || manifest.dataset.archivePath !== 'dataset' || typeof manifest.dataset.name !== 'string' ||
      manifest.dataset.name.length > 512 || typeof manifest.dataset.encrypted !== 'boolean' || !isRecord(manifest.source) ||
      manifest.source.app !== 'ai-toolkit' || typeof manifest.source.datasetName !== 'string' || typeof manifest.exportedAt !== 'string') {
    throw new Error('Unsupported dataset export archive');
  }
  await validateLayeredImageAssets(getExtractedDatasetPath(extractRoot, manifest.dataset.archivePath));
  return { format: DATASET_EXPORT_FORMAT, version: DATASET_EXPORT_VERSION, exportedAt: manifest.exportedAt,
    source: { app: 'ai-toolkit', datasetName: manifest.source.datasetName },
    dataset: { name: manifest.dataset.name, archivePath: 'dataset', encrypted: manifest.dataset.encrypted } };
}

export function getExtractedDatasetPath(extractRoot: string, archivePath: string) {
  const normalized = validateArchiveEntryName(archivePath);
  const resolved = path.resolve(extractRoot, ...normalized.split('/'));
  if (!isArchivePathInside(extractRoot, resolved)) {
    throw new Error(`Archive path escapes import folder: ${archivePath}`);
  }
  return resolved;
}
