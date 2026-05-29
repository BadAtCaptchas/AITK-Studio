import archiver from 'archiver';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import yauzl from 'yauzl';
import { isEncryptedDatasetFolder } from './encryptedDatasets';
import {
  isPathInside,
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

export async function extractZipSafely(zipPath: string, destination: string) {
  const destinationRoot = path.resolve(destination);
  await fsp.mkdir(destinationRoot, { recursive: true });

  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const fail = (error: Error) => {
      if (!settled) {
        settled = true;
        reject(error);
      }
    };

    yauzl.open(zipPath, { lazyEntries: true, validateEntrySizes: true }, (openError, zipFile) => {
      if (openError || !zipFile) {
        fail(openError || new Error('Could not open archive'));
        return;
      }

      zipFile.on('error', fail);
      zipFile.on('end', () => {
        if (!settled) {
          settled = true;
          resolve();
        }
      });

      zipFile.readEntry();
      zipFile.on('entry', entry => {
        let normalizedName: string;
        try {
          normalizedName = validateArchiveEntryName(entry.fileName);
        } catch (error) {
          zipFile.close();
          fail(error as Error);
          return;
        }

        const targetPath = path.resolve(destinationRoot, ...normalizedName.split('/'));
        if (!isPathInside(destinationRoot, targetPath)) {
          zipFile.close();
          fail(new Error(`Archive entry escapes import folder: ${entry.fileName}`));
          return;
        }

        if (/\/$/.test(normalizedName)) {
          fsp
            .mkdir(targetPath, { recursive: true })
            .then(() => zipFile.readEntry())
            .catch(error => {
              zipFile.close();
              fail(error);
            });
          return;
        }

        zipFile.openReadStream(entry, (streamError, readStream) => {
          if (streamError || !readStream) {
            zipFile.close();
            fail(streamError || new Error(`Could not read archive entry: ${entry.fileName}`));
            return;
          }

          fsp
            .mkdir(path.dirname(targetPath), { recursive: true })
            .then(() => {
              const writeStream = fs.createWriteStream(targetPath, { flags: 'wx' });
              writeStream.on('error', error => {
                zipFile.close();
                fail(error);
              });
              writeStream.on('close', () => zipFile.readEntry());
              readStream.on('error', error => {
                zipFile.close();
                fail(error);
              });
              readStream.pipe(writeStream);
            })
            .catch(error => {
              zipFile.close();
              fail(error);
            });
        });
      });
    });
  });
}

export async function readDatasetExportManifest(extractRoot: string) {
  const text = await fsp.readFile(path.join(extractRoot, 'manifest.json'), 'utf-8');
  const manifest = JSON.parse(text) as DatasetExportManifest;
  if (manifest.format !== DATASET_EXPORT_FORMAT || manifest.version !== DATASET_EXPORT_VERSION) {
    throw new Error('Unsupported dataset export archive');
  }
  return manifest;
}

export function getExtractedDatasetPath(extractRoot: string, archivePath: string) {
  const normalized = validateArchiveEntryName(archivePath);
  const resolved = path.resolve(extractRoot, ...normalized.split('/'));
  if (!isPathInside(extractRoot, resolved)) {
    throw new Error(`Archive path escapes import folder: ${archivePath}`);
  }
  return resolved;
}
