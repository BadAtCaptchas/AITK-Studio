import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { ROOT_CAPTION_FILE_NAME } from '../utils/folderImport';

export { ROOT_CAPTION_FILE_NAME };

export function isRootCaptionFileName(fileName: string) {
  return fileName.toLowerCase() === ROOT_CAPTION_FILE_NAME.toLowerCase();
}

export function isDatasetRootCaptionEntry(datasetFolder: string, currentDir: string, fileName: string) {
  return path.resolve(datasetFolder) === path.resolve(currentDir) && isRootCaptionFileName(fileName);
}

export async function findDatasetRootCaptionPath(datasetFolder: string) {
  const entries = await fsp.readdir(datasetFolder, { withFileTypes: true }).catch(() => []);
  const files = entries.filter(entry => entry.isFile() && isRootCaptionFileName(entry.name));
  const exact = files.find(entry => entry.name === ROOT_CAPTION_FILE_NAME);
  const selected = exact || files[0];
  return selected ? path.join(datasetFolder, selected.name) : null;
}

export async function readDatasetRootCaption(datasetFolder: string) {
  const captionPath = await findDatasetRootCaptionPath(datasetFolder);
  if (!captionPath) return { found: false, systemPrompt: '' };
  const stat = await fsp.stat(captionPath).catch(() => null);
  if (!stat?.isFile()) return { found: false, systemPrompt: '' };
  return {
    found: true,
    systemPrompt: (await fsp.readFile(captionPath, 'utf-8')).trim(),
  };
}

export function hasDatasetRootCaptionSync(datasetFolder: string) {
  const exact = path.join(datasetFolder, ROOT_CAPTION_FILE_NAME);
  if (fs.existsSync(exact) && fs.statSync(exact).isFile()) return true;
  try {
    return fs
      .readdirSync(datasetFolder, { withFileTypes: true })
      .some(entry => entry.isFile() && entry.name !== ROOT_CAPTION_FILE_NAME && isRootCaptionFileName(entry.name));
  } catch {
    return false;
  }
}
