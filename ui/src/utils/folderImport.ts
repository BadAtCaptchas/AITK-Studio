export function normalizeFolderImportRelativePath(relativePath: string) {
  return relativePath.replace(/\\/g, '/').split('/').filter(Boolean).join('/');
}

export function folderImportRootName(relativePath: string) {
  return normalizeFolderImportRelativePath(relativePath).split('/').filter(Boolean)[0] || 'imported_folder';
}

export function folderImportExtension(relativePath: string) {
  const fileName = normalizeFolderImportRelativePath(relativePath).split('/').pop() || '';
  const dotIndex = fileName.lastIndexOf('.');
  return dotIndex > 0 ? fileName.slice(dotIndex).toLowerCase() : '';
}

export function stripFolderImportRoot(relativePath: string, fallbackFileName: string) {
  const parts = normalizeFolderImportRelativePath(relativePath).split('/').filter(Boolean);
  if (parts.length > 1) return parts.slice(1).join('/');
  return fallbackFileName || parts[0] || 'file';
}

function folderImportFileName(relativePath: string) {
  return normalizeFolderImportRelativePath(relativePath).split('/').pop() || 'file';
}

function splitFileName(fileName: string) {
  const dotIndex = fileName.lastIndexOf('.');
  if (dotIndex <= 0) return { stem: fileName || 'file', extension: '' };
  return {
    stem: fileName.slice(0, dotIndex) || 'file',
    extension: fileName.slice(dotIndex),
  };
}

export function folderImportCaptionKey(relativePath: string) {
  const normalized = normalizeFolderImportRelativePath(relativePath).toLowerCase();
  const slashIndex = normalized.lastIndexOf('/');
  const directory = slashIndex >= 0 ? normalized.slice(0, slashIndex + 1) : '';
  const fileName = slashIndex >= 0 ? normalized.slice(slashIndex + 1) : normalized;
  const { stem } = splitFileName(fileName);
  return `${directory}${stem}`;
}

export function createFlattenedFileNameAllocator(existingNames?: Iterable<string>) {
  const usedNames = new Set(Array.from(existingNames || [], name => name.toLowerCase()));

  return (relativePath: string) => {
    const { stem, extension } = splitFileName(folderImportFileName(relativePath));
    let candidate = `${stem}${extension}`;
    let suffix = 2;
    while (usedNames.has(candidate.toLowerCase())) {
      candidate = `${stem}_${suffix}${extension}`;
      suffix += 1;
    }
    usedNames.add(candidate.toLowerCase());
    return candidate;
  };
}

export function flattenedFolderImportFileNames(relativePaths: string[]) {
  const allocate = createFlattenedFileNameAllocator();
  return relativePaths.map(relativePath => allocate(relativePath));
}
