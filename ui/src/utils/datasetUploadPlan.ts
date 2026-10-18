import { FOLDER_IMPORT_MEDIA_EXTENSIONS, folderImportCaptionKey, folderImportExtension } from './folderImport';

class DatasetUploadPlanError extends Error { readonly status = 400; }

/** Match by original name before sanitizing; never infer a caption pairing from a renamed file. */
export function planDatasetUploads(paths: string[], createID: () => string) {
  const groups = new Map<string, { id: string; indexes: number[]; media: number[] }>();
  const destinations = new Map<string, string>();
  const filenames = new Set<string>();
  for (const [index, path] of paths.entries()) {
    const original = path.replace(/\\/g, '/');
    if (original.startsWith('/') || original.split('/').some(part => part === '..')) throw new DatasetUploadPlanError('Invalid upload path');
    const normalized = original.split('/').filter(Boolean).map((part, i, parts) => {
      const ext = i === parts.length - 1 ? folderImportExtension(part) : '';
      const stem = ext ? part.slice(0, -ext.length) : part;
      return (stem.trim().replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').replace(/\s+/g, '_').replace(/^\.+/, '').slice(0, 120) || 'file') + ext;
    }).join('/');
    const originalExtension = folderImportExtension(original);
    const key = originalExtension ? original.slice(0, -originalExtension.length) : original;
    const destination = folderImportCaptionKey(normalized);
    if (filenames.has(normalized.toLowerCase()) || (destinations.has(destination) && destinations.get(destination) !== key)) {
      throw new DatasetUploadPlanError(`Ambiguous upload names: ${path}. Rename the files and their captions together.`);
    }
    filenames.add(normalized.toLowerCase());
    destinations.set(destination, key);
    const group: { id: string; indexes: number[]; media: number[] } = groups.get(key) ?? { id: createID(), indexes: [], media: [] };
    group.indexes.push(index);
    if (FOLDER_IMPORT_MEDIA_EXTENSIONS.includes(folderImportExtension(original))) group.media.push(index);
    if (group.media.length > 1) throw new DatasetUploadPlanError(`Multiple media files share a caption name: ${path}`);
    groups.set(key, group);
  }
  return [...groups.values()].map(group => ({ id: group.id,
    indexes: [...group.media, ...group.indexes.filter(index => !group.media.includes(index))] }));
}
