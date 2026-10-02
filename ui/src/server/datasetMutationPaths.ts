import fsp from 'fs/promises';
import path from 'path';
import { isPathWithinRoot } from './pathContainment';

export class DatasetMutationPathError extends Error {
  constructor() {
    super('Invalid dataset path');
    this.name = 'DatasetMutationPathError';
  }
}

function isMissing(error: unknown): boolean {
  return !!error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT';
}

/** Resolve aliases before mutation; only a new final filename may be missing. */
export async function resolveDatasetMutationPath(
  datasetsRoot: string,
  target: string,
  options: { allowMissingLeaf?: boolean } = {},
): Promise<{ canonicalRoot: string; canonicalPath: string; entryPath: string }> {
  const root = path.resolve(datasetsRoot);
  const resolved = path.resolve(root, target);
  if (resolved === root || !isPathWithinRoot(root, resolved)) throw new DatasetMutationPathError();

  const canonicalRoot = await fsp.realpath(root);
  const canonicalParent = await fsp.realpath(path.dirname(resolved));
  if (!isPathWithinRoot(canonicalRoot, canonicalParent)) throw new DatasetMutationPathError();
  // Retain the final directory entry so deleting an in-root alias removes the
  // alias itself, while removing junction aliases from all of its ancestors.
  const entryPath = path.join(canonicalParent, path.basename(resolved));
  let canonicalPath: string;
  try {
    canonicalPath = await fsp.realpath(entryPath);
  } catch (error) {
    if (!options.allowMissingLeaf || !isMissing(error)) throw error;
    const entry = await fsp.lstat(entryPath).catch(error => {
      if (isMissing(error)) return null;
      throw error;
    });
    if (entry) throw new DatasetMutationPathError(); // A dangling link is not a new file.
    canonicalPath = entryPath;
  }
  if (canonicalPath === canonicalRoot || !isPathWithinRoot(canonicalRoot, canonicalPath)) {
    throw new DatasetMutationPathError();
  }
  return { canonicalRoot, canonicalPath, entryPath };
}
