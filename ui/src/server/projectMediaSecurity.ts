import fsp from 'fs/promises';
import path from 'path';
import { db } from './db';
import { getProjectRoots } from './projects';

let pendingProjectRoots: Promise<string[]> | null = null;

async function canonicalRegisteredProjectRoots() {
  if (!pendingProjectRoots) {
    pendingProjectRoots = (async () =>
      (
        await Promise.all(
          (await db.projects.list()).map(async project => {
            try {
              return await fsp.realpath((await getProjectRoots(project)).root);
            } catch {
              return null;
            }
          }),
        )
      ).filter((root): root is string => !!root))();
    void pendingProjectRoots
      .finally(() => {
        pendingProjectRoots = null;
      })
      .catch(() => undefined);
  }
  return pendingProjectRoots;
}

export async function isRegisteredProjectPath(filePath: string) {
  const candidate = path.resolve(filePath);
  return (await canonicalRegisteredProjectRoots()).some(projectRoot => {
    const relativePath = path.relative(projectRoot, candidate);
    return relativePath === '' || (!relativePath.startsWith('..') && !path.isAbsolute(relativePath));
  });
}
