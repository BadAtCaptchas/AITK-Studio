import path from 'node:path';
import { buildServer, uiRoot } from './build-server.mjs';

/** Serialize rebuilds; publish a new handler only when its build succeeds. */
export function watchApi(vite, publish) {
  let pending = false;
  let building = false;
  let closed = false;
  let timer;
  async function rebuild() {
    if (building || closed) return;
    building = true;
    while (pending && !closed) {
      pending = false;
      try {
        await buildServer();
        if (!closed) publish();
      } catch (error) {
        console.error('API rebuild failed:', error);
      }
    }
    building = false;
  }
  function changed(filename) {
    const relative = path.relative(uiRoot, filename).replaceAll(path.sep, '/');
    if (!/^src\/(server\/|app\/api\/|utils\/|generated\/|proxy\.ts$|paths\.ts$)/.test(relative)) return;
    pending = true;
    clearTimeout(timer);
    timer = setTimeout(() => void rebuild(), 100);
  }
  vite.watcher.on('add', changed).on('change', changed).on('unlink', changed);
  return () => {
    closed = true;
    clearTimeout(timer);
    vite.watcher.off('add', changed).off('change', changed).off('unlink', changed);
  };
}
