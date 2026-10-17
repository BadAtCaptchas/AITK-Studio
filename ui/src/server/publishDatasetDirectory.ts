import fs from 'fs/promises';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getToolkitPythonPath } from './pythonPath';

// Node's POSIX rename replaces an empty destination directory. Migration must
// reject even that race, so use the OS no-replace primitive there.
const noReplace = `import ctypes, errno, os, sys
libc = ctypes.CDLL(None, use_errno=True)
source, destination = map(os.fsencode, sys.argv[1:3])
if sys.platform == 'darwin':
    result = libc.renamex_np(source, destination, 4)
else:
    result = libc.renameat2(-100, source, -100, destination, 1)
if result != 0:
    code = ctypes.get_errno()
    raise OSError(code, os.strerror(code))
`;

export async function publishDatasetDirectory(source: string, destination: string): Promise<void> {
  if (process.platform === 'win32') {
    await fs.rename(source, destination);
    return;
  }
  await promisify(execFile)(getToolkitPythonPath(), ['-c', noReplace, source, destination], {
    timeout: 30000,
    windowsHide: true,
  });
}
