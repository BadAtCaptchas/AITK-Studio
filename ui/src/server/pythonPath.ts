import fs from 'fs';
import path from 'path';
import { TOOLKIT_ROOT } from '../paths';

export function getToolkitPythonPath() {
  const isWindows = process.platform === 'win32';
  const venvDir = fs.existsSync(path.join(TOOLKIT_ROOT, '.venv')) ? '.venv' : fs.existsSync(path.join(TOOLKIT_ROOT, 'venv')) ? 'venv' : null;

  if (!venvDir) {
    return isWindows ? 'python.exe' : 'python3';
  }

  if (isWindows) {
    return path.join(TOOLKIT_ROOT, venvDir, 'Scripts', 'python.exe');
  }

  return path.join(TOOLKIT_ROOT, venvDir, 'bin', 'python');
}
