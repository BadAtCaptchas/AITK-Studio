import { execFile } from 'child_process';
import { promisify } from 'util';

const execute = promisify(execFile);

/** OS process start time in Unix seconds, compatible with existing psutil lease records. */
export async function getProcessBirth(pid: number): Promise<number | null> {
  if (!Number.isSafeInteger(pid) || pid <= 0 || pid > 0x7fffffff) {
    throw new Error('Invalid process id');
  }

  try {
    const windows = process.platform === 'win32';
    const result = await execute(
      windows ? 'powershell.exe' : 'ps',
      windows
        ? [
            '-NoProfile',
            '-NonInteractive',
            '-Command',
            `$ErrorActionPreference = 'Stop'; (Get-Process -Id ${pid} -ErrorAction Stop).StartTime.ToUniversalTime().ToString('o', [System.Globalization.CultureInfo]::InvariantCulture)`,
          ]
        : ['-p', String(pid), '-o', 'lstart='],
      {
        timeout: 5_000,
        maxBuffer: 4096,
        windowsHide: true,
        env: { ...process.env, LC_ALL: 'C', TZ: 'UTC0' },
      },
    );
    const output = result.stdout.trim();
    const format = windows
      ? /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,7})?Z$/
      : /^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}$/;
    if (!format.test(output)) throw new Error('Cannot verify process ownership');
    const birth = Date.parse(windows ? output : `${output} GMT`) / 1000;
    if (!Number.isFinite(birth) || birth <= 0) throw new Error('Cannot verify process ownership');
    return birth;
  } catch (error) {
    // A failed query (including access denied) does not prove that an owner is gone.
    try {
      process.kill(pid, 0);
    } catch (probeError: unknown) {
      if (probeError && typeof probeError === 'object' && 'code' in probeError && probeError.code === 'ESRCH') {
        return null;
      }
    }
    throw error;
  }
}

export function processBirthMatches(actual: number, recorded: number): boolean {
  // POSIX ps reports whole seconds; retain live owners recorded by psutil with fractions.
  return Math.abs(actual - recorded) < (process.platform === 'win32' ? 0.01 : 1);
}
