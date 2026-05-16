import { spawn } from 'child_process';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { NextRequest, NextResponse } from 'next/server';
import { TOOLKIT_ROOT } from '@/paths';
import { getDatabaseConfig } from '@/server/db';
import { getTrainingFolder } from '@/server/settings';
import { getToolkitPythonPath } from '@/server/tensorboard';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const isWindows = process.platform === 'win32';
const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp']);

function ensureApiAccess(request: NextRequest): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) {
    return null;
  }

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

function sanitizeName(value: unknown) {
  const raw = typeof value === 'string' ? value : '';
  const sanitized = raw
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 60);
  return sanitized || 'inline_generate';
}

function getGenerateConfig(jobConfig: any) {
  const processList = jobConfig?.config?.process;
  if (jobConfig?.job !== 'generate' || !Array.isArray(processList) || processList.length !== 1) {
    return null;
  }
  const processConfig = processList[0];
  if (!processConfig || typeof processConfig !== 'object' || !processConfig.generate) {
    return null;
  }
  return { processConfig, generateConfig: processConfig.generate };
}

function normalizePromptItems(generateConfig: any) {
  const images = generateConfig?.images;
  if (Array.isArray(images)) {
    return images.filter(item => {
      if (typeof item === 'string') return item.trim().length > 0;
      return item && typeof item === 'object' && typeof item.prompt === 'string' && item.prompt.trim().length > 0;
    });
  }

  const prompts = generateConfig?.prompts;
  if (Array.isArray(prompts)) {
    return prompts.filter(item => typeof item === 'string' && item.trim().length > 0);
  }

  return [];
}

function getRequestedImageCount(generateConfig: any) {
  const repeatCount = Number(generateConfig?.num_repeats ?? 1);
  const numRepeats = Number.isFinite(repeatCount) && repeatCount > 0 ? Math.floor(repeatCount) : 1;
  return normalizePromptItems(generateConfig).length * numRepeats;
}

async function findNewestGeneratedImage(outputFolder: string) {
  const entries = await fsp.readdir(outputFolder, { withFileTypes: true }).catch(() => []);
  const files = await Promise.all(
    entries
      .filter(entry => entry.isFile() && imageExtensions.has(path.extname(entry.name).toLowerCase()))
      .map(async entry => {
        const filePath = path.join(outputFolder, entry.name);
        const stat = await fsp.stat(filePath);
        return { filePath, mtimeMs: stat.mtimeMs };
      }),
  );

  files.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return files[0]?.filePath || null;
}

function readLogTail(logPath: string, maxChars = 8000) {
  try {
    const log = fs.readFileSync(logPath, 'utf8');
    return log.slice(Math.max(0, log.length - maxChars));
  } catch {
    return '';
  }
}

function runInlineGenerate(
  pythonPath: string,
  args: string[],
  env: NodeJS.ProcessEnv,
  launchLogPath: string,
  logPath: string,
) {
  return new Promise<void>((resolve, reject) => {
    const logFd = fs.openSync(launchLogPath, 'a');
    const closeLog = () => {
      try {
        fs.closeSync(logFd);
      } catch {
        // The descriptor may already be closed if launch failed early.
      }
    };
    const subprocess = spawn(pythonPath, args, {
      cwd: TOOLKIT_ROOT,
      env,
      windowsHide: isWindows,
      stdio: ['ignore', logFd, logFd],
    });

    subprocess.once('error', error => {
      closeLog();
      reject(error);
    });
    subprocess.once('exit', (code, signal) => {
      closeLog();
      if (code === 0 && signal == null) {
        resolve();
        return;
      }
      const reason = signal ? `signal ${signal}` : `exit code ${code ?? 'unknown'}`;
      const logTail = readLogTail(logPath) || readLogTail(launchLogPath);
      reject(new Error(`Inline generation failed with ${reason}.${logTail ? `\n${logTail}` : ''}`));
    });
  });
}

export async function POST(request: NextRequest) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  try {
    const body = await request.json();
    const jobConfig = JSON.parse(JSON.stringify(body.job_config || null));
    const generateContext = getGenerateConfig(jobConfig);
    if (!generateContext) {
      return NextResponse.json({ error: 'Inline generation requires a generate job config.' }, { status: 400 });
    }

    const requestedImageCount = getRequestedImageCount(generateContext.generateConfig);
    if (requestedImageCount !== 1) {
      return NextResponse.json(
        { error: 'Inline generation only supports one image. Create a generate job for multiple images.' },
        { status: 400 },
      );
    }

    const gpuIds = typeof body.gpu_ids === 'string' && body.gpu_ids.trim() ? body.gpu_ids.trim() : '0';
    const trainingRoot = await getTrainingFolder();
    const dbConfig = getDatabaseConfig();
    const baseName = sanitizeName(jobConfig.config?.name);
    const runName = `${baseName}_${Date.now()}`;
    const runFolder = path.join(trainingRoot, '.inline_generate', runName);
    const outputFolder = path.join(runFolder, 'samples');
    const configPath = path.join(runFolder, '.job_config.json');
    const logPath = path.join(runFolder, 'log.txt');
    const launchLogPath = path.join(runFolder, 'launch.log');
    const hfDownloadProgressPath = path.join(runFolder, '.hf_download_progress.json');

    await fsp.mkdir(outputFolder, { recursive: true });

    jobConfig.config.name = runName;
    for (const processConfig of jobConfig.config.process) {
      processConfig.output_folder = outputFolder;
      processConfig.sqlite_db_path = dbConfig.sqlitePath;
      processConfig.training_folder = trainingRoot;
    }

    await fsp.writeFile(configPath, JSON.stringify(jobConfig, null, 2));

    const pythonPath = getToolkitPythonPath();
    const runFilePath = path.join(TOOLKIT_ROOT, 'run.py');
    if (!fs.existsSync(runFilePath)) {
      return NextResponse.json({ error: `run.py not found at ${runFilePath}` }, { status: 500 });
    }

    const additionalEnv: NodeJS.ProcessEnv = {
      ...process.env,
      AITK_JOB_ID: `inline_${runName}`,
      AITK_DB_PROVIDER: dbConfig.provider,
      AITK_SQLITE_PATH: dbConfig.sqlitePath,
      AITK_MONGODB_URI: dbConfig.mongoUri || '',
      AITK_MONGODB_DB: dbConfig.mongoDb,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: gpuIds,
      IS_AI_TOOLKIT_UI: '1',
      AITK_HF_DOWNLOAD_PROGRESS_PATH: hfDownloadProgressPath,
      PYTHONUNBUFFERED: '1',
      HF_HUB_ENABLE_HF_TRANSFER: isWindows ? '0' : process.env.HF_HUB_ENABLE_HF_TRANSFER || '1',
    };

    await runInlineGenerate(
      pythonPath,
      [runFilePath, configPath, '--log', logPath],
      additionalEnv,
      launchLogPath,
      logPath,
    );
    const imagePath = await findNewestGeneratedImage(outputFolder);
    if (!imagePath) {
      return NextResponse.json(
        { error: 'Generation finished, but no image file was found.', log: readLogTail(logPath) },
        { status: 500 },
      );
    }

    return NextResponse.json({
      image_path: imagePath,
      imagePath,
      output_folder: outputFolder,
      log_path: logPath,
    });
  } catch (error: any) {
    console.error('Inline generation failed:', error);
    return NextResponse.json({ error: error?.message || 'Inline generation failed.' }, { status: 500 });
  }
}
