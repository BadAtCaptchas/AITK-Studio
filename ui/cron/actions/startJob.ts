import { db, getDatabaseConfig } from '../../src/server/db';
import type { Job } from '../../src/types';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getHFToken, getOpenRouterApiKey, getTrainingFolder } from '../paths';
import { getTensorBoardLogDir, isTensorBoardEnabled } from '../../src/server/tensorboard';
import { getToolkitPythonPath } from '../../src/server/pythonPath';
import { prepareHfTokenEnv } from '../../src/server/hfTokenEnv';
import { resolvePathWithinRoot } from '../../src/server/pathSafety';
import {
  getEncryptedDatasetsForJobConfig,
  getKeyForRequiredDataset,
  normalizeEncryptedKeyMap,
} from '../../src/server/encryptedDatasets';
import {
  clearDurableEncryptedDatasetKeys,
  getDurableEncryptedDatasetKeys,
} from '../../src/server/encryptedDatasetSecrets';
import { getSecureRemoteOllamaWorkerId } from '../../src/server/secureRemoteCaptionJobs';
import type { EncryptedDatasetStartKey } from '../../src/types';

const isWindows = process.platform === 'win32';
const LAUNCH_LOG_FILE = 'launch.log';

type StartJobOptions = {
  encryptedDatasetKeys?: EncryptedDatasetStartKey[];
};

function normalizeWorkerBaseUrl(baseUrl: string) {
  const trimmed = baseUrl.trim().replace(/\/+$/, '');
  if (!/^https?:\/\//i.test(trimmed)) {
    throw new Error('Worker base URL must start with http:// or https://');
  }
  return trimmed;
}

async function getSecureRemoteOllamaWorker(workerId: string) {
  const worker = await db.workerNodes.findById(workerId);
  if (!worker) throw new Error(`Remote worker not found: ${workerId}`);
  if (!worker.enabled) throw new Error(`Remote worker is disabled: ${worker.name}`);
  if (!worker.api_token) throw new Error(`Remote worker has no API token: ${worker.name}`);
  return {
    id: worker.id,
    base_url: normalizeWorkerBaseUrl(worker.base_url),
    api_token: worker.api_token,
  };
}

function appendLaunchLog(launchLogPath: string, message: string) {
  try {
    fs.appendFileSync(launchLogPath, `${message}\n`);
  } catch (error) {
    console.error('Error writing launch log:', error);
  }
}

function archiveExistingLog(filePath: string, logsFolder: string, suffix: string) {
  if (!fs.existsSync(filePath)) return;
  if (!fs.existsSync(logsFolder)) {
    fs.mkdirSync(logsFolder, { recursive: true });
  }

  let num = 0;
  while (fs.existsSync(path.join(logsFolder, `${num}_${suffix}`))) {
    num++;
  }

  fs.renameSync(filePath, path.join(logsFolder, `${num}_${suffix}`));
}

const startAndWatchJob = (job: Job, options: StartJobOptions = {}) => {
  return new Promise<void>(async resolve => {
    const jobID = job.id;
    let launchLogPath = '';
    let launchLogFd: number | null = null;
    let cleanupHfTokenEnv: (() => Promise<void>) | null = null;

    const closeLaunchLog = () => {
      if (launchLogFd == null) return;
      try {
        fs.closeSync(launchLogFd);
      } catch {
        // The descriptor may already be closed if spawn failed early.
      } finally {
        launchLogFd = null;
      }
    };
    const cleanupSensitiveEnv = () => {
      if (!cleanupHfTokenEnv) return;
      const cleanup = cleanupHfTokenEnv;
      cleanupHfTokenEnv = null;
      void cleanup().catch(error => console.error('Error cleaning Hugging Face token file:', error));
    };

    try {
      const trainingRoot = await getTrainingFolder();
      const hfToken = await getHFToken();
      const openRouterApiKey = await getOpenRouterApiKey();
      const tensorBoardEnabled = isTensorBoardEnabled();
      const tensorBoardLogDir = getTensorBoardLogDir(trainingRoot);

      const trainingFolder = resolvePathWithinRoot(trainingRoot, job.name);
      if (!trainingFolder) {
        throw new Error('Invalid job path');
      }

      if (!fs.existsSync(trainingFolder)) {
        fs.mkdirSync(trainingFolder, { recursive: true });
      }

      const configPath = path.join(trainingFolder, '.job_config.json');
      const logPath = path.join(trainingFolder, 'log.txt');
      launchLogPath = path.join(trainingFolder, LAUNCH_LOG_FILE);
      const hfDownloadProgressPath = path.join(trainingFolder, '.hf_download_progress.json');
      const comfyInstallProgressPath = path.join(trainingFolder, '.comfy_install_progress.json');

      try {
        const logsFolder = path.join(trainingFolder, 'logs');
        archiveExistingLog(logPath, logsFolder, 'log.txt');
        archiveExistingLog(launchLogPath, logsFolder, LAUNCH_LOG_FILE);
      } catch (error) {
        console.error('Error moving log file:', error);
      }

      const dbConfig = getDatabaseConfig();
      const jobConfig = JSON.parse(job.job_config);
      const secureRemoteOllamaWorkerId = getSecureRemoteOllamaWorkerId(jobConfig);
      const secureRemoteOllamaEnv: Record<string, string> = {};
      if (secureRemoteOllamaWorkerId) {
        const worker = await getSecureRemoteOllamaWorker(secureRemoteOllamaWorkerId);
        secureRemoteOllamaEnv.AITK_SECURE_CAPTION_REMOTE_BASE_URL = worker.base_url;
        secureRemoteOllamaEnv.AITK_SECURE_CAPTION_REMOTE_TOKEN = worker.api_token;
        secureRemoteOllamaEnv.AITK_SECURE_CAPTION_REMOTE_WORKER_ID = worker.id;
      }
      const requiredEncryptedDatasets = await getEncryptedDatasetsForJobConfig(jobConfig);
      const durableEncryptedDatasetKeys = await getDurableEncryptedDatasetKeys(jobID);
      const encryptedKeyMap = normalizeEncryptedKeyMap([
        ...durableEncryptedDatasetKeys,
        ...(options.encryptedDatasetKeys || []),
      ]);
      const encryptedDatasetKeys = requiredEncryptedDatasets.map(dataset => {
        const keyB64 = getKeyForRequiredDataset(encryptedKeyMap, dataset);
        if (!keyB64) {
          throw new Error(`decryption key required for encrypted dataset: ${dataset.name}`);
        }
        return { datasetPath: dataset.path, keyB64 };
      });
      jobConfig.config.name = job.name;
      if (Array.isArray(jobConfig.config?.process)) {
        jobConfig.config.process.forEach((processConfig: any) => {
          processConfig.sqlite_db_path = dbConfig.sqlitePath;
          processConfig.training_folder = trainingRoot;
          if (tensorBoardEnabled && processConfig.log_dir == null) {
            processConfig.log_dir = tensorBoardLogDir;
          }
        });
      }

      fs.writeFileSync(configPath, JSON.stringify(jobConfig, null, 2));
      try {
        fs.rmSync(hfDownloadProgressPath, { force: true });
      } catch (error) {
        console.error('Error clearing Hugging Face download progress file:', error);
      }
      try {
        fs.rmSync(comfyInstallProgressPath, { force: true });
      } catch (error) {
        console.error('Error clearing ComfyUI install progress file:', error);
      }

      const pythonPath = getToolkitPythonPath();
      const runFilePath = path.join(TOOLKIT_ROOT, 'run.py');
      if (!fs.existsSync(runFilePath)) {
        const message = `Error launching job: run.py not found`;
        appendLaunchLog(launchLogPath, `[launcher] run.py not found at path: ${runFilePath}`);
        await db.jobs.update(jobID, { status: 'error', pid: null, info: message });
        resolve();
        return;
      }

      const additionalEnv: Record<string, string> = {
        AITK_JOB_ID: jobID,
        AITK_DB_PROVIDER: dbConfig.provider,
        AITK_SQLITE_PATH: dbConfig.sqlitePath,
        AITK_MONGODB_URI: dbConfig.mongoUri || '',
        AITK_MONGODB_DB: dbConfig.mongoDb,
        CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
        IS_AI_TOOLKIT_UI: '1',
        AITK_HF_DOWNLOAD_PROGRESS_PATH: hfDownloadProgressPath,
        AITK_COMFY_INSTALL_PROGRESS_PATH: comfyInstallProgressPath,
        PYTHONUNBUFFERED: '1',
        HF_HUB_ENABLE_HF_TRANSFER: isWindows ? '0' : process.env.HF_HUB_ENABLE_HF_TRANSFER || '1',
        ...secureRemoteOllamaEnv,
      };
      if (openRouterApiKey) {
        additionalEnv.OPENROUTER_API_KEY = openRouterApiKey;
        additionalEnv.AITK_OPENROUTER_API_KEY = openRouterApiKey;
      }
      if (encryptedDatasetKeys.length > 0) {
        additionalEnv.AITK_ENCRYPTED_DATASET_KEYS_B64 = Buffer.from(
          JSON.stringify(encryptedDatasetKeys),
          'utf-8',
        ).toString('base64');
      }

      const args = [runFilePath, configPath, '--log', logPath];
      launchLogFd = fs.openSync(launchLogPath, 'a');
      appendLaunchLog(launchLogPath, `[launcher] ${new Date().toISOString()} starting job ${jobID}`);
      appendLaunchLog(launchLogPath, `[launcher] cwd: ${TOOLKIT_ROOT}`);
      appendLaunchLog(
        launchLogPath,
        `[launcher] command: ${pythonPath} ${args.map(arg => JSON.stringify(arg)).join(' ')}`,
      );

      const preparedHfEnv = await prepareHfTokenEnv({
        env: {
          ...process.env,
          ...additionalEnv,
        },
        token: hfToken,
        tokenFilePrefix: `job-${jobID}`,
      });
      cleanupHfTokenEnv = preparedHfEnv.cleanup;

      const subprocess = spawn(pythonPath, args, {
        env: preparedHfEnv.env,
        cwd: TOOLKIT_ROOT,
        detached: true,
        windowsHide: isWindows,
        stdio: ['ignore', launchLogFd, launchLogFd] as any,
      });

      const pid = subprocess.pid ?? null;
      const handleLaunchFailure = async (message: string) => {
        appendLaunchLog(launchLogPath, `[launcher] ${message}`);
        const currentJob = await db.jobs.findById(jobID).catch(() => null);
        if (currentJob?.status === 'running' && (pid == null || currentJob.pid == null || currentJob.pid === pid)) {
          await db.jobs
            .update(jobID, {
              status: 'error',
              pid: null,
              info: message,
            })
            .catch(error => console.error('Error updating failed job status:', error));
        }
      };

      subprocess.once('error', error => {
        closeLaunchLog();
        cleanupSensitiveEnv();
        void handleLaunchFailure(`Error launching job: ${error.message}`);
      });

      subprocess.once('exit', (code, signal) => {
        closeLaunchLog();
        cleanupSensitiveEnv();
        if (code === 0 && signal == null) {
          void db.jobs
            .findById(jobID)
            .then(currentJob => {
              if (currentJob?.status === 'running' && (pid == null || currentJob.pid == null || currentJob.pid === pid)) {
                return db.jobs.update(jobID, {
                  status: 'completed',
                  pid: null,
                  info: 'Job completed',
                }).then(() => clearDurableEncryptedDatasetKeys(jobID));
              }
              return null;
            })
            .catch(error => console.error('Error reconciling completed job process:', error));
          return;
        }

        const reason = signal ? `signal ${signal}` : `exit code ${code ?? 'unknown'}`;
        void handleLaunchFailure(`Job process exited with ${reason}. Check the job log for details.`);
      });

      if (pid != null) {
        await db.jobs.update(jobID, { pid });
      }
      try {
        fs.writeFileSync(path.join(trainingFolder, 'pid.txt'), String(pid ?? ''), { flag: 'w' });
      } catch (error) {
        console.error('Error writing pid file:', error);
      }

      subprocess.unref?.();
    } catch (error: any) {
      closeLaunchLog();
      cleanupSensitiveEnv();
      console.error('Error launching process:', error);
      if (launchLogPath) {
        appendLaunchLog(launchLogPath, `[launcher] Error launching process: ${error?.stack || error?.message || error}`);
      }
      await db.jobs
        .update(jobID, {
          status: 'error',
          pid: null,
          info: `Error launching job: ${error?.message || 'Unknown error'}`,
        })
        .catch(updateError => console.error('Error updating failed job status:', updateError));
    }

    resolve();
  });
};

export async function startJobNow(jobID: string, options: StartJobOptions = {}) {
  const job: Job | null = await db.jobs.findById(jobID);
  if (!job) {
    console.error(`Job with ID ${jobID} not found`);
    return;
  }
  if (job.worker_id && job.worker_id !== 'local') {
    console.error(`Job ${jobID} belongs to remote worker ${job.worker_id}; local cron will not start it.`);
    return;
  }

  await db.jobs.update(jobID, {
    status: 'running',
    stop: false,
    info: 'Starting job...',
  });

  startAndWatchJob(job, options).catch(async (error: any) => {
    console.error('Error preparing job launch:', error);
    await db.jobs
      .update(jobID, {
        status: 'error',
        pid: null,
        info: `Error launching job: ${error?.message || 'Unknown error'}`,
      })
      .catch(updateError => console.error('Error updating failed job status:', updateError));
  });
}

export default async function startJob(jobID: string) {
  await startJobNow(jobID);
}
