import { db, getDatabaseConfig } from '../../src/server/db';
import type { Job } from '../../src/types';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getTrainingFolder } from '../paths';
import { getTensorBoardLogDir, getToolkitPythonPath, isTensorBoardEnabled } from '../../src/server/tensorboard';
const isWindows = process.platform === 'win32';

const startAndWatchJob = (job: Job) => {
  // starts and watches the job asynchronously
  return new Promise<void>(async (resolve, reject) => {
    const jobID = job.id;

    // setup the training
    const trainingRoot = await getTrainingFolder();
    const tensorBoardEnabled = isTensorBoardEnabled();
    const tensorBoardLogDir = getTensorBoardLogDir(trainingRoot);

    const trainingFolder = path.join(trainingRoot, job.name);
    if (!fs.existsSync(trainingFolder)) {
      fs.mkdirSync(trainingFolder, { recursive: true });
    }

    // make the config file
    const configPath = path.join(trainingFolder, '.job_config.json');

    //log to path
    const logPath = path.join(trainingFolder, 'log.txt');
    const hfDownloadProgressPath = path.join(trainingFolder, '.hf_download_progress.json');

    try {
      // if the log path exists, move it to a folder called logs and rename it {num}_log.txt, looking for the highest num
      // if the log path does not exist, create it
      if (fs.existsSync(logPath)) {
        const logsFolder = path.join(trainingFolder, 'logs');
        if (!fs.existsSync(logsFolder)) {
          fs.mkdirSync(logsFolder, { recursive: true });
        }

        let num = 0;
        while (fs.existsSync(path.join(logsFolder, `${num}_log.txt`))) {
          num++;
        }

        fs.renameSync(logPath, path.join(logsFolder, `${num}_log.txt`));
      }
    } catch (e) {
      console.error('Error moving log file:', e);
    }

    // update runtime-local paths before launch. Imported jobs may come from a
    // different system, and multi-process jobs need every process patched.
    const dbConfig = getDatabaseConfig();
    const jobConfig = JSON.parse(job.job_config);
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

    // write the config file
    fs.writeFileSync(configPath, JSON.stringify(jobConfig, null, 2));
    try {
      fs.rmSync(hfDownloadProgressPath, { force: true });
    } catch (e) {
      console.error('Error clearing Hugging Face download progress file:', e);
    }

    const pythonPath = getToolkitPythonPath();

    const runFilePath = path.join(TOOLKIT_ROOT, 'run.py');
    if (!fs.existsSync(runFilePath)) {
      console.error(`run.py not found at path: ${runFilePath}`);
      await db.jobs.update(jobID, {
        status: 'error',
        info: `Error launching job: run.py not found`,
      });
      return;
    }

    const additionalEnv: any = {
      AITK_JOB_ID: jobID,
      AITK_DB_PROVIDER: dbConfig.provider,
      AITK_SQLITE_PATH: dbConfig.sqlitePath,
      AITK_MONGODB_URI: dbConfig.mongoUri || '',
      AITK_MONGODB_DB: dbConfig.mongoDb,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
      IS_AI_TOOLKIT_UI: '1',
      AITK_HF_DOWNLOAD_PROGRESS_PATH: hfDownloadProgressPath,
    };

    // Add the --log argument to the command
    const args = [runFilePath, configPath, '--log', logPath];

    try {
      let subprocess;

      if (isWindows) {
        // Spawn Python directly on Windows so the process can survive parent exit
        subprocess = spawn(pythonPath, args, {
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
          detached: true,
          windowsHide: true,
          stdio: 'ignore', // don't tie stdio to parent
        });
      } else {
        // For non-Windows platforms, fully detach and ignore stdio so it survives daemon-like
        subprocess = spawn(pythonPath, args, {
          detached: true,
          stdio: 'ignore',
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
        });
      }

      // Save the PID to the database and a file for future management (stop/inspect)
      const pid = subprocess.pid ?? null;
      if (pid != null) {
        await db.jobs.update(jobID, { pid });
      }
      try {
        fs.writeFileSync(path.join(trainingFolder, 'pid.txt'), String(pid ?? ''), { flag: 'w' });
      } catch (e) {
        console.error('Error writing pid file:', e);
      }

      // Important: let the child run independently of this Node process.
      if (subprocess.unref) {
        subprocess.unref();
      }

      // (No stdout/stderr listeners — logging should go to --log handled by your Python)
      // (No monitoring loop — the whole point is to let it live past this worker)
    } catch (error: any) {
      // Handle any exceptions during process launch
      console.error('Error launching process:', error);

      await db.jobs.update(jobID, {
        status: 'error',
        info: `Error launching job: ${error?.message || 'Unknown error'}`,
      });
      return;
    }
    // Resolve the promise immediately after starting the process
    resolve();
  });
};

export default async function startJob(jobID: string) {
  const job: Job | null = await db.jobs.findById(jobID);
  if (!job) {
    console.error(`Job with ID ${jobID} not found`);
    return;
  }
  // update job status to 'running', this will run sync so we don't start multiple jobs.
  await db.jobs.update(jobID, {
    status: 'running',
    stop: false,
    info: 'Starting job...',
  });
  // start and watch the job asynchronously so the cron can continue
  startAndWatchJob(job);
}
