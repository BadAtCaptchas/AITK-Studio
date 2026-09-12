import { cleanupAuthRecords } from '../src/server/runtimeRetention';
import processQueue from './actions/processQueue';
import { db, disconnectDb } from '../src/server/db';
import { startTensorBoard, stopTensorBoard } from '../src/server/tensorboard';
import { getTrainingFolder } from './paths';
import { getCloudflaredConfig, startCloudflared, stopCloudflared } from '../src/server/cloudflared';
import { purgeLegacyDurableEncryptedDatasetKeys } from '../src/server/encryptedDatasetSecrets';
import { syncRemoteCaptionResults } from '../src/server/remoteCaptionResults';
import { runDatasetWatchersIfDue } from '../src/server/datasetWatchers';
import { synchronizeJobPage } from '../src/server/jobReadModel';
import { discoverRemoteJobs } from '../src/server/remoteClient';
import { withProcessLease, LeaseBusyError } from '../src/server/processLease';
import { processOperations } from '../src/server/operationWorker';

const SHUTDOWN_TIMEOUT_MS = 10000;

class CronWorker {
  interval: number;
  is_running: boolean;
  intervalId: NodeJS.Timeout;
  currentRun: Promise<void> | null;
  is_stopping: boolean;
  background = new Map<string, Promise<void>>();
  nextBackground = new Map<string, number>();

  constructor() {
    this.interval = 1000; // Default interval of 1 second
    this.is_running = false;
    this.currentRun = null;
    this.is_stopping = false;
    this.intervalId = setInterval(() => {
      this.run();
      this.startBackground('remote-caption', syncRemoteCaptionResults, 5000);
      this.startBackground('dataset-watchers', runDatasetWatchersIfDue, 5000);
      this.startBackground('operations', processOperations, 1000);
      this.startBackground('job-sync', synchronizeJobPage, 5000);
      this.startBackground('auth-retention', cleanupAuthRecords, 60000);
      this.startBackground('remote-discovery', () => discoverRemoteJobs(), 15000);
    }, this.interval);
  }

  async run() {
    if (this.is_running || this.is_stopping) {
      return;
    }
    this.is_running = true;
    this.currentRun = this.loop();
    try {
      await this.currentRun;
    } catch (error) {
      console.error('Error in cron worker loop:', error);
    } finally {
      this.currentRun = null;
      this.is_running = false;
    }
  }

  async loop() {
    const heartbeat = await db.runtime.get('worker-heartbeat');
    await db.runtime.compareAndSwap('worker-heartbeat', heartbeat?.version ?? null, { at: Date.now(), pid: process.pid });
    await processQueue();
  }

  startBackground(name: string, task: () => Promise<unknown>, interval: number) {
    if (this.is_stopping || this.background.has(name) || Date.now() < (this.nextBackground.get(name) || 0)) return;
    this.nextBackground.set(name, Date.now() + interval);
    const running = withProcessLease('background:' + name, task).then(() => undefined).catch(error => { if (!(error instanceof LeaseBusyError)) console.error(`Background ${name} failed:`, error); })
      .finally(() => this.background.delete(name));
    this.background.set(name, running);
  }

  async stop() {
    this.is_stopping = true;
    clearInterval(this.intervalId);

    if (this.currentRun) {
      await this.currentRun.catch(() => undefined);
    }
    await Promise.allSettled(this.background.values());
  }
}

let cronWorker: CronWorker | null = null;

async function startCronWorker() {
  try {
    const purged = await purgeLegacyDurableEncryptedDatasetKeys();
    if (purged > 0) {
      console.warn(`Purged ${purged} legacy plaintext durable encrypted dataset key setting(s).`);
    }
  } catch (error) {
    console.error('Error purging legacy durable encrypted dataset keys:', error);
  }

  cronWorker = new CronWorker();
  console.log('Cron worker started with interval:', cronWorker.interval, 'ms');
}

void startCronWorker();

async function startOptionalTensorBoard() {
  const trainingRoot = await getTrainingFolder();
  const status = await startTensorBoard(trainingRoot);
  if (!status.enabled) {
    return;
  }

  if (status.running) {
    console.log(`TensorBoard available at ${status.url}`);
  } else {
    console.error(`TensorBoard is enabled but did not start on port ${status.port}`);
  }
}

void startOptionalTensorBoard().catch(error => {
  console.error('Error starting TensorBoard:', error);
});

async function startOptionalCloudflared() {
  if (!getCloudflaredConfig().enabled) return;
  const status = await startCloudflared();
  if (status.running) {
    console.log(`cloudflared ${status.mode} tunnel started${status.publicUrl ? ` for ${status.publicUrl}` : ''}`);
  } else if (status.error) {
    console.error(`cloudflared did not start: ${status.error}`);
  }
}

void startOptionalCloudflared().catch(error => {
  console.error('Error starting cloudflared:', error);
});

let shutdownPromise: Promise<void> | null = null;

function waitWithTimeout(promise: Promise<void>, timeoutMs: number) {
  return new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(resolve, timeoutMs);

    promise.then(resolve, reject).finally(() => clearTimeout(timeout));
  });
}

async function shutdown(signal: NodeJS.Signals) {
  if (shutdownPromise) {
    return;
  }

  console.log(`Cron worker received ${signal}, shutting down...`);
  shutdownPromise = (async () => {
    if (cronWorker) {
      await waitWithTimeout(cronWorker.stop(), SHUTDOWN_TIMEOUT_MS);
    }
    await waitWithTimeout(
      Promise.allSettled([stopTensorBoard(), stopCloudflared()]).then(() => undefined),
      SHUTDOWN_TIMEOUT_MS,
    );
    await disconnectDb();
  })();

  try {
    await shutdownPromise;
    process.exit(0);
  } catch (error) {
    console.error('Error while shutting down cron worker:', error);
    process.exit(1);
  }
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
