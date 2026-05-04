import processQueue from './actions/processQueue';
import prisma from './prisma';

const SHUTDOWN_TIMEOUT_MS = 3000;

class CronWorker {
  interval: number;
  is_running: boolean;
  intervalId: NodeJS.Timeout;
  currentRun: Promise<void> | null;
  is_stopping: boolean;

  constructor() {
    this.interval = 1000; // Default interval of 1 second
    this.is_running = false;
    this.currentRun = null;
    this.is_stopping = false;
    this.intervalId = setInterval(() => {
      this.run();
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
    await processQueue();
  }

  async stop() {
    this.is_stopping = true;
    clearInterval(this.intervalId);

    if (this.currentRun) {
      await this.currentRun.catch(() => undefined);
    }
  }
}

// it automatically starts the loop
const cronWorker = new CronWorker();
console.log('Cron worker started with interval:', cronWorker.interval, 'ms');

let shutdownPromise: Promise<void> | null = null;

function waitWithTimeout(promise: Promise<void>, timeoutMs: number) {
  return new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(resolve, timeoutMs);

    promise
      .then(resolve, reject)
      .finally(() => clearTimeout(timeout));
  });
}

async function shutdown(signal: NodeJS.Signals) {
  if (shutdownPromise) {
    return;
  }

  console.log(`Cron worker received ${signal}, shutting down...`);
  shutdownPromise = (async () => {
    await waitWithTimeout(cronWorker.stop(), SHUTDOWN_TIMEOUT_MS);
    await prisma.$disconnect();
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
