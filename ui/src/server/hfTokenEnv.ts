import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { TOOLKIT_ROOT } from '../paths';

const HF_TOKEN_ENV_KEYS = ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'] as const;
const DEFAULT_TOKEN_DIR = path.join(TOOLKIT_ROOT, '.tmp', 'hf_tokens');

type PrepareHfTokenEnvOptions = {
  env?: NodeJS.ProcessEnv;
  token?: string | null;
  tokenDir?: string;
  tokenFilePrefix?: string;
};

type PreparedHfTokenEnv = {
  env: NodeJS.ProcessEnv;
  tokenPath: string | null;
  cleanup: () => Promise<void>;
};

function normalizeToken(value: string | null | undefined) {
  const token = value?.trim();
  return token || '';
}

function makeTokenFileName(prefix: string) {
  const safePrefix = prefix.replace(/[^a-zA-Z0-9_.-]+/g, '_').slice(0, 48) || 'hf-token';
  return `${safePrefix}-${process.pid}-${Date.now()}-${crypto.randomBytes(8).toString('hex')}.token`;
}

export async function prepareHfTokenEnv(options: PrepareHfTokenEnvOptions = {}): Promise<PreparedHfTokenEnv> {
  const sourceEnv = options.env ?? process.env;
  const childEnv: NodeJS.ProcessEnv = { ...sourceEnv };
  const effectiveToken =
    normalizeToken(options.token) ||
    normalizeToken(sourceEnv.HF_TOKEN) ||
    normalizeToken(sourceEnv.HUGGING_FACE_HUB_TOKEN);

  for (const key of HF_TOKEN_ENV_KEYS) {
    delete childEnv[key];
  }

  if (!effectiveToken) {
    return {
      env: childEnv,
      tokenPath: null,
      cleanup: async () => undefined,
    };
  }

  const tokenDir = options.tokenDir ?? DEFAULT_TOKEN_DIR;
  await fs.mkdir(tokenDir, { recursive: true, mode: 0o700 });

  const tokenPath = path.join(tokenDir, makeTokenFileName(options.tokenFilePrefix ?? 'hf-token'));
  await fs.writeFile(tokenPath, effectiveToken, { encoding: 'utf8', mode: 0o600 });
  childEnv.HF_TOKEN_PATH = tokenPath;

  let cleaned = false;
  return {
    env: childEnv,
    tokenPath,
    cleanup: async () => {
      if (cleaned) return;
      cleaned = true;
      await fs.rm(tokenPath, { force: true });
    },
  };
}
