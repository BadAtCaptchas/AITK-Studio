import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { ReadableStream } from 'stream/web';
import { TOOLKIT_ROOT } from '../paths';
import { getToolkitPythonPath } from './pythonPath';

export const SCRIPT_TIMEOUT_MS = 20 * 60 * 1000;
export const ALLOWED_SCRIPT = 'merge_loras.py';

const UI_SCRIPTS_ROOT = path.join(TOOLKIT_ROOT, 'ui_scripts');
const SAVE_DTYPES = new Set(['float32', 'fp32', 'float16', 'fp16', 'bfloat16', 'bf16']);
const DEVICE_RE = /^(cpu|mps|cuda(?::\d+)?)$/;

export class ScriptValidationError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'ScriptValidationError';
    this.status = status;
  }
}

export interface ScriptRunResult {
  ok: boolean;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
  result: unknown;
  timedOut: boolean;
  error?: string;
}

export interface ScriptInvocation {
  scriptPath: string;
  args: string[];
}

interface MergeLoraEntry {
  path: string;
  strength?: number;
}

function canonicalExistingPath(value: string) {
  const resolved = path.resolve(value);
  try {
    return fs.realpathSync.native(resolved);
  } catch {
    return resolved;
  }
}

export function isPathInside(parent: string, candidate: string) {
  const root = canonicalExistingPath(parent);
  const target = canonicalExistingPath(candidate);
  const relative = path.relative(root, target);
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

function requireString(value: unknown, label: string) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new ScriptValidationError(`${label} must be a non-empty string`);
  }
  return value.trim();
}

function resolveAllowedScriptPath(rawScript: unknown) {
  const script = requireString(rawScript, 'script');
  if (script !== ALLOWED_SCRIPT) {
    throw new ScriptValidationError('Unknown script');
  }

  const scriptPath = path.resolve(UI_SCRIPTS_ROOT, script);
  if (!isPathInside(UI_SCRIPTS_ROOT, scriptPath) || !fs.existsSync(scriptPath) || !fs.statSync(scriptPath).isFile()) {
    throw new ScriptValidationError('Script is not available');
  }
  return scriptPath;
}

function parseLoras(raw: unknown) {
  let parsed = raw;
  if (typeof raw === 'string') {
    try {
      parsed = JSON.parse(raw);
    } catch {
      throw new ScriptValidationError('args.loras must be valid JSON');
    }
  }

  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new ScriptValidationError('args.loras must be a non-empty list');
  }

  return parsed as MergeLoraEntry[];
}

function normalizeMergeLorasArgs(rawArgs: unknown, trainingRoot: string) {
  if (rawArgs == null || Array.isArray(rawArgs) || typeof rawArgs !== 'object') {
    throw new ScriptValidationError('args must be an object');
  }

  const args = rawArgs as Record<string, unknown>;
  const loras = parseLoras(args.loras).map((entry, index) => {
    if (!entry || typeof entry !== 'object') {
      throw new ScriptValidationError(`args.loras[${index}] must be an object`);
    }

    const loraPath = requireString(entry.path, `args.loras[${index}].path`);
    if (!loraPath.toLowerCase().endsWith('.safetensors')) {
      throw new ScriptValidationError(`args.loras[${index}].path must be a .safetensors file`);
    }
    if (!fs.existsSync(loraPath) || !fs.statSync(loraPath).isFile()) {
      throw new ScriptValidationError(`LoRA file not found: ${loraPath}`);
    }
    if (!isPathInside(trainingRoot, loraPath)) {
      throw new ScriptValidationError('LoRA inputs must be inside the local training folder');
    }

    const strength = entry.strength == null ? 1 : Number(entry.strength);
    if (!Number.isFinite(strength)) {
      throw new ScriptValidationError(`args.loras[${index}].strength must be a finite number`);
    }

    return { path: canonicalExistingPath(loraPath), strength };
  });

  const output = path.resolve(requireString(args.output, 'args.output'));
  if (!output.toLowerCase().endsWith('.safetensors')) {
    throw new ScriptValidationError('args.output must be a .safetensors file');
  }
  const outputParent = path.dirname(output);
  if (!fs.existsSync(outputParent) || !fs.statSync(outputParent).isDirectory()) {
    throw new ScriptValidationError('args.output parent folder does not exist');
  }
  if (!isPathInside(trainingRoot, outputParent)) {
    throw new ScriptValidationError('Output must be inside the local training folder');
  }

  const saveDtype = typeof args.save_dtype === 'string' && args.save_dtype.trim() ? args.save_dtype.trim() : 'bfloat16';
  if (!SAVE_DTYPES.has(saveDtype)) {
    throw new ScriptValidationError('args.save_dtype is invalid');
  }

  const device = typeof args.device === 'string' && args.device.trim() ? args.device.trim() : 'cpu';
  if (!DEVICE_RE.test(device)) {
    throw new ScriptValidationError('args.device is invalid');
  }

  return [
    '--loras',
    JSON.stringify(loras),
    '--output',
    output,
    '--save_dtype',
    saveDtype,
    '--device',
    device,
  ];
}

export function buildScriptInvocation(body: unknown, trainingRoot: string): ScriptInvocation {
  if (!body || typeof body !== 'object') {
    throw new ScriptValidationError('Invalid JSON body');
  }

  const payload = body as Record<string, unknown>;
  return {
    scriptPath: resolveAllowedScriptPath(payload.script),
    args: normalizeMergeLorasArgs(payload.args, trainingRoot),
  };
}

function parseResult(stdout: string): unknown {
  const lines = stdout.trimEnd().split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (!line) continue;
    if (line.startsWith('{') || line.startsWith('[')) {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    }
    return null;
  }
  return null;
}

export function runScriptBuffered(invocation: ScriptInvocation): Promise<ScriptRunResult> {
  return new Promise(resolve => {
    const child = spawn(getToolkitPythonPath(), ['-u', invocation.scriptPath, ...invocation.args], {
      cwd: TOOLKIT_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, SCRIPT_TIMEOUT_MS);

    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf-8');
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf-8');
    });

    child.on('error', error => {
      clearTimeout(timer);
      resolve({
        ok: false,
        exitCode: null,
        signal: null,
        stdout,
        stderr,
        result: null,
        timedOut,
        error: error.message,
      });
    });

    child.on('close', (code, signal) => {
      clearTimeout(timer);
      resolve({
        ok: !timedOut && code === 0,
        exitCode: code,
        signal,
        stdout,
        stderr,
        result: parseResult(stdout),
        timedOut,
        error: timedOut ? 'Script timed out after 20 minutes' : undefined,
      });
    });
  });
}

export function runScriptStreaming(invocation: ScriptInvocation): Response {
  const child = spawn(getToolkitPythonPath(), ['-u', invocation.scriptPath, ...invocation.args], {
    cwd: TOOLKIT_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
    windowsHide: true,
  });

  const encoder = new TextEncoder();
  let stdoutBuf = '';
  let stderrBuf = '';
  let timedOut = false;
  let streamClosed = false;
  let childSettled = false;
  let timer: NodeJS.Timeout | null = null;
  let streamController: { enqueue(chunk: Uint8Array): void; close(): void } | null = null;

  const clearRunTimer = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const detachOutputListeners = () => {
    child.stdout.off('data', handleStdout);
    child.stderr.off('data', handleStderr);
  };

  const finishChildListeners = () => {
    clearRunTimer();
    detachOutputListeners();
    child.off('error', handleError);
    child.off('close', handleClose);
    streamController = null;
  };

  const stopStreaming = () => {
    streamClosed = true;
    streamController = null;
    clearRunTimer();
    detachOutputListeners();
  };

  const send = (obj: unknown) => {
    if (streamClosed || !streamController) return false;
    try {
      streamController.enqueue(encoder.encode(JSON.stringify(obj) + '\n'));
      return true;
    } catch {
      stopStreaming();
      if (!child.killed) child.kill('SIGKILL');
      return false;
    }
  };

  const closeStream = () => {
    if (streamClosed) return;
    const controller = streamController;
    streamClosed = true;
    streamController = null;
    try {
      controller?.close();
    } catch {
      // The client may have disconnected between the last send and close.
    }
  };

  function handleStdout(chunk: Buffer) {
    const text = chunk.toString('utf-8');
    stdoutBuf += text;
    send({ type: 'stdout', data: text });
  }

  function handleStderr(chunk: Buffer) {
    const text = chunk.toString('utf-8');
    stderrBuf += text;
    send({ type: 'stderr', data: text });
  }

  function handleError(error: Error) {
    if (childSettled) return;
    childSettled = true;
    clearRunTimer();
    if (!streamClosed) {
      send({ type: 'error', message: error.message });
      closeStream();
    }
    finishChildListeners();
  }

  function handleClose(code: number | null, signal: NodeJS.Signals | null) {
    if (childSettled) return;
    childSettled = true;
    clearRunTimer();
    if (!streamClosed) {
      send({
        type: 'exit',
        exitCode: code,
        signal,
        ok: !timedOut && code === 0,
        timedOut,
        result: parseResult(stdoutBuf),
        stderr: stderrBuf,
      });
      closeStream();
    }
    finishChildListeners();
  }

  const stream = new ReadableStream({
    start(controller) {
      streamController = controller;

      timer = setTimeout(() => {
        timedOut = true;
        send({ type: 'error', message: 'Script timed out after 20 minutes' });
        child.kill('SIGKILL');
      }, SCRIPT_TIMEOUT_MS);

      child.stdout.on('data', handleStdout);
      child.stderr.on('data', handleStderr);
      child.on('error', handleError);
      child.on('close', handleClose);
    },
    cancel() {
      stopStreaming();
      if (!child.killed) child.kill('SIGKILL');
    },
  });

  return new Response(stream as unknown as BodyInit, {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
    },
  });
}
