import fs from 'node:fs/promises';
import type { FileHandle } from 'node:fs/promises';

// Bound allocations before parsing untrusted headers; never read tensor payloads.
export const MAX_HEADER_BYTES = 16 * 1024 * 1024;
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const integer = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;

export class SafetensorsInfoError extends Error {}
const invalid = (): never => { throw new SafetensorsInfoError('Invalid or unsupported safetensors header'); };

async function readExactly(handle: FileHandle, size: number, position: number) {
  const buffer = Buffer.alloc(size);
  let offset = 0;
  while (offset < size) {
    const { bytesRead } = await handle.read(buffer, offset, size - offset, position + offset);
    if (!bytesRead) invalid();
    offset += bytesRead;
  }
  return buffer;
}

export async function readSafetensorsInfo(filePath: string) {
  const handle = await fs.open(filePath, 'r');
  try {
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size < 10 || !Number.isSafeInteger(stat.size)) invalid();
    const prefix = await readExactly(handle, 8, 0);
    const length = prefix.readBigUInt64LE();
    if (length < BigInt(2) || length > BigInt(MAX_HEADER_BYTES) || length > BigInt(stat.size - 8)) invalid();
    const headerSize = Number(length);
    let header: unknown;
    try {
      header = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(await readExactly(handle, headerSize, 8)));
    } catch { return invalid(); }
    if (!record(header)) return invalid();
    const metadata: Record<string, string> = Object.create(null);
    if (header.__metadata__ !== undefined) {
      if (!record(header.__metadata__)) return invalid();
      for (const [key, value] of Object.entries(header.__metadata__)) {
        if (typeof value !== 'string') return invalid();
        metadata[key] = value;
      }
    }
    const tensors: { name: string; dtype: string; shape: number[]; numel: number; bytes: number }[] = [];
    const dtypeCounts: Record<string, { count: number; numel: number; bytes: number }> = Object.create(null);
    const ranges: [number, number][] = [];
    let totalParams = 0, totalBytes = 0;
    for (const [name, tensor] of Object.entries(header)) {
      if (name === '__metadata__') continue;
      if (!record(tensor) || typeof tensor.dtype !== 'string' || !/^[A-Z][A-Z0-9_]{0,31}$/.test(tensor.dtype)
        || !Array.isArray(tensor.shape) || !tensor.shape.every(integer) || tensor.shape.length > 32
        || !Array.isArray(tensor.data_offsets) || tensor.data_offsets.length !== 2 || !tensor.data_offsets.every(integer)) return invalid();
      const shape: number[] = tensor.shape;
      const numel = shape.includes(0) ? 0 : shape.reduce((a, b) => a * b, 1);
      const [start, end] = tensor.data_offsets;
      if (!Number.isSafeInteger(numel) || end < start || end > stat.size - 8 - headerSize) return invalid();
      const bytes = end - start;
      ranges.push([start, end]);
      tensors.push({ name, dtype: tensor.dtype, shape, numel, bytes });
      totalParams += numel;
      totalBytes += bytes;
      if (!Number.isSafeInteger(totalParams) || !Number.isSafeInteger(totalBytes)) return invalid();
      const count = dtypeCounts[tensor.dtype] ??= { count: 0, numel: 0, bytes: 0 };
      count.count++; count.numel += numel; count.bytes += bytes;
    }
    ranges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let end = 0;
    for (const range of ranges) {
      if (range[0] !== end) return invalid();
      end = range[1];
    }
    if (end !== stat.size - 8 - headerSize) return invalid();
    tensors.sort((a, b) => a.name.localeCompare(b.name));
    return { filePath, fileSize: stat.size, headerSize, metadata, tensors, tensorCount: tensors.length, totalParams, totalBytes, dtypeCounts };
  } finally { await handle.close(); }
}
