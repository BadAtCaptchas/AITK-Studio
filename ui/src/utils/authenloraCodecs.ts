import type { SelectOption } from '@/types';

export const AUTHENLORA_BUILTIN_CODEC_BITS: Record<string, number> = {
  'builtin:authenlora_48bits': 48,
  'builtin:authenlora_80bits': 80,
  'builtin:authenlora_100bits': 100,
};

export const AUTHENLORA_CODEC_OPTIONS: SelectOption[] = [
  { value: 'builtin:authenlora_48bits', label: 'Built-in 48-bit' },
  { value: 'builtin:authenlora_80bits', label: 'Built-in 80-bit' },
  { value: 'builtin:authenlora_100bits', label: 'Built-in 100-bit' },
  { value: 'custom', label: 'Custom path' },
];

export function getAuthenloraCodecBits(codecPath: string | null | undefined) {
  return AUTHENLORA_BUILTIN_CODEC_BITS[codecPath || ''] || null;
}

export function getAuthenloraCodecSelectValue(codecPath: string | null | undefined) {
  return getAuthenloraCodecBits(codecPath) ? codecPath || 'builtin:authenlora_48bits' : 'custom';
}
