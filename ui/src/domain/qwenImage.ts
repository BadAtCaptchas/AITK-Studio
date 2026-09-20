import type { SampleItem } from '@/types';

export const QWEN_IMAGE_PRESETS = [
  { label: 'Quick preview (1024 x 1024, 20 steps)', width: 1024, height: 1024, steps: 20 },
  { label: '2K square (1:1)', width: 2048, height: 2048, steps: 40 },
  { label: '2K landscape (4:3)', width: 2400, height: 1792, steps: 40 },
  { label: '2K portrait (3:4)', width: 1792, height: 2400, steps: 40 },
  { label: '2K landscape (3:2)', width: 2528, height: 1696, steps: 40 },
  { label: '2K portrait (2:3)', width: 1696, height: 2528, steps: 40 },
  { label: '2K widescreen (16:9)', width: 2752, height: 1536, steps: 40 },
  { label: '2K tall (9:16)', width: 1536, height: 2752, steps: 40 },
] as const;

export function transparentQwenPrompt(prompt: string): string {
  if (prompt.startsWith('This is an RGBA image with transparency.')) return prompt;
  return `This is an RGBA image with transparency. ${prompt.trim()} The image has alpha channel and the background is transparent.`;
}

export function qwenSampleReferences(sample: Pick<SampleItem, 'ctrl_imgs' | 'ctrl_img' | 'ctrl_img_1' | 'ctrl_img_2' | 'ctrl_img_3'>): string[] {
  if (sample.ctrl_imgs) return sample.ctrl_imgs;
  return [sample.ctrl_img, sample.ctrl_img_1 !== sample.ctrl_img ? sample.ctrl_img_1 : null, sample.ctrl_img_2, sample.ctrl_img_3]
    .filter((path): path is string => typeof path === 'string' && path.length > 0);
}
