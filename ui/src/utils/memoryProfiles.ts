export type LayerOffloadingBackend = 'block' | 'legacy';

export interface LayerOffloadingMemoryProfile {
  backend: LayerOffloadingBackend;
  transformerPercent: number;
  textEncoderPercent: number;
}

const BLOCK_SUPPORTED_ARCHES = new Set([
  'flux',
  'flux_kontext',
  'flux2',
  'flux2_klein_4b',
  'flux2_klein_9b',
  'asymflux2_klein_9b',
  'wan21',
  'wan21_i2v',
  'wan22_14b',
  'wan22_14b_i2v',
  'wan22_5b',
  'zimage',
  'zimage_l2p',
  'zeta_chroma',
  'glm_image',
  'qwen_image',
  'qwen_image_edit',
  'qwen_image_edit_plus',
  'hidream',
  'hidream_e1',
  'hidream_o1',
  'ltx2',
  'ltx2.3',
  'nucleus_image',
  'ernie_image',
  'ideogram4',
  'prx_pixel',
]);

const LARGE_TEXT_ENCODER_ARCHES = new Set([
  'flux',
  'flux_kontext',
  'flux2',
  'flux2_klein_4b',
  'flux2_klein_9b',
  'asymflux2_klein_9b',
  'wan21',
  'wan21_i2v',
  'zimage',
  'zimage_l2p',
  'zeta_chroma',
  'glm_image',
  'qwen_image',
  'qwen_image_edit',
  'qwen_image_edit_plus',
  'ltx2',
  'ltx2.3',
  'nucleus_image',
  'ernie_image',
  'ideogram4',
  'prx_pixel',
]);

export function normalizeMemoryProfileArch(archName?: string | null) {
  return String(archName || '').split(':')[0];
}

export function supportsBlockLayerOffloading(archName?: string | null) {
  return BLOCK_SUPPORTED_ARCHES.has(normalizeMemoryProfileArch(archName));
}

export function getLayerOffloadingMemoryProfile(archName?: string | null): LayerOffloadingMemoryProfile {
  const arch = normalizeMemoryProfileArch(archName);
  if (!BLOCK_SUPPORTED_ARCHES.has(arch)) {
    return {
      backend: 'legacy',
      transformerPercent: 1.0,
      textEncoderPercent: 1.0,
    };
  }

  return {
    backend: 'block',
    transformerPercent: 0.7,
    textEncoderPercent: LARGE_TEXT_ENCODER_ARCHES.has(arch) ? 0.5 : 0.25,
  };
}
