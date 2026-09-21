import type { ModelConfig, NetworkConfig } from '@/types';

const compactExclusions = ['txtfusion.', 'txtmlp'] as const;
const expandedExclusions = [
  'txtfusion.layerwise_blocks.0',
  'txtfusion.layerwise_blocks.1',
  'txtfusion.projector',
  'txtfusion.refiner_blocks.0',
  'txtfusion.refiner_blocks.1',
  'txtmlp',
] as const;
const presetExclusions = new Set<string>([...compactExclusions, ...expandedExclusions]);

type Exclusions = Readonly<NetworkConfig['network_kwargs']['ignore_if_contains']>;
type ModelOptions = Pick<ModelConfig, 'arch'> & { model_kwargs?: Record<string, unknown> };

export function supportsKrea2TextFusionExclusion(
  model: ModelOptions,
  network?: Pick<NetworkConfig, 'type'>,
): boolean {
  return (
    (model.arch === 'krea2' || model.arch === 'krea2:turbo') &&
    network?.type === 'lora' &&
    !model.model_kwargs?.edit
  );
}

// This reports the recognized preset, not arbitrary custom substring filters.
export function hasKrea2TextFusionExclusion(exclusions?: Exclusions | null): boolean {
  return (
    compactExclusions.every(entry => exclusions?.includes(entry)) ||
    expandedExclusions.every(entry => exclusions?.includes(entry))
  );
}

export function setKrea2TextFusionExclusion(
  exclusions: Exclusions | null | undefined,
  enabled: boolean,
): NetworkConfig['network_kwargs']['ignore_if_contains'] {
  const customExclusions = (exclusions ?? []).filter(entry => !presetExclusions.has(entry));
  return enabled ? [...customExclusions, ...compactExclusions] : customExclusions;
}
