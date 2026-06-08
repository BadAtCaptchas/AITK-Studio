export type FluxGuidanceBypassPolicy = 'required' | 'forbidden' | 'unspecified';

export type FluxGuidancePolicyInput = {
  arch?: string | null;
  name_or_path?: string | null;
  use_flux_cfg?: boolean | null;
};

const FLEX_GUIDANCE_BYPASS_REFS = [
  'ostris/flex.1-alpha',
  'flex.1-alpha',
  'ostris/flex.2-preview',
  'flex.2-preview',
];

const OFFICIAL_FLUX_NO_GUIDANCE_BYPASS_REFS = [
  'black-forest-labs/flux.1-dev',
  'flux.1-dev',
  'black-forest-labs/flux.1-schnell',
  'flux.1-schnell',
  'black-forest-labs/flux.1-kontext-dev',
  'flux.1-kontext-dev',
];

function normalizeModelRef(value?: string | null): string {
  return String(value ?? '').trim().replace(/\\/g, '/').toLowerCase();
}

function matchesKnownRef(modelRef: string, refs: string[]): boolean {
  return refs.some(ref => modelRef.includes(ref));
}

export function getFluxGuidanceBypassPolicy(input: FluxGuidancePolicyInput): FluxGuidanceBypassPolicy {
  if (input.use_flux_cfg) {
    return 'required';
  }

  const arch = String(input.arch ?? '').trim().toLowerCase();
  const baseArch = arch.split(':')[0];
  const modelRef = normalizeModelRef(input.name_or_path);

  if (
    baseArch === 'flex1' ||
    baseArch === 'flex2' ||
    matchesKnownRef(modelRef, FLEX_GUIDANCE_BYPASS_REFS)
  ) {
    return 'required';
  }

  if (
    baseArch === 'flux' ||
    baseArch === 'flux_kontext' ||
    matchesKnownRef(modelRef, OFFICIAL_FLUX_NO_GUIDANCE_BYPASS_REFS)
  ) {
    return 'forbidden';
  }

  return 'unspecified';
}

export function expectedFluxGuidanceBypass(input: FluxGuidancePolicyInput): boolean | null {
  const policy = getFluxGuidanceBypassPolicy(input);
  if (policy === 'required') {
    return true;
  }
  if (policy === 'forbidden') {
    return false;
  }
  return null;
}
