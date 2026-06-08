export type FluxGuidanceBypassPolicy = 'required' | 'forbidden' | 'unspecified';

export type FluxGuidancePolicyInput = {
  arch?: string | null;
  name_or_path?: string | null;
  use_flux_cfg?: boolean | null;
};

function normalizeModelRef(value?: string | null): string {
  return String(value ?? '').trim().replace(/\\/g, '/').toLowerCase();
}

function looksLikeFlexRef(modelRef: string): boolean {
  return /(^|[/\\])flex\.[12](?:[^a-z0-9]|$)/i.test(modelRef);
}

export function getFluxGuidanceBypassPolicy(input: FluxGuidancePolicyInput): FluxGuidanceBypassPolicy {
  const arch = String(input.arch ?? '').trim().toLowerCase();
  const baseArch = arch.split(':')[0];
  const modelRef = normalizeModelRef(input.name_or_path);

  if (input.use_flux_cfg && (baseArch === 'flux' || baseArch === 'flux_kontext')) {
    return 'required';
  }

  if (baseArch === 'flex1' || baseArch === 'flex2' || looksLikeFlexRef(modelRef)) {
    return 'required';
  }

  if (baseArch) {
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
