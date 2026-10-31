function emaPass(ys: (number | null)[], alpha: number, reverse: boolean) {
  const vals: (number | null)[] = new Array(ys.length).fill(null);
  const weights: number[] = new Array(ys.length).fill(0);
  const start = reverse ? ys.length - 1 : 0;
  const step = reverse ? -1 : 1;
  let s = 0;
  let n = 0;
  for (let i = start; i >= 0 && i < ys.length; i += step) {
    const v = ys[i];
    if (v === null || !Number.isFinite(v)) {
      continue;
    }
    s = alpha * (v as number) + (1 - alpha) * s;
    n += 1;
    const w = 1 - Math.pow(1 - alpha, n);
    vals[i] = s / w;
    weights[i] = w;
  }
  return { vals, weights };
}

export function emaWithNulls(ys: (number | null)[], alpha: number): (number | null)[] {
  const fwd = emaPass(ys, alpha, false);
  const bwd = emaPass(ys, alpha, true);
  const out: (number | null)[] = new Array(ys.length);
  for (let i = 0; i < ys.length; i++) {
    const f = fwd.vals[i];
    const b = bwd.vals[i];
    if (f === null || b === null) {
      out[i] = null;
      continue;
    }
    const wf = fwd.weights[i];
    const wb = bwd.weights[i];
    const wsum = wf + wb;
    out[i] = wsum > 0 ? (wf * f + wb * b) / wsum : (f + b) / 2;
  }
  return out;
}
