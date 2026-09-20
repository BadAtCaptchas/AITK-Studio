import { createHmac } from 'crypto';

/** Per-attempt credentials stay in process memory and never enter job configs or endpoint files. */
export function inferenceToken(jobID: string, attemptID: string): string {
  const secret = process.env.AITK_INTERNAL_TOKEN;
  if (!secret || !attemptID) throw new Error('Live inference requires the managed Studio app stack');
  return createHmac('sha256', secret).update(`aitk-engine:${jobID}:${attemptID}`).digest('base64url');
}
