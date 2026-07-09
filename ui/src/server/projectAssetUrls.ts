import { createHmac, timingSafeEqual } from 'crypto';

export const PROJECT_ASSET_SIGNATURE_CONTEXT = 'project-asset-v1';
export const PROJECT_ASSET_DEFAULT_TTL_MS = 15 * 60 * 1000;

export type ProjectAssetDisposition = 'inline' | 'attachment';

function signingSecret() {
  return process.env.AI_TOOLKIT_AUTH || 'aitk-local-project-assets';
}

export function normalizeProjectAssetPath(value: string) {
  const segments = value.replace(/\\/g, '/').split('/').filter(Boolean);
  if (segments.length === 0 || segments.some(segment => segment === '.' || segment === '..' || segment.includes('\0'))) {
    throw new Error('Invalid project asset path');
  }
  return segments.join('/');
}

export function projectAssetSignaturePayload(
  projectID: string,
  relativePath: string,
  disposition: ProjectAssetDisposition,
  expires: number,
) {
  return [PROJECT_ASSET_SIGNATURE_CONTEXT, projectID, relativePath, disposition, String(expires)].join('\n');
}

export function signProjectAsset(
  projectID: string,
  relativePath: string,
  disposition: ProjectAssetDisposition,
  expires: number,
) {
  return createHmac('sha256', signingSecret())
    .update(projectAssetSignaturePayload(projectID, relativePath, disposition, expires))
    .digest('base64url');
}

export function verifyProjectAssetSignature(options: {
  projectID: string;
  relativePath: string;
  disposition: ProjectAssetDisposition;
  expires: number;
  signature: string;
}) {
  if (!Number.isSafeInteger(options.expires) || options.expires <= Date.now() || !options.signature) return false;
  const expected = signProjectAsset(
    options.projectID,
    options.relativePath,
    options.disposition,
    options.expires,
  );
  const left = Buffer.from(expected);
  const right = Buffer.from(options.signature);
  return left.length === right.length && timingSafeEqual(left, right);
}

export function createProjectAssetUrl(
  projectID: string,
  relativePathInput: string,
  disposition: ProjectAssetDisposition,
  expires = Date.now() + PROJECT_ASSET_DEFAULT_TTL_MS,
) {
  const relativePath = normalizeProjectAssetPath(relativePathInput);
  const signature = signProjectAsset(projectID, relativePath, disposition, expires);
  const encodedPath = relativePath.split('/').map(encodeURIComponent).join('/');
  const query = new URLSearchParams({
    project_id: projectID,
    path: relativePath,
    disposition,
    expires: String(expires),
    sig: signature,
  });
  return `/api/project-assets/${encodeURIComponent(projectID)}/${encodedPath}?${query.toString()}`;
}
