import fs from 'fs';

export const DATASET_CAPTION_SIDECAR_EXTENSIONS = ['.json', '.txt', '.caption', '.sdxl', '.md'];
export const DATASET_TEXT_CAPTION_EXTENSIONS = ['.txt', '.caption', '.sdxl', '.md'];

export function isTextCaptionFilePath(filePath: string) {
  const cleanPath = filePath.split(/[?#]/, 1)[0];
  const fileName = cleanPath.split(/[\\/]/).pop() || cleanPath;
  const dotIndex = fileName.lastIndexOf('.');
  const extension = dotIndex >= 0 ? fileName.slice(dotIndex).toLowerCase() : '';
  return DATASET_TEXT_CAPTION_EXTENSIONS.includes(extension);
}

export function captionStemForMediaPath(mediaPath: string) {
  return mediaPath.replace(/\.[^/.\\]+$/, '');
}

export function captionSidecarPath(mediaPath: string, extension: string) {
  const normalizedExtension = extension.startsWith('.') ? extension : `.${extension}`;
  return `${captionStemForMediaPath(mediaPath)}${normalizedExtension.toLowerCase()}`;
}

export function findExistingCaptionSidecar(mediaPath: string) {
  for (const extension of DATASET_CAPTION_SIDECAR_EXTENSIONS) {
    const candidate = captionSidecarPath(mediaPath, extension);
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

export function readCaptionSidecar(mediaPath: string) {
  if (isTextCaptionFilePath(mediaPath) && fs.existsSync(mediaPath) && fs.statSync(mediaPath).isFile()) {
    return fs.readFileSync(mediaPath, 'utf-8');
  }

  const captionPath = findExistingCaptionSidecar(mediaPath);
  if (!captionPath) return '';
  return fs.readFileSync(captionPath, 'utf-8');
}

export function resolveCaptionWritePath(mediaPath: string, caption: string) {
  if (isTextCaptionFilePath(mediaPath)) return mediaPath;

  const existingPath = findExistingCaptionSidecar(mediaPath);
  if (existingPath) return existingPath;

  try {
    const parsed = JSON.parse(caption);
    if (parsed && typeof parsed === 'object') {
      return captionSidecarPath(mediaPath, '.json');
    }
  } catch {
    // Plain text captions still default to .txt.
  }

  return captionSidecarPath(mediaPath, '.txt');
}

export function deleteCaptionSidecars(mediaPath: string) {
  for (const extension of DATASET_CAPTION_SIDECAR_EXTENSIONS) {
    const candidate = captionSidecarPath(mediaPath, extension);
    if (fs.existsSync(candidate)) {
      fs.unlinkSync(candidate);
    }
  }
}
