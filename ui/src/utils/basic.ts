export const objectCopy = <T>(obj: T): T => {
  return JSON.parse(JSON.stringify(obj)) as T;
};

export const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export type PollVisibilitySource = {
  readonly visibilityState: string;
  addEventListener(type: 'visibilitychange', listener: () => void): void;
  removeEventListener(type: 'visibilitychange', listener: () => void): void;
};

export function startSettledPollLoop(
  callback: (signal: AbortSignal) => void | Promise<unknown>,
  intervalMs: number | null | undefined,
  onError: (error: unknown) => void = error => console.error('Polling request failed:', error),
  visibility?: PollVisibilitySource,
) {
  let stopped = false;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let activeController: AbortController | null = null;
  let hidden = visibility?.visibilityState === 'hidden';
  let refreshOnSettle = false;

  const clearTimer = () => {
    if (timeout !== undefined) clearTimeout(timeout);
    timeout = undefined;
  };

  const tick = async () => {
    // A transport may take time to settle after abort. Foregrounding queues a
    // refresh rather than overlapping that outstanding request.
    if (stopped || hidden || activeController) return;
    refreshOnSettle = false;
    const controller = new AbortController();
    activeController = controller;
    try {
      await callback(controller.signal);
    } catch (error) {
      if (!controller.signal.aborted) onError(error);
    } finally {
      if (activeController === controller) activeController = null;
    }

    if (stopped || hidden) return;
    if (refreshOnSettle) {
      void tick();
    } else if (intervalMs) {
      timeout = setTimeout(() => {
        timeout = undefined;
        void tick();
      }, intervalMs);
    }
  };

  const onVisibilityChange = () => {
    const nextHidden = visibility?.visibilityState === 'hidden';
    if (stopped || nextHidden === hidden) return;
    hidden = nextHidden;
    clearTimer();
    if (hidden) {
      refreshOnSettle = false;
      activeController?.abort();
    } else {
      refreshOnSettle = true;
      void tick();
    }
  };

  visibility?.addEventListener('visibilitychange', onVisibilityChange);
  void tick();

  return () => {
    stopped = true;
    visibility?.removeEventListener('visibilitychange', onVisibilityChange);
    activeController?.abort();
    clearTimer();
  };
}

export const imgExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.jxl', '.svg', '.bmp'];
export const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'];
export const audioExtensions = ['.mp3', '.wav', '.flac', '.ogg'];
export const textCaptionExtensions = ['.json', '.txt', '.caption', '.sdxl', '.md'];

export const getFileExtension = (filePath: string) => {
  const cleanPath = filePath.split(/[?#]/, 1)[0];
  const fileName = cleanPath.split(/[\\/]/).pop() || cleanPath;
  const dotIndex = fileName.lastIndexOf('.');
  return dotIndex >= 0 ? fileName.slice(dotIndex).toLowerCase() : '';
};

export const isVideo = (filePath: string) => videoExtensions.includes(getFileExtension(filePath));
export const isImage = (filePath: string) => imgExtensions.includes(getFileExtension(filePath));
export const isAudio = (filePath: string) => audioExtensions.includes(getFileExtension(filePath));
export const isTextCaption = (filePath: string) => textCaptionExtensions.includes(getFileExtension(filePath));

/** Keep the basename readable while encoding the directory as one catch-all segment. */
export const encodeFilePathForUrl = (filePath: string) => {
  const index = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'));
  if (index === -1) return encodeURIComponent(filePath);
  return `${encodeURIComponent(filePath.slice(0, index))}/${encodeURIComponent(filePath.slice(index + 1))}`;
};

export const tagsToObj = (tagStr: string): Record<string, any> => {
  const result: Record<string, any> = {};
  const regex = /<([A-Z_][A-Z0-9_]*)>([\s\S]*?)<\/\1>/g;
  let match;
  while ((match = regex.exec(tagStr)) !== null) {
    const value = match[2].trim();
    try {
      result[match[1]] = JSON.parse(value);
    } catch {
      result[match[1]] = value;
    }
  }
  return result;
};

export const objToTags = (obj: Record<string, any>): string => {
  return Object.entries(obj)
    .map(([key, value]) => {
      const content = typeof value === 'string' ? value : JSON.stringify(value);
      return `<${key}>${content}</${key}>`;
    })
    .join('\n');
};

export const pathJoin = (...parts: string[]) => {
  const sep = parts.length > 0 && parts[0].includes('\\') ? '\\' : '/';
  const leadingTrailing = sep === '\\' ? /^\\+|\\+$/g : /^\/+|\/+$/g;
  const trailing = sep === '\\' ? /\\+$/ : /\/+$/;
  return parts
    .map((part, index) => {
      if (index === 0) {
        return part.replace(trailing, '');
      } else {
        return part.replace(leadingTrailing, '');
      }
    })
    .filter(part => part.length > 0)
    .join(sep);
}

export const isText = (value: string): boolean => /\.txt(?:[?#]|$)/i.test(value);
