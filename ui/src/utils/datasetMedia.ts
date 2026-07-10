import { getMediaUrl } from './media';

export type DatasetImageListItem = {
  img_path: string;
  media_url?: string | null;
  added_at?: string | null;
  captioned_at?: string | null;
  size_bytes?: number | null;
};

export function normalizeDatasetImageListItem(item: unknown, root: string | null): DatasetImageListItem | null {
  if (!item || typeof item !== 'object') return null;
  const record = item as Record<string, unknown>;
  const imagePath = record.img_path;
  if (typeof imagePath !== 'string') return null;

  return {
    img_path: root ? root + imagePath : imagePath,
    media_url: typeof record.media_url === 'string' ? record.media_url : null,
    added_at: typeof record.added_at === 'string' ? record.added_at : null,
    captioned_at: typeof record.captioned_at === 'string' ? record.captioned_at : null,
    size_bytes: typeof record.size_bytes === 'number' ? record.size_bytes : null,
  };
}

export function getDatasetImageMediaUrl(item: Pick<DatasetImageListItem, 'img_path' | 'media_url'>) {
  return getMediaUrl(item.media_url || item.img_path);
}
