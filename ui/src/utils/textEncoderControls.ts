import type { DatasetConfig, ProcessConfig, TrainConfig } from '../types';

export type TextEncoderControl = 'training' | 'cache' | 'unload';
type TextEncoderConfig = Pick<ProcessConfig, 'train' | 'datasets'>;

/** Match the trainer's per-dataset override of the global cache default. */
export function hasTextEmbeddingCache(
  train: Pick<TrainConfig, 'cache_text_embeddings'>,
  datasets: Pick<DatasetConfig, 'cache_text_embeddings'>[],
): boolean {
  if (!datasets.length) return Boolean(train.cache_text_embeddings);
  return datasets.some(dataset => dataset.cache_text_embeddings ?? train.cache_text_embeddings);
}

export function updateTextEncoderControls(
  config: TextEncoderConfig,
  control: TextEncoderControl,
  enabled: boolean,
): TextEncoderConfig {
  const train = { ...config.train };
  let datasets = config.datasets;
  const setCache = (value: boolean) => {
    train.cache_text_embeddings = value;
    datasets = datasets.map(dataset => ({ ...dataset, cache_text_embeddings: value }));
  };

  if (control === 'training') {
    train.train_text_encoder = enabled;
    if (enabled) {
      train.unload_text_encoder = false;
      setCache(false);
    }
  } else if (enabled && train.train_text_encoder) {
    return config;
  } else if (control === 'cache') {
    setCache(enabled);
    // Disabling a cached preset must restore caption encoding, not trigger-only training.
    train.unload_text_encoder = false;
  } else {
    train.unload_text_encoder = enabled;
    if (enabled) setCache(false);
  }

  return { train, datasets };
}
