import type React from 'react';

/**
 * UI database records
 */

export type DbDate = string | Date;

export interface Queue {
  id: number;
  gpu_ids: string;
  is_running: boolean;
}

export interface Job {
  id: string;
  name: string;
  gpu_ids: string;
  job_config: string;
  created_at: DbDate;
  updated_at: DbDate;
  status: string;
  stop: boolean;
  return_to_queue: boolean;
  step: number;
  info: string;
  speed_string: string;
  queue_position: number;
  pid: number | null;
  job_type: string;
  job_ref: string | null;
}

/**
 * GPU API response
 */

export interface GpuUtilization {
  gpu: number;
  memory: number;
}

export interface GpuMemory {
  total: number;
  free: number;
  used: number;
}

export interface GpuPower {
  draw: number;
  limit: number;
}

export interface GpuClocks {
  graphics: number;
  memory: number;
}

export interface GpuFan {
  speed: number;
}

export interface GpuInfo {
  index: number;
  name: string;
  driverVersion: string;
  temperature: number;
  utilization: GpuUtilization;
  memory: GpuMemory;
  power: GpuPower;
  clocks: GpuClocks;
  fan: GpuFan;
}

export interface CpuInfo {
  name: string;
  cores: number;
  temperature: number;
  totalMemory: number;
  freeMemory: number;
  availableMemory: number;
  currentLoad: number;
}

export interface SystemMetricSample {
  id: string;
  created_at: DbDate;
  scope: string;
  device_id: string;
  metric: string;
  value: number;
  unit: string;
  metadata: string;
}

export interface ModelArtifact {
  id: string;
  kind: string;
  name: string;
  path: string;
  source: string;
  job_id: string | null;
  step: number | null;
  exists: boolean;
  size: number | null;
  modified_at: DbDate | null;
  metadata: string;
  created_at: DbDate;
  updated_at: DbDate;
}

export interface EvaluationRun {
  id: string;
  name: string;
  status: string;
  job_ids: string;
  artifact_ids: string;
  reference_path: string | null;
  metrics: string;
  error: string | null;
  created_at: DbDate;
  updated_at: DbDate;
  completed_at: DbDate | null;
}

export interface EvaluationItem {
  id: string;
  run_id: string;
  item_type: string;
  item_id: string;
  sample_path: string | null;
  reference_path: string | null;
  metrics: string;
  error: string | null;
  status: string;
  created_at: DbDate;
  updated_at: DbDate;
}

export interface AlertRule {
  id: string;
  name: string;
  metric: string;
  operator: string;
  threshold: number;
  duration_seconds: number;
  severity: string;
  enabled: boolean;
  config: string;
  created_at: DbDate;
  updated_at: DbDate;
}

export interface AlertEvent {
  id: string;
  rule_id: string | null;
  title: string;
  message: string;
  severity: string;
  status: string;
  resource_type: string;
  resource_id: string;
  fingerprint: string;
  created_at: DbDate;
  updated_at: DbDate;
  acknowledged_at: DbDate | null;
  snoozed_until: DbDate | null;
  resolved_at: DbDate | null;
  metadata: string;
}

export interface GPUApiResponse {
  hasNvidiaSmi: boolean;
  isMac: boolean;
  gpus: GpuInfo[];
  error?: string;
}

/**
 * Training configuration
 */

export interface NetworkConfig {
  type: string;
  linear: number;
  linear_alpha: number;
  conv: number;
  conv_alpha: number;
  lokr_full_rank: boolean;
  lokr_factor: number;
  network_kwargs: {
    ignore_if_contains: string[];
  };
}

export interface SaveConfig {
  dtype: string;
  save_every: number;
  max_step_saves_to_keep: number;
  save_format: string;
  push_to_hub: boolean;
}

export interface DatasetConfig {
  folder_path: string;
  mask_path: string | null;
  mask_min_value: number;
  default_caption: string;
  caption_ext: string;
  caption_dropout_rate: number;
  shuffle_tokens?: boolean;
  is_reg: boolean;
  network_weight: number;
  cache_latents_to_disk?: boolean;
  resolution: number[];
  controls: string[];
  control_path?: string | null;
  num_frames: number;
  shrink_video_to_frames: boolean;
  do_i2v?: boolean;
  do_audio?: boolean;
  audio_normalize?: boolean;
  audio_preserve_pitch?: boolean;
  fps?: number;
  flip_x: boolean;
  flip_y: boolean;
  num_repeats?: number;
  control_path_1?: string | null;
  control_path_2?: string | null;
  control_path_3?: string | null;
  auto_frame_count?: boolean;
}

export interface EMAConfig {
  use_ema: boolean;
  ema_decay: number;
}

export interface TrainConfig {
  batch_size: number;
  bypass_guidance_embedding?: boolean;
  steps: number;
  gradient_accumulation: number;
  train_unet: boolean;
  train_text_encoder: boolean;
  gradient_checkpointing: boolean;
  noise_scheduler: string;
  timestep_type: string;
  content_or_style: string;
  optimizer: string;
  lr: number;
  ema_config?: EMAConfig;
  dtype: string;
  unload_text_encoder: boolean;
  cache_text_embeddings: boolean;
  optimizer_params: {
    weight_decay: number;
  };
  skip_first_sample: boolean;
  force_first_sample: boolean;
  disable_sampling: boolean;
  diff_output_preservation: boolean;
  diff_output_preservation_multiplier: number;
  diff_output_preservation_class: string;
  blank_prompt_preservation?: boolean;
  blank_prompt_preservation_multiplier?: number;
  switch_boundary_every: number;
  loss_type: 'mse' | 'mae' | 'wavelet' | 'stepped';
  do_differential_guidance?: boolean;
  differential_guidance_scale?: number;
  audio_loss_multiplier?: number;
}

export interface QuantizeKwargsConfig {
  exclude: string[];
}

export interface ModelConfig {
  name_or_path: string;
  quantize: boolean;
  quantize_te: boolean;
  qtype: string;
  qtype_te: string;
  quantize_kwargs?: QuantizeKwargsConfig;
  arch: string;
  low_vram: boolean;
  model_kwargs: { [key: string]: any };
  layer_offloading?: boolean;
  layer_offloading_transformer_percent?: number;
  layer_offloading_text_encoder_percent?: number;
  assistant_lora_path?: string;
}

export interface SampleItem {
  prompt: string;
  width?: number;
  height?: number;
  neg?: string;
  seed?: number;
  guidance_scale?: number;
  sample_steps?: number;
  fps?: number;
  num_frames?: number;
  ctrl_img?: string | null;
  ctrl_idx?: number;
  network_multiplier?: number;
  ctrl_img_1?: string | null;
  ctrl_img_2?: string | null;
  ctrl_img_3?: string | null;
}

export interface SampleConfig {
  sampler: string;
  sample_every: number;
  width: number;
  height: number;
  prompts?: string[];
  samples: SampleItem[];
  neg: string;
  seed: number;
  walk_seed: boolean;
  guidance_scale: number;
  sample_steps: number;
  num_frames: number;
  fps: number;
}

export interface LoggingConfig {
  log_every: number;
  use_ui_logger: boolean;
}

export interface SliderConfig {
  guidance_strength?: number;
  anchor_strength?: number;
  positive_prompt?: string;
  negative_prompt?: string;
  target_class?: string;
  anchor_class?: string | null;
}

export interface ProcessConfig {
  type: string;
  sqlite_db_path?: string;
  training_folder: string;
  performance_log_every: number;
  trigger_word: string | null;
  device: string;
  network?: NetworkConfig;
  slider?: SliderConfig;
  save: SaveConfig;
  datasets: DatasetConfig[];
  train: TrainConfig;
  logging: LoggingConfig;
  model: ModelConfig;
  sample: SampleConfig;
}

export interface ConfigObject {
  name: string;
  process: ProcessConfig[];
}

export interface MetaConfig {
  name: string;
  version: string;
}

export interface JobConfig {
  job: string;
  config: ConfigObject;
  meta: MetaConfig;
}

export interface CaptionProcessConfig {
  type: string;
  sqlite_db_path?: string;
  device: string;
  caption: {
    model_name_or_path: string;
    model_name_or_path2?: string;
    dtype: string;
    quantize: boolean;
    qtype: string;
    low_vram: boolean;
    extensions: string[];
    path_to_caption: string;
    recaption: boolean;
    caption_prompt?: string;
    max_res?: number;
    max_new_tokens?: number;
    fixed_caption?: string;
  }
}

export interface CaptionConfigObject {
  name: string;
  process: CaptionProcessConfig[];
}

export interface CaptionJobConfig {
  job: string;
  config: CaptionConfigObject;
}

export interface ConfigDoc {
  title: string | React.ReactNode;
  description: React.ReactNode;
}

export interface SelectOption {
  readonly value: string;
  readonly label: string;
}
export interface GroupedSelectOption {
  readonly label: string;
  readonly options: SelectOption[];
}

export type JobStatus = 'queued' | 'running' | 'stopping' | 'stopped' | 'completed' | 'error';
