import type { NetworkConfig, TrainConfig, TrainingPhaseConfig } from '@/types';

export type AutoTrainingProfile = {
  id: string;
  name: string;
  modelArchs?: string[];
  train?: Partial<TrainConfig>;
  network?: Partial<NetworkConfig>;
  phases: TrainingPhaseConfig[];
};

const defaultAutoAdvance = {
  type: 'loss_plateau' as const,
  metric: 'loss/loss',
  mode: 'min' as const,
  min_steps: 300,
  window: 100,
  patience: 2,
  min_delta_pct: 1.0,
};

export const builtInAutoTrainingProfiles: AutoTrainingProfile[] = [
  {
    id: 'glm-image-balanced-lora',
    name: 'GLM-Image Balanced LoRA',
    modelArchs: ['glm_image'],
    network: {
      type: 'lora',
      linear: 32,
      linear_alpha: 32,
      transformer_only: true,
    },
    train: {
      batch_size: 1,
      gradient_accumulation: 1,
      cache_text_embeddings: true,
      optimizer: 'adamw8bit',
      lr: 0.00005,
      timestep_type: 'weighted',
      content_or_style: 'content',
      loss_type: 'mse',
      save_on_phase_change: true,
      optimizer_params: {
        weight_decay: 0.0001,
      },
    },
    phases: [
      {
        name: 'Teach subject',
        optimizer: 'adamw8bit',
        lr: 0.00005,
        timestep_type: 'weighted',
        content_or_style: 'content',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: { ...defaultAutoAdvance },
      },
      {
        name: 'Stabilize',
        optimizer: 'adamw8bit',
        lr: 0.00003,
        timestep_type: 'weighted',
        content_or_style: 'balanced',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: { ...defaultAutoAdvance },
      },
      {
        name: 'Polish style',
        optimizer: 'adamw8bit',
        lr: 0.000015,
        timestep_type: 'weighted',
        content_or_style: 'style',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: { ...defaultAutoAdvance },
      },
    ],
  },
  {
    id: 'glm-image-low-vram-lora',
    name: 'GLM-Image Low VRAM LoRA',
    modelArchs: ['glm_image'],
    network: {
      type: 'lora',
      linear: 16,
      linear_alpha: 16,
      dropout: 0.05,
      transformer_only: true,
    },
    train: {
      batch_size: 1,
      gradient_accumulation: 2,
      cache_text_embeddings: true,
      optimizer: 'adamw8bit',
      lr: 0.00003,
      timestep_type: 'weighted',
      content_or_style: 'content',
      loss_type: 'mse',
      save_on_phase_change: true,
      optimizer_params: {
        weight_decay: 0.0001,
      },
    },
    phases: [
      {
        name: 'Teach subject',
        optimizer: 'adamw8bit',
        lr: 0.00003,
        timestep_type: 'weighted',
        content_or_style: 'content',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: { ...defaultAutoAdvance },
      },
      {
        name: 'Stabilize',
        optimizer: 'adamw8bit',
        lr: 0.00002,
        timestep_type: 'weighted',
        content_or_style: 'balanced',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: { ...defaultAutoAdvance },
      },
      {
        name: 'Polish style',
        optimizer: 'adamw8bit',
        lr: 0.00001,
        timestep_type: 'weighted',
        content_or_style: 'style',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: { ...defaultAutoAdvance },
      },
    ],
  },
  {
    id: 'anatomy-lokr',
    name: 'Anatomy LoKr',
    network: {
      type: 'lokr',
      dropout: 0.05,
      lokr_factor: 8,
      lokr_full_rank: false,
    },
    train: {
      optimizer: 'adamw',
      lr: 0.00002,
      timestep_type: 'weighted',
      content_or_style: 'content',
      loss_type: 'mse',
      optimizer_params: {
        weight_decay: 0.0001,
      },
    },
    phases: [
      {
        name: 'Teach anatomy',
        optimizer: 'adamw',
        lr: 0.00002,
        timestep_type: 'weighted',
        content_or_style: 'content',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: { ...defaultAutoAdvance },
      },
      {
        name: 'Stabilize',
        optimizer: 'adamw',
        lr: 0.00001,
        timestep_type: 'weighted',
        content_or_style: 'balanced',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: { ...defaultAutoAdvance },
      },
      {
        name: 'Fine detail cleanup',
        optimizer: 'adamw',
        lr: 0.000005,
        timestep_type: 'weighted',
        content_or_style: 'style',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: { ...defaultAutoAdvance },
      },
    ],
  },
];

