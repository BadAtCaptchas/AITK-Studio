import type { NetworkConfig, TrainConfig, TrainingPhaseConfig } from '@/types';

export type AutoTrainingProfile = {
  id: string;
  name: string;
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

