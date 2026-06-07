import type { NetworkConfig, PhaseAutoAdvanceConfig, TrainConfig, TrainingPhaseConfig } from '@/types';

type PlateauPreset = 'fast' | 'standard' | 'long';
type TimestepBias = 'content' | 'balanced' | 'style';

export type AutoTrainingProfile = {
  id: string;
  name: string;
  description?: string;
  compatibleArchs?: string[];
  modelArchs?: string[];
  train?: Partial<TrainConfig>;
  network?: Partial<NetworkConfig>;
  phases: TrainingPhaseConfig[];
};

const plateauDefaults: Record<PlateauPreset, PhaseAutoAdvanceConfig> = {
  fast: {
    type: 'loss_plateau',
    metric: 'loss/loss',
    mode: 'min',
    min_steps: 200,
    window: 80,
    patience: 2,
    min_delta_pct: 1.25,
  },
  standard: {
    type: 'loss_plateau',
    metric: 'loss/loss',
    mode: 'min',
    min_steps: 300,
    window: 100,
    patience: 2,
    min_delta_pct: 1.0,
  },
  long: {
    type: 'loss_plateau',
    metric: 'loss/loss',
    mode: 'min',
    min_steps: 500,
    window: 150,
    patience: 3,
    min_delta_pct: 0.75,
  },
};

const flowImageArchs = [
  'flux',
  'flux_kontext',
  'flex1',
  'flex2',
  'chroma',
  'zeta_chroma',
  'lumina2',
  'omnigen2',
  'zimage',
  'zimage:*',
  'ernie_image',
];

const qwenImageArchs = ['qwen_image', 'qwen_image:2512'];
const qwenEditArchs = ['qwen_image_edit', 'qwen_image_edit_plus', 'qwen_image_edit_plus:2511'];
const ideogram4Archs = ['ideogram4', 'ideogram4:fp8'];
const sdArchs = ['sd15', 'sdxl'];
const wanCommonArchs = ['wan21*', 'wan22_5b'];
const wan22A14BArchs = ['wan22_14b:*', 'wan22_14b_i2v'];
const ltxArchs = ['ltx2', 'ltx2.3'];
const aceArchs = ['ace_step_15', 'ace_step_15_xl'];
const anatomyRealismArchs = [
  ...flowImageArchs,
  ...sdArchs,
  ...qwenImageArchs,
  'flux2',
  'flux2_klein_4b',
  'flux2_klein_9b',
  'asymflux2_klein_9b',
];

function autoAdvance(preset: PlateauPreset): PhaseAutoAdvanceConfig {
  return { ...plateauDefaults[preset] };
}

function loraNetwork(rank: number, extra: Partial<NetworkConfig> = {}): Partial<NetworkConfig> {
  return {
    type: 'lora',
    linear: rank,
    linear_alpha: rank,
    conv: undefined,
    conv_alpha: undefined,
    dropout: undefined,
    lokr_factor: -1,
    lokr_full_rank: false,
    lokr_full_matrix: false,
    lokr_use_tucker: false,
    lokr_use_scalar: false,
    lokr_decompose_both: false,
    lokr_rank_dropout_scale: false,
    lokr_weight_decompose: false,
    lokr_wd_on_output: true,
    lokr_bypass_mode: false,
    lokr_rs_lora: false,
    lokr_unbalanced_factorization: false,
    lokr_legacy_factorization: false,
    ...extra,
  };
}

function trainDefaults(lr: number, extra: Partial<TrainConfig> = {}): Partial<TrainConfig> {
  return {
    optimizer: 'adamw8bit',
    lr,
    content_or_style: 'balanced',
    loss_type: 'mse',
    optimizer_params: {
      weight_decay: 0.0001,
    },
    ...extra,
  };
}

function phase(
  name: string,
  lr: number,
  contentOrStyle: TimestepBias,
  preset: PlateauPreset,
  extra: Partial<TrainingPhaseConfig> = {},
): TrainingPhaseConfig {
  return {
    name,
    optimizer: 'adamw8bit',
    lr,
    content_or_style: contentOrStyle,
    loss_type: 'mse',
    optimizer_params: {
      weight_decay: 0.0001,
    },
    auto_advance: autoAdvance(preset),
    ...extra,
  };
}

function threePhase(
  preset: PlateauPreset,
  firstLR: number,
  secondLR: number,
  thirdLR: number,
  extra: Partial<TrainingPhaseConfig> = {},
): TrainingPhaseConfig[] {
  return [
    phase('Teach structure', firstLR, 'content', preset, extra),
    phase('Stabilize concept', secondLR, 'balanced', preset, extra),
    phase('Refine detail', thirdLR, 'style', preset, extra),
  ];
}

function profileArchs(profile: AutoTrainingProfile): string[] | undefined {
  return [...(profile.compatibleArchs ?? []), ...(profile.modelArchs ?? [])];
}

export function isAutoTrainingProfileCompatible(profile: AutoTrainingProfile, currentArch?: string): boolean {
  const archs = profileArchs(profile);
  if (!archs?.length || !currentArch) return true;
  return archs.some(pattern => {
    if (pattern.endsWith('*')) {
      return currentArch.startsWith(pattern.slice(0, -1));
    }
    return pattern === currentArch;
  });
}

export const builtInAutoTrainingProfiles: AutoTrainingProfile[] = [
  {
    id: 'sd-subject',
    name: 'SD Subject',
    description: 'Conservative SD 1.5/SDXL subject learning with a short plateau window.',
    compatibleArchs: sdArchs,
    network: loraNetwork(16),
    train: trainDefaults(0.0001, { content_or_style: 'content' }),
    phases: threePhase('fast', 0.0001, 0.00005, 0.00002),
  },
  {
    id: 'sd-style',
    name: 'SD Style',
    description: 'Lower-pressure SD 1.5/SDXL style and finish learning.',
    compatibleArchs: sdArchs,
    network: loraNetwork(16),
    train: trainDefaults(0.00008, { content_or_style: 'balanced' }),
    phases: [
      phase('Capture style', 0.00008, 'balanced', 'fast'),
      phase('Tune finish', 0.00004, 'style', 'fast'),
      phase('Polish details', 0.000015, 'style', 'fast'),
    ],
  },
  {
    id: 'flow-subject',
    name: 'Flow Subject',
    description: 'General subject/concept LoRA for flow-matching image models.',
    compatibleArchs: flowImageArchs,
    network: loraNetwork(16),
    train: trainDefaults(0.0001, { content_or_style: 'content' }),
    phases: threePhase('standard', 0.0001, 0.00005, 0.00002),
  },
  {
    id: 'flow-style-detail',
    name: 'Flow Style Detail',
    description: 'Higher-rank style and detail pass for flow-matching image models.',
    compatibleArchs: flowImageArchs,
    network: loraNetwork(32),
    train: trainDefaults(0.000075, { content_or_style: 'balanced' }),
    phases: [
      phase('Capture style', 0.000075, 'balanced', 'standard'),
      phase('Strengthen detail', 0.00004, 'style', 'standard'),
      phase('Polish finish', 0.000015, 'style', 'standard'),
    ],
  },
  {
    id: 'glm-image-balanced-lora',
    name: 'GLM-Image Balanced LoRA',
    description: 'Balanced GLM-Image transformer LoRA profile for general subject, style, and detail learning.',
    modelArchs: ['glm_image'],
    network: loraNetwork(32, { transformer_only: true }),
    train: trainDefaults(0.00005, {
      batch_size: 1,
      gradient_accumulation: 1,
      cache_text_embeddings: true,
      timestep_type: 'weighted',
      content_or_style: 'content',
      save_on_phase_change: true,
    }),
    phases: [
      phase('Teach subject', 0.00005, 'content', 'standard', { timestep_type: 'weighted' }),
      phase('Stabilize', 0.00003, 'balanced', 'standard', { timestep_type: 'weighted' }),
      phase('Polish style', 0.000015, 'style', 'standard', { timestep_type: 'weighted' }),
    ],
  },
  {
    id: 'glm-image-low-vram-lora',
    name: 'GLM-Image Low VRAM LoRA',
    description: 'Lower-rank GLM-Image LoRA profile with accumulation for tighter VRAM budgets.',
    modelArchs: ['glm_image'],
    network: loraNetwork(16, { dropout: 0.05, transformer_only: true }),
    train: trainDefaults(0.00003, {
      batch_size: 1,
      gradient_accumulation: 2,
      cache_text_embeddings: true,
      timestep_type: 'weighted',
      content_or_style: 'content',
      save_on_phase_change: true,
    }),
    phases: [
      phase('Teach subject', 0.00003, 'content', 'standard', { timestep_type: 'weighted' }),
      phase('Stabilize', 0.00002, 'balanced', 'standard', { timestep_type: 'weighted' }),
      phase('Polish style', 0.00001, 'style', 'standard', { timestep_type: 'weighted' }),
    ],
  },
  {
    id: 'ideogram4-balanced-lora',
    name: 'Ideogram 4 Balanced LoRA',
    description: 'Transformer-only Ideogram 4 LoRA profile for JSON-caption datasets.',
    modelArchs: ideogram4Archs,
    network: loraNetwork(32, { transformer_only: true }),
    train: trainDefaults(0.00004, {
      batch_size: 1,
      gradient_accumulation: 1,
      cache_text_embeddings: true,
      timestep_type: 'weighted',
      content_or_style: 'content',
      save_on_phase_change: true,
    }),
    phases: [
      phase('Teach JSON layout', 0.00004, 'content', 'long', { timestep_type: 'weighted' }),
      phase('Stabilize composition', 0.000025, 'balanced', 'long', { timestep_type: 'weighted' }),
      phase('Refine style detail', 0.00001, 'style', 'long', { timestep_type: 'weighted' }),
    ],
  },
  {
    id: 'anatomy-lokr',
    name: 'Anatomy LoKr',
    description: 'LoKr profile for anatomy-heavy concepts that need broad structure before fine detail.',
    compatibleArchs: [...flowImageArchs, ...sdArchs],
    network: {
      type: 'lokr',
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
        auto_advance: autoAdvance('standard'),
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
        auto_advance: autoAdvance('standard'),
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
        auto_advance: autoAdvance('standard'),
      },
    ],
  },
  {
    id: 'qwen-subject',
    name: 'Qwen Subject',
    description: 'Conservative Qwen-Image subject profile with longer plateau checks.',
    compatibleArchs: qwenImageArchs,
    network: loraNetwork(16),
    train: trainDefaults(0.00005, { content_or_style: 'content' }),
    phases: threePhase('long', 0.00005, 0.000025, 0.00001),
  },
  {
    id: 'qwen-text-layout',
    name: 'Qwen Text/Layout',
    description: 'Higher-rank Qwen-Image profile for text, layout, and structured visual concepts.',
    compatibleArchs: qwenImageArchs,
    network: loraNetwork(32),
    train: trainDefaults(0.00004, { content_or_style: 'content' }),
    phases: threePhase('long', 0.00004, 0.00002, 0.000008),
  },
  {
    id: 'qwen-edit-consistency',
    name: 'Qwen Edit Consistency',
    description: 'Qwen-Image-Edit profile for edit consistency and controlled subject/style transfer.',
    compatibleArchs: qwenEditArchs,
    network: loraNetwork(16),
    train: trainDefaults(0.00005, { content_or_style: 'content' }),
    phases: threePhase('long', 0.00005, 0.000025, 0.00001),
  },
  {
    id: 'flux2-klein',
    name: 'FLUX.2 Klein',
    description: 'FLUX.2 Klein profile that keeps model-selected layer targeting intact.',
    compatibleArchs: ['flux2_klein_4b', 'flux2_klein_9b'],
    network: loraNetwork(32),
    train: trainDefaults(0.0001, { content_or_style: 'content' }),
    phases: threePhase('standard', 0.0001, 0.00005, 0.00002),
  },
  {
    id: 'flux2-sega-distill',
    name: 'FLUX.2 SEGA Distill',
    description: 'FLUX.2 LoRA profile that keeps the dataset loss and adds an online SEGA teacher for higher-resolution behavior.',
    compatibleArchs: ['flux2', 'flux2_klein_4b', 'flux2_klein_9b'],
    network: loraNetwork(32),
    train: trainDefaults(0.000075, {
      content_or_style: 'content',
      sega_distill: true,
      sega_distill_weight: 1.0,
      sega_distill_base_resolution: 1024,
      sega_distill_strength: 1.0,
      sega_distill_min_scale: 0.5,
      sega_distill_max_scale: 2.0,
      sega_distill_on_reg: false,
    }),
    phases: threePhase('standard', 0.000075, 0.000035, 0.000015),
  },
  {
    id: 'zimage-sega-distill',
    name: 'Z-Image SEGA Distill',
    description: 'Z-Image LoRA profile that keeps the dataset loss and adds an online SEGA teacher for higher-resolution behavior.',
    compatibleArchs: ['zimage', 'zimage:*'],
    network: loraNetwork(32),
    train: trainDefaults(0.000075, {
      content_or_style: 'content',
      sega_distill: true,
      sega_distill_weight: 0.25,
      sega_distill_base_resolution: 1024,
      sega_distill_strength: 1.0,
      sega_distill_min_scale: 0.5,
      sega_distill_max_scale: 2.0,
      sega_distill_on_reg: false,
    }),
    phases: threePhase('standard', 0.000075, 0.000035, 0.000015),
  },
  {
    id: 'asymflux2-klein',
    name: 'AsymFLUX.2',
    description: 'AsymFLUX.2 profile using shift timesteps.',
    compatibleArchs: ['asymflux2_klein_9b'],
    network: loraNetwork(32),
    train: trainDefaults(0.0001, {
      timestep_type: 'shift',
      content_or_style: 'content',
    }),
    phases: threePhase('standard', 0.0001, 0.00005, 0.00002, { timestep_type: 'shift' }),
  },
  {
    id: 'hidream-i1',
    name: 'HiDream I1',
    description: 'HiDream I1 profile matching the higher learning-rate defaults used by the model family.',
    compatibleArchs: ['hidream'],
    network: loraNetwork(32),
    train: trainDefaults(0.0002, {
      timestep_type: 'shift',
      content_or_style: 'content',
    }),
    phases: threePhase('long', 0.0002, 0.0001, 0.00005, { timestep_type: 'shift' }),
  },
  {
    id: 'hidream-e1',
    name: 'HiDream E1',
    description: 'HiDream edit profile with half-strength I1 learning rates.',
    compatibleArchs: ['hidream_e1'],
    network: loraNetwork(32),
    train: trainDefaults(0.0001, {
      timestep_type: 'weighted',
      content_or_style: 'content',
    }),
    phases: threePhase('long', 0.0001, 0.00005, 0.000025, { timestep_type: 'weighted' }),
  },
  {
    id: 'hidream-o1',
    name: 'HiDream-O1',
    description: 'HiDream-O1 profile that keeps x0-target loss and no-conv defaults.',
    compatibleArchs: ['hidream_o1'],
    network: loraNetwork(32, {
      conv: undefined,
      conv_alpha: undefined,
    }),
    train: trainDefaults(0.00003, {
      batch_size: 2,
      gradient_accumulation: 1,
      timestep_type: 'sigmoid',
      content_or_style: 'balanced',
      t0_loss_target: true,
      max_loss: 1.0,
    }),
    phases: [
      phase('Learn O1 target', 0.00003, 'balanced', 'long', { timestep_type: 'sigmoid' }),
      phase('Stabilize O1 target', 0.00002, 'balanced', 'long', { timestep_type: 'sigmoid' }),
      phase('Refine O1 detail', 0.00001, 'balanced', 'long', { timestep_type: 'sigmoid' }),
    ],
  },
  {
    id: 'nucleus-image',
    name: 'Nucleus Image',
    description: 'High-rank Nucleus profile using linear timesteps.',
    compatibleArchs: ['nucleus_image'],
    network: loraNetwork(128),
    train: trainDefaults(0.00008, {
      timestep_type: 'linear',
      content_or_style: 'content',
    }),
    phases: threePhase('long', 0.00008, 0.00004, 0.00002, { timestep_type: 'linear' }),
  },
  {
    id: 'wan-motion',
    name: 'Wan Motion',
    description: 'Wan motion profile that leaves the selected model timestep default in place.',
    compatibleArchs: wanCommonArchs,
    network: loraNetwork(32),
    train: trainDefaults(0.0001, { content_or_style: 'content' }),
    phases: threePhase('long', 0.0001, 0.00005, 0.00002),
  },
  {
    id: 'wan22-a14b-motion',
    name: 'Wan 2.2 A14B Motion',
    description: 'Wan 2.2 A14B motion profile with linear timesteps.',
    compatibleArchs: wan22A14BArchs,
    network: loraNetwork(32),
    train: trainDefaults(0.0001, {
      timestep_type: 'linear',
      content_or_style: 'content',
    }),
    phases: threePhase('long', 0.0001, 0.00005, 0.00002, { timestep_type: 'linear' }),
  },
  {
    id: 'ltx-audio-video',
    name: 'LTX Audio-Video',
    description: 'LTX audio-video profile that keeps audio loss enabled.',
    compatibleArchs: ltxArchs,
    network: loraNetwork(32),
    train: trainDefaults(0.000075, {
      audio_loss_multiplier: 1.0,
      content_or_style: 'content',
    }),
    phases: threePhase('long', 0.000075, 0.00004, 0.00002),
  },
  {
    id: 'ace-music-style',
    name: 'ACE Music Style',
    description: 'ACE-Step music style profile using linear timesteps.',
    compatibleArchs: aceArchs,
    network: loraNetwork(32),
    train: trainDefaults(0.00005, {
      timestep_type: 'linear',
      content_or_style: 'content',
    }),
    phases: threePhase('long', 0.00005, 0.000025, 0.00001, { timestep_type: 'linear' }),
  },
  {
    id: 'anatomy-realism-lora',
    name: 'Anatomy Realism LoRA',
    description: 'Photoreal anatomy profile for body proportions, pose stability, and natural skin detail.',
    compatibleArchs: anatomyRealismArchs,
    network: loraNetwork(32, { dropout: 0.03 }),
    train: trainDefaults(0.000075, { content_or_style: 'content' }),
    phases: [
      phase('Learn body proportions', 0.000075, 'content', 'long'),
      phase('Stabilize pose and skin', 0.000035, 'balanced', 'long'),
      phase('Photoreal detail cleanup', 0.000012, 'style', 'long'),
    ],
  },
  {
    id: 'anatomy-realism-high-rank',
    name: 'Anatomy Realism High Rank',
    description: 'Higher-capacity anatomy realism profile for larger datasets or stubborn body-detail concepts.',
    compatibleArchs: anatomyRealismArchs,
    network: loraNetwork(64, { dropout: 0.05 }),
    train: trainDefaults(0.00005, { content_or_style: 'content' }),
    phases: [
      phase('Map realistic structure', 0.00005, 'content', 'long'),
      phase('Balance form and pose', 0.000025, 'balanced', 'long'),
      phase('Preserve skin texture', 0.00001, 'style', 'long'),
    ],
  },
  {
    id: 'anatomy-realism-lokr',
    name: 'Anatomy Realism LoKr',
    description: 'Lower-LR LoKr realism profile for anatomy datasets that need shape control without harsh style drift.',
    compatibleArchs: [...anatomyRealismArchs, 'hidream'],
    network: {
      type: 'lokr',
      lokr_factor: 8,
      lokr_full_rank: false,
    },
    train: {
      optimizer: 'adamw',
      lr: 0.000025,
      content_or_style: 'content',
      loss_type: 'mse',
      optimizer_params: {
        weight_decay: 0.0001,
      },
    },
    phases: [
      {
        name: 'Teach realistic proportions',
        optimizer: 'adamw',
        lr: 0.000025,
        content_or_style: 'content',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: autoAdvance('long'),
      },
      {
        name: 'Stabilize realistic form',
        optimizer: 'adamw',
        lr: 0.000012,
        content_or_style: 'balanced',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: autoAdvance('long'),
      },
      {
        name: 'Clean skin detail',
        optimizer: 'adamw',
        lr: 0.000005,
        content_or_style: 'style',
        loss_type: 'mse',
        optimizer_params: {
          weight_decay: 0.0001,
        },
        auto_advance: autoAdvance('long'),
      },
    ],
  },
  {
    id: 'anatomy-realism-o1',
    name: 'Anatomy Realism O1',
    description: 'HiDream-O1 anatomy realism profile that keeps O1 in its x0-target loss space.',
    compatibleArchs: ['hidream_o1'],
    network: loraNetwork(32, {
      conv: undefined,
      conv_alpha: undefined,
    }),
    train: trainDefaults(0.000025, {
      batch_size: 2,
      gradient_accumulation: 1,
      timestep_type: 'sigmoid',
      content_or_style: 'balanced',
      t0_loss_target: true,
      max_loss: 1.0,
    }),
    phases: [
      phase('Learn realistic anatomy', 0.000025, 'balanced', 'long', { timestep_type: 'sigmoid' }),
      phase('Stabilize body detail', 0.000015, 'balanced', 'long', { timestep_type: 'sigmoid' }),
      phase('Clean photoreal texture', 0.000008, 'balanced', 'long', { timestep_type: 'sigmoid' }),
    ],
  },
];
