'use client';

import { Checkbox, NumberInput, SelectInput, TextInput } from '@/components/formInputs';
import type { PhaseAutoAdvanceConfig, TrainConfig, TrainingPhaseConfig, TrainLossType } from '@/types';
import { Copy, Plus, Trash2 } from 'lucide-react';

type Props = {
  train: TrainConfig;
  setJobConfig: (value: any, key: string) => void;
  disableTimestepType?: boolean;
};

const optimizerOptions = [
  { value: 'adafactor', label: 'Adafactor' },
  { value: 'adam', label: 'Adam' },
  { value: 'adamw', label: 'AdamW' },
  { value: 'adamw8bit', label: 'AdamW8Bit' },
  { value: 'automagic', label: 'Automagic' },
  { value: 'automagic2', label: 'Automagic v2' },
  { value: 'prodigyopt', label: 'Prodigy' },
  { value: 'prodigy8bit', label: 'Prodigy8Bit' },
];

const timestepTypeOptions = [
  { value: 'sigmoid', label: 'Sigmoid' },
  { value: 'linear', label: 'Linear' },
  { value: 'shift', label: 'Shift' },
  { value: 'weighted', label: 'Weighted' },
];

const timestepBiasOptions = [
  { value: 'balanced', label: 'Balanced' },
  { value: 'content', label: 'High Noise' },
  { value: 'style', label: 'Low Noise' },
];

const lossTypeOptions = [
  { value: 'mse', label: 'Mean Squared Error' },
  { value: 'mae', label: 'Mean Absolute Error' },
  { value: 'wavelet', label: 'Wavelet' },
  { value: 'stepped', label: 'Stepped Recovery' },
  { value: 'pseudo_huber', label: 'Pseudo Huber' },
  { value: 'mean_flow', label: 'Mean Flow' },
];

const plateauModeOptions = [
  { value: 'min', label: 'Min' },
  { value: 'max', label: 'Max' },
];

function sumPhaseSteps(phases: TrainingPhaseConfig[]) {
  return phases.reduce((sum, phase) => sum + Math.max(1, Number(phase.steps) || 1), 0);
}

function defaultAutoAdvance(): PhaseAutoAdvanceConfig {
  return {
    type: 'loss_plateau',
    metric: 'loss/loss',
    mode: 'min',
    window: 100,
    patience: 2,
    min_delta_pct: 1.0,
  };
}

function clonePhase(phase: TrainingPhaseConfig): TrainingPhaseConfig {
  return {
    ...phase,
    optimizer_params: phase.optimizer_params ? { ...phase.optimizer_params } : undefined,
    lr_scheduler_params: phase.lr_scheduler_params ? { ...phase.lr_scheduler_params } : undefined,
    auto_advance: phase.auto_advance ? { ...phase.auto_advance } : undefined,
  };
}

function normalizePhase(phase: TrainingPhaseConfig, index: number): TrainingPhaseConfig {
  return {
    ...phase,
    name: phase.name?.trim() || `Phase ${index + 1}`,
    steps: Math.max(1, Number(phase.steps) || 1),
  };
}

export default function TrainingPhasesEditor({ train, setJobConfig, disableTimestepType = false }: Props) {
  const phases = train.phases ?? [];
  const phaseTotal = sumPhaseSteps(phases);

  const buildPhase = (index: number, steps?: number): TrainingPhaseConfig => ({
    name: `Phase ${index + 1}`,
    steps: steps ?? Math.max(1, Math.round((train.steps || 1000) / Math.max(1, phases.length + 1))),
    optimizer: train.optimizer,
    lr: train.lr,
    timestep_type: train.timestep_type,
    content_or_style: train.content_or_style,
    loss_type: train.loss_type,
    optimizer_params: {
      weight_decay: Number(train.optimizer_params?.weight_decay ?? 0),
    },
  });

  const setPhases = (nextPhases: TrainingPhaseConfig[]) => {
    if (nextPhases.length === 0) {
      setJobConfig(undefined, 'config.process[0].train.phases');
      return;
    }
    const normalized = nextPhases.map(normalizePhase);
    setJobConfig(normalized, 'config.process[0].train.phases');
    setJobConfig(sumPhaseSteps(normalized), 'config.process[0].train.steps');
    if (train.save_on_phase_change === undefined) {
      setJobConfig(true, 'config.process[0].train.save_on_phase_change');
    }
  };

  const updatePhase = (index: number, patch: Partial<TrainingPhaseConfig>) => {
    setPhases(phases.map((phase, phaseIndex) => (phaseIndex === index ? { ...clonePhase(phase), ...patch } : phase)));
  };

  const updateAutoAdvance = (index: number, patch: Partial<PhaseAutoAdvanceConfig>) => {
    const phase = phases[index];
    const autoAdvance = { ...(phase.auto_advance ?? defaultAutoAdvance()), ...patch, type: 'loss_plateau' as const };
    updatePhase(index, { auto_advance: autoAdvance });
  };

  if (phases.length === 0) {
    return (
      <div className="mt-6 border-t border-gray-800 pt-4">
        <button
          type="button"
          onClick={() => setPhases([buildPhase(0, train.steps || 1000)])}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-sm border border-gray-700 bg-gray-900 text-sm text-gray-200 hover:bg-gray-800"
        >
          <Plus className="h-4 w-4" />
          Add Training Phase
        </button>
      </div>
    );
  }

  return (
    <div className="mt-6 border-t border-gray-800 pt-4 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-gray-100">Training Phases</h3>
          <div className="text-xs text-gray-400">{phaseTotal.toLocaleString()} synchronized steps</div>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setPhases([...phases.map(clonePhase), buildPhase(phases.length)])}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-sm border border-gray-700 bg-gray-900 text-sm text-gray-200 hover:bg-gray-800"
          >
            <Plus className="h-4 w-4" />
            Add
          </button>
          <button
            type="button"
            onClick={() => setPhases([])}
            className="px-3 py-2 rounded-sm border border-gray-700 bg-gray-900 text-sm text-gray-300 hover:bg-gray-800"
          >
            Disable
          </button>
        </div>
      </div>

      <Checkbox
        label="Save on phase change"
        checked={train.save_on_phase_change ?? true}
        onChange={value => setJobConfig(value, 'config.process[0].train.save_on_phase_change')}
      />

      <div className="space-y-3">
        {phases.map((phase, index) => {
          const autoAdvance = phase.auto_advance;
          return (
            <div key={index} className="rounded-sm border border-gray-800 bg-gray-950 p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 flex-1">
                  <TextInput
                    label="Name"
                    value={phase.name ?? ''}
                    onChange={value => updatePhase(index, { name: value })}
                    placeholder="Phase name"
                  />
                  <NumberInput
                    label="Steps"
                    value={phase.steps}
                    onChange={value => updatePhase(index, { steps: value ?? 1 })}
                    placeholder="eg. 1000"
                    min={1}
                  />
                  <SelectInput
                    label="Optimizer"
                    value={phase.optimizer ?? train.optimizer}
                    onChange={value => updatePhase(index, { optimizer: value })}
                    options={optimizerOptions}
                  />
                  <NumberInput
                    label="Learning Rate"
                    value={phase.lr ?? train.lr}
                    onChange={value => updatePhase(index, { lr: value ?? train.lr })}
                    placeholder="eg. 0.00003"
                    min={0}
                  />
                  <NumberInput
                    label="Weight Decay"
                    value={Number(phase.optimizer_params?.weight_decay ?? train.optimizer_params?.weight_decay ?? 0)}
                    onChange={value =>
                      updatePhase(index, {
                        optimizer_params: {
                          ...(phase.optimizer_params ?? {}),
                          weight_decay: value ?? 0,
                        },
                      })
                    }
                    placeholder="eg. 0.0001"
                    min={0}
                  />
                  {!disableTimestepType && (
                    <SelectInput
                      label="Timestep Type"
                      value={phase.timestep_type ?? train.timestep_type}
                      onChange={value => updatePhase(index, { timestep_type: value })}
                      options={timestepTypeOptions}
                    />
                  )}
                  <SelectInput
                    label="Timestep Bias"
                    value={phase.content_or_style ?? train.content_or_style}
                    onChange={value => updatePhase(index, { content_or_style: value })}
                    options={timestepBiasOptions}
                  />
                  <SelectInput
                    label="Loss Type"
                    value={phase.loss_type ?? train.loss_type}
                    onChange={value => updatePhase(index, { loss_type: value as TrainLossType })}
                    options={lossTypeOptions}
                  />
                </div>
                <div className="flex gap-2 pt-8">
                  <button
                    type="button"
                    title="Duplicate phase"
                    onClick={() => {
                      const clone = clonePhase(phase);
                      clone.name = `${clone.name} Copy`;
                      const next = phases.map(clonePhase);
                      next.splice(index + 1, 0, clone);
                      setPhases(next);
                    }}
                    className="p-2 rounded-sm border border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-800"
                  >
                    <Copy className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    title="Delete phase"
                    onClick={() => setPhases(phases.filter((_, phaseIndex) => phaseIndex !== index).map(clonePhase))}
                    className="p-2 rounded-sm border border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-800"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="mt-3 border-t border-gray-800 pt-3">
                <Checkbox
                  label="Auto advance on plateau"
                  checked={!!autoAdvance}
                  onChange={value => updatePhase(index, { auto_advance: value ? defaultAutoAdvance() : undefined })}
                />
                {autoAdvance && (
                  <div className="mt-2 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-6 gap-3">
                    <TextInput
                      label="Metric"
                      value={autoAdvance.metric ?? 'loss/loss'}
                      onChange={value => updateAutoAdvance(index, { metric: value })}
                      placeholder="loss/loss"
                    />
                    <SelectInput
                      label="Mode"
                      value={autoAdvance.mode ?? 'min'}
                      onChange={value => updateAutoAdvance(index, { mode: value as 'min' | 'max' })}
                      options={plateauModeOptions}
                    />
                    <NumberInput
                      label="Min Steps"
                      value={autoAdvance.min_steps ?? null}
                      onChange={value => updateAutoAdvance(index, { min_steps: value ?? undefined })}
                      placeholder="auto"
                      min={1}
                    />
                    <NumberInput
                      label="Window"
                      value={autoAdvance.window ?? 100}
                      onChange={value => updateAutoAdvance(index, { window: value ?? 100 })}
                      min={1}
                    />
                    <NumberInput
                      label="Patience"
                      value={autoAdvance.patience ?? 2}
                      onChange={value => updateAutoAdvance(index, { patience: value ?? 2 })}
                      min={1}
                    />
                    <NumberInput
                      label="Min Delta %"
                      value={autoAdvance.min_delta_pct ?? 1.0}
                      onChange={value => updateAutoAdvance(index, { min_delta_pct: value ?? 1.0 })}
                      min={0}
                    />
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
