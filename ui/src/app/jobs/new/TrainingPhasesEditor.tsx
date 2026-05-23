'use client';

import { useEffect, useMemo, useState } from 'react';
import { Checkbox, NumberInput, SelectInput, TextInput } from '@/components/formInputs';
import type {
  NetworkConfig,
  PhaseAutoAdvanceConfig,
  TrainConfig,
  TrainingPhaseConfig,
  TrainLossType,
} from '@/types';
import { Copy, Plus, Save, Trash2 } from 'lucide-react';
import {
  builtInAutoTrainingProfiles,
  isAutoTrainingProfileCompatible,
  type AutoTrainingProfile,
} from './autoTrainingProfiles';

type Props = {
  train: TrainConfig;
  network?: NetworkConfig;
  currentArch?: string;
  setJobConfig: (value: any, key: string) => void;
  disableTimestepType?: boolean;
  modelArchName?: string;
  defaultAutoTrainingProfileId?: string;
};

const CUSTOM_PROFILE_STORAGE_KEY = 'aitk.autoTrainingProfiles.v1';

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
    min_steps: 300,
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

function clonePlainValue<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map(item => clonePlainValue(item)) as T;
  }
  if (value && typeof value === 'object') {
    const clone: Record<string, unknown> = {};
    for (const [key, nestedValue] of Object.entries(value as Record<string, unknown>)) {
      clone[key] = clonePlainValue(nestedValue);
    }
    return clone as T;
  }
  return value;
}

function normalizePhase(phase: TrainingPhaseConfig, index: number, autoTrain: boolean): TrainingPhaseConfig {
  const normalized: TrainingPhaseConfig = {
    ...phase,
    name: phase.name?.trim() || `Phase ${index + 1}`,
  };

  if (autoTrain) {
    delete normalized.steps;
    normalized.auto_advance = normalized.auto_advance ? { ...normalized.auto_advance } : defaultAutoAdvance();
  } else {
    normalized.steps = Math.max(1, Number(phase.steps) || 1);
  }

  return normalized;
}

function cloneProfile(profile: AutoTrainingProfile): AutoTrainingProfile {
  return {
    ...profile,
    compatibleArchs: profile.compatibleArchs ? [...profile.compatibleArchs] : undefined,
    modelArchs: profile.modelArchs ? [...profile.modelArchs] : undefined,
    train: profile.train ? clonePlainValue(profile.train) : undefined,
    network: profile.network ? clonePlainValue(profile.network) : undefined,
    phases: profile.phases.map(clonePhase),
  };
}

function supportsNormalNetworkDropout(networkType?: string): boolean {
  return networkType?.toLowerCase() !== 'lokr';
}

function sanitizeStoredProfiles(raw: unknown): AutoTrainingProfile[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((profile): profile is AutoTrainingProfile => {
      return (
        !!profile &&
        typeof profile === 'object' &&
        typeof (profile as AutoTrainingProfile).id === 'string' &&
        typeof (profile as AutoTrainingProfile).name === 'string' &&
        Array.isArray((profile as AutoTrainingProfile).phases)
      );
    })
    .map(cloneProfile);
}

export default function TrainingPhasesEditor({
  train,
  network,
  currentArch,
  setJobConfig,
  disableTimestepType = false,
  modelArchName,
  defaultAutoTrainingProfileId,
}: Props) {
  const autoTrain = !!train.auto_train;
  const phases = train.phases ?? [];
  const phaseTotal = sumPhaseSteps(phases);
  const [customProfiles, setCustomProfiles] = useState<AutoTrainingProfile[]>([]);
  const [selectedProfileID, setSelectedProfileID] = useState(defaultAutoTrainingProfileId ?? builtInAutoTrainingProfiles[0]?.id ?? '');
  const [customProfileName, setCustomProfileName] = useState('');
  const activeArch = modelArchName ?? currentArch;

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(CUSTOM_PROFILE_STORAGE_KEY);
      if (raw) setCustomProfiles(sanitizeStoredProfiles(JSON.parse(raw)));
    } catch {
      setCustomProfiles([]);
    }
  }, []);

  const availableBuiltInProfiles = useMemo(
    () => builtInAutoTrainingProfiles.filter(profile => isAutoTrainingProfileCompatible(profile, activeArch)),
    [activeArch],
  );

  const availableCustomProfiles = useMemo(
    () => customProfiles.filter(profile => isAutoTrainingProfileCompatible(profile, activeArch)),
    [customProfiles, activeArch],
  );

  const allProfiles = useMemo(
    () => [...availableBuiltInProfiles, ...availableCustomProfiles],
    [availableBuiltInProfiles, availableCustomProfiles],
  );

  const preferredProfileID = useMemo(
    () =>
      defaultAutoTrainingProfileId && allProfiles.some(profile => profile.id === defaultAutoTrainingProfileId)
        ? defaultAutoTrainingProfileId
        : allProfiles[0]?.id ?? '',
    [allProfiles, defaultAutoTrainingProfileId],
  );

  useEffect(() => {
    if (preferredProfileID) setSelectedProfileID(preferredProfileID);
  }, [activeArch, defaultAutoTrainingProfileId, preferredProfileID]);

  useEffect(() => {
    if (!allProfiles.length) {
      setSelectedProfileID('');
      return;
    }
    if (!allProfiles.some(profile => profile.id === selectedProfileID)) {
      setSelectedProfileID(preferredProfileID);
    }
  }, [allProfiles, selectedProfileID, preferredProfileID]);

  const profileOptions = useMemo(
    () => [
      {
        label: 'Built in',
        options: availableBuiltInProfiles.map(profile => ({ value: profile.id, label: profile.name })),
      },
      {
        label: 'Custom',
        options: availableCustomProfiles.map(profile => ({ value: profile.id, label: profile.name })),
      },
    ],
    [availableBuiltInProfiles, availableCustomProfiles],
  );

  const persistCustomProfiles = (profiles: AutoTrainingProfile[]) => {
    setCustomProfiles(profiles);
    window.localStorage.setItem(CUSTOM_PROFILE_STORAGE_KEY, JSON.stringify(profiles));
  };

  const buildPhase = (index: number, steps?: number): TrainingPhaseConfig => {
    const phase: TrainingPhaseConfig = {
      name: `Phase ${index + 1}`,
      optimizer: train.optimizer,
      lr: train.lr,
      timestep_type: train.timestep_type,
      content_or_style: train.content_or_style,
      loss_type: train.loss_type,
      optimizer_params: {
        weight_decay: Number(train.optimizer_params?.weight_decay ?? 0),
      },
    };

    if (autoTrain) {
      phase.auto_advance = defaultAutoAdvance();
    } else {
      phase.steps = steps ?? Math.max(1, Math.round((train.steps || 1000) / Math.max(1, phases.length + 1)));
    }

    return phase;
  };

  const setPhases = (nextPhases: TrainingPhaseConfig[], forceAutoTrain = autoTrain) => {
    if (nextPhases.length === 0) {
      setJobConfig(undefined, 'config.process[0].train.phases');
      setJobConfig(false, 'config.process[0].train.auto_train');
      return;
    }
    const normalized = nextPhases.map((phase, index) => normalizePhase(phase, index, forceAutoTrain));
    setJobConfig(normalized, 'config.process[0].train.phases');
    setJobConfig(forceAutoTrain, 'config.process[0].train.auto_train');
    if (!forceAutoTrain) {
      setJobConfig(sumPhaseSteps(normalized), 'config.process[0].train.steps');
    }
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

  const applyAutoProfile = (profile: AutoTrainingProfile) => {
    const nextProfile = cloneProfile(profile);
    const profileNetworkType = nextProfile.network?.type ?? network?.type;
    setSelectedProfileID(nextProfile.id);
    setJobConfig(true, 'config.process[0].train.auto_train');

    for (const [key, value] of Object.entries(nextProfile.train ?? {})) {
      if (key === 'steps' || key === 'phases' || key === 'auto_train') continue;
      if (key === 'timestep_type' && disableTimestepType) continue;
      setJobConfig(value, `config.process[0].train.${key}`);
    }

    if (!supportsNormalNetworkDropout(profileNetworkType)) {
      setJobConfig(undefined, 'config.process[0].network.dropout');
    }

    for (const [key, value] of Object.entries(nextProfile.network ?? {})) {
      if (key === 'dropout' && !supportsNormalNetworkDropout(profileNetworkType)) continue;
      setJobConfig(value, `config.process[0].network.${key}`);
    }

    const nextPhases = nextProfile.phases.map((phase, index) => {
      const normalized = normalizePhase(phase, index, true);
      if (disableTimestepType) delete normalized.timestep_type;
      return normalized;
    });
    setPhases(nextPhases, true);
  };

  useEffect(() => {
    if (allProfiles.some(profile => profile.id === selectedProfileID)) return;

    const fallbackProfile = allProfiles[0];
    if (!fallbackProfile) {
      setSelectedProfileID('');
      return;
    }

    if (autoTrain) {
      applyAutoProfile(fallbackProfile);
      return;
    }

    setSelectedProfileID(fallbackProfile.id);
  }, [allProfiles, autoTrain, selectedProfileID]);

  const toggleAutoTrain = (enabled: boolean) => {
    if (!enabled) {
      const fallbackPhaseSteps = Math.max(1, Math.round((train.steps || 1000) / Math.max(1, phases.length)));
      const normalized = phases.length
        ? phases.map((phase, index) =>
            normalizePhase({ ...phase, steps: phase.steps ?? fallbackPhaseSteps }, index, false),
          )
        : [];
      setJobConfig(false, 'config.process[0].train.auto_train');
      if (normalized.length) {
        setJobConfig(normalized, 'config.process[0].train.phases');
        setJobConfig(sumPhaseSteps(normalized), 'config.process[0].train.steps');
      }
      return;
    }

    const profile = allProfiles.find(candidate => candidate.id === selectedProfileID) ?? allProfiles[0];
    if (profile) {
      applyAutoProfile(profile);
    } else {
      setPhases(phases.length ? phases.map(clonePhase) : [buildPhase(0)], true);
    }
  };

  const saveCustomProfile = () => {
    const name = customProfileName.trim();
    if (!name || phases.length === 0) return;

    const profile: AutoTrainingProfile = {
      id: `custom:${Date.now()}`,
      name,
      modelArchs: activeArch ? [activeArch] : undefined,
      train: {
        optimizer: train.optimizer,
        lr: train.lr,
        timestep_type: train.timestep_type,
        content_or_style: train.content_or_style,
        loss_type: train.loss_type,
        optimizer_params: train.optimizer_params ? clonePlainValue(train.optimizer_params) : undefined,
        lr_scheduler: train.lr_scheduler,
        lr_scheduler_params: train.lr_scheduler_params ? clonePlainValue(train.lr_scheduler_params) : undefined,
        audio_loss_multiplier: train.audio_loss_multiplier,
        batch_size: train.batch_size,
        gradient_accumulation: train.gradient_accumulation,
        t0_loss_target: train.t0_loss_target,
        max_loss: train.max_loss,
      },
      network: network
        ? {
            type: network.type,
            linear: network.linear,
            linear_alpha: network.linear_alpha,
            conv: network.conv,
            conv_alpha: network.conv_alpha,
            dropout: supportsNormalNetworkDropout(network.type) ? network.dropout : undefined,
            lokr_factor: network.lokr_factor,
            lokr_full_rank: network.lokr_full_rank,
            network_kwargs: network.network_kwargs ? clonePlainValue(network.network_kwargs) : undefined,
            transformer_only: network.transformer_only,
          }
        : undefined,
      phases: phases.map((phase, index) => normalizePhase(clonePhase(phase), index, true)),
    };

    const nextProfiles = [...customProfiles.filter(candidate => candidate.name !== name), profile];
    persistCustomProfiles(nextProfiles);
    setSelectedProfileID(profile.id);
    setCustomProfileName('');
  };

  const deleteSelectedCustomProfile = () => {
    if (!selectedProfileID.startsWith('custom:')) return;
    const nextProfiles = customProfiles.filter(profile => profile.id !== selectedProfileID);
    persistCustomProfiles(nextProfiles);
    setSelectedProfileID(availableBuiltInProfiles[0]?.id ?? '');
  };

  const autoControls = (
    <div className="space-y-3">
      <Checkbox label="Auto learn" checked={autoTrain} onChange={toggleAutoTrain} />
      {autoTrain && (
        <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto_auto] gap-3 items-end">
          <SelectInput
            label="Profile"
            value={selectedProfileID}
            onChange={value => {
              const profile = allProfiles.find(candidate => candidate.id === value);
              if (profile) applyAutoProfile(profile);
            }}
            options={profileOptions}
          />
          <TextInput
            label="Custom Profile"
            value={customProfileName}
            onChange={setCustomProfileName}
            placeholder="Profile name"
          />
          <button
            type="button"
            title="Save current profile"
            onClick={saveCustomProfile}
            disabled={!customProfileName.trim() || phases.length === 0}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-sm border border-gray-700 bg-gray-900 text-sm text-gray-200 hover:bg-gray-800 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Save className="h-4 w-4" />
            Save
          </button>
          <button
            type="button"
            title="Delete selected custom profile"
            onClick={deleteSelectedCustomProfile}
            disabled={!selectedProfileID.startsWith('custom:')}
            className="p-2 rounded-sm border border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-800 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );

  if (phases.length === 0) {
    return (
      <div className="mt-6 border-t border-gray-800 pt-4 space-y-4">
        {autoControls}
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
      {autoControls}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-gray-100">Training Phases</h3>
          <div className="text-xs text-gray-400">
            {autoTrain ? 'Open-ended plateau stages' : `${phaseTotal.toLocaleString()} synchronized steps`}
          </div>
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
          const autoAdvance = phase.auto_advance ?? (autoTrain ? defaultAutoAdvance() : undefined);
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
                  {!autoTrain && (
                    <NumberInput
                      label="Steps"
                      value={phase.steps ?? 1}
                      onChange={value => updatePhase(index, { steps: value ?? 1 })}
                      placeholder="eg. 1000"
                      min={1}
                    />
                  )}
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
                {autoTrain ? null : (
                  <Checkbox
                    label="Auto advance on plateau"
                    checked={!!autoAdvance}
                    onChange={value => updatePhase(index, { auto_advance: value ? defaultAutoAdvance() : undefined })}
                  />
                )}
                {autoAdvance && (
                  <div className={autoTrain ? 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3' : 'mt-2 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-6 gap-3'}>
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
                    {!autoTrain && (
                      <NumberInput
                        label="Min Steps"
                        value={autoAdvance.min_steps ?? null}
                        onChange={value => updateAutoAdvance(index, { min_steps: value ?? undefined })}
                        placeholder="auto"
                        min={1}
                      />
                    )}
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
