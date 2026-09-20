'use client';
import React, { useEffect, useMemo } from 'react';
import * as jsx from 'react/jsx-runtime';
import * as jsxDev from 'react/jsx-dev-runtime';
import Link from 'next/link';
import { createGlobalState } from 'react-global-hooks';
import * as defaultSamples from '@/helpers/defaultSamples';
import * as formInputs from '@/components/formInputs';
import * as options from '@/domain/modelOptions';
import type { ModelArch } from '@/domain/modelOptions';
import { apiClient } from '@/utils/api';

const shims: Record<string, unknown> = {
  react: React,
  'react/jsx-runtime': jsx,
  'react/jsx-dev-runtime': jsxDev,
  'next/link': { __esModule: true, default: Link },
  '@/helpers/defaultSamples': defaultSamples,
  '@/components/formInputs': formInputs,
  '@/app/jobs/new/options': options,
  '@/types': {},
};
const record = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const strings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every(item => typeof item === 'string');
const groups = ['image', 'instruction', 'video', 'experimental', 'audio', 'llm'] as const;
function parseArch(value: unknown): ModelArch {
  if (
    !record(value) ||
    typeof value.name !== 'string' ||
    typeof value.label !== 'string' ||
    !groups.some(group => group === value.group)
  )
    throw new Error('Invalid model name, label, or group');
  for (const key of ['disableSections', 'additionalSections', 'allowedNetworkTypes', 'controls'])
    if (value[key] !== undefined && !strings(value[key])) throw new Error(`Invalid ${key}`);
  if (
    value.defaults !== undefined &&
    (!record(value.defaults) || !Object.values(value.defaults).every(v => Array.isArray(v) && v.length === 2))
  )
    throw new Error('Model defaults require selected/unselected pairs');
  for (const key of ['isVideoModel', 'hasMultiLinePrompts'])
    if (value[key] !== undefined && typeof value[key] !== 'boolean') throw new Error(`Invalid ${key}`);
  if (
    value.sampleTags !== undefined &&
    (!record(value.sampleTags) ||
      !Object.values(value.sampleTags).every(
        v => record(v) && typeof v.title === 'string' && ['text', 'multiline', 'number'].includes(String(v.type)),
      ))
  )
    throw new Error('Invalid sample tags');
  // Only installed local extension code is evaluated. This is the same trust boundary as its Python package.
  const parsed: ModelArch = { name: value.name, label: value.label, group: value.group as ModelArch['group'] };
  if (record(value.defaults)) parsed.defaults = value.defaults;
  if (strings(value.disableSections)) parsed.disableSections = value.disableSections as ModelArch['disableSections'];
  if (strings(value.additionalSections))
    parsed.additionalSections = value.additionalSections as ModelArch['additionalSections'];
  if (strings(value.allowedNetworkTypes)) parsed.allowedNetworkTypes = value.allowedNetworkTypes;
  if (strings(value.controls)) parsed.controls = value.controls as ModelArch['controls'];
  if (typeof value.gateUrl === 'string' && /^https:\/\//.test(value.gateUrl)) parsed.gateUrl = value.gateUrl;
  if (typeof value.isVideoModel === 'boolean') parsed.isVideoModel = value.isVideoModel;
  if (typeof value.hasMultiLinePrompts === 'boolean') parsed.hasMultiLinePrompts = value.hasMultiLinePrompts;
  if (record(value.sampleTags)) parsed.sampleTags = value.sampleTags as ModelArch['sampleTags'];
  if (React.isValidElement(value.modelNotes) || typeof value.modelNotes === 'string')
    parsed.extensionNotes = value.modelNotes;
  if (typeof value.customSections === 'function')
    parsed.customSections = value.customSections as NonNullable<ModelArch['customSections']>;
  return parsed;
}
const state = createGlobalState({ archs: options.modelArchs, errors: [] as string[], isLoaded: false });
let pending: Promise<void> | null = null;
export function loadModelArchs(): Promise<void> {
  if (pending) return pending;
  pending = (async () => {
    const errors: string[] = [],
      byName = new Map(options.modelArchs.map(arch => [arch.name, arch]));
    try {
      const { data } = await apiClient.get<unknown>('/api/model_archs');
      if (!record(data) || !Array.isArray(data.modules)) throw new Error('Invalid model extension response');
      if (strings(data.errors)) errors.push(...data.errors);
      for (const mod of data.modules) {
        if (!record(mod) || typeof mod.id !== 'string' || typeof mod.code !== 'string') {
          errors.push('Invalid model UI module');
          continue;
        }
        try {
          const evaluatedModule: { exports: Record<string, unknown> } = { exports: {} };
          new Function('require', 'module', 'exports', mod.code)(
            (name: string) => {
              if (Object.hasOwn(shims, name)) return shims[name];
              throw new Error(`Unsupported extension import: ${name}`);
            },
            evaluatedModule,
            evaluatedModule.exports,
          );
          const models = evaluatedModule.exports.AI_TOOLKIT_UI_MODELS;
          if (!Array.isArray(models)) throw new Error('Expected AI_TOOLKIT_UI_MODELS');
          for (const value of models) {
            const arch = parseArch(value);
            byName.set(arch.name, { ...byName.get(arch.name), ...arch });
          }
        } catch (error) {
          errors.push(`${mod.id}: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
    state.set({ archs: [...byName.values()].sort((a, b) => a.label.localeCompare(b.label)), errors, isLoaded: true });
  })().finally(() => {
    pending = null;
  });
  return pending;
}
export function useModelArchs() {
  const [current] = state.use();
  useEffect(() => {
    void loadModelArchs();
  }, []);
  const groupedModelOptions = useMemo(() => options.groupModelOptions(current.archs), [current.archs]);
  return { ...current, groupedModelOptions, refresh: loadModelArchs };
}
