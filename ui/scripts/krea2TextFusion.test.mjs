import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import YAML from 'yaml';

const helper = {};
const code = ts.transpileModule(
  fs.readFileSync(new URL('../src/utils/krea2TextFusion.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } },
).outputText;
new Function('exports', code)(helper);
const {
  hasKrea2TextFusionExclusion,
  setKrea2TextFusionExclusion,
  supportsKrea2TextFusionExclusion,
} = helper;

const compact = ['txtfusion.', 'txtmlp'];
const expanded = [
  'txtfusion.layerwise_blocks.0',
  'txtfusion.layerwise_blocks.1',
  'txtfusion.projector',
  'txtfusion.refiner_blocks.0',
  'txtfusion.refiner_blocks.1',
  'txtmlp',
];

test('text-fusion control is limited to standard Raw/Turbo LoRA jobs without edit conditioning', () => {
  for (const arch of ['krea2', 'krea2:turbo']) {
    assert.equal(supportsKrea2TextFusionExclusion({ arch }, { type: 'lora' }), true);
    assert.equal(supportsKrea2TextFusionExclusion({ arch, model_kwargs: { edit: false } }, { type: 'lora' }), true);
    assert.equal(supportsKrea2TextFusionExclusion({ arch, model_kwargs: { edit: true } }, { type: 'lora' }), false);
    for (const type of ['lokr', 'dora', 'locon', 'lycoris', 'lorm', 'full']) {
      assert.equal(supportsKrea2TextFusionExclusion({ arch }, { type }), false);
    }
    assert.equal(supportsKrea2TextFusionExclusion({ arch }), false);
  }
  for (const arch of ['krea2:o_edit', 'krea2:o_edit_turbo', 'krea2:custom', 'flux', '']) {
    assert.equal(supportsKrea2TextFusionExclusion({ arch }, { type: 'lora' }), false);
  }
});

test('preset is off by default and partial or arbitrary exclusions do not imply it is enabled', () => {
  for (const exclusions of [undefined, null, [], ['blocks.0'], ['txtfusion.'], ['txtmlp'], expanded.slice(1), ['txtfusion']]) {
    assert.equal(hasKrea2TextFusionExclusion(exclusions), false);
  }
});

test('saved compact and expanded presets are recognized without rewriting their entries', () => {
  for (const preset of [compact, expanded]) {
    const exclusions = Object.freeze(['blocks.0.attn', ...preset].reverse());
    const original = [...exclusions];
    assert.equal(hasKrea2TextFusionExclusion(exclusions), true);
    assert.deepEqual(exclusions, original);
  }
});

test('enabling fills missing configuration and repeated toggles do not accumulate entries', () => {
  for (const original of [undefined, null, [], ['txtfusion.']]) {
    const enabled = setKrea2TextFusionExclusion(original, true);
    assert.deepEqual(enabled, compact);
    assert.equal(hasKrea2TextFusionExclusion(enabled), true);
    assert.deepEqual(setKrea2TextFusionExclusion(enabled, true), compact);
    const disabled = setKrea2TextFusionExclusion(enabled, false);
    assert.deepEqual(disabled, []);
    assert.equal(hasKrea2TextFusionExclusion(disabled), false);
    assert.deepEqual(setKrea2TextFusionExclusion(disabled, false), []);
    assert.deepEqual(setKrea2TextFusionExclusion(disabled, true), compact);
  }
});

test('explicit toggles normalize both presets and preserve exact custom entries and ordering', () => {
  const custom = ['blocks.0.attn', 'txtfusion', 'txtfusion.refiner_blocks.0.mlp', 'txtmlp.1', 'blocks.0.attn'];
  for (const preset of [compact, expanded, [...expanded, ...compact, ...expanded]]) {
    const original = Object.freeze([...custom, ...preset]);
    assert.deepEqual(setKrea2TextFusionExclusion(original, true), [...custom, ...compact]);
    assert.deepEqual(setKrea2TextFusionExclusion(original, false), custom);
    assert.deepEqual(original, [...custom, ...preset]);
  }
});

test('only the exclusion field changes and its state survives job JSON and YAML round trips', () => {
  for (const preset of [compact, expanded]) {
    const job = {
      job: 'extension',
      config: {
        name: 'krea2-text-fusion-fixture',
        process: [{
          type: 'diffusion_trainer',
          model: {
            arch: 'krea2:turbo',
            name_or_path: 'krea/Krea-2-Turbo',
            assistant_lora_path: 'fixture/unchanged-adapter.safetensors',
          },
          network: {
            type: 'lora',
            linear: 32,
            linear_alpha: 32,
            transformer_only: true,
            network_kwargs: {
              ignore_if_contains: ['blocks.0', ...preset],
              only_if_contains: ['blocks.'],
              full_if_contains: ['blocks.1.mlp'],
              custom_option: 'preserve',
            },
          },
        }],
      },
    };
    const original = structuredClone(job);
    const reloaded = JSON.parse(JSON.stringify(job));
    const kwargs = reloaded.config.process[0].network.network_kwargs;
    assert.equal(hasKrea2TextFusionExclusion(kwargs.ignore_if_contains), true);
    kwargs.ignore_if_contains = setKrea2TextFusionExclusion(kwargs.ignore_if_contains, true);
    const imported = YAML.parse(YAML.stringify(reloaded));
    assert.deepEqual(imported, reloaded);
    assert.equal(hasKrea2TextFusionExclusion(imported.config.process[0].network.network_kwargs.ignore_if_contains), true);
    imported.config.process[0].network.network_kwargs.ignore_if_contains = setKrea2TextFusionExclusion(
      imported.config.process[0].network.network_kwargs.ignore_if_contains,
      false,
    );
    const expected = structuredClone(original);
    expected.config.process[0].network.network_kwargs.ignore_if_contains = ['blocks.0'];
    assert.deepEqual(imported, expected);
    assert.deepEqual(job, original);
  }
});
