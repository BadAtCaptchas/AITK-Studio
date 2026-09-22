import contract from './modelCapabilities.json';

export const CAPABILITY_VERSION = contract.version;
const record = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
export type DeviceBackend = 'cuda' | 'mps';
export function configContractErrors(value: unknown, context: { deviceBackend?: DeviceBackend } = {}): string[] {
  const errors: string[] = [];
  if (
    !record(value) ||
    !record(value.config) ||
    !Array.isArray(value.config.process) ||
    !value.config.process.length ||
    value.config.process.length > 32
  )
    return ['Config requires between 1 and 32 processes.'];
  if (value.capability_version !== undefined && value.capability_version !== CAPABILITY_VERSION)
    errors.push('Unsupported capability_version.');
  const extraProcesses = new Set<string>(),
    extraModels = new Set<string>();
  if (value.extensions !== undefined) {
    if (!record(value.extensions)) errors.push('extensions must be a namespaced object.');
    else
      for (const [name, declaration] of Object.entries(value.extensions)) {
        if (!/^[a-z][a-z0-9_-]*\.[a-z][a-z0-9_.-]*$/.test(name) || !record(declaration)) {
          errors.push('Invalid extension namespace.');
          continue;
        }
        for (const [key, target] of [
          ['processKinds', extraProcesses],
          ['modelArches', extraModels],
        ] as const) {
          const entries = declaration[key];
          if (entries === undefined) continue;
          if (
            !Array.isArray(entries) ||
            entries.length > 32 ||
            entries.some(entry => typeof entry !== 'string' || entry.length > 100)
          )
            errors.push(`Invalid extension ${key}.`);
          else for (const entry of entries) target.add(entry);
        }
      }
  }
  const finite = (object: Record<string, unknown>, key: string, min: number, max: number, integer = false) => {
    const number = object[key];
    if (
      number !== undefined &&
      (typeof number !== 'number' ||
        !Number.isFinite(number) ||
        number < min ||
        number > max ||
        (integer && !Number.isInteger(number)))
    )
      errors.push(`${key} must be ${integer ? 'an integer' : 'a finite number'} between ${min} and ${max}.`);
  };
  for (const process of value.config.process) {
    if (!record(process) || typeof process.type !== 'string') {
      errors.push('Every process requires a type.');
      continue;
    }
    if (!contract.processKinds.includes(process.type) && !extraProcesses.has(process.type))
      errors.push(`Unknown process type: ${process.type}. Declare custom processes in a namespaced extensions entry.`);
    const train = process.train;
    if (train !== undefined && !record(train)) errors.push('train must be an object.');
    if (record(train)) {
      finite(train, 'steps', 1, 1_000_000_000, true);
      finite(train, 'batch_size', 1, 4096, true);
      finite(train, 'lr', 0, 10);
      finite(train, 'gradient_accumulation', 1, 1_000_000, true);
    }
    const model = process.model;
    if (model !== undefined && !record(model)) errors.push('model must be an object.');
    if (record(model)) {
      if (typeof model.arch !== 'string' || !model.arch) {
        errors.push('model.arch is required.');
        continue;
      }
      const arch = model.arch.split(':')[0];
      if (arch === 'ming_image_design' || arch === 'ming_image_design_layer') {
        if (record(train) && train.batch_size !== undefined && train.batch_size !== 1) errors.push('Ming Image requires batch size 1.');
        if (record(train) && train.train_text_encoder === true) errors.push('Ming Image conditioning must remain frozen.');
        if (model.quantize_te === true) errors.push('Ming Image text encoder quantization is not supported.');
        if (record(process.sample)) {
          const prompts = [process.sample, ...(Array.isArray(process.sample.samples) ? process.sample.samples.filter(record) : [])];
          if (prompts.some(sample => typeof sample.neg === 'string' && sample.neg.trim())) errors.push('Ming Image does not support nonempty negative prompts.');
          for (const sample of prompts) for (const key of ['width', 'height'] as const) {
            const dimension = sample[key];
            if (dimension !== undefined && (typeof dimension !== 'number' || !Number.isInteger(dimension) || dimension < 16 || dimension > 8192 || dimension % 16 !== 0)) errors.push(`Ming Image sample ${key} must be a multiple of 16 between 16 and 8192.`);
          }
        }
        if (Array.isArray(process.datasets)) for (const dataset of process.datasets) {
          if (!record(dataset)) continue;
          if (arch === 'ming_image_design_layer' && dataset.type !== 'layered_image') errors.push('Ming Design-Layer requires layered_image datasets.');
          if (arch === 'ming_image_design' && dataset.type === 'layered_image') errors.push('Select Ming Design-Layer for layered_image datasets.');
        }
      }
      if (arch === 'qwen25_omni' && record(train) && train.batch_size !== undefined && train.batch_size !== 1) errors.push('Qwen2.5-Omni requires batch size 1.');
      if (arch === 'yue2' && record(model.model_kwargs) && model.model_kwargs.do_separation === true) errors.push('YuE2 separation training is not supported.');
      const choice =
        contract.choices.find(item => item.name === model.arch) || contract.choices.find(item => item.name === arch);
      if (!choice && !contract.models.some(item => item.arch === arch) && !extraModels.has(arch))
        errors.push(`Unknown model architecture: ${model.arch}.`);
      const network = process.network;
      if (
        choice?.allowedNetworkTypes.length &&
        record(network) &&
        typeof network.type === 'string' &&
        !choice.allowedNetworkTypes.some(type => type === network.type)
      )
        errors.push(`${model.arch} does not support network ${network.type}.`);
      const backend =
        context.deviceBackend || (typeof process.device === 'string' ? process.device.split(':')[0] : undefined);
      if (choice && backend && !choice.deviceBackends.includes(backend))
        errors.push(`${model.arch} does not support device backend ${backend}.`);
      const dtype = model.dtype ?? (record(train) ? train.dtype : undefined) ?? process.dtype;
      if (choice && dtype !== undefined && (typeof dtype !== 'string' || !choice.precisions.includes(dtype)))
        errors.push('Unsupported model precision.');
      if (
        backend === 'mps' &&
        ['qtype', 'qtype_te'].some(
          key => typeof model[key] === 'string' && contract.cudaOnlyQuantization.includes(model[key] as string),
        )
      )
        errors.push('The selected quantization requires a CUDA worker.');
      if (choice && model.layer_offloading === true && !choice.offloading.layers)
        errors.push(`${model.arch} does not support layer offloading.`);
      if (
        choice &&
        record(process.sample) &&
        typeof process.sample.num_frames === 'number' &&
        (process.sample.num_frames - 1) % choice.frames.multipleAfterFirst !== 0
      )
        errors.push(`Frame count for ${model.arch} must be 1 plus a multiple of ${choice.frames.multipleAfterFirst}.`);
      if (model.arch.startsWith('minimax_h3') && model.layer_offloading === true)
        errors.push('MiniMax H3 does not support layer offloading.');
    }
    if (process.sample !== undefined && !record(process.sample)) errors.push('sample must be an object.');
    if (record(process.sample)) {
      finite(process.sample, 'num_frames', 1, 100000, true);
      finite(process.sample, 'num_layers', 1, 32, true);
      finite(process.sample, 'fps', 0.001, 1000);
      finite(process.sample, 'width', 1, 32768, true);
      finite(process.sample, 'height', 1, 32768, true);
      finite(process.sample, 'sample_steps', 1, 10000, true);
      finite(process.sample, 'guidance_scale', 0, 1000);
      if (Array.isArray(process.sample.samples)) {
        for (const sample of process.sample.samples) if (record(sample)) finite(sample, 'num_layers', 1, 32, true);
      }
    }
  }
  return errors;
}
