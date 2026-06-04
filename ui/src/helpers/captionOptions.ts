import { GroupedSelectOption, SelectOption } from "@/types";

type CaptionGroup = 'image' | 'music';
type AdditionalSections =
    | 'caption.model_name_or_path2'
    | 'caption.caption_prompt'
    | 'caption.max_res'
    | 'caption.max_new_tokens'
    | 'caption.fixed_caption'
    | 'caption.system_prompt'
    | 'caption.remote_worker_id';

export interface CaptionOption {
    name: string;
    label: string;
    group: CaptionGroup;
    hasMultiLinePrompts?: boolean;
    usesGpu?: boolean;
    usesQuantization?: boolean;
    usesOpenRouter?: boolean;
    usesRemoteWorker?: boolean;
    defaults?: { [key: string]: any };
    additionalSections?: AdditionalSections[];
    name_or_path_options?: SelectOption[];
    name_or_path2_options?: SelectOption[];
}

const defaultNameOrPath = '';

const extensionsAudio = ['mp3', 'wav', 'flac', 'ogg'];
const extensionsImage = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];

const defaultExtensions = [...extensionsImage];

export const defaultImageCaptionPrompt = "Caption this image as if you were going to try to generate it with an image generator. Be thurough and describe everything in the image. Be decisive by stating things as they are. Do not say things like \"It appears that\" Or \"possibly\". Start out with things like \"A person on the beach\" or \"A black dragon\". No preamble. Just get to the point.";
export const defaultIdeogramJsonCaptionPrompt = `Create an Ideogram 4 training caption for this image as a JSON object.
Return only valid JSON. Use high_level_description, style_description, and compositional_deconstruction. Use [ymin, xmin, ymax, xmax] bounding boxes normalized to 0-1000 when useful.
Preserve and refine this existing caption when present:
{existing_caption}`;

export const captionerTypes: CaptionOption[] = [
    {
        name: 'AceStepCaptioner',
        label: 'Ace Step',
        group: 'music',
        usesGpu: true,
        usesQuantization: true,
        defaults: {
            'config.process[0].caption.model_name_or_path': ['ACE-Step/acestep-transcriber', defaultNameOrPath],
            'config.process[0].caption.model_name_or_path2': ['ACE-Step/acestep-captioner', undefined],
            'config.process[0].caption.extensions': [extensionsAudio, defaultExtensions],
        },
        name_or_path_options: [
            { value: 'ACE-Step/acestep-transcriber', label: 'ACE-Step/acestep-transcriber' },
        ],
        name_or_path2_options: [
            { value: 'ACE-Step/acestep-captioner', label: 'ACE-Step/acestep-captioner' },
        ],
        additionalSections: [
            'caption.model_name_or_path2',
            'caption.fixed_caption',
        ],
    },
    {
        name: 'Qwen3VLCaptioner',
        label: 'Qwen3-VL',
        group: 'image',
        usesGpu: true,
        usesQuantization: true,
        defaults: {
            'config.process[0].caption.model_name_or_path': ['Qwen/Qwen3-VL-8B-Instruct', defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultImageCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [512, undefined],
            'config.process[0].caption.max_new_tokens': [128, undefined],

        },
        name_or_path_options: [
            { value: 'Qwen/Qwen3-VL-2B-Instruct', label: 'Qwen/Qwen3-VL-2B-Instruct' },
            { value: 'Qwen/Qwen3-VL-4B-Instruct', label: 'Qwen/Qwen3-VL-4B-Instruct' },
            { value: 'Qwen/Qwen3-VL-8B-Instruct', label: 'Qwen/Qwen3-VL-8B-Instruct' },
            { value: 'Qwen/Qwen3-VL-30B-A3B-Instruct', label: 'Qwen/Qwen3-VL-30B-A3B-Instruct' },
        ],
        additionalSections: [
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
        ],
    },
    {
        name: 'OllamaCaptioner',
        label: 'Ollama',
        group: 'image',
        defaults: {
            'config.process[0].device': ['cpu', 'cuda'],
            'config.process[0].caption.model_name_or_path': ['llava:latest', defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultImageCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [768, undefined],
            'config.process[0].caption.max_new_tokens': [180, undefined],
            'config.process[0].caption.quantize': [false, true],
            'config.process[0].caption.low_vram': [false, true],
        },
        name_or_path_options: [
            { value: 'llava:latest', label: 'llava:latest' },
            { value: 'llama3.2-vision:latest', label: 'llama3.2-vision:latest' },
            { value: 'qwen2.5vl:latest', label: 'qwen2.5vl:latest' },
        ],
        additionalSections: [
            'caption.caption_prompt',
            'caption.system_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
        ],
    },
    {
        name: 'SecureRemoteOllamaCaptioner',
        label: 'Remote Ollama',
        group: 'image',
        usesRemoteWorker: true,
        defaults: {
            'config.process[0].device': ['cpu', 'cuda'],
            'config.process[0].caption.model_name_or_path': ['llava:latest', defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultImageCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [768, undefined],
            'config.process[0].caption.max_new_tokens': [180, undefined],
            'config.process[0].caption.quantize': [false, true],
            'config.process[0].caption.low_vram': [false, true],
            'config.process[0].caption.remote_worker_id': ['', undefined],
        },
        name_or_path_options: [
            { value: 'llava:latest', label: 'llava:latest' },
            { value: 'llama3.2-vision:latest', label: 'llama3.2-vision:latest' },
            { value: 'qwen2.5vl:latest', label: 'qwen2.5vl:latest' },
        ],
        additionalSections: [
            'caption.remote_worker_id',
            'caption.caption_prompt',
            'caption.system_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
        ],
    },
    {
        name: 'OpenRouterCaptioner',
        label: 'OpenRouter',
        group: 'image',
        usesOpenRouter: true,
        defaults: {
            'config.process[0].device': ['cpu', 'cuda'],
            'config.process[0].caption.model_name_or_path': ['x-ai/grok-4.3', defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultImageCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [1024, undefined],
            'config.process[0].caption.max_new_tokens': [220, undefined],
            'config.process[0].caption.quantize': [false, true],
            'config.process[0].caption.low_vram': [false, true],
        },
        name_or_path_options: [
            { value: 'x-ai/grok-4.3', label: 'x-ai/grok-4.3 (recommended, $1.25/M in, $2.50/M out)' },
            { value: 'x-ai/grok-4-fast', label: 'x-ai/grok-4-fast (cheaper, $0.20/M in, $0.50/M out)' },
        ],
        additionalSections: [
            'caption.caption_prompt',
            'caption.system_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
        ],
    },

].sort((a, b) => {
    // Sort by label, case-insensitive
    return a.label.localeCompare(b.label, undefined, { sensitivity: 'base' });
}) as any;

export const groupedCaptionerTypes: GroupedSelectOption[] = captionerTypes.reduce((acc, arch) => {
    const group = acc.find(g => g.label === arch.group);
    if (group) {
        group.options.push({ value: arch.name, label: arch.label });
    } else {
        acc.push({
            label: arch.group,
            options: [{ value: arch.name, label: arch.label }],
        });
    }
    return acc;
}, [] as GroupedSelectOption[]);

export const quantizationOptions: SelectOption[] = [
    { value: '', label: '- NONE -' },
    { value: 'float8', label: 'float8 (default)' },
    { value: 'uint7', label: '7 bit' },
    { value: 'uint6', label: '6 bit' },
    { value: 'uint5', label: '5 bit' },
    { value: 'uint4', label: '4 bit' },
    { value: 'uint3', label: '3 bit' },
    { value: 'uint2', label: '2 bit' },
];

export const maxResOptions: SelectOption[] = [
    { value: '256', label: '256' },
    { value: '512', label: '512 (default)' },
    { value: '768', label: '768' },
    { value: '1024', label: '1024' },
];
export const maxNewTokensOptions: SelectOption[] = [
    { value: '64', label: '64' },
    { value: '128', label: '128 (default)' },
    { value: '256', label: '256' },
    { value: '512', label: '512' },
    { value: '1024', label: '1024' },
];

export const defaultQtype = 'float8';
