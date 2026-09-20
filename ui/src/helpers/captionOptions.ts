import type { CloudLora } from '@/utils/loraTypes';
import { GroupedSelectOption, SelectOption } from "@/types";

type CaptionGroup = 'image' | 'music' | 'video' | 'image/video/sound';
type AdditionalSections =
    | 'caption.caption_format'
    | 'caption.extract_vocals_before_transcribe'
    | 'caption.keep_timestamps'
    | 'caption.model_name_or_path2'
    | 'caption.caption_prompt'
    | 'caption.max_res'
    | 'caption.max_new_tokens'
    | 'caption.batch_size'
    | 'caption.layer_offloading'
    | 'caption.thinking'
    | 'caption.fixed_caption'
    | 'caption.system_prompt'
    | 'caption.remote_ollama_worker_id';

export interface CaptionOption {
    name: string;
    label: string;
    group: CaptionGroup;
    hasMultiLinePrompts?: boolean;
    minNewTokens?: number;
    supportsLoras?: boolean;
    cloudLoras?: CloudLora[];
    usesGpu?: boolean;
    usesQuantization?: boolean;
    usesOpenRouter?: boolean;
    defaults?: { [key: string]: any };
    additionalSections?: AdditionalSections[];
    name_or_path_options?: SelectOption[];
    name_or_path2_options?: SelectOption[];
    // named caption prompts the user can swap between; the picker only shows
    // when there is more than one, and selecting one just fills caption_prompt
    captionPrompts?: { [name: string]: string };
}

const defaultNameOrPath = '';
const defaultOllamaVisionModel = 'qwen3.5:35b';

const extensionsAudio = ['mp3', 'wav', 'flac', 'ogg'];
const extensionsImage = ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'jxl'];
const extensionsVideo = ['mp4', 'mov', 'webm', 'mkv', 'avi'];

const defaultExtensions = [...extensionsImage];

export const legacyDefaultImageCaptionPrompt = "Caption this image as if you were going to try to generate it with an image generator. Be thurough and describe everything in the image. Be decisive by stating things as they are. Do not say things like \"It appears that\" Or \"possibly\". Start out with things like \"A person on the beach\" or \"A black dragon\". No preamble. Just get to the point.";
export const defaultImageCaptionPrompt = "Caption this image as if you were going to try to generate it with an image generator. Be thurough and describe everything in the image. Be decisive by stating things as they are. Do not say things like \"It appears that\" Or \"possibly\". Start out with things like \"A person on the beach\" or \"A black dragon\". NSFW content is allowed; if present, describe it directly and clinically. No preamble. Just get to the point.";
const defaultVideoCaptionPrompt = "Caption this video for a video generator. Describe the visual content, motion over time, camera work, and audio. Transcribe spoken dialogue verbatim in quotes. Be decisive and include no preamble.";
const minimaxT2VCaptionPrompt = `Caption this video as a MiniMax T2VA training prompt. Output exactly these fields: integrated_multimodal_description, overall_soundscape, and non_diegetic_music. Use [Shot N] timeline markers with timestamps for cuts, stable (S1)/(S2) speaker IDs, verbatim dialogue in <d>[Language] ...</d>, precise camera motion, and quoted visible text. Describe only what is seen and heard.`;
const minimaxImageCaptionPrompt = `Caption this still image as a MiniMax training prompt. Output integrated_multimodal_description beginning with [Shot 1], followed by overall_soundscape: N/A and non_diegetic_music: N/A. Describe style, framing, subjects, lighting, props, spatial relationships, and visible text without inventing motion or audio.`;
export const defaultIdeogramJsonCaptionPrompt = `Create an Ideogram 4 training caption for this image as a JSON object.
Return only valid JSON. Do not wrap it in markdown.

high_level_description should be a concise but detailed one-paragraph description.
NSFW content is allowed. If present, describe visible nudity, sexual content, fetish content, anatomy, fluids, clothing state, and explicit poses directly and clinically instead of censoring or omitting them.
For each important visible element, include type ("obj" or "text"), desc, optional color_palette, and bbox when you can estimate it. For text elements, include text with the readable text content when known, or an empty string when unreadable.

Use this exact JSON contract:
- Top-level key order: high_level_description, style_description, compositional_deconstruction.
- For photo captions, style_description key order must be: aesthetics, lighting, photo, medium, color_palette.
- For non-photo captions, style_description key order must be: aesthetics, lighting, medium, art_style, color_palette.
- Include exactly one of style_description.photo or style_description.art_style.
- compositional_deconstruction key order must be: background, elements.
- Object element key order must be: type, bbox, desc, color_palette.
- Text element key order must be: type, bbox, text, desc, color_palette.
- Omit bbox or color_palette only when unavailable; if present, keep them in the listed position.
- Bounding boxes must be [ymin, xmin, ymax, xmax] normalized to 0-1000.
- Colors must be uppercase #RRGGBB hex strings.

Preserve and refine this existing caption when present:
{existing_caption}`;

const yue2CaptionPrompt = `Listen to this song and write a YuE2 training caption for it. Output exactly two parts and nothing else.

Part 1, the first line only: comma-separated style tags describing the music. Cover, in this order where applicable: genre and sub-genre, mood, vocal type (male vocal, female vocal, duet, choir, rap, or instrumental), vocal delivery (breathy, belted, falsetto, spoken, whispered, harmonized), lead instruments and production (acoustic guitar, piano, synth pads, 808 bass, live drums, drum machine, strings, lo-fi, reverb-heavy), tempo feel (slow ballad, mid-tempo, upbeat, fast), and era or scene (90s alt rock, modern trap, bedroom pop). Use concrete lowercase tags, no sentences, no hedging, no artist or song names. Example: alternative rock, melancholic, male vocal, falsetto, electric guitar, piano, live drums, slow ballad, 2000s

Part 2: on the next line write [Lyrics] and then the complete lyrics transcribed verbatim, one sung line per line. Split the song into sections with a bracketed header on its own line before each one, using these names: [Intro], [Verse 1], [Verse 2], [Pre-Chorus], [Chorus], [Bridge], [Instrumental], [Solo], [Outro]. A short descriptor may follow the name inside the brackets, e.g. [Intro choir] or [Chorus harmonized]. Leave one blank line between sections. Repeat a chorus each time it is sung, but write each repeated line once per time it is sung, never more. Wordless vocalizing (na, la, oh, ah, hums, vocal chops) is never transcribed syllable by syllable: describe it once in the section header instead, e.g. [Intro wordless vocals] or [Bridge oohs], and if you cannot make out real words in a passage, treat it as wordless. Do not add timestamps, speaker labels, quotation marks, translations, or commentary, and do not annotate ad-libs beyond the section header. If the song has no vocals at all, Part 2 is [Lyrics] followed by a single [Instrumental] line.

Transcribe only what is actually sung. Be decisive. No preamble, no explanations, no markdown - output only the tag line and the lyrics block.`;

const mossMusicCaptionPrompt = "Write a prompt that a text-to-music generator could use to recreate this track. One paragraph, under 80 words: genre, mood, instrumentation, tempo feel, production style, vocal style. No timestamps, section timings, chord names, or lyric quotes.";

const mossMusicTagsPrompt = "Describe this music as a single line of comma-separated style tags: genre, mood, vocal type and delivery, lead instruments, production style, tempo feel, era. Lowercase tags only, no sentences.";

export const captionerTypes: CaptionOption[] = [
{
        name: 'MossMusicCaptioner',
        label: 'MOSS-Music',
        group: 'music', usesGpu: true, usesQuantization: true,
        defaults: {
            'config.process[0].caption.model_name_or_path': ['OpenMOSS-Team/MOSS-Music-8B-Instruct', defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsAudio, defaultExtensions],
            'config.process[0].caption.caption_format': ['ace_step', undefined],
            'config.process[0].caption.caption_prompt': [mossMusicCaptionPrompt, undefined],
            'config.process[0].caption.keep_timestamps': [false, undefined],
            'config.process[0].caption.compile': [true, false],
        },
        name_or_path_options: [
            { value: 'OpenMOSS-Team/MOSS-Music-8B-Instruct', label: 'OpenMOSS-Team/MOSS-Music-8B-Instruct' },
        ],
        captionPrompts: {
            'Description (ACE-Step)': mossMusicCaptionPrompt,
            'Style tags (YuE2)': mossMusicTagsPrompt,
        },
        additionalSections: [
            'caption.fixed_caption',
            'caption.caption_format',
            'caption.keep_timestamps',
            'caption.caption_prompt',
        ],
    },
{
        name: 'Qwen25OmniCaptioner',
        label: 'Qwen2.5-Omni',
        group: 'image/video/sound', usesGpu: true, usesQuantization: true,
        supportsLoras: true,
        cloudLoras: [
            {
                path: 'ai-toolkit/Qwen2.5-Omni-7B/qwen2_5_omni_7b_lora_caption_this_song.safetensors',
                name: 'Caption This Song',
            },
        ],
        defaults: {
            'config.process[0].caption.loras': [[], undefined],
            'config.process[0].caption.model_name_or_path': ['ai-toolkit/Qwen2.5-Omni-7B/qwen2_5_omni_7b_convrot8.safetensors', defaultNameOrPath],
            'config.process[0].caption.extensions': [[...extensionsVideo, ...extensionsImage, ...extensionsAudio], defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultVideoCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [512, undefined],
            'config.process[0].caption.max_new_tokens': [512, undefined],
            'config.process[0].caption.batch_size': [1, undefined],
            'config.process[0].caption.compile': [true, false],
        },
        name_or_path_options: [
            { value: 'ai-toolkit/Qwen2.5-Omni-7B/qwen2_5_omni_7b_convrot8.safetensors', label: 'ai-toolkit/Qwen2.5-Omni-7B (convrot8)' },
            { value: 'Qwen/Qwen2.5-Omni-7B', label: 'Qwen/Qwen2.5-Omni-7B' },
            { value: 'Qwen/Qwen2.5-Omni-3B', label: 'Qwen/Qwen2.5-Omni-3B' },
        ],
        captionPrompts: {
            'General': defaultVideoCaptionPrompt,
            'MiniMax H4 T2V': minimaxT2VCaptionPrompt,
            'MiniMax H4 Image': minimaxImageCaptionPrompt,
            'YuE2': yue2CaptionPrompt,
        },
        additionalSections: [
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
            'caption.batch_size',
        ],
    },
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
            'config.process[0].caption.caption_format': ['ace_step', undefined],
            'config.process[0].caption.compile': [true, false],
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
            'caption.caption_format',
            'caption.extract_vocals_before_transcribe',
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
            { value: 'huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated', label: 'huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated' },
            { value: 'Qwen/Qwen3-VL-30B-A3B-Instruct', label: 'Qwen/Qwen3-VL-30B-A3B-Instruct' },
            { value: 'Qwen/Qwen3.6-27B', label: 'Qwen/Qwen3.6-27B' },
            { value: 'huihui-ai/Huihui-Qwen3.6-27B-abliterated', label: 'huihui-ai/Huihui-Qwen3.6-27B-abliterated' },
        ],
        additionalSections: [
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
            'caption.thinking',
        ],
    },
    {
        name: 'Qwen3OmniCaptioner',
        label: 'Qwen3-Omni',
        group: 'image/video/sound',
        usesGpu: true,
        usesQuantization: true,
        defaults: {
            'config.process[0].caption.model_name_or_path': ['ai-toolkit/Qwen3-Omni-30B-A3B-Thinking', defaultNameOrPath],
            'config.process[0].caption.extensions': [[...extensionsVideo, ...extensionsImage, ...extensionsAudio], defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultVideoCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [512, undefined],
            'config.process[0].caption.max_new_tokens': [512, undefined],
            'config.process[0].caption.batch_size': [1, undefined],
            'config.process[0].caption.compile': [true, false],
        },
        name_or_path_options: [
            { value: 'ai-toolkit/Qwen3-Omni-30B-A3B-Instruct', label: 'ai-toolkit/Qwen3-Omni-30B-A3B-Instruct' },
            { value: 'ai-toolkit/Qwen3-Omni-30B-A3B-Thinking', label: 'ai-toolkit/Qwen3-Omni-30B-A3B-Thinking' },
            { value: 'ai-toolkit/Huihui-Qwen3-Omni-30B-A3B-Thinking-abliterated', label: 'ai-toolkit/Huihui-Qwen3-Omni-30B-A3B-Thinking-abliterated' },
        ],
        captionPrompts: {
            'YuE2': yue2CaptionPrompt,
            General: defaultVideoCaptionPrompt,
            'MiniMax H3 T2V': minimaxT2VCaptionPrompt,
            'MiniMax H3 Image': minimaxImageCaptionPrompt,
        },
        additionalSections: [
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
            'caption.batch_size',
            'caption.layer_offloading',
            'caption.thinking',
        ],
    },
    {
        name: 'OllamaCaptioner',
        label: 'Ollama',
        group: 'image',
        defaults: {
            'config.process[0].device': ['cpu', 'cuda'],
            'config.process[0].caption.model_name_or_path': [defaultOllamaVisionModel, defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultImageCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [768, undefined],
            'config.process[0].caption.max_new_tokens': [180, undefined],
            'config.process[0].caption.quantize': [false, true],
            'config.process[0].caption.low_vram': [false, true],
        },
        name_or_path_options: [
            { value: 'qwen3.5:122b', label: 'qwen3.5:122b (best quality, high VRAM)' },
            { value: 'qwen3.5:35b', label: 'qwen3.5:35b (recommended)' },
            { value: 'qwen3.5:27b', label: 'qwen3.5:27b (backup)' },
            { value: 'qwen3.5:9b', label: 'qwen3.5:9b (small backup)' },
            { value: 'gemma4:31b', label: 'gemma4:31b' },
            { value: 'gemma4:26b', label: 'gemma4:26b' },
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
        defaults: {
            'config.process[0].device': ['cpu', 'cuda'],
            'config.process[0].caption.model_name_or_path': [defaultOllamaVisionModel, defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultImageCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [768, undefined],
            'config.process[0].caption.max_new_tokens': [180, undefined],
            'config.process[0].caption.quantize': [false, true],
            'config.process[0].caption.low_vram': [false, true],
            'config.process[0].caption.remote_ollama_worker_id': ['', undefined],
        },
        name_or_path_options: [
            { value: 'qwen3.5:122b', label: 'qwen3.5:122b (best quality, high VRAM)' },
            { value: 'qwen3.5:35b', label: 'qwen3.5:35b (recommended)' },
            { value: 'qwen3.5:27b', label: 'qwen3.5:27b (backup)' },
            { value: 'qwen3.5:9b', label: 'qwen3.5:9b (small backup)' },
            { value: 'gemma4:31b', label: 'gemma4:31b' },
            { value: 'gemma4:26b', label: 'gemma4:26b' },
        ],
        additionalSections: [
            'caption.remote_ollama_worker_id',
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
    { value: 'convrot8', label: 'ConvRot W8A8' },
    { value: 'convrot4', label: 'ConvRot NVFP4 W4A4' },
    { value: 'convrotint7', label: 'ConvRot 7-bit weights' },
    { value: 'convrotint6', label: 'ConvRot 6-bit weights' },
    { value: 'convrotint5', label: 'ConvRot 5-bit weights' },
    { value: 'convrotint4', label: 'ConvRot 4-bit weights' },
    { value: 'convrotint3', label: 'ConvRot 3-bit weights' },
    { value: 'convrotint2', label: 'ConvRot 2-bit weights' },
    { value: 'convrotbitnet', label: 'ConvRot BitNet 1.58-bit weights' },
    { value: 'uint7', label: '7 bit' },
    { value: 'uint6', label: '6 bit' },
    { value: 'uint5', label: '5 bit' },
    { value: 'uint4', label: '4 bit' },
    { value: 'uint3', label: '3 bit' },
    { value: 'uint2', label: '2 bit' },
];

export const batchSizeOptions: SelectOption[] = [
    { value: '1', label: '1 (default)' },
    { value: '2', label: '2' },
    { value: '4', label: '4' },
    { value: '8', label: '8' },
    { value: '12', label: '12' },
    { value: '16', label: '16' },
    { value: '24', label: '24' },
    { value: '32', label: '32' },
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
    { value: '2048', label: '2048' },
];

export const defaultQtype = 'float8';

export const captionFormatOptions: SelectOption[] = [
    { value: 'ace_step', label: 'ACE-Step (caption, lyrics, bpm, key, time signature, duration)' },
    { value: 'yue2', label: 'YuE2 (description, then [Lyrics] block)' },
];
