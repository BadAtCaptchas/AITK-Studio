import { SampleConfig } from "@/types";

export const defaultSampleConfig: SampleConfig = {
  backend: 'native',
  sampler: 'flowmatch',
  sample_every: 250,
  sample_start_step: 0,
  width: 1024,
  height: 1024,
  samples: [
    {
      prompt: 'woman with red hair, playing chess at the park, bomb going off in the background',
    },
    {
      prompt: 'a woman holding a coffee cup, in a beanie, sitting at a cafe',
    },
    {
      prompt: 'a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini',
    },
    {
      prompt:
        'a man showing off his cool new t shirt at the beach, a shark is jumping out of the water in the background',
    },
    {
      prompt: 'a bear building a log cabin in the snow covered mountains',
    },
    {
      prompt: 'woman playing the guitar, on stage, singing a song, laser lights, punk rocker',
    },
    {
      prompt: 'hipster man with a beard, building a chair, in a wood shop',
    },
    {
      prompt:
        'photo of a man, white background, medium shot, modeling clothing, studio lighting, white backdrop',
    },
    {
      prompt: "a man holding a sign that says, 'this is a sign'",
    },
    {
      prompt:
        'a bulldog, in a post apocalyptic world, with a shotgun, in a leather jacket, in a desert, with a motorcycle',
    },
  ],
  neg: '',
  seed: 42,
  walk_seed: true,
  guidance_scale: 4,
  sample_steps: 30,
  keep_low_vram_for_samples: false,
  num_frames: 1,
  fps: 1,
}

export const defaultAudioSampleConfig: SampleConfig = {
  backend: 'native',
  sampler: 'flowmatch',
  sample_every: 250,
  sample_start_step: 0,
  width: 1024,
  height: 1024,
  samples: [
    {
      prompt: `
<CAPTION>my style song</CAPTION>
<LYRICS>
[Intro choir]
Laura
Laura
Laura
Laura Training

[Verse 1]
A new open model, she been training it nightly
AI tool kit, she configures it tightly,
Loss curves dropping down to the floor
Wondering if she's done or she should train it some more.

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Verse 4]
She's caching all the latents
now she doesn't need a vay
Training on some voices
What will she make them say

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Guitar Solo]

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Outro]
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
</LYRICS>
<BPM>112</BPM>
<KEYSCALE>A minor</KEYSCALE>
<TIMESIGNATURE>4</TIMESIGNATURE>
<DURATION>180</DURATION>
<LANGUAGE>en</LANGUAGE>
`,
    }, {
      prompt: `
<CAPTION>my style song</CAPTION>
<LYRICS>
[Intro choir]
Laura
Laura
Laura
Laura Training

[Verse 1]
A new open model, she been training it nightly
AI tool kit, she configures it tightly,
Loss curves dropping down to the floor
Wondering if she's done or she should train it some more.

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Verse 4]
She's caching all the latents
now she doesn't need a vay
Training on some voices
What will she make them say

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Guitar Solo]

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Outro]
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
</LYRICS>
<BPM>112</BPM>
<KEYSCALE>A minor</KEYSCALE>
<TIMESIGNATURE>4</TIMESIGNATURE>
<DURATION>180</DURATION>
<LANGUAGE>en</LANGUAGE>
`,
    }, {
      prompt: `
<CAPTION>my style song</CAPTION>
<LYRICS>
[Intro choir]
Laura
Laura
Laura
Laura Training

[Verse 1]
A new open model, she been training it nightly
AI tool kit, she configures it tightly,
Loss curves dropping down to the floor
Wondering if she's done or she should train it some more.

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Verse 4]
She's caching all the latents
now she doesn't need a vay
Training on some voices
What will she make them say

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Guitar Solo]

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Outro]
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
</LYRICS>
<BPM>112</BPM>
<KEYSCALE>A minor</KEYSCALE>
<TIMESIGNATURE>4</TIMESIGNATURE>
<DURATION>180</DURATION>
<LANGUAGE>en</LANGUAGE>
`,
    }, {
      prompt: `
<CAPTION>my style song</CAPTION>
<LYRICS>
[Intro choir]
Laura
Laura
Laura
Laura Training

[Verse 1]
A new open model, she been training it nightly
AI tool kit, she configures it tightly,
Loss curves dropping down to the floor
Wondering if she's done or she should train it some more.

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Verse 4]
She's caching all the latents
now she doesn't need a vay
Training on some voices
What will she make them say

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Guitar Solo]

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Outro]
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
</LYRICS>
<BPM>112</BPM>
<KEYSCALE>A minor</KEYSCALE>
<TIMESIGNATURE>4</TIMESIGNATURE>
<DURATION>180</DURATION>
<LANGUAGE>en</LANGUAGE>
`,
    },
  ],
  neg: '',
  seed: 42,
  walk_seed: true,
  guidance_scale: 4,
  sample_steps: 30,
  keep_low_vram_for_samples: false,
  num_frames: 1,
  fps: 1,
};

const tagValue = (prompt: string, tag: string) => {
  const m = prompt.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`));
  return m ? m[1].trim() : '';
};

export const defaultQwen25OmniSampleConfig: SampleConfig = {
  ...defaultSampleConfig,
  width: 512,
  height: 512,
  guidance_scale: 1,
  sample_steps: 1,
  samples: [{ prompt: 'Describe this in detail.' }, { prompt: 'Describe this in detail.' }],
};

export const defaultYue2SampleConfig: SampleConfig = {
  ...defaultAudioSampleConfig,
  samples: defaultAudioSampleConfig.samples.map(s => ({
    ...s,
    prompt: `${tagValue(s.prompt, 'CAPTION')}\n[Lyrics]\n${tagValue(s.prompt, 'LYRICS')}\n`,
  })),
  guidance_scale: 1,
  sample_steps: 32,
  // max seconds per sample; the AR stops earlier when the song ends
  duration: 120,
};
