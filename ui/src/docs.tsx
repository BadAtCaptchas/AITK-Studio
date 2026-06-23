import React from 'react';
import { ConfigDoc } from '@/types';
import { IoFlaskSharp } from 'react-icons/io5';

const docs: { [key: string]: ConfigDoc } = {
  'config.name': {
    title: 'Training Name',
    description: (
      <>
        The name of the training job. This name will be used to identify the job in the system and will the the filename
        of the final model. It must be unique and can only contain alphanumeric characters, underscores, and dashes. No
        spaces or special characters are allowed.
      </>
    ),
  },
  gpuids: {
    title: 'GPU ID',
    description: (
      <>
        This is the GPU that will be used for training. Only one GPU can be used per job at a time via the UI currently.
        However, you can start multiple jobs in parallel, each using a different GPU.
      </>
    ),
  },
  'config.process[0].trigger_word': {
    title: 'Trigger Word',
    description: (
      <>
        Optional: This will be the word or token used to trigger your concept or character.
        <br />
        <br />
        When using a trigger word, If your captions do not contain the trigger word, it will be added automatically the
        beginning of the caption. If you do not have captions, the caption will become just the trigger word. If you
        want to have variable trigger words in your captions to put it in different spots, you can use the{' '}
        <code>{'[trigger]'}</code> placeholder in your captions. This will be automatically replaced with your trigger
        word.
        <br />
        <br />
        Trigger words will not automatically be added to your test prompts, so you will need to either add your trigger
        word manually or use the
        <code>{'[trigger]'}</code> placeholder in your test prompts as well.
      </>
    ),
  },
  'config.process[0].model.name_or_path': {
    title: 'Name or Path',
    description: (
      <>
        The name of a diffusers repo on Huggingface or the local path to the base model you want to train from. The
        folder needs to be in diffusers format for most models. For some models, such as SDXL and SD1, you can put the
        path to an all in one safetensors checkpoint here.
      </>
    ),
  },
  'datasets.control_path': {
    title: 'Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be
        matching file pairs. These images are fed as control/input images during training. The control images will be
        resized to match the training images.
      </>
    ),
  },
  'datasets.multi_control_paths': {
    title: 'Multi Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be
        matching file pairs. These images are fed as control/input images during training.
        <br />
        <br />
        For multi control datasets, the controls will all be applied in the order they are listed. If the model does not
        require the images to be the same aspect ratios, such as with Qwen/Qwen-Image-Edit-2509, then the control images
        do not need to match the aspect size or aspect ratio of the target image and they will be automatically resized
        to the ideal resolutions for the model / target images.
      </>
    ),
  },
  'datasets.num_frames': {
    title: 'Number of Frames',
    description: (
      <>
        This sets the number of frames to shrink videos to for a video dataset. If this dataset is images, set this to 1
        for one frame. If your dataset is only videos, frames will be extracted evenly spaced from the videos in the
        dataset.
        <br />
        <br />
        It is best to trim your videos to the proper length before training. Wan is 16 frames a second. Doing 81 frames
        will result in a 5 second video. So you would want all of your videos trimmed to around 5 seconds for best
        results.
        <br />
        <br />
        Example: Setting this to 81 and having 2 videos in your dataset, one is 2 seconds and one is 90 seconds long,
        will result in 81 evenly spaced frames for each video making the 2 second video appear slow and the 90second
        video appear very fast.
      </>
    ),
  },
  'datasets.do_i2v': {
    title: 'Do I2V',
    description: (
      <>
        For video models that can handle both I2V (Image to Video) and T2V (Text to Video), this option sets this
        dataset to be trained as an I2V dataset. This means that the first frame will be extracted from the video and
        used as the start image for the video. If this option is not set, the dataset will be treated as a T2V dataset.
      </>
    ),
  },
  'datasets.do_audio': {
    title: 'Do Audio',
    description: (
      <>
        For models that support audio with video, this option will load the audio from the video and resize it to match
        the video sequence. Since the video is automatically resized, the audio may drop or raise in pitch to match the
        new speed of the video. It is important to prep your dataset to have the proper length before training.
      </>
    ),
  },
  'datasets.audio_normalize': {
    title: 'Audio Normalize',
    description: (
      <>
        When loading audio, this will normalize the audio volume to the max peaks. Useful if your dataset has varying
        audio volumes. Warning, do not use if you have clips with full silence you want to keep, as it will raise the
        volume of those clips.
      </>
    ),
  },
  'datasets.audio_preserve_pitch': {
    title: 'Audio Preserve Pitch',
    description: (
      <>
        When loading audio to match the number of frames requested, this option will preserve the pitch of the audio if
        the length does not match training target. It is recommended to have a dataset that matches your target length,
        as this option can add sound distortions.
      </>
    ),
  },
  'datasets.flip': {
    title: 'Flip X and Flip Y',
    description: (
      <>
        You can augment your dataset on the fly by flipping the x (horizontal) and/or y (vertical) axis. Flipping a
        single axis will effectively double your dataset. It will result it training on normal images, and the flipped
        versions of the images. This can be very helpful, but keep in mind it can also be destructive. There is no
        reason to train people upside down, and flipping a face can confuse the model as a person's right side does not
        look identical to their left side. For text, obviously flipping text is not a good idea.
        <br />
        <br />
        Control images for a dataset will also be flipped to match the images, so they will always match on the pixel
        level.
      </>
    ),
  },
  'train.unload_text_encoder': {
    title: 'Unload Text Encoder',
    description: (
      <>
        Unloading text encoder will cache the trigger word and the sample prompts and unload the text encoder from the
        GPU. Captions in for the dataset will be ignored. This cannot be used while training the text encoder.
      </>
    ),
  },
  'train.train_text_encoder': {
    title: 'Train Text Encoder',
    description: (
      <>
        Trains the text encoder along with the selected training target. This uses more VRAM and cannot be combined
        with unloading the text encoder or cached text embeddings.
      </>
    ),
  },
  'train.text_encoder_lr': {
    title: 'Text Encoder Learning Rate',
    description: (
      <>
        Learning rate used for text encoder parameters. When unset, the trainer uses the main learning rate.
      </>
    ),
  },
  'train.cache_text_embeddings': {
    title: 'Cache Text Embeddings',
    description: (
      <>
        <small>(experimental)</small>
        <br />
        Caching text embeddings will process and cache all the text embeddings from the text encoder to the disk. The
        text encoder will be unloaded from the GPU. This does not work with things that dynamically change the prompt
        such as trigger words, caption dropout, etc. This cannot be used while training the text encoder.
      </>
    ),
  },
  'model.multistage': {
    title: 'Stages to Train',
    description: (
      <>
        Some models have multi stage networks that are trained and used separately in the denoising process. Most
        common, is to have 2 stages. One for high noise and one for low noise. You can choose to train both stages at
        once or train them separately. If trained at the same time, The trainer will alternate between training each
        model every so many steps and will output 2 different LoRAs. If you choose to train only one stage, the trainer
        will only train that stage and output a single LoRA.
      </>
    ),
  },
  'train.switch_boundary_every': {
    title: 'Switch Boundary Every',
    description: (
      <>
        When training a model with multiple stages, this setting controls how often the trainer will switch between
        training each stage.
        <br />
        <br />
        For low vram settings, the model not being trained will be unloaded from the gpu to save memory. This takes some
        time to do, so it is recommended to alternate less often when using low vram. A setting like 10 or 20 is
        recommended for low vram settings.
        <br />
        <br />
        The swap happens at the batch level, meaning it will swap between a gradient accumulation steps. To train both
        stages in a single step, set them to switch every 1 step and set gradient accumulation to 2.
      </>
    ),
  },
  'train.force_first_sample': {
    title: 'Force First Sample',
    description: (
      <>
        This option will force the trainer to generate samples when it starts. The trainer will normally only generate a
        first sample when nothing has been trained yet, but will not do a first sample when resuming from an existing
        checkpoint. This option forces a first sample every time the trainer is started. This can be useful if you have
        changed sample prompts and want to see the new prompts right away.
      </>
    ),
  },
  'sample.keep_low_vram_for_samples': {
    title: 'Keep Low VRAM During Samples',
    description: (
      <>
        Leave low VRAM and legacy layer offloading enabled while generating training samples. By default, the trainer
        temporarily disables those memory-saving modes during samples and restores them afterward.
      </>
    ),
  },
  'model.layer_offloading': {
    title: (
      <>
        Layer Offloading{' '}
        <span className="text-yellow-500">
          ( <IoFlaskSharp className="inline text-yellow-500" name="Experimental" /> Experimental)
        </span>
      </>
    ),
    description: (
      <>
        This is an experimental whole-block CPU offloading feature for supported CUDA models. The default block backend
        moves ordered transformer/text-encoder blocks between CPU RAM and GPU VRAM as they are needed, while the legacy
        backend keeps the older per-module offloader available for unsupported models.
        <br />
        <br />
        Layer Offloading uses the CPU RAM instead of the GPU ram to hold most of the model weights. This allows training
        a much larger model on a smaller GPU, assuming you have enough CPU RAM. This is slower than training on pure GPU
        RAM, but CPU RAM is cheaper and upgradeable. You will still need GPU RAM to hold the optimizer states and LoRA
        weights, so a larger card is usually still needed.
        <br />
        <br />
        You can select the percentage of whole blocks to offload. It is generally best to offload as few as possible
        (close to 0%) for best performance, but you can offload more if you need the memory. Low VRAM is still a
        separate mode for broader component-level unloading and may be slower.
      </>
    ),
  },
  'model.qie.match_target_res': {
    title: 'Match Target Res',
    description: (
      <>
        This setting will make the control images match the resolution of the target image. The official inference
        example for Qwen-Image-Edit-2509 feeds the control image is at 1MP resolution, no matter what size you are
        generating. Doing this makes training at lower res difficult because 1MP control images are fed in despite how
        large your target image is. Match Target Res will match the resolution of your target to feed in the control
        images allowing you to use less VRAM when training with smaller resolutions. You can still use different aspect
        ratios, the image will just be resizes to match the amount of pixels in the target image.
      </>
    ),
  },
  'config.process[0].network.lokr_factor': {
    title: 'LoKr Factor',
    tooltip: 'Controls the Kronecker factorization shape; use -1 for automatic factorization.',
    description: (
      <>
        Controls the Kronecker factorization shape used by LoKr. With Upstream Factor enabled, positive factors keep the
        same ordering as upstream AI Toolkit. Use <code>-1</code> for automatic factorization, or set a specific factor
        such as <code>8</code> or <code>16</code> when you want a known LoKr layout.
      </>
    ),
  },
  'config.process[0].network.lokr_options': {
    title: 'LoKr Options',
    tooltip: 'Advanced LoKr behavior controls for capacity, scaling, compatibility, and quantized layers.',
    description: (
      <>
        Advanced LyCORIS LoKr settings. The defaults match upstream AI Toolkit for plain imported LoKr configs; only
        change these when you are matching a specific LoKr recipe or intentionally experimenting with capacity, scaling,
        factorization, or quantized-layer behavior.
      </>
    ),
  },
  'config.process[0].network.lokr_full_matrix': {
    title: 'Full Matrix',
    tooltip: 'Stores the larger Kronecker block as a full matrix; usually larger, sometimes higher capacity.',
    description: (
      <>
        Stores the larger LoKr Kronecker block as a full matrix instead of decomposing it. This can increase capacity
        but usually makes the adapter larger. The old <code>lokr_full_rank</code> option maps to this behavior.
      </>
    ),
  },
  'config.process[0].network.lokr_use_tucker': {
    title: 'Tucker Conv',
    tooltip: 'Uses Tucker decomposition for non-1x1 convolution LoKr weights.',
    description: (
      <>
        Uses Tucker decomposition for convolution kernels larger than 1x1. This is LyCORIS <code>use_tucker</code> and
        is mainly useful when training convolution layers with LoKr.
      </>
    ),
  },
  'config.process[0].network.lokr_use_scalar': {
    title: 'Scalar',
    tooltip: 'Trains an extra scalar multiplier in front of the LoKr weight difference.',
    description: (
      <>
        Trains an additional scalar multiplier in front of the LoKr weight difference and uses the LyCORIS scalar
        initialization path. It gives the adapter another learned strength control.
      </>
    ),
  },
  'config.process[0].network.lokr_decompose_both': {
    title: 'Decompose Both',
    tooltip: 'Low-rank decomposes both Kronecker blocks; smaller adapters, potentially less capacity.',
    description: (
      <>
        Low-rank decomposes both LoKr Kronecker blocks instead of only decomposing the larger block. This can shrink the
        adapter, but it can also reduce capacity.
      </>
    ),
  },
  'config.process[0].network.lokr_weight_decompose': {
    title: 'DoRA',
    tooltip: 'Enables DoRA-style weight decomposition with a learned magnitude scale.',
    description: (
      <>
        Enables LyCORIS weight decomposition, also known as DoRA. The adapter learns a magnitude scale in addition to
        the LoKr direction, which may improve stability for some recipes.
      </>
    ),
  },
  'config.process[0].network.lokr_wd_on_output': {
    title: 'DoRA On Output',
    tooltip: 'When DoRA is enabled, scales output channels instead of the input side.',
    description: (
      <>
        Chooses the normalization axis for LoKr DoRA. When enabled, DoRA scales output channels; when disabled, it scales
        the input side. This only applies when DoRA is enabled.
      </>
    ),
  },
  'config.process[0].network.lokr_bypass_mode': {
    title: 'Bypass Mode',
    tooltip: 'Computes base and LoKr delta outputs separately, useful for quantized or low-bit layers.',
    description: (
      <>
        Computes the base layer output and LoKr delta output separately instead of building a merged weight for the
        forward pass. This is intended for quantized or low-bit layers where merging can be awkward.
      </>
    ),
  },
  'config.process[0].network.lokr_rs_lora': {
    title: 'Rank-Stabilized',
    tooltip: 'Uses sqrt(rank) scaling so higher ranks start less weakly.',
    description: (
      <>
        Uses rank-stabilized scaling for the LoKr alpha ratio, dividing by the square root of rank instead of rank. This
        can make higher-rank adapters behave less weakly at initialization.
      </>
    ),
  },
  'config.process[0].network.lokr_rank_dropout_scale': {
    title: 'Rank Dropout Scale',
    tooltip: 'Rescales remaining rows during rank dropout to preserve expected adapter strength.',
    description: (
      <>
        Rescales the remaining LoKr rows during rank dropout so the expected adapter strength stays closer to the
        no-dropout value. This only matters when rank dropout is configured.
      </>
    ),
  },
  'config.process[0].network.lokr_unbalanced_factorization': {
    title: 'Unbalanced Factor',
    tooltip: 'Swaps output factor order for specific LyCORIS layouts or checkpoint compatibility.',
    description: (
      <>
        Swaps the output-side LoKr factor order after factorization. This changes how the Kronecker blocks are shaped
        and is mostly useful for matching specific LyCORIS experiments or checkpoints.
      </>
    ),
  },
  'config.process[0].network.lokr_legacy_factorization': {
    title: 'Upstream Factor',
    tooltip: 'Uses upstream AI Toolkit LoKr factor ordering for imported configs or checkpoints.',
    description: (
      <>
        Uses upstream AI Toolkit LoKr factor ordering instead of AITK Studio's balanced factorization. Keep this enabled
        for plain imported LoKr configs, or disable it only when you intentionally want the newer balanced layout.
      </>
    ),
  },
  'train.diff_output_preservation': {
    title: 'Differential Output Preservation',
    description: (
      <>
        Differential Output Preservation (DOP) is a technique to help preserve class of the trained concept during
        training. For this, you must have a trigger word set to differentiate your concept from its class. For instance,
        You may be training a woman named Alice. Your trigger word may be "Alice". The class is "woman", since Alice is
        a woman. We want to teach the model to remember what it knows about the class "woman" while teaching it what is
        different about Alice. During training, the trainer will make a prediction with your LoRA bypassed and your
        trigger word in the prompt replaced with the class word. Making "photo of Alice" become "photo of woman". This
        prediction is called the prior prediction. Each step, we will do the normal training step, but also do another
        step with this prior prediction and the class prompt in order to teach our LoRA to preserve the knowledge of the
        class. This should not only improve the performance of your trained concept, but also allow you to do things
        like "Alice standing next to a woman" and not make both of the people look like Alice.
      </>
    ),
  },
  'train.blank_prompt_preservation': {
    title: 'Blank Prompt Preservation',
    description: (
      <>
        Blank Prompt Preservation (BPP) is a technique to help preserve the current models knowledge when unprompted.
        This will not only help the model become more flexible, but will also help the quality of your concept during
        inference, especially when a model uses CFG (Classifier Free Guidance) on inference. At each step during
        training, a prior prediction is made with a blank prompt and with the LoRA disabled. This prediction is then
        used as a target on an additional training step with a blank prompt, to preserve the model's knowledge when no
        prompt is given. This helps the model to not overfit to the prompt and retain its generalization capabilities.
      </>
    ),
  },
  'train.sega_distill': {
    title: 'SEGA Distillation',
    description: (
      <>
        Runs an online frozen teacher with SEGA-modulated Flux2 RoPE and adds a teacher-matching loss beside the normal
        dataset loss. This is currently limited to FLUX.2, FLUX.2 Klein, and Z-Image LoRA training and adds an extra
        teacher prediction each step.
      </>
    ),
  },
  'train.sega_distill_weight': {
    title: 'SEGA Distillation Weight',
    description: (
      <>
        Multiplies the auxiliary SEGA teacher matching loss before it is added to the normal dataset loss.
      </>
    ),
  },
  'train.sega_distill_base_resolution': {
    title: 'SEGA Base Resolution',
    description: (
      <>
        Resolution used as the identity point for SEGA scaling. Larger training resolutions receive stronger
        frequency-aware RoPE scaling.
      </>
    ),
  },
  'train.sega_distill_strength': {
    title: 'SEGA Strength',
    description: (
      <>
        Controls how strongly the latent FFT energy changes the teacher RoPE scaling. Set to zero for identity scaling.
      </>
    ),
  },
  'train.sega_distill_scale': {
    title: 'SEGA Scale Clamp',
    description: (
      <>
        Minimum and maximum bounds for the RoPE scale factors produced by the SEGA teacher.
      </>
    ),
  },
  'train.sega_distill_on_reg': {
    title: 'Apply SEGA To Reg',
    description: (
      <>
        Applies SEGA distillation to regularization batches. By default regularization batches keep the normal training
        target.
      </>
    ),
  },
  'train.do_differential_guidance': {
    title: 'Differential Guidance',
    description: (
      <>
        Differential Guidance will amplify the difference of the model prediction and the target during training to make
        a new target. Differential Guidance Scale will be the multiplier for the difference. This is still experimental,
        but in my tests, it makes the model train faster, and learns details better in every scenario I have tried with
        it.
        <br />
        <br />
        The idea is that normal training inches closer to the target but never actually gets there, because it is
        limited by the learning rate. With differential guidance, we amplify the difference for a new target beyond the
        actual target, this would make the model learn to hit or overshoot the target instead of falling short.
        <br />
        <br />
        <img src="/imgs/diff_guidance.png" alt="Differential Guidance Diagram" className="max-w-full mx-auto" />
      </>
    ),
  },
  'dataset.num_repeats': {
    title: 'Num Repeats',
    description: (
      <>
        Number of Repeats will allow you to repeate the items in a dataset multiple times. This is useful when you are
        using multiple datasets and want to balance the number of samples from each dataset. For instance, if you have a
        small dataset of 10 images and a large dataset of 100 images, you can set the small dataset to have 10 repeats
        to effectively make it 100 images, making the two datasets occour equally during training.
      </>
    ),
  },
  'train.timestep_type': {
    title: 'Timestep Type',
    description: (
      <>
        Controls how training timesteps are sampled. LTX-2 can use <code>shifted_logit_normal</code>, which follows the
        Lightricks trainer sampler by shifting the logit-normal distribution based on the latent token count.
      </>
    ),
  },
  'train.audio_loss_multiplier': {
    title: 'Audio Loss Multiplier',
    description: (
      <>
        When training audio and video, sometimes the video loss is so great that it outweights the audio loss, causing
        the audio to become distorted. If you are noticing this happen, you can increase the audio loss multiplier to
        give more weight to the audio loss. You could try something like 2.0, 10.0 etc. Warning, setting this too high
        could overfit and damage the model.
      </>
    ),
  },
  'train.ltx_strategy': {
    title: 'LTX Strategy',
    description: (
      <>
        Optional advanced LTX-2 strategy object. This supports generated video, generated or frozen audio,
        and <code>first_frame</code>, <code>prefix</code>, or <code>suffix</code> conditions. Reference IC-LoRA, masks,
        spatial crops, and audio-only/V2A training are not supported yet.
      </>
    ),
  },
  'datasets.auto_frame_count': {
    title: 'Auto Frame Count',
    description: (
      <>
        This will automatically determine the number of frames to use for each video in your dataset instead of relying
        on a fixed num_frames. This allows you to include videos of different lengths in the dataset, and each video
        will be processed without speeding up or slowing down. Be careful about adding long videos into your dataset, as
        they use up more VRAM. This currently will not work with a batch size greater than 1.
      </>
    ),
  },
};

export const getDoc = (key: string | null | undefined): ConfigDoc | null => {
  if (key && key in docs) {
    return docs[key];
  }
  return null;
};

export default docs;
