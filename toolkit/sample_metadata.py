"""Recorded generation settings and transactional publication of sample files."""
import json
import hashlib
import os
from pathlib import Path
import shutil


def write_sample_metadata(config, media_path, count):
    """Allowlist effective scalar settings; never serialize arbitrary model kwargs."""
    prompt_identity = json.dumps({
        'prompt': config.prompt,
        'prompt_2': config.prompt_2,
        'negative_prompt': config.negative_prompt or '',
        'negative_prompt_2': config.negative_prompt_2 or '',
        'controls': config.ctrl_imgs or [config.ctrl_img_1, config.ctrl_img_2, config.ctrl_img_3],
    }, sort_keys=True, ensure_ascii=False)
    metadata = {
        'version': 1,
        'prompt': config.prompt,
        'prompt_2': config.prompt_2,
        'negative_prompt': config.negative_prompt or '',
        'negative_prompt_2': config.negative_prompt_2,
        'seed': config.seed,
        'width': config.width,
        'height': config.height,
        'guidance_scale': config.guidance_scale,
        'guidance_rescale': config.guidance_rescale,
        'sample_steps': config.num_inference_steps,
        'network_multiplier': config.network_multiplier,
        'num_frames': config.num_frames,
        'fps': config.fps,
        'step': config.sample_step,
        'prompt_index': config.sample_prompt_index if config.sample_prompt_index is not None else count,
        'sampler': config.sample_sampler,
        'iteration_id': config.sample_iteration_id,
        'prompt_identity': hashlib.sha256(prompt_identity.encode('utf-8')).hexdigest(),
    }
    with open(str(media_path) + '.metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, allow_nan=False)


def publish_sample_files(staging_folder, destination_folder, media_filename, thumbnail=None):
    """Commit metadata before media, restoring previous sidecars on any failure."""
    staging = Path(staging_folder).resolve(strict=True)
    destination = Path(destination_folder).resolve(strict=True)
    if Path(media_filename).name != media_filename:
        raise ValueError('Sample media filename must be a basename')
    rollback = staging / '.rollback'
    rollback.mkdir()
    published = []

    def publish(source, target):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError('Staged sample must be a regular file')
        if target.is_symlink() or target.parent.resolve(strict=True) != target.parent:
            raise ValueError('Sample publication target must not be a symbolic link')
        previous = None
        if target.exists():
            if not target.is_file():
                raise ValueError('Sample publication target must be a regular file')
            previous = rollback / str(len(published))
            shutil.copyfile(target, previous)
        os.replace(str(source), str(target))
        published.append((target, previous))

    try:
        if thumbnail:
            source, relative_target = thumbnail
            if Path(relative_target).is_absolute() or '..' in Path(relative_target).parts:
                raise ValueError('Sample thumbnail path must stay inside the destination')
            target = destination / relative_target
            target.parent.mkdir(exist_ok=True)
            publish(source, target)
        for source in sorted(staging.iterdir()):
            if source.is_file() and source.name not in (media_filename, media_filename + '.thumb'):
                publish(source, destination / source.name)
        publish(staging / media_filename, destination / media_filename)
    except BaseException:
        for target, previous in reversed(published):
            if previous is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(str(previous), str(target))
        raise
