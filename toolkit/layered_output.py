"""A still-image layer set is distinct from a video or a batch of samples."""
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from PIL import Image


@dataclass(frozen=True)
class LayeredImageOutput:
    composite: Image.Image
    layers: list[Image.Image]

    def __post_init__(self):
        if not self.layers or len(self.layers) > 32:
            raise ValueError("A layered image requires between 1 and 32 layers")
        if any(layer.size != self.composite.size for layer in self.layers):
            raise ValueError("All generated layers must share the composite canvas")


def validate_num_layers(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32):
        raise ValueError("num_layers must be an integer between 1 and 32")
    return value


def publish_layered_output(staging_folder, destination_folder, media_filename, *, thumbnail=None):
    """Publish a staged sample, restoring replaced files if any commit step fails.

    Layer directories have unique IDs. Sidecars and thumbnails may replace an
    earlier sample at the same name, so their previous contents must survive a
    failure publishing the composite, which is always the final commit step.
    """
    staging = Path(staging_folder).resolve(strict=True)
    destination = Path(destination_folder).resolve(strict=True)
    if Path(media_filename).name != media_filename:
        raise ValueError('Layered sample media filename must be a basename')
    layer_root = staging / '.layers'
    output_layers = destination / '.layers'
    output_layers.mkdir(exist_ok=True)
    if output_layers.resolve() != output_layers:
        raise ValueError('Sample layer directory must not be a symbolic link')
    rollback = staging / '.rollback'
    rollback.mkdir()
    published_files = []
    published_directories = []

    def publish_file(source, target):
        if target.parent.resolve(strict=True) != target.parent or target.is_symlink():
            raise ValueError('Sample publication target must not be a symbolic link')
        previous = None
        if target.exists():
            if not target.is_file():
                raise ValueError('Sample publication target is not a regular file')
            previous = rollback / str(len(published_files))
            shutil.copyfile(target, previous)
        # Record only completed replacements. A failed os.replace leaves the
        # prior target intact, and rollback must not remove that untouched file.
        os.replace(source, target)
        published_files.append((target, previous))

    try:
        for entry in layer_root.iterdir():
            if not entry.is_dir() or entry.is_symlink():
                raise ValueError('Unexpected layered sample asset')
            target = output_layers / entry.name
            # Reserve this generated ID without replacing a pre-existing tree.
            target.mkdir()
            published_directories.append(target)
            for asset in entry.iterdir():
                if not asset.is_file() or asset.is_symlink():
                    raise ValueError('Unexpected layered sample asset')
                os.replace(asset, target / asset.name)
        if thumbnail is not None:
            source, relative_target = thumbnail
            thumb_target = destination / relative_target
            thumb_target.parent.mkdir(exist_ok=True)
            publish_file(Path(source), thumb_target)
        for source in sorted(staging.iterdir()):
            if source.is_file() and source.name not in (media_filename, media_filename + '.thumb'):
                publish_file(source, destination / source.name)
        publish_file(staging / media_filename, destination / media_filename)
    except BaseException:
        for target, previous in reversed(published_files):
            if previous is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(previous, target)
        for directory in reversed(published_directories):
            shutil.rmtree(directory)
        raise
