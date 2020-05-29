from pathlib import Path
import imageio


def list_source_image_paths() -> Path:
    source_dir = Path.cwd().joinpath('images').joinpath('source')
    return [p for p in source_dir.glob('*.jpg') if p.is_file()]


def read_images(source_file_paths):
    return [imageio.imread(f) for f in source_file_paths]
