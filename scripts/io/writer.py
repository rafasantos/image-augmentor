from pathlib import Path
from typing import List

from PIL import Image

from scripts.builders.pascal import build_label_file_text
from scripts.structs.meta_image import MetaImage


def write_images(m_images: List[MetaImage]):
    for m_image in m_images:
        image_pil = Image.fromarray(m_image.data.astype('uint8'))
        target_image_path = Path.cwd().joinpath('images').joinpath('target').joinpath(m_image.name)
        image_pil.save('{}.jpg'.format(str(target_image_path)))


def write_labels(m_images: List[MetaImage]):
    for m_image in m_images:
        target_label_path = Path.cwd()\
            .joinpath('images')\
            .joinpath('target')\
            .joinpath(m_image.name + '.xml')
        label_text = build_label_file_text(m_image)
        target_label_path.write_text(label_text)
