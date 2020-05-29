from pathlib import Path
from typing import List

from scripts.augment.augment import augment
from scripts.builders.meta_image import build_meta_image
from scripts.io.reader import list_source_image_paths
from scripts.io.writer import write_images, write_labels
from scripts.structs.meta_image import MetaImage


def main(args=None):
    source_image_paths: List[Path] = list_source_image_paths()
    m_image: List[MetaImage] = build_meta_image(source_image_paths)
    m_image = augment(m_image)
    write_images(m_image)
    write_labels(m_image)


if __name__ == '__main__':
    main()
