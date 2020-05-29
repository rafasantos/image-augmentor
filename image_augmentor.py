from pathlib import Path
from typing import List

from augments.augment import augment
from builders.meta_image import build_meta_image
from io_utils.reader import list_source_image_paths
from io_utils.writer import write_images, write_labels
from struts.meta_image import MetaImage


def main(args=None):
    print('Starting augmentation...')
    source_image_paths: List[Path] = list_source_image_paths()
    print('Files to augment:')
    print(source_image_paths)
    m_image: List[MetaImage] = build_meta_image(source_image_paths)
    print('Augmenting images')
    m_image = augment(m_image)
    print('Writing augmented image files')
    write_images(m_image)
    print('Writing label files')
    write_labels(m_image)
    print('Augmentation completed. {} images augmented'.format(len(source_image_paths)))


if __name__ == '__main__':
    main()
