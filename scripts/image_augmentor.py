from pathlib import Path
from typing import List

from scripts.augment.augment import augment
from scripts.builders.image_struct import build_image_structs
from scripts.io.reader import list_source_image_paths
from scripts.io.writer import write_images, write_labels
from scripts.structs.image_struct import ImageStruct


def main(args=None):
    source_image_paths: List[Path] = list_source_image_paths()
    image_structs: List[ImageStruct] = build_image_structs(source_image_paths)
    image_structs = augment(image_structs)
    write_images(image_structs)
    write_labels(image_structs)


if __name__ == '__main__':
    main()
