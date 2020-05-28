from typing import List

import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pathlib

from scripts.structs.image_struct import ImageStruct


def main(args=None):
    source_image_paths = build_source_file_paths()
    images = read_images(source_image_paths)
    # TODO: List of BoundingBoxes
    image_structs = build_image_structs(source_image_paths, images)
    image_structs = augment(image_structs)
    save_images(image_structs)


def save_images(image_structs: List[ImageStruct]):
    for image_struct in image_structs:
        image_pil = Image.fromarray(image_struct.data)
        target_image_path = pathlib.Path.cwd().joinpath('images').joinpath('target').joinpath(image_struct.name)
        image_pil.save('{}.jpg'.format(str(target_image_path)))


def augment(image_structs: List[ImageStruct]):
    augmentations: List[ImageStruct] = []
    # Augmentations
    for image_struct in image_structs:
        noise_images_structs = augment_noise(image_struct)
        augmentations.extend(noise_images_structs)

    # Rotate augmentations
    result: List[ImageStruct] = []
    for image_struct in augmentations:
        rotated_images_by_name = augment_rotation(image_struct)
        result.extend(rotated_images_by_name)

    return result


def augment_rotation(image_struct: ImageStruct):
    result: List[ImageStruct] = []
    images = [image_struct.data, image_struct.data, image_struct.data, image_struct.data]
    rotate = iaa.Affine(rotate=(-30, 30))
    augmentations = rotate(images=images)
    for i in range(len(augmentations)):
        image_name = '{}_rot_{}'.format(image_struct.name, i)
        result.append(ImageStruct(image_name, augmentations[i], None))
    return result


def augment_noise(image_struct: ImageStruct):
    images = [image_struct.data, image_struct.data, image_struct.data]
    gaussian_noise = iaa.AdditiveGaussianNoise(scale=(10, 30))
    augmented_images = gaussian_noise(images=images)
    result: List[ImageStruct] = []
    for i in range(len(augmented_images)):
        image_name = '{}_ns_{}'.format(image_struct.name, i)
        result.append(ImageStruct(image_name, augmented_images[i], None))
    return result


def build_image_structs(image_paths, images):
    result: List[ImageStruct] = []
    for i in range(len(image_paths)):
        image_name = image_paths[i].name.replace('.jpg', '')
        result.append(ImageStruct(image_name, images[i], None))
    return result


def read_images(source_file_paths):
    return [imageio.imread(f) for f in source_file_paths]


def build_source_file_paths():
    source_dir = pathlib.Path.cwd().joinpath('images').joinpath('source')
    return [p for p in source_dir.glob('*.jpg') if p.is_file()]

if __name__ == '__main__':
    main()
