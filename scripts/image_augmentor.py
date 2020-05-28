import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pathlib


def main(args=None):
    source_image_paths = build_source_file_paths()
    images = read_images(source_image_paths)
    source_images_by_name = build_images_by_name(source_image_paths, images)
    augmented_images_by_name = augment(source_images_by_name)
    save_images(augmented_images_by_name)


def save_images(augmented_images_by_name):
    for image_name in augmented_images_by_name:
        image_array = augmented_images_by_name[image_name]
        image_pil = Image.fromarray(image_array)
        target_image_path = pathlib.Path.cwd().joinpath('images').joinpath('target').joinpath(image_name)
        image_pil.save(target_image_path)


def augment(images_by_name):
    result = {}
    for image_name in images_by_name:
        image = images_by_name[image_name]
        rotated_images_by_name = augment_rotation(image_name, image)
        result.update(rotated_images_by_name)
    return result


def augment_rotation(image_name, image):
    result = {}
    images = [image, image, image, image, image]
    rotate = iaa.Affine(rotate=(-30, 30))
    augmented_images = rotate(images=images)
    for i in range(len(augmented_images)):
        augmented_image_name = '{}_rotation_{}.jpg'.format(image_name, i)
        result[augmented_image_name] = augmented_images[i]
    return result


def build_images_by_name(image_paths, images):
    result = {}
    for i in range(len(image_paths)):
        image_name = image_paths[i].name.replace('.jpg', '')
        result[image_name] = images[i]
    return result


def read_images(source_file_paths):
    return [imageio.imread(f) for f in source_file_paths]


def build_source_file_paths():
    source_dir = pathlib.Path.cwd().joinpath('images').joinpath('source')
    return [p for p in source_dir.glob('*') if p.is_file()]

if __name__ == '__main__':
    main()
