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
    source_images_by_name = build_images_by_name(images, source_image_paths)
    augmented_images_by_name = augment(source_images_by_name)
    save_images(augmented_images_by_name)


def save_images(augmented_images_by_name):
    for image_name in augmented_images_by_name:
        image = augmented_images_by_name[image_name]
        image_obj = Image.fromarray(image)
        target_image_path = pathlib.Path.cwd().joinpath('images').joinpath('target').joinpath(image_name)
        image_obj.save(target_image_path)


def augment(images_by_name):
    augmented_images_by_name = {}
    for image_name in images_by_name:
        image = images_by_name[image_name]
        image_aug = augment_rotation(image_name, image)
        for name in image_aug:
            augmented_images_by_name[name] = image_aug[name]
    return augmented_images_by_name


def augment_rotation(image_name, image):
    result = {}
    images = [image, image, image, image, image]
    rotate = iaa.Affine(rotate=(-30, 30))
    augmented_images = rotate(images=images)
    for i in range(len(augmented_images)):
        augmented_image_name = '{}_rotation_{}.jpg'.format(image_name, i)
        result[augmented_image_name] = augmented_images[i]
    return result


def build_images_by_name(images, source_image_paths):
    images_by_target_path = {}
    for i in range(len(source_image_paths)):
        images_by_target_path[source_image_paths[i].name.replace('.jpg', '')] = images[i]
    return images_by_target_path


def read_images(source_file_paths):
    images = []
    for f in source_file_paths:
        images.append(imageio.imread(f))
    return images


def build_source_file_paths():
    source_dir = pathlib.Path.cwd().joinpath('images').joinpath('source')
    source_files = [p for p in source_dir.glob('*') if p.is_file()]
    for f in source_files:
        print(f)
    return source_files


if __name__ == '__main__':
    main()
