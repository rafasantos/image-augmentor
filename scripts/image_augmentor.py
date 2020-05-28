import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pathlib

def main(args=None):
    source_file_paths = obtain_source_file_paths()
    images = read_images(source_file_paths)
    rotate = iaa.Affine(rotate=(-25, 25))
    image_aug = rotate(image=images[0])
    target_image_name = source_file_paths[0].name
    image_obj = Image.fromarray(image_aug)
    target_image_path = pathlib.Path.cwd().joinpath('images').joinpath('target').joinpath(target_image_name)
    image_obj.save(target_image_path)

def read_images(source_file_paths):
    images = []
    for f in source_file_paths:
        images.append(imageio.imread(f))
    return images

def obtain_source_file_paths():
    source_dir = pathlib.Path.cwd().joinpath('images').joinpath('source')
    source_files = [p for p in source_dir.glob('*') if p.is_file()]
    for f in source_files:
        print(f)
    return source_files


if __name__ == '__main__':
    main()