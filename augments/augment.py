import random
from typing import List
from imgaug.augmenters import AdditiveGaussianNoise, Rotate, PerspectiveTransform, CropAndPad, Affine
from imgaug.augmenters.imgcorruptlike import ElasticTransform, Fog
from builders.meta_image import to_bounding_boxes_on_image, to_labeled_boxes
from struts.meta_image import MetaImage


def augment(m_images: List[MetaImage]) -> List[MetaImage]:
    augmentations: List[MetaImage] = []
    for m_image in m_images:
        augmentations.append(m_image)
        augmentations.append(augment_perspective(m_image))
        augmentations.append(augment_affine(augment_elastic_transformation(m_image)))
        augmentations.append(augment_affine(augment_fog(m_image)))
        augmentations.append(augment_affine(augment_noise(m_image)))
    return augmentations


def augment_affine(m_image: MetaImage) -> MetaImage:
    scale = random.randrange(35, 100) / 100
    rotate = random.randint(0, 359)
    shear_x = random.randint(-45, 45)
    shear_y = random.randint(-45, 45)
    affine = Affine(scale=scale, rotate=rotate, shear=[shear_x, shear_y], mode='edge', fit_output=True)
    image_aug, bbs_aug = affine(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_aff{}-{}-{}-{}'.format(m_image.name, int(scale * 100), rotate, shear_x, shear_y)
    return MetaImage(image_name, image_aug, bbs_aug)


def augment_elastic_transformation(m_image: MetaImage) -> MetaImage:
    augmentation = ElasticTransform(severity=[1, 1])
    augmented_image = augmentation(image=m_image.data)
    image_name = '{}_elas'.format(m_image.name)
    return MetaImage(image_name, augmented_image, m_image.labeled_boxes)


def augment_fog(m_image: MetaImage) -> MetaImage:
    augmentation = Fog(severity=1)
    augmented_image = augmentation(image=m_image.data)
    image_name = '{}_fog'.format(m_image.name)
    return MetaImage(image_name, augmented_image, m_image.labeled_boxes)


def augment_perspective(m_image: MetaImage) -> MetaImage:
    image_aug, bbs_aug = PerspectiveTransform(scale=(.04, .12), fit_output=True)\
        (image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_pers'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))


def augment_noise(m_image: MetaImage) -> MetaImage:
    image_aug, bbs_aug = AdditiveGaussianNoise(scale=(10, 30))\
        (image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_noise'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))


def augment_rot(m_image: MetaImage, base_angle: int) -> MetaImage:
    angles = (base_angle - 15, base_angle + 15)
    image_aug, bbs_aug = Rotate(angles)(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_rot{}'.format(m_image.name, base_angle)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))
