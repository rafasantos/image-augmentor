import random
from typing import List

from imgaug.augmenters import Affine, AdditiveGaussianNoise, ScaleX, ScaleY
from imgaug.augmenters.imgcorruptlike import ElasticTransform, Fog

from builders.meta_image import to_bounding_boxes_on_image, to_labeled_boxes
from struts.meta_image import MetaImage


def augment(m_images: List[MetaImage]) -> List[MetaImage]:
    augmentations: List[MetaImage] = []
    # Augmentations
    for m_image in m_images:
        augmentations.append(augment_elastic_transformation(m_image))
        augmentations.append(augment_fog(m_image))
        augmentations.extend(augment_noise(m_image))
        augmentations.append(augment_scale_x(m_image))
        augmentations.append(augment_scale_y(m_image))

    # Rotate augmentations
    result: List[MetaImage] = []
    for m_image in augmentations:
        rotated_m_images = augment_rotation(m_image)
        result.extend(rotated_m_images)

    return result
    return augmentations


def augment_elastic_transformation(m_image: MetaImage) -> MetaImage:
    augmentation = ElasticTransform(severity=[1, 1])
    augmented_image = augmentation(image=m_image.data)
    image_name = '{}_elas_'.format(m_image.name)
    return MetaImage(image_name, augmented_image, m_image.labeled_boxes)


def augment_fog(m_image: MetaImage) -> MetaImage:
    severity = random.randint(1, 3)
    augmentation = Fog(severity=severity)
    augmented_image = augmentation(image=m_image.data)
    image_name = '{}_fog_'.format(m_image.name)
    return MetaImage(image_name, augmented_image, m_image.labeled_boxes)


def augment_scale_x(m_image: MetaImage) -> MetaImage:
    augmentation = ScaleX((0.75, 1.5))
    augmented_image = augmentation(image=m_image.data)
    image_name = '{}_scalex_'.format(m_image.name)
    return MetaImage(image_name, augmented_image, m_image.labeled_boxes)


def augment_scale_y(m_image: MetaImage) -> MetaImage:
    augmentation = ScaleY((0.75, 1.5))
    augmented_image = augmentation(image=m_image.data)
    image_name = '{}_scaley_'.format(m_image.name)
    return MetaImage(image_name, augmented_image, m_image.labeled_boxes)


def augment_noise(m_image: MetaImage):
    images = [m_image.data, m_image.data, m_image.data]
    gaussian_noise = AdditiveGaussianNoise(scale=(10, 30))
    augmented_images = gaussian_noise(images=images)
    result: List[MetaImage] = []
    for i in range(len(augmented_images)):
        image_name = '{}_ns_{}'.format(m_image.name, i)
        result.append(MetaImage(image_name, augmented_images[i], m_image.labeled_boxes))
    return result


def augment_rotation(m_image: MetaImage):
    result: List[MetaImage] = []
    rotate = Affine(rotate=(-30, 30))
    images = [m_image.data]
    boxes = [to_bounding_boxes_on_image(m_image)]
    augmentations, augmented_boxes = rotate(images=images, bounding_boxes=boxes)
    for i in range(len(augmentations)):
        image_name = '{}_rot_{}'.format(m_image.name, i)
        result.append(MetaImage(image_name, augmentations[i], to_labeled_boxes(augmented_boxes[i])))
    return result
