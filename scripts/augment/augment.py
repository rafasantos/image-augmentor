from typing import List
from imgaug.augmenters import Affine, AdditiveGaussianNoise
from scripts.builders.meta_image import to_bounding_boxes_on_image, to_labeled_boxes
from scripts.structs.meta_image import MetaImage


def augment(m_images: List[MetaImage]) -> List[MetaImage]:
    augmentations: List[MetaImage] = []
    # Augmentations
    for m_image in m_images:
        noise_images_structs = augment_noise(m_image)
        augmentations.extend(noise_images_structs)

    # Rotate augmentations
    result: List[MetaImage] = []
    for m_image in augmentations:
        rotated_m_images = augment_rotation(m_image)
        result.extend(rotated_m_images)

    return result


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
    images = [m_image.data, m_image.data, m_image.data, m_image.data]
    boxes = [to_bounding_boxes_on_image(m_image),
             to_bounding_boxes_on_image(m_image),
             to_bounding_boxes_on_image(m_image),
             to_bounding_boxes_on_image(m_image)]
    rotate = Affine(rotate=(-30, 30))
    augmentations, augmented_boxes = rotate(images=images, bounding_boxes=boxes)
    for i in range(len(augmentations)):
        image_name = '{}_rot_{}'.format(m_image.name, i)
        result.append(MetaImage(image_name, augmentations[i], to_labeled_boxes(augmented_boxes[i])))
    return result
