from typing import List

from imgaug.augmenters import AdditiveGaussianNoise, ScaleX, ScaleY, ShearX, ShearY, Rotate, PerspectiveTransform, \
    CropAndPad
from imgaug.augmenters.imgcorruptlike import ElasticTransform, Fog

from builders.meta_image import to_bounding_boxes_on_image, to_labeled_boxes
from struts.meta_image import MetaImage


def augment(m_images: List[MetaImage]) -> List[MetaImage]:
    augmentations: List[MetaImage] = []

    # # Add augmentations
    for m_image in m_images:
        augmentations.append(augment_elastic_transformation(m_image))
        augmentations.append(augment_fog(m_image))
        augmentations.append(augment_noise(m_image))
        augmentations.append(augment_scale_x(m_image))
        augmentations.append(augment_scale_y(m_image))
        augmentations.append(augment_shear_x(m_image))
        augmentations.append(augment_shear_y(m_image))
        augmentations.append(augment_perspective(m_image))

    # Rotate augmentations
    rotated_augmentations: List[MetaImage] = []
    for m_image in augmentations:
        rotated_augmentations.append(m_image)
        rotated_augmentations.append(augment_rot(m_image, 90))
        rotated_augmentations.append(augment_rot(m_image, 180))
        rotated_augmentations.append(augment_rot(m_image, 270))

    return rotated_augmentations


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
    image_aug, bbs_aug = CropAndPad(percent=.25)(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_aug, bbs_aug = PerspectiveTransform(scale=(.08, .12))(image=image_aug, bounding_boxes=bbs_aug)
    image_name = '{}_pers'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))


def augment_scale_x(m_image: MetaImage) -> MetaImage:
    image_aug, bbs_aug = ScaleX((0.75, 1.25))(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_scalex'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))


def augment_scale_y(m_image: MetaImage) -> MetaImage:
    image_aug, bbs_aug = ScaleY((0.75, 1.25))(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_scaley'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))


def augment_shear_x(m_image: MetaImage) -> MetaImage:
    image_aug, bbs_aug = ShearX((-20, 20))(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_shearx'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))


def augment_shear_y(m_image: MetaImage) -> MetaImage:
    image_aug, bbs_aug = ShearY((-20, 20))(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_sheary'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))


def augment_noise(m_image: MetaImage) -> MetaImage:
    image_aug, bbs_aug = AdditiveGaussianNoise(scale=(10, 30))(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_noise'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))


def augment_rot(m_image: MetaImage, base_angle: int) -> MetaImage:
    angles = (base_angle - 15, base_angle + 15)
    image_aug, bbs_aug = Rotate(angles)(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_rot{}'.format(m_image.name, base_angle)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))
