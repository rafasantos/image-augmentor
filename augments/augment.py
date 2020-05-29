import random
from typing import List

from imgaug.augmenters import AdditiveGaussianNoise, ScaleX, ScaleY, ShearX, ShearY, Rotate, PerspectiveTransform, \
    CropAndPad
from imgaug.augmenters.imgcorruptlike import ElasticTransform, Fog

from builders.meta_image import to_bounding_boxes_on_image, to_labeled_boxes
from struts.meta_image import MetaImage


def augment(m_images: List[MetaImage]) -> List[MetaImage]:
    augmentations: List[MetaImage] = []

    # Add augmentations
    for m_image in m_images:
        augmentations.append(augment_elastic_transformation(m_image))
        augmentations.append(augment_fog(m_image))
        augmentations.append(augment_noise(m_image))
        augmentations.append(augment_scale_x(m_image))
        augmentations.append(augment_scale_y(m_image))
        augmentations.append(augment_shear_x(m_image))
        augmentations.append(augment_shear_y(m_image))
        augmentations.append(augment_perspective(m_image))

    # Apply two combined augmentations at random
    # Do not augmentations that change the size of the image in this session because we don't wont to bleed the images
    for m_image in m_images:
        augmented_image = m_image
        for i in range(2):
            random_int = random.randint(0, 2)
            if random_int == 0:
                augmented_image = augment_elastic_transformation(augmented_image)
            elif random_int == 1:
                augmented_image = augment_fog(augmented_image)
            elif random_int == 2:
                augmented_image = augment_noise(augmented_image)
        augmentations.append(augmented_image)

    # Rotate augmentations
    result: List[MetaImage] = []
    for m_image in augmentations:
        result.append(augment_rotation(m_image))

    return result


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


def augment_rotation(m_image: MetaImage) -> MetaImage:
    image_aug, bbs_aug = Rotate((-30, 30))(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_rot'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))
