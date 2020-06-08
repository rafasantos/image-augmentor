import random
from typing import List
from imgaug.augmenters import AdditiveGaussianNoise, Rotate, PerspectiveTransform, CropAndPad, Affine, Resize
from imgaug.augmenters.imgcorruptlike import ElasticTransform, Fog
from builders.meta_image import to_bounding_boxes_on_image, to_labeled_boxes
from struts.meta_image import MetaImage


def augment(m_images: List[MetaImage]) -> List[MetaImage]:
    augmentations: List[MetaImage] = []
    for crop in m_images:
        crop = augment_crop_and_pad(crop)
        augmentations.append(augment_resize(crop))

        perspective = augment_perspective(crop)
        perspective = augment_resize(perspective)
        perspective = augment_rot(perspective)
        augmentations.append(perspective)

        elastic_transformation = augment_elastic_transformation(crop)
        elastic_transformation = augment_resize(elastic_transformation)
        elastic_transformation = augment_rot(elastic_transformation)
        augmentations.append(elastic_transformation)

        # fog = augment_fog(crop)
        # fog = augment_resize(fog)
        # augmentations.append(fog)

        noise = augment_noise(crop)
        noise = augment_rot(noise)
        noise = augment_resize(noise)
        augmentations.append(noise)

        affine = augment_affine(crop)
        affine = augment_resize(affine)
        augmentations.append(affine)
    return augmentations


def augment_resize(m_image: MetaImage) -> MetaImage:
    augmentation = Resize({"longer-side": 1200, "shorter-side": "keep-aspect-ratio"})
    image_aug, bbs_aug = augmentation(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_res'.format(m_image.name)
    return MetaImage(image_name, image_aug, bbs_aug)


def augment_crop_and_pad(m_image: MetaImage) -> MetaImage:
    percent = -.075
    augmentation = CropAndPad(percent=percent, keep_size=False)
    image_aug, bbs_aug = augmentation(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_cnp'.format(m_image.name)
    return MetaImage(image_name, image_aug, bbs_aug)


def augment_affine(m_image: MetaImage) -> MetaImage:
    scale = random.randrange(35, 100) / 100
    rotate = random.randint(0, 359)
    shear_x = random.randint(-35, 35)
    shear_y = random.randint(-35, 35)
    affine = Affine(scale=scale, rotate=rotate, shear=[shear_x, shear_y], mode='edge', fit_output=True)
    image_aug, bbs_aug = affine(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_aff{}-{}-{}-{}'.format(m_image.name, int(scale * 100), rotate, shear_x, shear_y)
    return MetaImage(image_name, image_aug, bbs_aug)


def augment_elastic_transformation(m_image: MetaImage) -> MetaImage:
    augmentation = ElasticTransform(severity=[5, 5])
    augmented_image = augmentation(image=m_image.data)
    image_name = '{}_elas'.format(m_image.name)
    return MetaImage(image_name, augmented_image, m_image.labeled_boxes)


def augment_fog(m_image: MetaImage) -> MetaImage:
    augmentation = Fog(severity=1)
    augmented_image = augmentation(image=m_image.data)
    image_name = '{}_fog'.format(m_image.name)
    return MetaImage(image_name, augmented_image, m_image.labeled_boxes)


def augment_perspective(m_image: MetaImage) -> MetaImage:
    image_aug, bbs_aug = PerspectiveTransform(scale=(.07, .10), fit_output=True)\
        (image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_pers'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))


def augment_noise(m_image: MetaImage) -> MetaImage:
    image_aug, bbs_aug = AdditiveGaussianNoise(scale=(10, 30))\
        (image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_noise'.format(m_image.name)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))


def augment_rot(m_image: MetaImage, base_angle=0) -> MetaImage:
    angles = (base_angle - 10, base_angle + 10)
    image_aug, bbs_aug = Rotate(angles)(image=m_image.data, bounding_boxes=to_bounding_boxes_on_image(m_image))
    image_name = '{}_rot{}'.format(m_image.name, base_angle)
    return MetaImage(image_name, image_aug, to_labeled_boxes(bbs_aug))
