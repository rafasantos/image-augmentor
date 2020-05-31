from pathlib import Path
from typing import List

from PIL import Image, ExifTags
from imgaug import BoundingBoxesOnImage, BoundingBox, imageio
from imgaug.augmenters import Rotate

from builders.pascal import build_labeled_boxes
from struts.labeled_box import LabeledBox
from struts.meta_image import MetaImage


def to_bounding_boxes_on_image(m_image: MetaImage) -> BoundingBoxesOnImage:
    bounding_boxes: List[BoundingBox] = []
    for lbox in m_image.labeled_boxes:
        bounding_boxes.append(BoundingBox(lbox.x1, lbox.y1, lbox.x2, lbox.y2, lbox.label))
    return BoundingBoxesOnImage(bounding_boxes, m_image.data.shape)


def to_labeled_boxes(bounding_boxes_on_image: BoundingBoxesOnImage) -> List[LabeledBox]:
    result: List[LabeledBox] = []
    for bbox in bounding_boxes_on_image.bounding_boxes:
        result.append(LabeledBox(bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.label))
    return result


def __obtain_exif_rotation(image_path: Path) -> int:
    image = Image.open(image_path)
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    exif = dict(image._getexif().items())
    if exif[orientation] == 3:
        return 180
    elif exif[orientation] == 6:
        return 270
    elif exif[orientation] == 8:
        return 90
    return None


def __rotate(exif_rotation: int, image):
    if exif_rotation:
        return Rotate(rotate=exif_rotation, fit_output=True)(image=image)
    else:
        return image


def build_meta_image(image_paths: List[Path]) -> List[MetaImage]:
    result: List[MetaImage] = []
    for image_path in image_paths:
        image = imageio.imread(image_path)
        exif_rotation = __obtain_exif_rotation(image_path)
        image = __rotate(exif_rotation, image)
        image_name = image_path.name.replace('.jpg', '')
        label_path = image_path.parent.joinpath(image_name + '.xml')
        image_boxes = build_labeled_boxes(label_path)
        result.append(MetaImage(image_name, image, image_boxes))
    return result
