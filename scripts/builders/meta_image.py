from pathlib import Path
from typing import List
from imgaug import BoundingBoxesOnImage, BoundingBox, imageio
from scripts.builders.pascal import build_labeled_boxes
from scripts.structs.meta_image import MetaImage, LabeledBox


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


def build_meta_image(image_paths: List[Path]) -> List[MetaImage]:
    result: List[MetaImage] = []
    for image_path in image_paths:
        image = imageio.imread(image_path)
        image_name = image_path.name.replace('.jpg', '')
        label_path = image_path.parent.joinpath(image_name + '.xml')
        image_boxes = build_labeled_boxes(label_path)
        result.append(MetaImage(image_name, image, image_boxes))
    return result

