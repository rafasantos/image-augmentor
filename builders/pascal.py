import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from struts.labeled_box import LabeledBox
from struts.meta_image import MetaImage


def build_labeled_boxes(pascal_xml_paths: Path) -> List[LabeledBox]:
    result: List[LabeledBox] = []
    tree = ET.parse(pascal_xml_paths)
    root = tree.getroot()
    for obj in root.iter('object'):
        x1 = y1 = x2 = y2 = 0
        label = obj.find('name').text
        for bndbox in obj.iter('bndbox'):
            x1 = int(bndbox.find('xmin').text)
            y1 = int(bndbox.find('ymin').text)
            x2 = int(bndbox.find('xmax').text)
            y2 = int(bndbox.find('ymax').text)
        result.append(LabeledBox(x1, y1, x2, y2, label))

    return result


def build_label_file_text(m_image: MetaImage) -> str:
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'FOLDER'
    filename = ET.SubElement(annotation, 'filename')
    filename.text = m_image.name + '.jpg'
    path = ET.SubElement(annotation, 'path')
    path.text = filename.text
    size = ET.SubElement(annotation, 'size')
    size_height = ET.SubElement(size, 'height')
    size_height.text = str(m_image.data.shape[0])
    size_width = ET.SubElement(size, 'width')
    size_width.text = str(m_image.data.shape[1])
    size_depth = ET.SubElement(size, 'depth')
    size_depth.text = str(m_image.data.shape[2])
    for labeled_box in m_image.labeled_boxes:
        obj = ET.SubElement(annotation, 'object')
        obj_name = ET.SubElement(obj, 'name')
        obj_name.text = labeled_box.label
        obj_pose = ET.SubElement(obj, 'pose')
        obj_pose.text = 'Unspecified'
        obj_truncated = ET.SubElement(obj, 'truncated')
        obj_truncated.text = '0'
        obj_difficult = ET.SubElement(obj, 'difficult')
        obj_difficult.text = '0'
        bdnbox = ET.SubElement(obj, 'bndbox')
        bdnbox_xmin = ET.SubElement(bdnbox, 'xmin')
        bdnbox_xmin.text = str(int(labeled_box.x1))
        bdnbox_ymin = ET.SubElement(bdnbox, 'ymin')
        bdnbox_ymin.text = str(int(labeled_box.y1))
        bdnbox_xmax = ET.SubElement(bdnbox, 'xmax')
        bdnbox_xmax.text = str(int(labeled_box.x2))
        bdnbox_ymax = ET.SubElement(bdnbox, 'ymax')
        bdnbox_ymax.text = str(int(labeled_box.y2))
    return ET.tostring(annotation, encoding='utf8', method='xml').decode()
