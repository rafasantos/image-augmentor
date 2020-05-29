from typing import List
from struts.labeled_box import LabeledBox


class MetaImage:
    def __init__(self, name: str, data, labeled_boxes: List[LabeledBox]):
        self.name = name
        self.data = data
        self.labeled_boxes = labeled_boxes
