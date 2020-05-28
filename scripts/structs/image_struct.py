from typing import List


class LabeledBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class ImageStruct:
    def __init__(self, name: str, data, labeled_boxes: List[LabeledBox]):
        self.name = name
        self.data = data
        self.labeled_boxes = labeled_boxes
