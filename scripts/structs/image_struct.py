import numpy as np


class BoundingBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class ImageStruct:
    def __init__(self, name: str, data, bounding_box: BoundingBox):
        self.name = name
        self.data = data
        self.bounding_box = bounding_box
