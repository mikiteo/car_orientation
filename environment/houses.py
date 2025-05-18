# environment/houses.py
import cv2 as cv
import numpy as np

class Houses:
    def __init__(self, img: np.ndarray, height: int):
        self.img = img
        self.height = height

    def draw(self):
        for y in range(0, self.height, 120):
            cv.rectangle(self.img, (50, y + 20), (150, y + 100), (0, 255, 255), -1)
            cv.rectangle(self.img, (50, y + 20), (150, y + 100), (0, 0, 0), 2)
            cv.rectangle(self.img, (650, y + 20), (750, y + 100), (0, 255, 255), -1)
            cv.rectangle(self.img, (650, y + 20), (750, y + 100), (0, 0, 0), 2)
