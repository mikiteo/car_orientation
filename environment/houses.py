# environment/houses.py
import cv2 as cv
import numpy as np
from config import CONFIG

class Houses:
    def __init__(self, img: np.ndarray, height: int):
        self.img = img
        self.height = height
        self.cfg = CONFIG

        self.road_x_start = (self.cfg["width"] - self.cfg["road_width"]) // 2
        self.road_x_end = self.road_x_start + self.cfg["road_width"]

        self.house_width = 80
        self.margin = 20 

        self.left_x1 = self.road_x_start - self.margin - self.house_width
        self.left_x2 = self.road_x_start - self.margin

        self.right_x1 = self.road_x_end + self.margin
        self.right_x2 = self.road_x_end + self.margin + self.house_width

    def draw(self):
        for y in range(0, self.height, 120):
            cv.rectangle(self.img, (self.left_x1, y + 20), (self.left_x2, y + 100), (0, 255, 255), -1)
            cv.rectangle(self.img, (self.left_x1, y + 20), (self.left_x2, y + 100), (0, 0, 0), 2)

            cv.rectangle(self.img, (self.right_x1, y + 20), (self.right_x2, y + 100), (0, 255, 255), -1)
            cv.rectangle(self.img, (self.right_x1, y + 20), (self.right_x2, y + 100), (0, 0, 0), 2)
