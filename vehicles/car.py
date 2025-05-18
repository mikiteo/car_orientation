# vehicles/car.py
from typing import List, Tuple
import cv2 as cv
import numpy as np

class Car:
    def __init__(self, car_id: int, position: List[int], direction: List[int], color: Tuple[int, int, int] = (255, 0, 0)):
        self.id = car_id
        self.position = list(position)
        self.direction = direction
        self.color = color

    def move(self):
        self.position[0] += self.direction[0]
        self.position[1] += self.direction[1]

    def draw(self, img: np.ndarray, car_width: int, car_height: int):
        x, y = self.position
        cv.rectangle(img, (x, y), (x + car_width, y + car_height), self.color, -1)
        cv.rectangle(img, (x, y), (x + car_width, y + car_height), (0, 0, 0), 2)