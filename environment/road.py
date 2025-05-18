# environment/road.py
import cv2 as cv
import numpy as np

class Road:
    def __init__(self, img: np.ndarray, height: int, road_x_start: int, road_width: int, num_lanes: int):
        self.img = img
        self.height = height
        self.road_x_start = road_x_start
        self.road_width = road_width
        self.num_lanes = num_lanes
        self.lane_width = road_width // num_lanes

    def draw(self):
        self.img[:] = (34, 139, 34)
        road_x_end = self.road_x_start + self.road_width
        cv.rectangle(self.img, (self.road_x_start, 0), (road_x_end, self.height), (50, 50, 50), -1)
        for i in range(1, self.num_lanes):
            x = self.road_x_start + i * self.lane_width
            for y in range(0, self.height, 40):
                cv.line(self.img, (x, y), (x, y + 20), (255, 255, 255), 2)