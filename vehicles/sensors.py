# vehicles/sensors.py
from typing import List, Tuple, Dict
from matplotlib.patches import Rectangle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class SensorSystem:
    def __init__(self, car_width: int, car_height: int, sensor_radius: int):
        self.car_width = car_width
        self.car_height = car_height
        self.sensor_radius = sensor_radius
        self.step = 1
        self.sensor_angle = 120
        self.start_angle = [165, 255, 75, 345]
        self.sensor_keys = ["front_left", "front_right", "rear_left", "rear_right"]
        self.display_radius = 50 

    @staticmethod
    def in_sector(angle, start, span):
        end = (start + span) % 360
        if span >= 360:
            return True
        if start <= end:
            return start <= angle <= end
        else:
            return angle >= start or angle <= end
        
    def get_visible_sides(self, main_pos, other_pos):
        mx1, my1 = main_pos
        mx2, my2 = mx1 + self.car_width, my1 + self.car_height
        ox1, oy1 = other_pos
        ox2, oy2 = ox1 + self.car_width, oy1 + self.car_height

        visible = []

        if oy2 < my1:
            visible.append("bottom")
        elif oy1 > my2:
            visible.append("top")

        if ox2 < mx1:
            visible.append("right")
        elif ox1 > mx2:
            visible.append("left")

        return visible

    def get_side_segments(self, position: List[int]) -> Dict[str, List[Tuple[int, int]]]:
        x, y = position
        return {
            "top": [(x + i, y) for i in range(0, self.car_width, self.step)],
            "bottom": [(x + i, y + self.car_height) for i in range(0, self.car_width, self.step)],
            "left": [(x, y + i) for i in range(0, self.car_height, self.step)],
            "right": [(x + self.car_width, y + i) for i in range(0, self.car_height, self.step)]
        }

    def add_sensors_and_process(self, img: np.ndarray, main_pos: List[int], other_cars: List[object]) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, str, Tuple[int, int]]]]]:
        sensor_positions = [
            (main_pos[0], main_pos[1]),
            (main_pos[0] + self.car_width, main_pos[1]),
            (main_pos[0], main_pos[1] + self.car_height),
            (main_pos[0] + self.car_width, main_pos[1] + self.car_height)
        ]

        sensor_arrays = {key: [] for key in self.sensor_keys}

        for car in other_cars:
            visible = self.get_visible_sides(main_pos, car.position)
            sides = self.get_side_segments(car.position)
            filtered_sides = {k: v if k in visible else [] for k, v in sides.items()}

            for i, (sx, sy) in enumerate(sensor_positions):
                end_angle = (self.start_angle[i] + self.sensor_angle) % 360
                if self.start_angle[i] > end_angle:
                    end_angle += 360

                cv.ellipse(img, (sx, sy), (self.display_radius, self.display_radius), 0,
                        self.start_angle[i], end_angle, (0, 255, 0), 1)

                for side, points in filtered_sides.items():
                    for px, py in points:
                        dx, dy = px - sx, py - sy
                        dist = np.hypot(dx, dy)
                        angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
                        if self.in_sector(angle, self.start_angle[i], self.sensor_angle) and dist <= self.sensor_radius:
                            is_blocked = False
                            for other_car in other_cars:
                                if other_car != car:
                                    if self.is_between(sx, sy, px, py, other_car.position):
                                        is_blocked = True
                                        break

                            if not is_blocked:
                                sensor_arrays[self.sensor_keys[i]].append((car.id, side, (px, py)))

        return img, sensor_arrays

    def is_between(self, sx, sy, px, py, car_position):
        car_x, car_y = car_position
        rect = {
            "left": car_x,
            "right": car_x + self.car_width,
            "top": car_y,
            "bottom": car_y + self.car_height
        }

        def intersects(x1, y1, x2, y2, rx1, ry1, rx2, ry2):
            edges = [
                ((rx1, ry1), (rx2, ry1)),  
                ((rx2, ry1), (rx2, ry2)),  
                ((rx2, ry2), (rx1, ry2)),  
                ((rx1, ry2), (rx1, ry1))  
            ]
            for (ex1, ey1), (ex2, ey2) in edges:
                if segments_intersect(x1, y1, x2, y2, ex1, ey1, ex2, ey2):
                    return True
            return False

        def ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

        def segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
            return (ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and
                    ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy))

        return intersects(sx, sy, px, py,
                        rect["left"], rect["top"],
                        rect["right"], rect["bottom"])



    def analyze_and_plot(self, sensor_arrays: Dict[str, List[Tuple[int, str, Tuple[int, int]]]], polar_radius: int, plot_center: Tuple[int, int]):
        cx, cy = plot_center
        points = []

        for data in sensor_arrays.values():
            for _, _, (x, y) in data:
                dx, dy = x - cx, y - cy
                dist = np.hypot(dx, dy)
                angle = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
                points.append((dist, angle, x, y))

        closest = [None] * 360
        for dist, angle, x, y in points:
            deg = int(np.degrees(angle))
            if closest[deg] is None or dist < closest[deg][0]:
                closest[deg] = (dist, angle, x, y)

        filtered = [p for p in closest if p is not None]

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        if filtered:
            distances, angles = zip(*[(p[0], p[1]) for p in filtered])
            ax.scatter(angles, distances, c='r', s=10, label="Closest Points")

        car_w = polar_radius * 0.1
        car_h = polar_radius * 0.2
        car_rect = Rectangle(
            (-car_w / 2, -car_h / 2), 
            car_w, car_h,
            facecolor='red',
            edgecolor='black',
            linewidth=1,
            transform=ax.transData._b, 
            zorder=5
        )
        ax.add_patch(car_rect)

        ax.set_ylim(0, polar_radius * 0.85)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_title("Closest Points to the Main Car")
        ax.legend()
        plt.show(block=False)
        return fig
