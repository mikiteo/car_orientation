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

    def analyze_and_plot(self, sensor_arrays: Dict[str, List[Tuple[int, str, Tuple[int, int]]]], polar_radius: int, plot_center: Tuple[int, int], overlay_data: Dict[int, Dict[str, float]]):
        cx, cy = plot_center
        points = []

        for direction, data in sensor_arrays.items():
            for car_id, side, (x, y) in data:
                dx, dy = x - cx, y - cy
                dist = np.hypot(dx, dy)
                angle = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
                points.append((dist, angle, car_id))

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

        if points:
            ax.scatter([p[1] for p in points], [p[0] for p in points], c='r', s=10, label='Closest Points')

        cars: Dict[int, List[Tuple[float, float]]] = {}
        for r, th, cid in points:
            cars.setdefault(cid, []).append((r, th))

        NEAR_DISTANCE_THRESHOLD = polar_radius * 0.43

        for car_id, vals in cars.items():
            if car_id not in overlay_data:
                continue

            d = overlay_data[car_id]['d']
            v_val = overlay_data[car_id]['prev_v']
            a = overlay_data[car_id]['a']

            x_list = [r * np.cos(th) for r, th in vals]
            y_list = [r * np.sin(th) for r, th in vals]
            x_avg = np.mean(x_list)
            y_avg = np.mean(y_list)
            th_center = (np.arctan2(y_avg, x_avg) + 2 * np.pi) % (2 * np.pi)
            r_center = np.mean([r for r, _ in vals])

            if r_center < NEAR_DISTANCE_THRESHOLD:
                r_shifted = r_center + 65
                th_shifted = th_center
            else:
                norm = r_center
                dx = x_avg / norm * 10 if norm != 0 else 0
                dy = y_avg / norm * 10 if norm != 0 else 0
                x_shifted = x_avg + dx
                y_shifted = y_avg + dy
                r_shifted = np.hypot(x_shifted, y_shifted)
                th_shifted = (np.arctan2(y_shifted, x_shifted) + 2 * np.pi) % (2 * np.pi)

            angle_deg = np.degrees(th_shifted)
            if 45 <= angle_deg <= 135:
                ha, va = 'center', 'bottom'
            elif 225 <= angle_deg <= 315:
                ha, va = 'center', 'top'
            elif angle_deg < 45 or angle_deg > 315:
                ha, va = 'left', 'center'
            else:
                ha, va = 'right', 'center'

            fontsize = 10 if r_shifted > polar_radius * 0.25 else 8
            label = f"d={d:.1f}\nv={v_val:.1f}\na={a:.1f}"
            ax.text(th_shifted, r_shifted, label, fontsize=fontsize, ha=ha, va=va)

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
        ax.legend(loc='upper right')

        plt.show(block=False)
        return fig
