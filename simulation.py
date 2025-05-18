from vehicles.car import Car
from vehicles.sensors import SensorSystem
from environment.road import Road
from environment.houses import Houses
from config import CONFIG
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class TrafficSimulation:
    def __init__(self):
        self.cfg = CONFIG
        self.road_x_start = (self.cfg["width"] - self.cfg["road_width"]) // 2
        self.lane_width = self.cfg["road_width"] // self.cfg["num_lanes"]
        self.lane_centers = [
            self.road_x_start + self.lane_width * i + self.lane_width // 2 - self.cfg["car_width"] // 2
            for i in range(self.cfg["num_lanes"])
        ]
        self.main_car = self._create_main_car()
        self.other_cars = self._create_other_cars()
        self.sensor = SensorSystem(
            self.cfg["car_width"],
            self.cfg["car_height"],
            self.cfg["sensor_radius"]
        )
        plt.ion()

    def _create_main_car(self):
        return Car(0, [self.lane_centers[2], 300], [0, -30], (0, 0, 255))

    def _create_other_cars(self):
        return [
            Car(1, [self.lane_centers[0], 100], [0, 20]),
            Car(2, [self.lane_centers[1], 400], [0, -15]),
            Car(3, [self.lane_centers[3], 200], [0, 10]),
            Car(4, [self.lane_centers[0], 500], [0, -25]),
            Car(5, [self.lane_centers[2], 100], [0, -30]),
            Car(6, [self.lane_centers[5], 150], [0, 15]),
            Car(7, [self.lane_centers[4], 100], [0, 20]),
            Car(8, [self.lane_centers[2], 450], [0, -30])
        ]

    def _simulate_frame(self, frame, prev_fig):
        image = np.zeros((self.cfg["height"], self.cfg["width"], 3), dtype=np.uint8)
        Road(image, self.cfg["height"], self.road_x_start, self.cfg["road_width"], self.cfg["num_lanes"]).draw()
        Houses(image, self.cfg["height"]).draw()

        self.main_car.move()
        for car in self.other_cars:
            car.move()

        self.main_car.draw(image, self.cfg["car_width"], self.cfg["car_height"])
        for car in self.other_cars:
            car.draw(image, self.cfg["car_width"], self.cfg["car_height"])

        image, sensor_data = self.sensor.add_sensors_and_process(image, self.main_car.position, self.other_cars)
        center = (
            self.main_car.position[0] + self.cfg["car_width"] // 2,
            self.main_car.position[1] + self.cfg["car_height"] // 2
        )

        if prev_fig is not None:
            plt.close(prev_fig)

        fig = self.sensor.analyze_and_plot(sensor_data, self.cfg["polar_radius"], center)
        plt.pause(0.001)

        cv.imshow(f"Traffic Frame {frame+1}", image)
        print("Press 'N' for next frame, or 'Esc' to exit.")

        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord('n') or key == ord('N'):
                break
            elif key == 27:
                cv.destroyAllWindows()
                plt.close('all')
                exit()

        cv.destroyAllWindows()
        return fig


    def run(self):
        prev_fig = None
        for frame in range(self.cfg["frames"]):
            prev_fig = self._simulate_frame(frame, prev_fig)
        plt.close('all')