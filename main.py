import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def draw_road(img, height, road_x_start, road_width, num_lanes, lane_width):
    img[:] = (34, 139, 34)  
    
    road_x_end = road_x_start + road_width
    cv.rectangle(img, (road_x_start, 0), (road_x_end, height), (50, 50, 50), -1) 

    for i in range(1, num_lanes):
        x = road_x_start + i * lane_width
        for y in range(0, height, 40):
            cv.line(img, (x, y), (x, y + 20), (255, 255, 255), 2)  


def draw_houses(img, height):

    for y in range(0, height, 120):
        cv.rectangle(img, (50, y + 20), (150, y + 100), (0, 255, 255), -1)
        cv.rectangle(img, (50, y + 20), (150, y + 100), (0, 0, 0), 2)
    
    for y in range(0, height, 120):
        cv.rectangle(img, (650, y + 20), (750, y + 100), (0, 255, 255), -1)
        cv.rectangle(img, (650, y + 20), (750, y + 100), (0, 0, 0), 2)


def draw_car(img, position, color, car_width, car_height):
    x, y = position
    cv.rectangle(img, (x, y), (x + car_width, y + car_height), color, -1)
    cv.rectangle(img, (x, y), (x + car_width, y + car_height), (0, 0, 0), 2)


def get_side_segments(position, car_width, car_height, step):
    x, y = position
    top_side = [(x + i, y) for i in range(0, car_width, step)]
    bottom_side = [(x + i, y + car_height) for i in range(0, car_width, step)]
    left_side = [(x, y + i) for i in range(0, car_height, step)]
    right_side = [(x + car_width, y + i) for i in range(0, car_height, step)]
    return top_side, bottom_side, left_side, right_side


def get_visible_sides(main_car_pos, other_car_pos, car_width, car_height):
    main_x, main_y = main_car_pos
    other_x, other_y = other_car_pos
    
    visible_sides = []

    if other_y + car_height < main_y: 
        visible_sides.append("bottom")
    elif other_y > main_y + car_height:
        visible_sides.append("top")

    if other_x + car_width < main_x: 
        visible_sides.append("right")
    elif other_x > main_x + car_width: 
        visible_sides.append("left")

    return visible_sides


def add_sensors_and_process(img, sensor_radius, main_car_pos, car_width, car_height, other_cars_pos, step):
    sensor_angle = 120  
    start_angle = [165, 255, 75, 345]    

    sensor_arrays = {
        "front_left": [],
        "front_right": [],
        "rear_left": [],
        "rear_right": []
    }

    sensor_positions = [
        (main_car_pos[0], main_car_pos[1]), 
        (main_car_pos[0] + car_width, main_car_pos[1]), 
        (main_car_pos[0], main_car_pos[1] + car_height), 
        (main_car_pos[0] + car_width, main_car_pos[1] + car_height) 
    ]

    for car in other_cars_pos:
        car_position = car['position']
        visible_sides = get_visible_sides(main_car_pos, car_position, car_width, car_height)
        top_side, bottom_side, left_side, right_side = get_side_segments(car_position, car_width, car_height, step)

        car_sides = {
            "top": top_side if "top" in visible_sides else [],
            "bottom": bottom_side if "bottom" in visible_sides else [],
            "left": left_side if "left" in visible_sides else [],
            "right": right_side if "right" in visible_sides else []
        }

        for i, (sensor_x, sensor_y) in enumerate(sensor_positions):

            end_angle = start_angle[i] + sensor_angle
            cv.ellipse(img, (sensor_x, sensor_y), (sensor_radius, sensor_radius), 0, start_angle[i], end_angle, (0, 255, 0), 1)
            end_angle = end_angle % 360


            for side, points in car_sides.items():
                for point in points:
                    dx = point[0] - sensor_x
                    dy = point[1] - sensor_y
                    distance = np.sqrt(dx**2 + dy**2)

                    angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

                    if (start_angle[i] <= angle <= end_angle) or (start_angle[i] > end_angle and (angle >= start_angle[i] or angle <= end_angle)):
                        if distance <= sensor_radius:
                            if i == 0:
                                sensor_arrays["front_left"].append((car['id'], side, point))
                            elif i == 1:
                                sensor_arrays["front_right"].append((car['id'], side, point))
                            elif i == 2:
                                sensor_arrays["rear_left"].append((car['id'], side, point))
                            elif i == 3:
                                sensor_arrays["rear_right"].append((car['id'], side, point))

    # print("Sensor Arrays:")
    # for key, array in sensor_arrays.items():
    #     print(f"{key}: {array}")

    return img, sensor_arrays


def analyze_and_plot_sensor_data(sensor_arrays, polar_radius, plot_center):
    points = []
    center_x, center_y = plot_center

    for _, sensor_data in sensor_arrays.items():
        for point_data in sensor_data:
            _, _, (x, y) = point_data
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx**2 + dy**2)
            angle = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)  
            points.append((distance, angle, x, y)) 

    closest_points = [None] * 360
    for distance, angle, x, y in points:
        degree = int(np.degrees(angle))
        if closest_points[degree] is None or distance < closest_points[degree][0]:
            closest_points[degree] = (distance, angle, x, y)

    closest_points = [p for p in closest_points if p is not None]

    
    _, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    if closest_points:
        distances, angles = zip(*[(p[0], p[1]) for p in closest_points])
        ax.scatter(angles, distances, c='r', s=10, label="Closest Points")
        print(f"Plotted {len(closest_points)} closest points.")


    ax.set_ylim(0, polar_radius)
    ax.set_theta_zero_location("E") 
    ax.set_theta_direction(-1)  
    ax.set_title("Closest Points to the Main Car")
    ax.legend()

    plt.show()


def main():
    width, height = 800, 600
    road_width = 400
    num_lanes = 6
    car_width, car_height = 40, 80
    polar_radius = 350
    sensor_radius = 300
    step = 1
    
    lane_width = road_width // num_lanes
    road_x_start = (width - road_width) // 2

    lane_centers = [road_x_start + lane_width * i + lane_width // 2 - car_width // 2 for i in range(num_lanes)]
    main_car_pos = [lane_centers[2], 300]

    center_x = main_car_pos[0] + car_width // 2
    center_y = main_car_pos[1] + car_height // 2
    main_car_center = (center_x, center_y)

    other_cars_pos = [
        {'id': 1, 'position': [lane_centers[0], 100], 'color': (255, 0, 0)},
        {'id': 2, 'position': [lane_centers[1], 400], 'color': (255, 0, 0)},
        {'id': 3, 'position': [lane_centers[3], 200], 'color': (255, 0, 0)},
        {'id': 4, 'position': [lane_centers[0], 500], 'color': (255, 0, 0)},
        {'id': 5, 'position': [lane_centers[2], 100], 'color': (255, 0, 0)},
        {'id': 6, 'position': [lane_centers[5], 150], 'color': (255, 0, 0)},
        {'id': 7, 'position': [lane_centers[4], 100], 'color': (255, 0, 0)},
        {'id': 8, 'position': [lane_centers[2], 450], 'color': (255, 0, 0)}
    ]

    img = np.zeros((height, width, 3), dtype=np.uint8)

    draw_road(img, height, road_x_start, road_width, num_lanes, lane_width)
    draw_houses(img, height)
    draw_car(img, main_car_pos, (0, 0, 255), car_width, car_height)

    for car in other_cars_pos:
        position = car['position']
        draw_car(img, position, car['color'], car_width, car_height)

    img, sensor_arrays = add_sensors_and_process(img, sensor_radius, main_car_pos, car_width, car_height, other_cars_pos, step)

    cv.imshow("Trafic", img)

    analyze_and_plot_sensor_data(sensor_arrays, polar_radius, main_car_center)


    
if __name__ == "__main__":
    main()


