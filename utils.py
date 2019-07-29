import numpy as np
import pandas as pd
from scipy import special
from functools import reduce
from typing import List, Dict, Tuple


def get_pose_cart(data: pd.DataFrame,
                  x_init: float,
                  y_init: float,
                  angle_init: float,
                  internal: float = 0.05) -> Dict[str, list]:
    """
    get the pose, as (x, y, angle), of the vehicle in the certain frame.

    :param data: the radar data includes host_speed and yaw_rate
    :param x_init: x coordinate of the initial position
    :param y_init: y coordinate of the initial position
    :param angle_init: (radius) the initial orientation
    :param internal: the time internal, default 0.05 seconds
    :return: (dictionary) {'x': list,
                           'y':list,
                           'angle':list}
    """
    data = data.drop_duplicates('FrameCnt', keep='first')
    speed = data['HostSpeed_ms'].tolist()
    yaw = np.array(data['YawRate_rads'].tolist()) * 0.05
    x_list = [x_init]
    y_list = [y_init]
    angle_list = [angle_init]

    for i in range(1, len(speed)):
        # theta = -(0.5*np.pi + angle_list[-1] + yaw[i-1])
        # theta = -(0.5*np.pi + angle_list[-1] + yaw[i-1])
        # x = x_list[-1] + internal*speed[i-1] * np.cos(theta)
        # y = y_list[-1] + internal*speed[i-1] * np.sin(theta)
        theta = 0.5 * np.pi + angle_list[-1] - yaw[i - 1]
        x = x_list[-1] + internal * speed[i - 1] * np.cos(theta)
        y = y_list[-1] - internal * speed[i - 1] * np.sin(theta)
        x_list.append(x)
        y_list.append(y)
        angle_list.append(yaw[i - 1] + angle_list[-1])

    return {'x': x_list,
            'y': y_list,
            'angle': angle_list}


def get_meas_vehicle_cart(data: pd.DataFrame) -> np.ndarray:
    '''
    get measures positions in vehicle Cartesian coordinate , as (x, y).

    :param data: all frames radar data.
    :return: measures positions matrix.
             the first row are all x coordinate, and the second are y.
    '''

    radar_offset = {'1': np.array([[3.23], [0.75]]),
                    '2': np.array([[3.23], [-0.75]]),
                    '3': np.array([[-0.64], [0.75]]),
                    '4': np.array([[-0.64], [-0.75]])}
    result = []
    for i in range(1, 5):
        tmp = data[data['ID'] == i]
        if tmp.empty:
            continue
        azimuth = (-1) * np.array(tmp['Azimuth_deg'].tolist())
        r = np.array(tmp['Range_m'].tolist())  # ranges of the detections
        meas_vehicle = radar_offset[str(
            i)] + r * np.stack([np.cos(azimuth), np.sin(azimuth)], 0)
        result.append(meas_vehicle)
    return reduce(lambda x, y: np.concatenate([x, y], 1), result)


def get_radar_cart(pose: List[float]) -> np.ndarray:
    '''
    obtain the radar global Cartesian positions.

    :param pose: the position of the vehicle in current frame.
    :return: the four radars' positions.
             each column indicates one radar position, from 1 to 4.
    '''
    radar_offset = np.array(
        [[3.23, 3.23, -0.64, -0.64], [0.75, -0.75, 0.75, -0.75]])
    rotate_angle = pose[2] + np.deg2rad(90)
    rotation = np.array([[np.cos(rotate_angle), np.sin(rotate_angle)],
                         [-np.sin(rotate_angle), np.cos(rotate_angle)]])
    radar_cart = np.array([[pose[0]], [pose[1]]]) + \
        np.matmul(rotation, radar_offset)
    return radar_cart


def meas_v2g(pose: List[float],
             meas_vehicle: np.ndarray) -> np.ndarray:
    '''
    Transform the measures positions from vehicle coordinate to global coordinate.

    :param pose: The current vehicle pose.
    :param meas_vehicle: the measures positions in vehicle coordinate.
    :return: the measures positions in global coordinate.
             each column indicates a certain measure's position, as [x, y].
    '''

    # pose_x, pose_y, pose_angle = pose[0], pose[1], pose[2]+np.deg2rad(90)
    pose_x, pose_y, pose_angle = pose[0], pose[1], pose[2]
    rotation_mat = np.array([[np.cos(pose_angle), -np.sin(pose_angle)],
                             [np.sin(pose_angle), np.cos(pose_angle)]])
    meas_global = np.array([[pose_x], [pose_y]]) + \
        np.matmul(rotation_mat, meas_vehicle)
    return meas_global


def get_line(start, end):
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    #     points = []
    x_list = []
    y_list = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else [x, y]
        #         points.append(coord)
        x_list.append(coord[0])
        y_list.append(coord[1])
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        x_list.reverse()
        y_list.reverse()
    return x_list, y_list


def get_free_grid(pose_grid, meas_grid):
    """

    :param pose_grid: (tuple) (pose_x, pose_y, pose_angle)
    :param meas_grid: (np.ndarray) the vector of meas_vehicle
    :return: (lists) the lists of the free grid position.
    """

    x_free = []
    y_free = []
    start = pose_grid
    # start = (0, 0)
    for i in range(meas_grid.shape[1]):
        end = (meas_grid[0][i], meas_grid[1][i])
        x, y = get_line(start, end)
        x_free.extend(x[1:-1])
        y_free.extend(y[1:-1])
    return np.abs(np.array([x_free, y_free]))


def cumulative_distribution(start: np.ndarray,
                            end: np.ndarray,
                            meas: float,
                            std: float) -> np.ndarray:
    '''
    implement the cdf function.

    :param start: min distances of all the tri_grid.
    :param end: max distances of all the tri_grid.
    :param meas: the measure's distance respective to a specific radar.
    :param std: the standard deviation.
                for r, it should be 0.017.
                for angle, it should be 0.33 degree.
    :return: cdf value for each grid in r or angle field.
    '''
    x1 = (start - meas) / (std * np.sqrt(2))
    x2 = (end - meas) / (std * np.sqrt(2))
    return 0.5 * (special.erf(x2) - special.erf(x1))


def occ_func(start: List[np.ndarray],
             end: List[np.ndarray],
             meas: List[float],
             std: List[float]) -> np.ndarray:
    '''
    implement the occupancy function described in <<3D Localization and Mapping using automative radar>>

    :param start: min_r and min_angle in two separate np.ndarray.
                  the first np.ndarray is min_rs, and the second one is min_angles.
    :param end: max_rs and max_angles also in two separate np.ndarray.
                the first np.ndarray is max_rs, and the second one is max_angles.
    :param meas: the measures, as [r, angle], in the current frame respective to the specific radar.
    :param std: the standard deviations for r and angle.
    :return: occupancy probability for each grid.
    '''

    occ_r = cumulative_distribution(start[0], end[0], meas[0], std[0])
    occ_angle = cumulative_distribution(start[1], end[1], meas[1], std[1])
    occ_prob = occ_r * occ_angle
    return occ_prob


def emp_func(start, end, meas, std):
    factor = np.exp((-start[0]**2 / (2 * (meas[0] / 4)**2)))
    return factor * cumulative_distribution(start[1], end[1], meas[1], std[1])


def get_triangle(vertexes: np.ndarray) -> np.ndarray:
    '''
    Draw a triangle in grid map given three vertexes.

    :param vertexes: the vertexes of the triangle which will be drawn.
                  each column, [x, y], indicates a node.
    :return: all grid in the triangle.
             each column indicates a grid.
    '''
    vertexes = np.hstack([vertexes, vertexes[:, 0][:, None]])
    pairs = set()
    for i in range(1, vertexes.shape[1]):
        x_temp, y_temp = get_line(vertexes[:, i - 1], vertexes[:, i])
        pairs.update(list(map(lambda x, y: (x, y), x_temp, y_temp)))
    df = pd.DataFrame(data=pairs, columns=['x', 'y'])
    x_min, x_max = df['x'].min(), df['x'].max()
    for x in range(x_min, x_max + 1):
        y_column = df[df['x'] == x]['y']
        y_list = y_column.tolist()
        y_min, y_max = y_column.min(), y_column.max()
        if len(y_column) != (y_max - y_min + 1):
            for y in range(y_min + 1, y_max):
                if y not in y_list:
                    df = df.append({'x': x, 'y': y}, ignore_index=True)
    return df.transpose().to_numpy()


def grid2polar(tri_grid, pose):
    """

    :rtype: (test)
    """
    angle = pose[2]
    radar_grid = get_radar_cart(pose)
    r = np.sqrt(np.sum((tri_grid - radar_grid)**2), axis=0)
    rotation_T = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    result = np.matmul(rotation_T, tri_grid - radar_grid) / r
    theta = result[1]
    print(theta)


def get_vertexes(r_meas: float,
                 azimuth: float,
                 radar_pos: np.ndarray,
                 grid_size: float) -> np.ndarray:
    '''
    calculate the vertexes grid position.

    :param r_meas: the detection range from radar.
    :param azimuth: the detection angle from radar.
    :param radar_pos: the global position of the radar.
    :param grid_size: the default grid_size.
    :return: two vertexes grid position that offset the detection due to uncertainties in range and angle.
             each column, [x, y], indicates a vertexes in grid map.
    '''
    azi_1 = np.deg2rad(azimuth + 1)
    tmp_1 = (r_meas + 0.05) * np.array([[np.cos(azi_1)], [np.sin(azi_1)]])
    meas_offset_1_grid = ((radar_pos + tmp_1) // grid_size).astype(int)
    azi_2 = np.deg2rad(azimuth - 1)
    tmp_2 = (r_meas + 0.05) * np.array([[np.cos(azi_1)], [np.sin(azi_2)]])
    meas_offset_2_grid = ((radar_pos + tmp_2) // grid_size).astype(int)
    vertexes = np.hstack([meas_offset_1_grid, meas_offset_2_grid])
    return vertexes


def get_r_and_angle_range(tri_grid: np.ndarray,
                          radar_pos: np.ndarray,
                          grid_size: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    calculate each grid's r-range and angle-range according to their relative positions of radar.

    :param tri_grid: triangle in grid map which presents the radar view.
    :param radar_pos: the global position of the radar.
    :param grid_size: default grid size.
    :return: two np.ndarrays which describe the r-range and angle-range of each tri_grid.
             the first is r-range matrix, each column indicates the min and the max r range of the grid.
             the second is angle-range matrix, each column indicates the min and max angle of the grid.
             (Note: the units of angle is degrees!)
    '''

    result_r = []
    result_angle = []
    for i in range(tri_grid.shape[1]):
        one_grid = tri_grid[:, i][:, None]
        cornors_global = (
            np.array([[0, 1, 0, 1], [0, 0, 1, 1]]) + one_grid) * grid_size
        distances = np.sqrt(np.sum((cornors_global - radar_pos)**2, 0))
        min_r, max_r = distances.min(), distances.max()

        angles = (cornors_global[0, :] - radar_pos[0]) / distances
        min_angle, max_angle = np.rad2deg(
            np.arccos(
                angles.max())), np.rad2deg(
            np.arccos(
                angles.min()))
        result_r.append(np.array([[min_r], [max_r]]))
        result_angle.append(np.array([[min_angle], [max_angle]]))

    r_range = reduce(lambda x, y: np.hstack([x, y]), result_r)
    angle_range = reduce(lambda x, y: np.hstack([x, y]), result_angle)
    return r_range, angle_range
