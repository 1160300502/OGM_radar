import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_pose_cart, \
    get_meas_vehicle_cart, \
    meas_v2g, \
    get_radar_cart, \
    get_triangle, \
    get_vertexes, \
    get_r_and_angle_range, \
    occ_func, emp_func


class Map():
    def __init__(self, x_size, y_size, grid_size):
        self.x_size = x_size + 2
        self.y_size = y_size + 2
        self.grid_size = grid_size
        # initialize the log-odds map to zeros
        self.log_prob_map = np.zeros((self.x_size, self.y_size))

        # self.track_map = np.zeros((self.x_size, self.y_size)) ## initialize
        # the log-odds map to zeros

        self.grid_position_map = np.array(
            [
                np.tile(
                    np.arange(
                        0,
                        self.x_size *
                        self.grid_size,
                        self.grid_size)[
                        :,
                        None],
                    (1,
                     self.y_size)),
                np.tile(
                    np.arange(
                        0,
                        self.y_size *
                        self.grid_size,
                        self.grid_size)[
                        None,
                        :],
                    (self.x_size,
                     1))])

        self.l_occ = np.log(0.65 / 0.35)
        self.l_free = np.log(0.35 / 0.65)

    def update_map(self, tri_grid, log_odds):
        """

        :param meas_vehicle: (lists) the positions of the measures in the current frame.
        :param free_grid: (lists) the free space grid positions.
        :return:
        """
        if type(log_odds) is float:
            self.log_prob_map[tri_grid[0,0]:tri_grid[0,1]+1, tri_grid[1,0]:tri_grid[1,1]+1] += log_odds
        else:
            self.log_prob_map[tri_grid[0], tri_grid[1]] += log_odds


def get_log_odds(r_range, angle_range, meas, std, det_prob):
    start = [r_range[0, :], angle_range[0, :]]
    end = [r_range[1, :], angle_range[1, :]]
    occ_prob = occ_func(start, end, meas, std)
    emp_prob = emp_func(start, end, meas, std)
    post_prob_grid = 0.5 * (1 + det_prob * (occ_prob - emp_prob))
    log_odds = np.log(post_prob_grid / (1 - post_prob_grid))
    return log_odds


if __name__ == "__main__":

    # configure
    grid_size = .2
    x_size = 100
    y_size = 100
    false_alarm_rate = 0.15
    maps = Map(int(x_size//grid_size), int(y_size//grid_size), grid_size)
    x_init = 40
    y_init = 72.5
    angle_init = np.deg2rad(90)

    # load radar data and parse
    path = "/home/aaron-ran/Documents/SLAM_internship/data/Test1_1/test1_1.csv"
    data = pd.read_csv(path)

    # get all frames pose
    pose = get_pose_cart(data, x_init, y_init, angle_init)

    # initial plot figure
    plt.figure(1)
    plt.ion()

    # update map in each frame
    for i in range(len(pose['x'])):
    # for index in range(400):
        # get current frame data and pose
        data_frame = data[data['FrameCnt'] == i]
        pose_frame = [pose['x'][i], pose['y'][i], pose['angle'][i]]
        x_grid = pose_frame[0] // grid_size
        y_grid = pose_frame[1] // grid_size
        pose_frame_grid = np.array([x_grid, y_grid, pose['angle'][i]])

        # get measures positions
        meas_vehicle = get_meas_vehicle_cart(data_frame)
        meas_global = meas_v2g(pose_frame, meas_vehicle)
        meas_grid = (meas_global // grid_size).astype(int)

        # radar position
        radar_global = get_radar_cart(pose_frame)
        radar_grid = (radar_global // grid_size).astype(int)
        # plt.ion()
        # radar_grid.T[[2, 3]] = radar_grid.T[[3, 2]]
        # radar_grid = np.hstack([radar_grid, radar_grid[:, 0][:, None]])
        # maps = np.zeros((500, 500))
        # maps[radar_grid[0, :], radar_grid[1, :]] = [0.2, .4, .6, .8]
        # plt.clf()
        # # plt.plot(radar_grid[0, :].tolist(), radar_grid[1, :].tolist(), 'r-')
        # plt.imshow(maps)
        # plt.pause(0.005)
        # # plt.show()

        min_x_radar_grid, max_x_radar_grid = radar_grid[0].min(), radar_grid[0].max()
        min_y_radar_grid, max_y_radar_grid = radar_grid[1].min(), radar_grid[1].max()
        vehicle_grid = np.array([[min_x_radar_grid, max_x_radar_grid], [min_y_radar_grid, max_y_radar_grid]])
        vehicle_log_odds = np.log(0.1/0.9)
        maps.update_map(vehicle_grid, vehicle_log_odds)

        radar_list = data_frame['ID'].drop_duplicates(keep='first').tolist()
        rad_index = radar_list[2] ## 只显示radar_1的结果。
        # for rad_index in radar_list:

        print(f"frames: {i}\n\tradars: {rad_index}")
        # sub dataset according to each radar
        sub_data = data_frame[data_frame['ID'] == rad_index]
        num_pcloud = sub_data.iloc[0, 2]
        if i == 148:
            print(f"num_pc: {num_pcloud}")
        for pc in range(num_pcloud):
            # get triangle grid
            radar_pos = radar_global[:, rad_index-1][:, None]
            r_meas = sub_data.iloc[pc, 4]
            azimuth = sub_data.iloc[pc, 6]
            vertexes = get_vertexes(r_meas, azimuth, radar_pos, grid_size)
            vertexes = np.hstack(
                [vertexes, radar_grid[:, rad_index-1][:, None]])
            tri_grid = get_triangle(vertexes)

            # calculate each grid's r-range and angle-range.
            r_range, angle_range = get_r_and_angle_range(
                tri_grid, radar_pos, grid_size)

            # probability of detection
            SNR = sub_data.iloc[pc, 3]
            det_prob = false_alarm_rate**(1.0 / (1.0 + SNR))

            # detection data and standard deviation.
            meas = [r_meas, abs(azimuth)]
            std = [0.017, 0.33]

            # update maps.
            log_odds = get_log_odds(
                r_range, angle_range, meas, std, det_prob)
            maps.update_map(tri_grid, log_odds)
            if (i % 20) == 0:
                plt.clf()
                plt.imshow(1.0 - 1./(1.+np.exp(maps.log_prob_map)), 'Greys')  ## the first arg is prob
                plt.pause(0.005)
plt.clf()
plt.imshow(1.0 - 1./(1.+np.exp(maps.log_prob_map)), 'Greys')  ## the first arg is prob
plt.pause(0.005)
plt.ioff()
plt.show()
