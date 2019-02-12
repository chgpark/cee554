import os
import pandas as pd
import csv
import numpy as np

import cv2
TARGET_DIR = "test/"
T_STEP = 'rosbagTimestamp'
RANGE = '-'
UWB_LIST = ["0x611b", "0x6119", "0x616d", "0x612b"]


# print (len(anchor_csv['rosbagTimestamp'])) #length of pure data

def find_min_length_uwb_csv(uwb_csv_list):
    global T_STEP
    min_len = 999999999999999
    i_th_uwb = 0
    for i, uwb_csv in enumerate(uwb_csv_list):
        csv_total_length = len(uwb_csv[T_STEP])
        if csv_total_length < min_len:
            i_th_uwb = i
            min_len = csv_total_length

    return i_th_uwb, min_len

def extract_distances_from_pozyx_csv(pozyx_csv):
    '''
    In pozyx_csv, sync is not distorted,
    so, if you check by yourself,
    then print below commented_line
    '''
    dist_0 = '-'
    dist_1 = 'RSS'
    dist_2 = 'dist.1'
    dist_3 = '-.2'
    dist_4 = 'RSS.2'
    dist_5 = 'dist.3'
    dist_6 = '-.4'
    dist_7 = 'RSS.4'

    # print(pozyx_csv['-'].iloc[i])
    # print(pozyx_csv['RSS'].iloc[i])
    # print(pozyx_csv['dist.1'].iloc[i])
    # print(pozyx_csv['-.2'].iloc[i])
    # print(pozyx_csv['RSS.2'].iloc[i])
    # print(pozyx_csv['dist.3'].iloc[i])
    # print(pozyx_csv['-.4'].iloc[i])
    # print(pozyx_csv['RSS.4'].iloc[i])

    pandas_index_list = [dist_0, dist_1, dist_2, dist_3, dist_4, dist_5, dist_6, dist_7]

    concatenated_pozyx_csv = np.array(pozyx_csv[dist_0])
    concatenated_pozyx_csv = np.reshape(concatenated_pozyx_csv, [-1, 1])

    for dist_idx in pandas_index_list[1:]:
        pozyx_dist_csv = np.array(pozyx_csv[dist_idx])
        pozyx_dist_csv = np.reshape(pozyx_dist_csv, [-1, 1])

        concatenated_pozyx_csv = np.concatenate((concatenated_pozyx_csv, pozyx_dist_csv), axis=1)

    return concatenated_pozyx_csv


def append_position_to_position_list(target_csv_loc, position_list):
    '''
    target_csv_loc[9:12] <- position x, y, z
    target_csv_loc[13:17] <- quaternion x,y,z,w
    '''

    position = list(target_csv_loc[9:11])
    position_list.append(position)

    return position_list

def search_nearest_pose(uwb_csv, target_csv):
    '''
    :param uwb_csv:
    :param target_csv: In this case, kobuki csv file is target
    :return: x,y
    '''
    global T_STEP
    i = 0
    position_list = []
    min_length = len(uwb_csv[T_STEP])
    nearest_index = 0

    for i in range(min_length):

        abs_list = []

        for j in range(nearest_index, len(target_csv[T_STEP])):
            j_th_time_diff = abs(uwb_csv[T_STEP][i] - target_csv[T_STEP][j])
            abs_list.append(j_th_time_diff)

            # In case of len(abs_list) == 2:
            if j == nearest_index + 1:
                # abs_list[0] is minimum
                if abs_list[0] < abs_list[1]:
                    break
            elif j > nearest_index + 1:
                # abs_list[-2] is minimum
                if abs_list[-3] > abs_list[-2] and abs_list[-2] < abs_list[-1]:
                    nearest_index = j-1
                    break

        position_list = append_position_to_position_list(target_csv.loc[nearest_index], position_list)

    assert len(position_list) == min_length

    position_list = np.array(position_list)

    return position_list

def write_file_data(result_csv, out_dir):
    result_file = open(out_dir, 'w', encoding='utf-8', newline='')

    wr = csv.writer(result_file)
    for i in result_csv:
        wr.writerow(i)

    result_file.close()

if __name__  == "__main__":
    '''Setting directories of object files'''
    # full_path = "/home/shapelim/Desktop/uio_net_data/190108/"
    # dir = full_path + "test/2019-01-08-08-31-38/"
    abs_path = "/home/shapelim/git_files/cee554/190209_range/range_02_11/"
    dir_list = os.listdir(abs_path)

    '''Remove files that are not target data folder files'''
    data_folder_dir_list = []
    for file_dir in dir_list:
        if '.' in file_dir:
            continue
        else:
            data_folder_dir_list.append(os.path.join(abs_path, file_dir))
    print(data_folder_dir_list)

    pozyx_file_name = '_slash_pozyx_slash_range.csv'
    mocap_file_name = '_slash_vrpn_client_node_slash_Kobuki_slash_pose.csv'

    for data_folder_dir in data_folder_dir_list:

        print("Target: ", data_folder_dir)
        pozyx_dir = data_folder_dir + '/' + pozyx_file_name
        mocap_dir = data_folder_dir + '/' + mocap_file_name

        pozyx_csv = pd.read_csv(pozyx_dir, delimiter = ',')
        mocap_csv = pd.read_csv(mocap_dir, delimiter = ',')

        # print(pozyx_csv['dist'].iloc[0])

        # print(pozyx_csv)
        #
        # """Align uwb sensors csvs."""
        distance_data = extract_distances_from_pozyx_csv(pozyx_csv)
        # print(distance_data)
        #
        # """
        # Find nearest pose on a standard of the shortest uwb csv.
        # This is for synchronization
        # """
        #
        print ("On synchronizing pose ...")
        position_list = search_nearest_pose(pozyx_csv, mocap_csv)
        # print(position_list)

        """
        Merge all things at the same time
        """
        # mean_pose = get_mean_pose_of_csv(anchor_csv)
        # anchor_transformation_mat = set_pose2transformation(mean_pose)
        # inv_anchor_transformation_mat = np.linalg.inv(anchor_transformation_mat)
        # transformed_pose_list = transform_pose_list(pose_list, inv_anchor_transformation_mat)


        # Merge all things

        result_csv = np.concatenate((distance_data, position_list), axis = 1)

        output_file_name = data_folder_dir.split("/")
        output_file_name = output_file_name[-1]
        output_file_name = output_file_name[-8:-3]

        output_dir_name = abs_path[:-12] + 'synced_02_11/' + output_file_name + '.csv'
        write_file_data(result_csv, output_dir_name)
