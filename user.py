

import glob
import os

import cv2
import pandas as pd
from charuco_utils import calculate_hand_eye_reprojection_error, calibrate_hand_eye_pnp_reprojection
from main_calibration_analysis import generate_board_table
import numpy as np

from user_study_2 import bin_and_sample, calib_frames_to_dataframe, perform_calibration
from utils import filter_and_merge_hand_eye_df


def main(data_path = 'results/user_study/mac',
         participant = 'matt',
         run_num = '0',
         img_ext = 'png',
         
         aruco_w = 13,
         aruco_h = 9,
         size_of_checkerboard = 20,
         aruco_size = 15,

         waitTime = 0,
         visualise_corner_detection = False,
         visualise_reprojection_error = False,

        num_images_for_calibration=10,
        grid_size_x=3, 
        grid_size_y=3, 
        min_positions=9, 
        min_distances=1, 
        min_angles=10, 
        max_distance_threshold=1000,
        min_distance_threshold=10, 
        min_angle_threshold=-40, 
        max_angle_threshold=40, 
         ): 
    

    
    aruco_dict = cv2.aruco.DICT_4X4_1000
    board = cv2.aruco.CharucoBoard(
        (aruco_w, aruco_h),  # 7x5
        size_of_checkerboard,
        aruco_size,
        cv2.aruco.getPredefinedDictionary(aruco_dict)
    )

    # generate table for reprojection error
    #table_data_pth = f'results/user_study/mac/aure/reprojection_dataset_endo_distance'
    #table_info_pth = f'results/user_study/mac/aure/reprojection_dataset_endo_distance_info'

    reproj_name = 'reprojection_dataset_endo_distance'
    calib_type = 'hand_eye'
    for cam in ['endo', 'rs']:

        for participant in ['aure', 'matt', 'mobarak', 'joao']:
            for run_num in [reproj_name,'1', '2', '3', '4', '5']:
                
                # check if reprojection has been done once and skip if so
                if run_num == reproj_name and participant != 'aure':
                    continue

                # load detected data to see img paths
                data = pd.read_pickle(f'{data_path}/{participant}/{run_num}/data_{cam}.pkl')
                image_pths = data.paths.values.tolist()
                
                # naming and folder creation
                results_data_pth = f'{data_path}/{participant}/{run_num}/{calib_type}' # pth where all participant run results stored (calibration and data)
                if not os.path.exists(f'{results_data_pth}/calibration'):
                    os.makedirs(f'{results_data_pth}/calibration')
                table_data_pth = f'{data_path}/{participant}/{run_num}/{calib_type}/{cam}_data.pkl'
                table_info_pth = f'{data_path}/{participant}/{run_num}/{calib_type}/{cam}_info.pkl'
                # if these exist, skip and print error
                if os.path.exists(table_data_pth) and os.path.exists(table_info_pth):
                    print(f'{table_data_pth} already exist, skipping')
                    if run_num != reproj_name and calib_type != 'hand_eye':
                        err = np.loadtxt(f'{results_data_pth}/calibration/err_{cam}.txt') 
                        print(f'error: {  err  }'   )
                    continue
                
                if calib_type == 'hand_eye':
                    # load intrinsics
                    intrinsics = np.loadtxt(f'{data_path}/{participant}/{run_num}/intrinsics/calibration/intrinsics_{cam}.txt')
                    distortion = np.loadtxt(f'{data_path}/{participant}/{run_num}/intrinsics/calibration/distortion_{cam}.txt')
                intrinsics, distortion = None, None
                # creates table of detected corners for all images with at least one corner detected
                data_df, _ = generate_board_table(image_pths, board, table_data_pth, table_info_pth,
                                                                    waiting_time=waitTime,
                                                                    visualise_corner_detection=visualise_corner_detection,
                                                                    intrinsics=intrinsics, distortion=distortion, main_format=False)
                

                

                # add poses
                # find frame_number of 
                data['frame_number'] = data_df['paths'].str.extract('(\d+).png')
                # add "poses" column to data_df from data['poses] where the 'frame_number' is the same
                data_df_merged = data_df.merge(data[['frame_number', 'poses']], on='frame_number', how='left')
                # add x,y,z,rxryrz columns
                data_df_merged = calib_frames_to_dataframe(data_df_merged, extension = '')
                data_df_merged.to_pickle(table_data_pth)

                if calib_type == 'hand_eye':
                    continue

                # select frames for calibration
                frames_for_calibration,remaining_frames  = bin_and_sample(data_df_merged, num_images_for_calibration=num_images_for_calibration, 
                                                              grid_size_x=grid_size_x, grid_size_y=grid_size_y, 
                                                              min_positions=min_positions, min_distances=min_distances, min_angles=min_angles, 
                                                              max_distance_threshold=max_distance_threshold,min_distance_threshold=min_distance_threshold, 
                                                              min_angle_threshold=min_angle_threshold, max_angle_threshold=max_angle_threshold)
                
                
                intrinsics_estimates_pth  = f'calibration_estimates/intrinsics_{cam}.txt'
                if run_num == reproj_name:
                    df_rs_reproj = remaining_frames
                else:
                    df_rs_reproj = pd.read_pickle(f'{data_path}/aure/{reproj_name}/{calib_type}/{cam}_data.pkl')

                intrinsics, distortion, err, num_corners_detected = perform_calibration(frames_for_calibration, df_rs_reproj, 
                                                                                            intrinsics_estimates_pth, visualise_reprojection_error = visualise_reprojection_error, waitTime = waitTime)
                
                print(f'{cam} {participant} {run_num} Calibration successful')
                print('intrinsics: ', intrinsics)
                print('distortion: ', distortion)
                print('err: ', err)
                print('num_corners_detected_rs: ', num_corners_detected)
                # save intrinsics and distortion as txt file 
                np.savetxt(f'{results_data_pth}/calibration/intrinsics_{cam}.txt', intrinsics)
                np.savetxt(f'{results_data_pth}/calibration/distortion_{cam}.txt', distortion)
                np.savetxt(f'{results_data_pth}/calibration/err_{cam}.txt', [err])

    
    # merge dataframes
    for participant in ['aure', 'matt',  'mobarak', 'joao']:
        for run_num in [reproj_name, '1', '2', '3', '4', '5']:
            if run_num == reproj_name and participant != 'aure':
                continue
            data_endo = pd.read_pickle(f'{data_path}/{participant}/{run_num}/hand_eye/endo_data.pkl')
            data_rs = pd.read_pickle(f'{data_path}/{participant}/{run_num}/hand_eye/rs_data.pkl')
            min_num_corners = int( 0.5 * (aruco_h * aruco_w) )
            data_df = filter_and_merge_hand_eye_df(data_endo, data_rs, min_num_corners, main_run = True)
            # save merged df
            data_df.to_pickle(f'{data_path}/{participant}/{run_num}/hand_eye/merged_data.pkl')
            # change column names x_endo to x, y_endo to y, z_endo to z, rx_endo to rx, ry_endo to ry, rz_endo to rz
            data_df.rename(columns={'x_endo': 'x', 'y_endo': 'y', 'z_endo': 'z', 'rx_endo': 'rx', 'ry_endo': 'ry', 'rz_endo': 'rz'}, inplace=True)
            # sample dataset
            num_images_for_he_calibration = 30
            min_angles = 10
            min_distances = 2
            min_positions = 5
            #df_combined = calib_frames_to_dataframe(df_combined, extension = '_endo')
            frames_for_he_calibration, remaining_frames  = bin_and_sample(data_df, num_images_for_calibration=num_images_for_he_calibration, 
                                                                grid_size_x=grid_size_x, grid_size_y=grid_size_y, 
                                                                min_positions=min_positions, min_distances=min_distances, min_angles=min_angles, 
                                                                max_distance_threshold=max_distance_threshold,min_distance_threshold=min_distance_threshold, 
                                                                min_angle_threshold=min_angle_threshold, max_angle_threshold=max_angle_threshold)


            if run_num == reproj_name:
                reprojection_data_df = remaining_frames
            else:
                reprojection_data_df = pd.read_pickle(f'{data_path}/aure/{reproj_name}/hand_eye/merged_data.pkl')
            intrinsics_endo = np.loadtxt(f'{data_path}/{participant}/{run_num}/intrinsics/calibration/intrinsics_endo.txt')
            distortion_endo = np.loadtxt(f'{data_path}/{participant}/{run_num}/intrinsics/calibration/distortion_endo.txt')
            # perform calibration
            hand_eye = calibrate_hand_eye_pnp_reprojection(frames_for_he_calibration,
                                                           reprojection_data_df, 
                                                           intrinsics_endo=intrinsics_endo, 
                                                           distortion_endo=distortion_endo, 
                                                           optimise=True, error_threshold=1)
            
            
            world2realsense = reprojection_data_df['T_rs'].values
            objPoints = reprojection_data_df['objPoints_rs'].values
            imgPoints = reprojection_data_df['imgPoints_endo'].values
            # calculate reprojection error
            if visualise_reprojection_error:
                endo_reprojection_images_pth = reprojection_data_df['paths_endo'].values
            else:
                endo_reprojection_images_pth = []
            err_np = calculate_hand_eye_reprojection_error(hand_eye, world2realsense,
                                                       objPoints, imgPoints,
                                                       intrinsics_endo, distortion_endo,
                                                       waitTime=waitTime,
                                                       endo_reprojection_images_pth=endo_reprojection_images_pth)
            reprojection_error_mean_final = pd.DataFrame(err_np).median()[0]

            print(f'{participant} {run_num} hand_eye calibration successful')
            print('reprojection_error_mean_final: ', reprojection_error_mean_final)
            np.savetxt(f'{data_path}/{participant}/{run_num}/hand_eye/calibration/hand_eye.txt', hand_eye)
            np.savetxt(f'{data_path}/{participant}/{run_num}/hand_eye/calibration/reprojection_error_mean_final.txt', [reprojection_error_mean_final])








    return 


if __name__=='__main__': 
    main() 