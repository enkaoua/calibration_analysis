

import glob
import os

import cv2
import pandas as pd
from main_calibration_analysis import generate_board_table
import numpy as np

from user_study_2 import bin_and_sample, calib_frames_to_dataframe, perform_calibration


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
    for cam in ['endo', 'realsense']:
        REPROJECTION_DONE = False
        for participant in ['aure', 'matt', 'mobarak', 'joao']:
            for run_num in [reproj_name,'1', '2', '3', '4', '5']:
                
                if run_num == reproj_name and REPROJECTION_DONE:
                    continue
                data = pd.read_pickle(f'{data_path}/{participant}/{run_num}/data_{cam}.pkl')
                image_pths = data.paths.values.tolist()
                """ image_pths = glob.glob(
                            f'{data_path}/{participant}/{run_num}/{cam}_images/*.{img_ext}') """
                
                # table_data_pth
                results_data_pth = f'{data_path}/{participant}/{run_num}/data'
                if not os.path.exists(results_data_pth):
                    os.makedirs(results_data_pth)
                table_data_pth = f'{data_path}/{participant}/{run_num}/data/{cam}_data.pkl'
                table_info_pth = f'{data_path}/{participant}/{run_num}/data/{cam}_info.pkl'
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
                #data_df_merged.to_pickle(f'{results_data_pth}/{cam}_data.pkl')
                data_df_merged.to_pickle(table_data_pth)

                if run_num == reproj_name:
                    REPROJECTION_DONE = True
                    continue 

                # add x,y,z,rxryrz columns
                data_df_merged = calib_frames_to_dataframe(data_df_merged, extension = '')

                # select frames for calibration
                frames_for_calibration,remaining_frames  = bin_and_sample(data_df_merged, num_images_for_calibration=num_images_for_calibration, 
                                                              grid_size_x=grid_size_x, grid_size_y=grid_size_y, 
                                                              min_positions=min_positions, min_distances=min_distances, min_angles=min_angles, 
                                                              max_distance_threshold=max_distance_threshold,min_distance_threshold=min_distance_threshold, 
                                                              min_angle_threshold=min_angle_threshold, max_angle_threshold=max_angle_threshold)
                
                
                intrinsics_estimates_pth  = f'calibration_estimates/intrinsics_{cam}.txt'
                df_rs_reproj = pd.read_pickle(f'{data_path}/{participant}/{reproj_name}/data_{cam}.pkl')
                intrinsics, distortion, err, num_corners_detected = perform_calibration(frames_for_calibration, df_rs_reproj, 
                                                                                            intrinsics_estimates_pth, visualise_reprojection_error = visualise_reprojection_error, waitTime = waitTime)
                
                print(f'{cam} {participant} {run_num} Calibration successful')
                print('intrinsics: ', intrinsics)
                print('distortion: ', distortion)
                print('err: ', err)
                print('num_corners_detected_rs: ', num_corners_detected)
                # save intrinsics and distortion as txt file 
                np.savetxt(f'{results_data_pth}/calibration/intrinsics_rs.txt', intrinsics)
                np.savetxt(f'{results_data_pth}/calibration/distortion_rs.txt', distortion)
                np.savetxt(f'{results_data_pth}/calibration/err_rs.txt', [err])

    
    # generate HAND-eye dataframe



    # merge dataframes
    """ for participant in ['matt', 'aure', 'mobarak', 'joao']:
        for run_num in ['1', '2', '3', '4', '5']:
            data_endo = pd.read_pickle(f'{data_path}/{participant}/{run_num}/data/endo_data.pkl')
            data_rs = pd.read_pickle(f'{data_path}/{participant}/{run_num}/data/realsense_data.pkl')
            
            data_df = filter_and_merge_hand_eye_df(data_df_endo, data_df_realsense, min_num_corners) """




    return 


if __name__=='__main__': 
    main() 