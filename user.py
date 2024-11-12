

import glob
import os

import cv2
import pandas as pd
from main_calibration_analysis import generate_board_table
import numpy as np


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
         ): 
    
    aruco_dict = cv2.aruco.DICT_4X4_1000
    board = cv2.aruco.CharucoBoard(
        (aruco_w, aruco_h),  # 7x5
        size_of_checkerboard,
        aruco_size,
        cv2.aruco.getPredefinedDictionary(aruco_dict)
    )
    for cam in ['endo', 'realsense']:
        for participant in ['matt', 'aure', 'mobarak', 'joao']:
            for run_num in ['1', '2', '3', '4', '5']:
            
                data = pd.read_pickle(f'{data_path}/{participant}/{run_num}/data_{cam}.pkl')
                image_pths = data.paths.values.tolist()
                """ image_pths = glob.glob(
                            f'{data_path}/{participant}/{run_num}/{cam}_images/*.{img_ext}') """
                
                # table_data_pth
                results_data_pth = f'{data_path}/{participant}/{run_num}/data/'
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

                # 

    
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