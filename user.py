

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
                # 1) find rows which match the frame_number between data_df and data
                matching_frame_num = pd.merge(data_df, data, on='frame_number', how='left')
                # 2) add the poses to the data_df
                # 3) save the data_df

                

                T =  np.array(data['poses'])

    return 


if __name__=='__main__': 
    main() 