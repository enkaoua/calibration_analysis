import glob
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from charuco_utils import detect_charuco_board_pose_images, detect_corners_charuco_cube_images, generate_charuco_board, perform_hand_eye_calibration_analysis, perform_analysis
import matplotlib.pyplot as plt

from utils import create_folders, filter_and_merge_hand_eye_df, find_best_intrinsics, sample_dataset


def generate_board_table(image_pths, board,table_data_pth,table_info_pth,  min_num_corners=6,percentage_of_corners=0.2, waiting_time=1, 
                         intrinsics=None, distortion=None):
    """
    Generate a table of the chessboard corners in the world coordinate system.
    """
    
    if intrinsics is not None and distortion is not None:
        # this will also return board pose information (for hand-eye calibration)
        updated_image_pths, min_corners, T, imgPoints, objPoints, ids =    detect_charuco_board_pose_images( board, image_pths,intrinsics, distortion,return_corners=True, min_num_corners=min_num_corners, waiting_time=waiting_time)
    else:
        # for intrinsics calibration
        updated_image_pths, imgPoints, objPoints, min_corners = detect_corners_charuco_cube_images( board, image_pths, min_num_corners=min_num_corners,percentage_of_corners=percentage_of_corners, waiting_time=waiting_time)

    # convert updated image paths- split path to get: [image number, pose number, degree number, going forward or backward]
    # convert to pandas dataframe
    data = {'paths':updated_image_pths, 'imgPoints':imgPoints, 'objPoints':objPoints}
    data_df = pd.DataFrame(data=data)
    # adding columns to describe pose, chess size, degree, direction
    data_df[['chess_size', 'pose', 'deg', 'direction']] = data_df['paths'].str.extract('acc_(\d+)_pos(\d+)_deg(\d+)_(\d+)')
    # also adding frame number
    data_df['frame_number'] = data_df['paths'].str.extract('(\d+).png')
    # convert to integers
    data_df[["pose", "chess_size", "deg", "direction"]] = data_df[["pose", "chess_size", "deg", "direction"]].apply(pd.to_numeric)
    # if intrinsics path, we want to also add the board pose
    if intrinsics is not None and distortion is not None:
        data_df['T'] = T
        data_df['ids'] = ids

    # save original number of images and the number of images with detected corners aswell as the number of corners detected in total
    original_number_of_images = len(image_pths)
    number_of_images_with_corners = len(updated_image_pths)
    number_of_corners_detected = sum([len(i) for i in imgPoints])
    titles = ['original_number_of_images', 'number_of_images_with_corners', 'number_of_corners_detected', 'minimum_corners_required']
    values = [original_number_of_images, number_of_images_with_corners, number_of_corners_detected, min_corners]
    info = {'titles':titles, 'data':values}
    info_df = pd.DataFrame(data=info)
    
    data_df.to_pickle(table_data_pth)
    info_df.to_csv(table_info_pth)

    return data_df, info_df



def main_hand_eye():
    data_path = '/Users/aure/Documents/CARES/data/massive_calibration_data'
    img_g_ext = 'png'
    cameras = ['endo', 'realsense']
    chess_sizes = [15,20, 25, 30]

    # analysis parameters
    reprojection_sample_size = 6
    min_num_corners = None # if none selected, the percentage of corners is used (with min 6 corners)
    percentage_of_corners = 0.2
    repeats=1 # number of repeats per number of images analysis
    num_images_start=1
    num_images_end=2 
    num_images_step=1
    visualise_reprojection_error=False
    waitTime = 0

    rec_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    rec_analysis = f'R{reprojection_sample_size}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}'

    table_pth = f'results/hand_eye/raw_corner_data/{rec_data}'
    filtered_table_pth = f'results/hand_eye/split_data/{rec_data}'
    analysis_results_pth = f'results/hand_eye/calibration_analysis/{rec_analysis}'

    create_folders([table_pth, filtered_table_pth, analysis_results_pth]) #, analysis_results_pth
    intrinsics_pth = f'results/intrinsics/calibration_analysis/R1000_N5_50_2_repeats_50'

    ######## GENERATE TABLES ########
    for size_chess in chess_sizes:
        for camera in cameras:
            image_pths = glob.glob(f'{data_path}/{size_chess}_charuco/pose*/acc_{size_chess}_pos*_deg*_*/raw/he_calibration_images/hand_eye_{camera}/*.{img_g_ext}')
            board= generate_charuco_board(size_chess)
            
            #analysis_results_pth = f'results/hand_eye/calibration_analysis/{rec_analysis}'
            
            intrinsics, distortion = find_best_intrinsics(intrinsics_pth, size_chess, camera, save_path=f'results/intrinsics/best_intrinsics')
            
            # path where to save tables of board data ['paths', 'imgPoints', 'objPoints', 'chess_size', 'pose', 'deg','direction', 'frame_number']
            table_data_pth = f'{table_pth}/{size_chess}_{camera}_corner_data.pkl'
            # path where to save info about the board data (original number of images, number of images with corners, number of corners detected)
            table_info_pth = f'{table_pth}/{size_chess}_{camera}_corner_info.csv'
            
            ###### TABLE GENERATION ######
            if os.path.isfile(table_data_pth) and os.path.isfile(table_info_pth):
                data_df = pd.read_pickle(table_data_pth)
                info_df = pd.read_csv(table_info_pth)
            else:
                data_df, info_df = generate_board_table(image_pths, board,table_data_pth,table_info_pth,  
                                 min_num_corners=min_num_corners,percentage_of_corners=percentage_of_corners, waiting_time=1, 
                                 intrinsics=intrinsics, distortion=distortion )
            print(f'table done for camera {camera}, size_chess {size_chess}')
    ##### H-E CALIBRATION AND ANALYSIS #####
    for size_chess in chess_sizes: 

        data_df_endo = pd.read_pickle(f'{table_pth}/{size_chess}_endo_corner_data.pkl')
        data_df_realsense = pd.read_pickle(f'{table_pth}/{size_chess}_realsense_corner_data.pkl')
        info_df_endo = pd.read_csv(f'{table_pth}/{size_chess}_endo_corner_info.csv')
        
        # FILTER TABLES AND MERGE
        data_df_combined = filter_and_merge_hand_eye_df(data_df_endo, data_df_realsense, info_df_endo)
        # Split to reprojection and calibration
        # filtered data (table data just split into reprojection and calibration)
        filtered_reprojection_dataset_pth = f'{filtered_table_pth}/{size_chess}_merged_corner_data_reprojection_dataset.pkl'
        filtered_calibration_dataset_pth = f'{filtered_table_pth}/{size_chess}_merged_corner_data_calibration_dataset.pkl'
        if os.path.isfile(filtered_reprojection_dataset_pth) and os.path.isfile(filtered_calibration_dataset_pth):
            reprojection_data_df = pd.read_pickle(filtered_reprojection_dataset_pth)
            remaining_samples = pd.read_pickle(filtered_calibration_dataset_pth)
        else:
            # sample dataset to split to reprojection and calibration
            reprojection_data_df, remaining_samples  = sample_dataset(data_df_combined, total_samples=reprojection_sample_size)
            # save the selected samples
            reprojection_data_df.to_pickle(filtered_reprojection_dataset_pth)
            remaining_samples.to_pickle(filtered_calibration_dataset_pth)

        ###### HAND-EYE CALIBRATION ######
        # analysis data (['errros', 'intrinsics', 'distortion', 'average_error', 'std_error'])
        calibration_analysis_results_save_pth = f'{analysis_results_pth}/{size_chess}_{camera}_calibration_data.pkl'

        if os.path.isfile(calibration_analysis_results_save_pth):
            calibration_data_df = pd.read_pickle(calibration_analysis_results_save_pth)
        else:
            # perform calibration analysis
            perform_hand_eye_calibration_analysis(remaining_samples, 
                                                        reprojection_data_df, 
                                                        intrinsics_pth, 
                                                        size_chess, repeats=repeats, 
                                                        num_images_start=num_images_start, 
                                                        num_images_end=num_images_end, 
                                                        num_images_step=num_images_step,  
                                                        waitTime=waitTime, 
                                                        visualise_reprojection_error=visualise_reprojection_error, 
                                                        results_pth=calibration_analysis_results_save_pth)
            
        print(f'analysis done for camera {camera}, size_chess {size_chess}')



def main_intrinsics(): 
    data_path = '/Users/aure/Documents/CARES/data/massive_calibration_data'
    img_ext = 'png'
    reprojection_sample_size = 100
    min_num_corners = None # if none selected, the percentage of corners is used (with min 6 corners)
    percentage_of_corners = 0.2

    # analysis parameters
    repeats=3 # number of repeats per number of images analysis
    num_images_start=1000
    num_images_end=10001
    num_images_step=1
    visualise_reprojection_error=False
    waitTime = 1

    rec_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    rec_analysis = f'R{reprojection_sample_size}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}'

    table_pth = f'results/intrinsics/raw_corner_data/{rec_data}'
    filtered_table_pth = f'results/intrinsics/split_data/{rec_data}'
    analysis_results_pth = f'results/intrinsics/calibration_analysis/{rec_analysis}'
    #analysis_results_pth = f'results/intrinsics/best_intrinsics/{rec_analysis}'

    create_folders([table_pth, filtered_table_pth, analysis_results_pth])
    
    chess_sizes = [15, 20, 25, 30] # 15, 20, 25, 30
    cameras = ['endo', 'realsense'] # 'endo', 'realsense'
    for camera in cameras:
        #plt.figure()
        for size_chess in chess_sizes:
            # path where to save tables of board data ['paths', 'imgPoints', 'objPoints', 'chess_size', 'pose', 'deg','direction', 'frame_number']
            table_data_pth = f'{table_pth}/{size_chess}_{camera}_corner_data.pkl'
            # path where to save info about the board data (original number of images, number of images with corners, number of corners detected)
            table_info_pth = f'{table_pth}/{size_chess}_{camera}_corner_info.csv'
    
            # filtered data (table data just split into reprojection and calibration)
            filtered_reprojection_dataset_pth = f'{filtered_table_pth}/{size_chess}_{camera}_corner_data_reprojection_dataset.pkl'
            filtered_calibration_dataset_pth = f'{filtered_table_pth}/{size_chess}_{camera}_corner_data_calibration_dataset.pkl'

            # analysis data (['errros', 'intrinsics', 'distortion', 'average_error', 'std_error'])
            calibration_analysis_results_save_pth = f'{analysis_results_pth}/{size_chess}_{camera}_calibration_data.pkl'

            image_pths = glob.glob(f'{data_path}/{size_chess}_charuco/pose*/acc_{size_chess}_pos*_deg*_*/raw/he_calibration_images/hand_eye_{camera}/*.{img_ext}')
            board= generate_charuco_board(size_chess)
            
            # board data table generation
            # generate the board data by detecting the corners in the images or load previously generated data
            if os.path.isfile(table_data_pth) and os.path.isfile(table_info_pth):
                data_df = pd.read_pickle(table_data_pth)
                info_df = pd.read_csv(table_info_pth)
            else:
                data_df, info_df = generate_board_table(image_pths,board,table_data_pth, table_info_pth, min_num_corners=min_num_corners,percentage_of_corners=percentage_of_corners, waiting_time=1)
            
            # split the data into reprojection and calibration
            if os.path.isfile(filtered_reprojection_dataset_pth) and os.path.isfile(filtered_calibration_dataset_pth):
                reprojection_data_df = pd.read_pickle(filtered_reprojection_dataset_pth)
                remaining_samples = pd.read_pickle(filtered_calibration_dataset_pth)
            else:
                # sample dataset to split to reprojection and calibration
                reprojection_data_df, remaining_samples  = sample_dataset(data_df, total_samples=reprojection_sample_size)
                # save the selected samples
                reprojection_data_df.to_pickle(filtered_reprojection_dataset_pth)
                remaining_samples.to_pickle(filtered_calibration_dataset_pth)
            print(f'table done for camera {camera}, size_chess {size_chess}')

            
            if os.path.isfile(calibration_analysis_results_save_pth):
                calibration_data_df = pd.read_pickle(calibration_analysis_results_save_pth)
            else:
                # perform calibration analysis
                perform_analysis(camera, size_chess, 
                                remaining_samples,reprojection_data_df, repeats=repeats, 
                                num_images_start=num_images_start, num_images_end=num_images_end, num_images_step=num_images_step,
                                visualise_reprojection_error=visualise_reprojection_error, waitTime=waitTime,
                                results_pth = calibration_analysis_results_save_pth)
            print(f'analysis done for camera {camera}, size_chess {size_chess}')

            

    
    return 


def main_poses():
    return


if __name__=='__main__': 
    #main_intrinsics() 
    main_hand_eye()