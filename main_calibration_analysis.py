import glob
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from charuco_utils import detect_charuco_board_pose_images, detect_corners_charuco_cube_images, generate_charuco_board, perform_hand_eye_calibration_analysis, perform_analysis
import matplotlib.pyplot as plt

from utils import create_folders, filter_and_merge_hand_eye_df, find_best_intrinsics, sample_dataset, select_min_num_corners


def generate_board_table(image_pths, board,table_data_pth,table_info_pth,  min_num_corners=6,percentage_of_corners=0.2, waiting_time=1, 
                         intrinsics=None, distortion=None, visualise_corner_detection=False):
    """
    Generate a table of the detected chessboard corners data in the world coordinate system.
    The pandas dataframe table is saved in table_data_pth and contains the following info:
    {
    'paths': the updated image paths excluding all images that were removed due to not enough corners detected
    'imgPoints': detected points in 2D image space
    'objPoints: detected points in 3D charuco space
    'num_detected_corners': number of detected corners
    'chess_size': size of chessboard squares
    'pose': which pose of the 9 possible grid poses this is (0-8)
    'deg': which possible angle of the 11 it is (0-10)
    'direction': whether the video is going forward (1) or backwards (2)
    'frame_number': frame number of image (00001)

    and if we are doing hand-eye, we would also have:
    'T': transform pose between board and camera
    'ids': ids of detected corners (allIDs3D_np_sorted_filtered)
    }

    There is also an info file saved with the following as a csv file:
    'titles': ['original_number_of_images', 'number_of_images_with_corners', 'number_of_corners_detected', 'minimum_corners_required']
    'data': values of the above info
    """
    
    if intrinsics is not None and distortion is not None:
        # this will also return board pose information (for hand-eye calibration)
        updated_image_pths, min_corners, T, imgPoints, objPoints,num_detected_corners, ids =  detect_charuco_board_pose_images( board, image_pths,intrinsics, distortion,return_corners=True, min_num_corners=min_num_corners,percentage_of_corners=percentage_of_corners, waiting_time=waiting_time, visualise_corner_detection=visualise_corner_detection)
    else:
        # for intrinsics calibration
        updated_image_pths, imgPoints, objPoints,num_detected_corners, min_corners = detect_corners_charuco_cube_images( board, image_pths, min_num_corners=min_num_corners,percentage_of_corners=percentage_of_corners, waiting_time=waiting_time, visualise_corner_detection=visualise_corner_detection)

    # convert updated image paths- split path to get: [image number, pose number, degree number, going forward or backward]
    # convert to pandas dataframe
    data = {'paths':updated_image_pths, 'imgPoints':imgPoints, 'objPoints':objPoints, 'num_detected_corners':num_detected_corners}
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



def main_hand_eye(data_path = '/Users/aure/Documents/CARES/data/massive_calibration_data',
    img_g_ext = 'png',
    cameras = ['endo', 'realsense'],
    chess_sizes = [15,20, 25, 30],
    visualise_corner_detection=False,
    # analysis parameters
    reprojection_sample_size = 6,
    min_num_corners = None, # if none selected, the percentage of corners is used (with min 6 corners)
    percentage_of_corners = 0.2,
    repeats=1, # number of repeats per number of images analysis
    num_images_start=1,
    num_images_end=2, 
    num_images_step=1,
    results_pth = 'results/intrinsics', 
    intrinsics_results_pth='results/intrinsics',
    visualise_reprojection_error=False,
    waitTime = 0):
    

    rec_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    rec_analysis = f'R{reprojection_sample_size}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}'

    table_pth = f'{results_pth}/raw_corner_data/{rec_data}'
    filtered_table_pth = f'{results_pth}/split_data/{rec_data}'
    analysis_results_pth = f'{results_pth}/calibration_analysis/{rec_analysis}'

    create_folders([table_pth, filtered_table_pth, analysis_results_pth]) #, analysis_results_pth
    intrinsics_pth = f'{intrinsics_results_pth}/calibration_analysis/R1000_N5_50_2_repeats_50'

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
                                 intrinsics=intrinsics, distortion=distortion, visualise_corner_detection=visualise_corner_detection )
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



def main_intrinsics(
        data_path = '/Users/aure/Documents/CARES/data/massive_calibration_data',
        img_ext = 'png',
        reprojection_sample_size = 100,
        min_num_corners = None, # if none selected, the percentage of corners is used (with min 6 corners)
        percentage_of_corners = 0.2,
        visualise_corner_detection=False,
        # analysis parameters
        repeats=3, # number of repeats per number of images analysis
        num_images_start=1000,
        num_images_end=10001,
        num_images_step=1,
        visualise_reprojection_error=False,
        waitTime = 1, 
        results_pth = 'results/intrinsics', 
        chess_sizes = [15, 20, 25, 30], # 15, 20, 25, 30,
        cameras = ['endo', 'realsense'], 
        use_different_board_for_reprojection=True ):
    

    # name of recording (MC- min num of corners, PC- percentage of corners) to be used for generated data (raw and filtered)
    rec_data = f'MC_None_PC_None'
    rec_filtered_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    # name of recording where analysis data is stored
    rec_analysis = f'R{reprojection_sample_size}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}'

    # generate paths where to save table data, split table data and analysis results
    table_pth = f'{results_pth}/raw_corner_data/{rec_data}'
    filtered_table_pth = f'{results_pth}/filtered_data/{rec_filtered_data}'
    split_table_pth = f'{results_pth}/split_data/{rec_filtered_data}'

    analysis_results_pth = f'{results_pth}/calibration_analysis/{rec_analysis}'
    create_folders([table_pth, filtered_table_pth,split_table_pth, analysis_results_pth])
    
    for camera in tqdm(cameras, desc='cameras'):
        #plt.figure()
        for size_chess in tqdm(chess_sizes, desc='chess_sizes', leave=True):
            # path where to save tables of board data ['paths', 'imgPoints', 'objPoints', 'chess_size', 'pose', 'deg','direction', 'frame_number']
            table_data_pth = f'{table_pth}/{size_chess}_{camera}_corner_data.pkl'
            # path where to save info about the board data (original number of images, number of images with corners, number of corners detected)
            table_info_pth = f'{table_pth}/{size_chess}_{camera}_corner_info.csv'
    
            image_pths = glob.glob(f'{data_path}/{size_chess}_charuco/pose*/acc_{size_chess}_pos*_deg*_*/raw/he_calibration_images/hand_eye_{camera}/*.{img_ext}')
            board= generate_charuco_board(size_chess)
            
            # board data table generation
            # generate the board data by detecting the corners in the images or load previously generated data
            if os.path.isfile(table_data_pth) and os.path.isfile(table_info_pth):
                data_df = pd.read_pickle(table_data_pth)
                info_df = pd.read_csv(table_info_pth)
            else:
                data_df, info_df = generate_board_table(image_pths,board,table_data_pth, table_info_pth, min_num_corners=None,percentage_of_corners=None, waiting_time=1, visualise_corner_detection=False)
            
            # filter data
            if os.path.isfile(filtered_table_pth):
                data_df = pd.read_pickle(filtered_table_pth)
            else:
                
                # select minimum number of corners to be detected 
                charuco_corners_3D = board.getChessboardCorners() # allCorners3D_np
                num_chess_corners = len(charuco_corners_3D) # number_of_corners_per_face
                selected_min_num_corners = select_min_num_corners(min_num_corners, percentage_of_corners,num_chess_corners)
                
                # filter whatever is less than min number of samples
                data_df = data_df[data_df['num_detected_corners']>selected_min_num_corners]
                data_df.to_pickle(f'{filtered_table_pth}/{size_chess}_{camera}_corner_data.pkl')
    

    for camera in tqdm(cameras, desc='cameras'):
        # merge all filtered_datasets into one large dataset
        all_data_df = pd.concat([pd.read_pickle(data_pth) for data_pth in glob.glob(f'{filtered_table_pth}/*_{camera}_corner_data.pkl')], ignore_index=True)

        for size_chess in tqdm(chess_sizes, desc='chess_sizes', leave=True):
            
            #data_df = pd.read_pickle(f'{filtered_table_pth}/{size_chess}_{camera}_corner_data.pkl')
            data_df = all_data_df[all_data_df['chess_size']==size_chess]

            # split filtered data (table filtered data just split into reprojection and calibration)
            split_reprojection_dataset_pth = f'{split_table_pth}/{size_chess}_{camera}_corner_data_reprojection_dataset.pkl'
            split_calibration_dataset_pth = f'{split_table_pth}/{size_chess}_{camera}_corner_data_calibration_dataset.pkl'

            # analysis data (['errros', 'intrinsics', 'distortion', 'average_error', 'std_error'])
            calibration_analysis_results_save_pth = f'{analysis_results_pth}/{size_chess}_{camera}_calibration_data.pkl'

            # split the data into reprojection and calibration, and filter out points
            if os.path.isfile(split_reprojection_dataset_pth) and os.path.isfile(split_calibration_dataset_pth):
                reprojection_data_df = pd.read_pickle(split_reprojection_dataset_pth)
                remaining_samples = pd.read_pickle(split_calibration_dataset_pth)
            else:                    
                
                if use_different_board_for_reprojection:
                    remaining_samples = data_df

                    reprojection_data_df = all_data_df[all_data_df['chess_size']!=size_chess]
                    """ if size_chess == 25:
                        reprojection_data_df = pd.read_pickle(f'{filtered_table_pth}/25_{camera}_corner_data.pkl')
                        data_df[data_df['num_detected_corners']>selected_min_num_corners]
                    else:
                        reprojection_data_df = pd.read_pickle(f'{filtered_table_pth}/30_{camera}_corner_data.pkl') """
                else:
                    # sample dataset to split to reprojection and calibration
                    reprojection_data_df, remaining_samples  = sample_dataset(data_df, total_samples=reprojection_sample_size)
                # save the selected samples
                reprojection_data_df.to_pickle(split_reprojection_dataset_pth)
                remaining_samples.to_pickle(split_calibration_dataset_pth)
            print(f'table done for camera {camera}, size_chess {size_chess}')

            
            if os.path.isfile(calibration_analysis_results_save_pth):
                calibration_data_df = pd.read_pickle(calibration_analysis_results_save_pth)
            else:
                # perform calibration analysis
                perform_analysis(camera, 
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