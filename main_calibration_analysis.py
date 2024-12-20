import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from charuco_utils import detect_charuco_board_pose_images, detect_corners_charuco_cube_images, generate_charuco_board, \
    perform_hand_eye_calibration_analysis, perform_analysis
from find_best_intrinsic import find_and_save_best_intrinsics
from plotting_results import plot_calibration_analysis_results
from utils import create_folders, filter_and_merge_hand_eye_df, find_best_intrinsics, sample_dataset, \
    select_min_num_corners


def generate_board_table(image_pths, board, table_data_pth, table_info_pth, 
                         #min_num_corners=1, percentage_of_corners=0.2, 
                         waiting_time=1,
                         intrinsics=None, distortion=None, visualise_corner_detection=False, main_format=True):
    """
    Generate a table of the detected chessboard corners data in the world coordinate system.
    The pandas dataframe table is saved in table_data_pth and contains the following info:
    {
    'paths': the updated image paths excluding all images that were removed due to not enough corners detected
    'imgPoints': detected points in 2D image space
    'objPoints: detected points in 3D charuco space
    'num_detected_corners': number of detected corners
    'ids': ids of detected corners (allIDs3D_np_sorted_filtered)
    'chess_size': size of chessboard squares
    'pose': which pose of the 9 possible grid poses this is (0-8)
    'deg': which possible angle of the 11 it is (0-10)
    'direction': whether the video is going forward (1) or backwards (2)
    'frame_number': frame number of image (00001)

    and if we are doing hand-eye, we would also have:
    'T': transform pose between board and camera
    }

    There is also an info file saved with the following as a csv file:
    'titles': ['original_number_of_images', 'number_of_images_with_corners', 'number_of_corners_detected', 'minimum_corners_required']
    'data': values of the above info
    """

    if intrinsics is not None and distortion is not None:
        # this will also return board pose information (for hand-eye calibration)
        updated_image_pths, min_corners, T, imgPoints, objPoints, num_detected_corners, ids = detect_charuco_board_pose_images(
            board, image_pths, intrinsics, distortion, return_corners=True,
             # min_num_corners=1, # min_num_corners set to 1 for raw images percentage_of_corners=percentage_of_corners, 
            waiting_time=waiting_time, visualise_corner_detection=visualise_corner_detection)
    else:
        # for intrinsics calibration
        updated_image_pths, min_corners,   imgPoints, objPoints, num_detected_corners, ids,  = detect_corners_charuco_cube_images(
            board, image_pths, return_corners=True,
            #min_num_corners=min_num_corners, percentage_of_corners=percentage_of_corners,
            waiting_time=waiting_time, visualise_corner_detection=visualise_corner_detection)

    # convert updated image paths- split path to get: [image number, pose number, degree number, going forward or backward]
    # convert to pandas dataframe
    data = {'paths': updated_image_pths, 'imgPoints': imgPoints, 'objPoints': objPoints,
            'num_detected_corners': num_detected_corners, 'ids':ids}
    data_df = pd.DataFrame(data=data)
    # adding columns to describe pose, chess size, degree, direction

    # also adding frame number
    data_df['frame_number'] = data_df['paths'].str.extract('(\d+).png')

    if main_format:
        data_df[['chess_size', 'pose', 'deg', 'direction']] = data_df['paths'].str.extract(
        'acc_(\d+)_pos(\d+)_deg(\d+)_(\d+)')
        # convert to integers
        data_df[["pose", "chess_size", "deg", "direction"]] = data_df[["pose", "chess_size", "deg", "direction"]].apply(
            pd.to_numeric)
    # if intrinsics path, we want to also add the board pose
    if intrinsics is not None and distortion is not None:
        data_df['T'] = T

    # save original number of images and the number of images with detected corners aswell as the number of corners detected in total
    original_number_of_images = len(image_pths)
    number_of_images_with_corners = len(updated_image_pths)
    number_of_corners_detected = sum([len(i) for i in imgPoints])
    titles = ['original_number_of_images', 'number_of_images_with_corners', 'number_of_corners_detected',
              'minimum_corners_required']
    values = [original_number_of_images, number_of_images_with_corners, number_of_corners_detected, min_corners]
    info = {'titles': titles, 'data': values}
    info_df = pd.DataFrame(data=info)

    data_df.to_pickle(table_data_pth)
    info_df.to_csv(table_info_pth)

    return data_df, info_df

""" 
def main_hand_eye(data_path='/Users/aure/Documents/CARES/data/massive_calibration_data',
                  img_g_ext='png',
                  cameras=['endo', 'realsense'],
                  chess_sizes=[15, 20, 25, 30],
                  visualise_corner_detection=False,
                  # analysis parameters
                  reprojection_sample_size=6,
                  min_num_corners=None,  # if none selected, the percentage of corners is used (with min 6 corners)
                  percentage_of_corners=0.2,
                  repeats=1,  # number of repeats per number of images analysis
                  num_images_start=1,
                  num_images_end=2,
                  num_images_step=1,
                  results_pth='results/intrinsics',
                  intrinsics_results_pth='results/intrinsics',
                  visualise_reprojection_error=False,
                  waitTime=0):
    # name of recording (MC- min num of corners, PC- percentage of corners) to be used for generated data (raw and filtered)
    rec_data = f'MC_None_PC_None'
    rec_filtered_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    # name of recording where analysis data is stored
    rec_analysis = f'R{reprojection_sample_size}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}_{rec_filtered_data}'

    # generate paths where to save table data, split table data and analysis results
    table_pth = f'{results_pth}/raw_corner_data/{rec_data}'
    filtered_table_pth = f'{results_pth}/filtered_data/{rec_filtered_data}'
    split_table_pth = f'{results_pth}/split_data/{rec_filtered_data}'

    analysis_results_pth = f'{results_pth}/calibration_analysis/{rec_analysis}'
    create_folders([table_pth, filtered_table_pth, split_table_pth, analysis_results_pth])

    intrinsics_pth = f'{intrinsics_results_pth}/calibration_analysis/R100_N5_60_5_repeats_10_MC_6.0_PC_0.5'

    # TODO FIX THE SPLITTING OF HAND-EYE INTO MERGE
    ######## GENERATE TABLES ########
    for size_chess in chess_sizes:
        for camera in cameras:
            image_pths = glob.glob(
                f'{data_path}/{size_chess}_charuco/pose*/acc_{size_chess}_pos*_deg*_*/raw/he_calibration_images/hand_eye_{camera}/*.{img_g_ext}')
            board = generate_charuco_board(size_chess)

            # analysis_results_pth = f'results/hand_eye/calibration_analysis/{rec_analysis}'

            intrinsics, distortion = find_best_intrinsics(intrinsics_pth, size_chess, camera,
                                                          save_path=f'results/intrinsics/best_intrinsics')

            # path where to save tables of board data ['paths', 'imgPoints', 'objPoints', 'chess_size', 'pose', 'deg','direction', 'frame_number']
            table_data_pth = f'{table_pth}/{size_chess}_{camera}_corner_data.pkl'
            # path where to save info about the board data (original number of images, number of images with corners, number of corners detected)
            table_info_pth = f'{table_pth}/{size_chess}_{camera}_corner_info.csv'

            ###### TABLE GENERATION ######
            if os.path.isfile(table_data_pth) and os.path.isfile(table_info_pth):
                data_df = pd.read_pickle(table_data_pth)
                info_df = pd.read_csv(table_info_pth)
            else:
                data_df, info_df = generate_board_table(image_pths, board, table_data_pth, table_info_pth,
                                                        min_num_corners=min_num_corners,
                                                        percentage_of_corners=percentage_of_corners, waiting_time=1,
                                                        intrinsics=intrinsics, distortion=distortion,
                                                        visualise_corner_detection=visualise_corner_detection)
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
            reprojection_data_df, remaining_samples = sample_dataset(data_df_combined,
                                                                     total_samples=reprojection_sample_size)
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
 """

def main_intrinsics(
        
        data_path='/Users/aure/Documents/CARES/data/massive_calibration_data',
        img_ext='png',
        reprojection_sample_size=None,
        min_num_corners=None,  # if none selected, the percentage of corners is used (with min 6 corners)
        percentage_of_corners=0.2,
        visualise_corner_detection=False,
        # analysis parameters
        repeats=3,  # number of repeats per number of images analysis
        num_images_start=1000,
        num_images_end=10001,
        num_images_step=1,
        visualise_reprojection_error=False,
        waitTime=1,
        results_pth='results/intrinsics',
        chess_sizes=[15, 20, 25, 30],  # 15, 20, 25, 30,
        cameras=['endo', 'realsense'],

        intrinsics_for_he='',
        optimise=True):
    """
    Perform intrinsic calibration analysis on a set of images.
        Args:
            data_path (str): 
                Path to the directory containing calibration data.
            img_ext (str): 
                Image file extension (e.g., 'png', 'jpg').
            reprojection_sample_size (int, optional): 
                Number of samples for reprojection. 
                If None, all samples of other boards are used.
                If number is given, that number of images is randomly selected from the same board dataset (evenly across angles and positions)
                If number is 0, the same dataset that was used for calibration will be used for reprojection 
            min_num_corners (int, optional): 
                Minimum number of corners to be detected. If None, the percentage is used.
            percentage_of_corners (float): Percentage of corners to be detected if min_num_corners is None.
                If both min_num_corners and percentage_of_corners are none, the min number of corners is 1. 
                If only the percentage corners is specified, we take the percentage_of_corners.
                If only the min corners is specified, we ake that as the number
                If both are specified, we take the larger one.
            
            visualise_corner_detection (bool): 
                Whether to visualize corner detection.
            repeats (int): 
                Number of repeats per number of images analysis.
            num_images_start (int): 
                Starting number of images for analysis.
            num_images_end (int): 
                Ending number of images for analysis.
            num_images_step (int): 
                Step size for the number of images in the analysis.
            visualise_reprojection_error (bool): 
                Whether to visualize reprojection error.
            waitTime (int): 
                Wait time for visualization.
            results_pth (str): 
                Path to save the results.
            chess_sizes (list): 
                List of chessboard sizes to be used in the analysis.
            cameras (list): 
                List of camera names to be used in the analysis.
            intrinsics_for_he (str): 
                Path to the directory containing intrinsic parameters for hand-eye calibration.
        Returns:
            None
    """

    # CREATING FOLDERS WHERE TO SAVE DATA 
    # name of recording (MC- min num of corners, PC- percentage of corners) to be used for generated data (raw and filtered)
    rec_data = f'MC_None_PC_None'
    rec_filtered_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    # name of recording where analysis data is stored
    rec_analysis = f'R{reprojection_sample_size}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}_{rec_filtered_data}'

    # generate paths where to save table data, split table data and analysis results
    table_pth = f'{results_pth}/raw_corner_data/{rec_data}'
    filtered_table_pth = f'{results_pth}/filtered_data/{rec_filtered_data}'
    split_table_pth = f'{results_pth}/split_data/R{reprojection_sample_size}_{rec_filtered_data}'

    analysis_results_pth = f'{results_pth}/calibration_analysis/{rec_analysis}'
    create_folders([table_pth, filtered_table_pth, split_table_pth, analysis_results_pth])

    ###### GENERATE TABLES AND GENERATE FILTERED DATA ########
    for camera in tqdm(cameras, desc='cameras'):
        # plt.figure()
        min_num_corners_dict = {}
        for size_chess in tqdm(chess_sizes, desc='chess_sizes', leave=True):
            
            # path where to save tables of board data ['paths', 'imgPoints', 'objPoints', 'chess_size', 'pose', 'deg','direction', 'frame_number']
            table_data_pth = f'{table_pth}/{size_chess}_{camera}_corner_data.pkl'
            # path where to save info about the board data (original number of images, number of images with corners, number of corners detected)
            table_info_pth = f'{table_pth}/{size_chess}_{camera}_corner_info.csv'

            image_pths = glob.glob(
                f'{data_path}/{size_chess}_charuco/pose*/acc_{size_chess}_pos*_deg*_*/raw/he_calibration_images/hand_eye_{camera}/*.{img_ext}')
            board = generate_charuco_board(size_chess)

            # load rough intrinsics for when calculating hand eye
            if len(intrinsics_for_he) > 0:
                """ intrinsics, distortion = find_best_intrinsics(intrinsics_for_he, size_chess, camera,
                                                              save_path=f'results/intrinsics/best_intrinsics') """
                
                # check if intrinsics and distortion exist, otherwise create them 
                if not os.path.exists(f'{intrinsics_for_he}/{size_chess}_{camera}_intrinsics.txt') and not os.path.exists(f'{intrinsics_for_he}/{size_chess}_{camera}_distortion.txt'):
                    find_and_save_best_intrinsics(data_pth = 'results/intrinsics/calibration_analysis/',
                            cameras=['endo', 'realsense'], 
                            chess_sizes= [15,20,25,30],
                            save_path = f'{intrinsics_for_he}')
                intrinsics = np.loadtxt(f'{intrinsics_for_he}/{size_chess}_{camera}_intrinsics.txt')
                distortion = np.loadtxt(f'{intrinsics_for_he}/{size_chess}_{camera}_distortion.txt')
                HAND_EYE = True
            else:
                intrinsics, distortion = None, None
                HAND_EYE = False

            # board data table generation
            # generate the board data by detecting the corners in the images or load previously generated data if exists
            if os.path.isfile(table_data_pth) and os.path.isfile(table_info_pth):
                data_df = pd.read_pickle(table_data_pth)
                #info_df = pd.read_csv(table_info_pth)
            else:
                # creates table of detected corners for all images with at least one corner detected
                data_df, _ = generate_board_table(image_pths, board, table_data_pth, table_info_pth,
                                                        #min_num_corners=None, percentage_of_corners=None,
                                                        waiting_time=waitTime,
                                                        visualise_corner_detection=visualise_corner_detection,
                                                        intrinsics=intrinsics, distortion=distortion)
            print(f'{camera} {size_chess} table done')

            # FILTER DATA
            # select minimum number of corners to be detected 
            charuco_corners_3D = board.getChessboardCorners()  # allCorners3D_np
            num_chess_corners = len(charuco_corners_3D)  # number_of_corners_per_face
            selected_min_num_corners = select_min_num_corners(min_num_corners, percentage_of_corners,
                                                                num_chess_corners)
            min_num_corners_dict[size_chess] = selected_min_num_corners
        
            if os.path.isfile(f'{filtered_table_pth}/{size_chess}_{camera}_corner_data.pkl'):
                data_df = pd.read_pickle(f'{filtered_table_pth}/{size_chess}_{camera}_corner_data.pkl')
            else:
                # filter whatever is less than min number of samples
                data_df = data_df[data_df['num_detected_corners'] > selected_min_num_corners]
                data_df.to_pickle(f'{filtered_table_pth}/{size_chess}_{camera}_corner_data.pkl')
            print(f'{camera} {size_chess} filtered table done')
    
    print('########### MERGING DATA ###################')
    # for HE- merge endo and realsense data and further filter if necessary
    if HAND_EYE == True:
        
        # load all filtered data of all boards for realsense and endo
        all_data_df_endo = pd.concat(
            [pd.read_pickle(data_pth) for data_pth in glob.glob(f'{filtered_table_pth}/*_endo_corner_data.pkl')],
            ignore_index=True)
        all_data_df_realsense = pd.concat(
            [pd.read_pickle(data_pth) for data_pth in glob.glob(f'{filtered_table_pth}/*_realsense_corner_data.pkl')],
            ignore_index=True)
        # merge endo and rs data for each board
        for size_chess in tqdm(chess_sizes, desc='chess_sizes', leave=True):
            # check if file exists already
            if os.path.isfile(f'{filtered_table_pth}/{size_chess}_merged_corner_data.pkl'):
                continue
            # filter data for each board size
            data_df_endo = all_data_df_endo[all_data_df_endo['chess_size'] == size_chess]
            data_df_realsense = all_data_df_realsense[all_data_df_realsense['chess_size'] == size_chess]
            min_num_corners = min_num_corners_dict[size_chess]
            # combined dataframe
            data_df = filter_and_merge_hand_eye_df(data_df_endo, data_df_realsense, min_num_corners)
            # save filtered and merged dataset
            data_df.to_pickle(f'{filtered_table_pth}/{size_chess}_merged_corner_data.pkl')

    print('########### ANALYSIS ###################')
    #### SPLIT DATA AND PERFORM ANALYSIS ########
    for camera in tqdm(cameras, desc='cameras'):
        # merge all chess sizes filtered_datasets into one large dataset
        if HAND_EYE:
            all_data_df = pd.concat(
                [pd.read_pickle(data_pth) for data_pth in glob.glob(f'{filtered_table_pth}/*_merged_corner_data.pkl')],
                ignore_index=True)
        else:
            all_data_df = pd.concat([pd.read_pickle(data_pth) for data_pth in
                                     glob.glob(f'{filtered_table_pth}/*_{camera}_corner_data.pkl')], ignore_index=True)

        for size_chess in tqdm(chess_sizes, desc='chess_sizes', leave=True):
            print(f'analysis for chess {size_chess} ')
            # Select data for specific chess size
            data_df = all_data_df[all_data_df['chess_size'] == size_chess]

            # split filtered data (table filtered data just split into reprojection and calibration)
            split_reprojection_dataset_pth = f'{split_table_pth}/{size_chess}_{camera}_corner_data_reprojection_dataset.pkl'
            split_calibration_dataset_pth = f'{split_table_pth}/{size_chess}_{camera}_corner_data_calibration_dataset.pkl'

            # analysis data (['errors', 'intrinsics', 'distortion', 'average_error', 'std_error'])
            calibration_analysis_results_save_pth = f'{analysis_results_pth}/{size_chess}_{camera}_calibration_data.pkl'

            # SPLIT DATA
            # split the data into reprojection and calibration, and filter out points
            if reprojection_sample_size==0:
                # will use same dataset for calibration and reprojection
                remaining_samples = data_df
                reprojection_data_df = None
            elif os.path.isfile(split_reprojection_dataset_pth) and os.path.isfile(split_calibration_dataset_pth):
                reprojection_data_df = pd.read_pickle(split_reprojection_dataset_pth)
                remaining_samples = pd.read_pickle(split_calibration_dataset_pth)
            else:
                # if no integer has been chosen for reprojection, use all reprojection files from other chessboard as reprojection dataset
                if reprojection_sample_size is None:
                    remaining_samples = data_df
                    reprojection_data_df = all_data_df[all_data_df['chess_size'] != size_chess]
                else:
                    # sample dataset to split to reprojection and calibration with size of reprojection dataset is reprojection_sample_size
                    reprojection_data_df, remaining_samples = sample_dataset(data_df,
                                                                             total_samples=reprojection_sample_size)
                # save the selected samples
                reprojection_data_df.to_pickle(split_reprojection_dataset_pth)
                remaining_samples.to_pickle(split_calibration_dataset_pth)
            print(f'table done for camera {camera}, size_chess {size_chess}')

            # PERFORM CALIBRATION
            if os.path.isfile(calibration_analysis_results_save_pth):
                calibration_data_df = pd.read_pickle(calibration_analysis_results_save_pth)
            else:
                # perform calibration analysis
                if HAND_EYE:
                    # perform calibration analysis

                    perform_hand_eye_calibration_analysis(remaining_samples,
                                                          reprojection_data_df,
                                                          intrinsics_for_he,
                                                          size_chess,
                                                          repeats=repeats,
                                                          num_images_start=num_images_start,
                                                          num_images_end=num_images_end,
                                                          num_images_step=num_images_step,
                                                          visualise_reprojection_error=visualise_reprojection_error,
                                                          waitTime=waitTime,
                                                          results_pth=calibration_analysis_results_save_pth,
                                                          optimise=optimise)

                else:
                    perform_analysis(camera,
                                     remaining_samples,
                                     reprojection_data_df,
                                     repeats=repeats,
                                     num_images_start=num_images_start,
                                     num_images_end=num_images_end,
                                     num_images_step=num_images_step,
                                     visualise_reprojection_error=visualise_reprojection_error,
                                     waitTime=waitTime,
                                     results_pth=calibration_analysis_results_save_pth)
            print(f'analysis done for camera {camera}, size_chess {size_chess}')

        if HAND_EYE:
            break


    # PLOT AND SAVE RESULTS
    plot_endo=True
    plot_rs=True
    if HAND_EYE:
        threshold = 30
        plot_rs = False
    else:
        threshold = 1.4
    plot_calibration_analysis_results(hand_eye=HAND_EYE, 
                                      calibration_pth = results_pth, 
                                      min_num_corners=min_num_corners, 
                                      percentage_of_corners=percentage_of_corners, 
                                      repeats=repeats, threshold=threshold, 
                                      endo=plot_endo, rs=plot_rs, shift=[0.3, 0.1],
                                      R=None, 
                                      num_images_start=num_images_start,
                                      num_images_end=num_images_end,
                                      num_images_step=num_images_step,
                                      ) 
    return


if __name__ == '__main__':
    # main_intrinsics()
    # main_hand_eye()
    print('done')
