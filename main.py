import glob
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from charuco_utils import analyse_calibration_data, detect_charuco_board_pose_images, detect_corners_charuco_cube_images, generate_charuco_board
import matplotlib.pyplot as plt

from utils import create_folders, sample_dataset


def generate_board_table(image_pths, board,table_data_pth,table_info_pth,  min_num_corners=6,percentage_of_corners=0.2, waiting_time=1, 
                         intrinsics=None, distortion=None):
    """
    Generate a table of the chessboard corners in the world coordinate system.
    """
    
    if intrinsics is not None and distortion is not None:
        # this will also return board pose information (for hand-eye calibration)
        updated_image_pths, min_corners, T, imgPoints, objPoints =    detect_charuco_board_pose_images( board, image_pths,intrinsics, distortion,return_corners=True, min_num_corners=min_num_corners, waiting_time=waiting_time)
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

def perform_analysis(camera, size_chess, data_df, reprojection_data_df, repeats=1000, num_images_start=5, num_images_end=60, num_images_step=2, 
                     visualise_reprojection_error=False, waitTime=1, results_pth='' ): #, info_df,board,image_pths
    
    # perform calibration analysis
    if camera =='realsense':
        intrinsics_initial_guess_pth = f'calibration_estimates/intrinsics_realsense.txt'
    else:
        intrinsics_initial_guess_pth = f'calibration_estimates/intrinsics_endo.txt'

    num_images_lst = np.arange(num_images_start, num_images_end, num_images_step)
    average_error_lst = []
    error_lst = []
    all_intrinsics = []
    all_distortion = []
    std_error_lst = []
    for num_images in num_images_lst:
        errors, intrinsics, distortion = analyse_calibration_data(data_df,
                             reprojection_data_df, # number of frames to use for calculating reprojection loss
                             n = num_images, # number of frames to use for calibration
                             repeats = repeats, # number of repeats for the calibration
                             #size_chess = size_chess,
                             #calibration_save_pth = '',
                             intrinsics_initial_guess_pth=intrinsics_initial_guess_pth,
                             visualise_reprojection_error=visualise_reprojection_error,
                             waitTime=waitTime # time to display each image for (in seconds) when showing reprojection
                             )
        # ignore infinite errors
        errors_filtered = [e for e in errors if not np.isinf(e)]
        # ignore anything larger than 20
        errors_filtered = [e for e in errors if e < 20]
        # print how many were infinite or larger than 20
        print(f'Number of larger errors: {len(errors)-len(errors_filtered)}')
        
        error_lst.append(errors)
        average_error_lst.append(np.mean(errors_filtered))
        std_error_lst.append(np.std(errors_filtered))
        all_intrinsics.append(intrinsics)
        all_distortion.append(distortion)

    # plot number of images vs average reprojection error with standard deviation
    plt.plot(num_images_lst, average_error_lst, label=f'{size_chess}_{camera}')
    plt.errorbar(num_images_lst, average_error_lst, yerr=std_error_lst)
    plt.xlabel('Number of images')
    plt.ylabel('Average reprojection error')
    
    # save intrinsics, distortion and errors
    data = {'num_images_lst':num_images_lst, 'errors':error_lst, 'intrinsics':all_intrinsics,'distortion':all_distortion,  'average_error':average_error_lst, 'std_error':std_error_lst}
    data_df = pd.DataFrame(data=data)
    data_df.to_pickle(results_pth)


def main_hand_eye():
    data_path = '/Users/aure/Documents/CARES/data/massive_calibration_data'
    img_g_ext = 'png'
    cameras = ['endo', 'realsense']
    chess_sizes = [15,20, 25, 30]

    # analysis parameters
    reprojection_sample_size = 1000
    min_num_corners = None # if none selected, the percentage of corners is used (with min 6 corners)
    percentage_of_corners = 0.2
    repeats=50 # number of repeats per number of images analysis
    num_images_start=5
    num_images_end=50 
    num_images_step=2
    visualise_reprojection_error=False
    waitTime = 1

    rec_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    #rec_analysis = f'R{reprojection_sample_size}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}'


    table_pth = f'results/hand_eye/raw_corner_data/{rec_data}'
    filtered_table_pth = f'results/hand_eye/split_data/{rec_data}'
    
    create_folders([table_pth, filtered_table_pth]) #, analysis_results_pth
    intrinsics_pth = f'results/intrinsics/calibration_analysis/R1000_N5_50_2_repeats_50'

    """ plt.figure() """
    for size_chess in chess_sizes:
        for camera in cameras:
            image_pths = glob.glob(f'{data_path}/{size_chess}_charuco/pose*/acc_{size_chess}_pos*_deg*_*/raw/he_calibration_images/hand_eye_{camera}/*.{img_g_ext}')
            board= generate_charuco_board(size_chess)
            
            #analysis_results_pth = f'results/hand_eye/calibration_analysis/{rec_analysis}'
            intrinsics_all_data = pd.read_pickle(f'{intrinsics_pth}/{size_chess}_{camera}_calibration_data.pkl')
            # find where average_error is smallest
            intrinsics_all_data = intrinsics_all_data[intrinsics_all_data.average_error == intrinsics_all_data.average_error.min()]
            errors_all = intrinsics_all_data['errors'].values[0]
            intrinsics = intrinsics_all_data['intrinsics'].values[0][errors_all.index(min(errors_all))]
            distortion = intrinsics_all_data['distortion'].values[0][errors_all.index(min(errors_all))]


            # path where to save tables of board data ['paths', 'imgPoints', 'objPoints', 'chess_size', 'pose', 'deg','direction', 'frame_number']
            table_data_pth = f'{table_pth}/{size_chess}_{camera}_corner_data.pkl'
            # path where to save info about the board data (original number of images, number of images with corners, number of corners detected)
            table_info_pth = f'{table_pth}/{size_chess}_{camera}_corner_info.csv'

            generate_board_table(image_pths, board,table_data_pth,table_info_pth,  
                                 min_num_corners=min_num_corners,percentage_of_corners=percentage_of_corners, waiting_time=1, 
                                 intrinsics=intrinsics, distortion=distortion )
        

        #plt.plot(size_chess, camera)
    """ plt.legend()
    plt.show() """


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
    analysis_results_pth = f'results/intrinsics/best_intrinsics/{rec_analysis}'

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
            if os.path.isfile(table_data_pth) and os.path.isfile(table_info_pth):
                data_df = pd.read_pickle(table_data_pth)
                info_df = pd.read_csv(table_info_pth)
            else:
                data_df, info_df = generate_board_table(image_pths,board,table_data_pth, table_info_pth, min_num_corners=min_num_corners,percentage_of_corners=percentage_of_corners, waiting_time=1)
            # generate the board data by detecting the corners in the images or load previously generated data
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

            
        """ plt.legend()
        plt.savefig(f'results/calibration_error_data/calibration_error_{camera}.png') """


    
    return 


if __name__=='__main__': 
    #main_intrinsics() 
    main_hand_eye()