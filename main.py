import glob
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from charuco_utils import analyse_calibration_data, detect_corners_charuco_cube_images, generate_charuco_board
import matplotlib.pyplot as plt


def generate_board_table(size_chess, camera, image_pths, board, min_num_corners=6, waiting_time=1):
    """
    Generate a table of the chessboard corners in the world coordinate system.
    """
    
    #_, imgPoints, objPoints, image_shape = detect_corners_charuco_cube_images( board, image_pths)
    updated_image_pths, imgPoints, objPoints, min_corners = detect_corners_charuco_cube_images( board, image_pths, min_num_corners=min_num_corners, waiting_time=waiting_time)
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
    
    # save original number of images and the number of images with detected corners aswell as the number of corners detected in total
    original_number_of_images = len(image_pths)
    number_of_images_with_corners = len(updated_image_pths)
    number_of_corners_detected = sum([len(i) for i in imgPoints])
    titles = ['original_number_of_images', 'number_of_images_with_corners', 'number_of_corners_detected', 'minimum_corners_required']
    values = [original_number_of_images, number_of_images_with_corners, number_of_corners_detected, min_corners]
    info = {'titles':titles, 'data':values}
    info_df = pd.DataFrame(data=info)
    

    data_df.to_pickle(f'results/corner_data/{size_chess}_{camera}_corner_data.pkl')
    info_df.to_csv(f'results/corner_data/{size_chess}_{camera}_corner_info.csv')

    return data_df, info_df


    # generate 11 random numbers that are in the range of the above loaded images
    #random_indices = random.sample(range(len(path_to_endo_images)), R+n)

def perform_analysis(camera, size_chess, data_df, reprojection_data_df, repeats=1000, num_images_start=5, num_images_end=60, num_images_step=2, 
                     visualise_reprojection_error=False, waitTime=1 ): #, info_df,board,image_pths
    
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
    data = {'errros':error_lst, 'intrinsics':all_intrinsics,'distortion':all_distortion,  'average_error':average_error_lst, 'std_error':std_error_lst}
    data_df = pd.DataFrame(data=data)
    data_df.to_pickle(f'results/calibration_error_data/{size_chess}_{camera}_calibration_data.pkl')



def main(): 
    data_path = '/Users/aure/Documents/CARES/data/massive_calibration_data'
    #analyse_calibration_data(data_path, img_ext='png')
    img_ext = 'png'
    #size_chess = 30
    #camera = 'endo' # realsense/endo
    

    chess_sizes = [15, 20, 25, 30] #15, 20, 25, 
    cameras = ['endo', 'realsense'] # 'endo', 'realsense'
    for camera in cameras:
        plt.figure()
        for size_chess in chess_sizes:
            print(f'camera {camera}, size_chess {size_chess}')
            image_pths = glob.glob(f'{data_path}/{size_chess}_charuco/pose*/acc_{size_chess}_pos*_deg*_*/raw/he_calibration_images/hand_eye_{camera}/*.{img_ext}')
            board= generate_charuco_board(size_chess)
        
            # generate the board data by detecting the corners in the images or load previously generated data
            if f'{size_chess}_{camera}_corner_data_calibration_dataset.pkl' in os.listdir('results/corner_data/filtered_data'):
                data_df = pd.read_pickle(f'results/corner_data/filtered_data/{size_chess}_{camera}_corner_data_calibration_dataset.pkl')
                reprojection_data_df = pd.read_pickle(f'results/corner_data/filtered_data/{size_chess}_{camera}_corner_data_reprojection_dataset.pkl')
                info_df = pd.read_csv(f'results/corner_data/{size_chess}_{camera}_corner_info.csv')
            else:
                data_df, info_df = generate_board_table(size_chess, camera, image_pths, board, min_num_corners=None, waiting_time=1)
    
            # perform calibration analysis
            perform_analysis(camera, size_chess, 
                             data_df,reprojection_data_df, repeats=1000, 
                             num_images_start=5, num_images_end=50, num_images_step=2,
                             visualise_reprojection_error=False, waitTime=1)
        
        
        plt.legend()
        plt.savefig(f'results/calibration_error_data/calibration_error_{camera}.png')
    # check if f'results/{size_chess}_{camera}_data.pkl' in os.listdir('results')
    # if not, generate the table and save it
    # if yes, load the table and continue

    ################################ DETECT CORNERS IN ALL FRAMES
    

    """ errors, intrinsics, distortion = analyse_calibration_data(data_df, 
                             R = 50, # number of frames to use for calculating reprojection loss
                             n = 10, # number of frames to use for calibration
                             repeats = 10, # number of repeats for the calibration
                             size_chess = size_chess, 
                             calibration_save_pth = '',
                             intrinsics_initial_guess_pth=intrinsics_initial_guess_pth,
                             ) """


    #np.savetxt(f'results/{size_chess}_info.txt', ['original_number_of_images', original_number_of_images, number_of_images_with_corners, number_of_corners_detected])
    #[pose, deg, back/forward, image_number]
    
    #df_loaded = pd.read_pickle('results/{size_chess}_data.pkl')

    # save in a file the updated image paths, corresponding image points and object points
    #np.savez(f'results/{size_chess}_charuco_corners.npz', updated_image_pths, imgPoints, objPoints)
    return 


if __name__=='__main__': 
    main() 