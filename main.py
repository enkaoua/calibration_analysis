import glob
import os
import random
import numpy as np
import pandas as pd
from charuco_utils import detect_corners_charuco_cube_images, generate_charuco_board


def generate_board_table(size_chess, camera, image_pths, board):
    """
    Generate a table of the chessboard corners in the world coordinate system.
    """
    
    #_, imgPoints, objPoints, image_shape = detect_corners_charuco_cube_images( board, image_pths)
    updated_image_pths, imgPoints, objPoints, image_shape = detect_corners_charuco_cube_images( board, image_pths, waiting_time=1)
    # convert updated image paths- split path to get: [image number, pose number, degree number, going forward or backward]
    # convert to pandas dataframe
    data = {'paths':updated_image_pths, 'imgPoints':imgPoints, 'objPoints':objPoints}
    data_df = pd.DataFrame(data=data)

    # save original number of images and the number of images with detected corners aswell as the number of corners detected in total
    original_number_of_images = len(image_pths)
    number_of_images_with_corners = len(updated_image_pths)
    number_of_corners_detected = sum([len(i) for i in imgPoints])
    titles = ['original_number_of_images', 'number_of_images_with_corners', 'number_of_corners_detected']
    values = [original_number_of_images, number_of_images_with_corners, number_of_corners_detected]
    info = {'titles':titles, 'data':values}
    info_df = pd.DataFrame(data=info)
    
    
    data_df.to_pickle(f'results/{size_chess}_{camera}_data.pkl')
    info_df.to_csv(f'results/{size_chess}_{camera}_info.csv')

    return data_df, info_df


    # generate 11 random numbers that are in the range of the above loaded images
    #random_indices = random.sample(range(len(path_to_endo_images)), R+n)

def main(): 
    data_path = '/Users/aure/Documents/CARES/data/massive_calibration_data'
    #analyse_calibration_data(data_path, img_ext='png')
    img_ext = 'png'
    size_chess = 15
    camera = 'realsense' # realsense
    image_pths = glob.glob(f'{data_path}/{size_chess}_charuco/pose0/acc_{size_chess}_pos*_deg*_*/raw/he_calibration_images/hand_eye_{camera}/*.{img_ext}')
    board= generate_charuco_board(size_chess)
    
    # check if f'results/{size_chess}_{camera}_data.pkl' in os.listdir('results')
    # if not, generate the table and save it
    # if yes, load the table and continue


    if f'{size_chess}_{camera}_data.pkl' in os.listdir('results'):
        data_df = pd.read_pickle(f'results/{size_chess}_{camera}_data.pkl')
        info_df = pd.read_csv(f'results/{size_chess}_{camera}_info.csv')
    else:
        data_df, info_df = generate_board_table(size_chess, camera, image_pths, board)
    
    # perform calibration analysis



    #np.savetxt(f'results/{size_chess}_info.txt', ['original_number_of_images', original_number_of_images, number_of_images_with_corners, number_of_corners_detected])
    #[pose, deg, back/forward, image_number]
    
    #df_loaded = pd.read_pickle('results/{size_chess}_data.pkl')

    # save in a file the updated image paths, corresponding image points and object points
    #np.savez(f'results/{size_chess}_charuco_corners.npz', updated_image_pths, imgPoints, objPoints)
    return 


if __name__=='__main__': 
    main() 