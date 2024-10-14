


import os
from utils import find_best_intrinsics
import glob
import numpy as np

def find_and_save_best_intrinsics(data_pth = 'results/intrinsics/calibration_analysis/',
        cameras=['endo', 'realsense'], 
         chess_sizes= [15,20,25,30],
         save_path = f'results/intrinsics/best_intrinsics'): 
    # create save pth if doesnt exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    best_intrinsics = None
    best_distortion = None
    best_error = np.inf
    for camera in cameras:
        for chess_size in chess_sizes:
            # get the data
            pths_with_intrinsics_analysis = glob.glob(f'{data_pth}/*')

            for intrinsics_analysis_pth in pths_with_intrinsics_analysis:
                print(f'pth:::::::. {intrinsics_analysis_pth}')
                intrinsics, distortion, error = find_best_intrinsics(intrinsics_analysis_pth, chess_size, camera,
                                                                save_path='')
                if error < best_error:
                    best_error = error
                    best_intrinsics = intrinsics
                    best_distortion = distortion
            # save best intrinsics
            print(f'camera: {camera}, chess_size: {chess_size}')
            print('##################################')
            print(f'best error: {best_error}')
            print(f'best intrinsics: {best_intrinsics}')
            print(f'best distortion: {best_distortion}')

            if len(save_path) > 0:
                # save as txt file
                np.savetxt(f'{save_path}/{chess_size}_{camera}_intrinsics.txt', intrinsics)
                np.savetxt(f'{save_path}/{chess_size}_{camera}_distortion.txt', distortion)

    return 


if __name__=='__main__': 
    find_and_save_best_intrinsics(
        data_pth = 'results/intrinsics/calibration_analysis/',
        cameras=['endo', 'realsense'], 
         chess_sizes= [15,20,25,30],
         save_path = f'results/intrinsics/best_intrinsics'
    ) 