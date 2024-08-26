import pandas as pd
import numpy as np
import glob

from charuco_utils import perform_analysis
from utils import T_to_xyz
import matplotlib.pyplot as plt

def print_and_show_data_3D(data, extension, ax, sizes, shapes, idx, chess_size):
    print(f'{extension} DATA STATS')

    print('RANGES:------------------')
    print('Z')
    print((data[f'T_z{extension}'].max() - data[f'T_z{extension}'].min()) / 1000)
    print('Y')
    print((data[f'T_y{extension}'].max() - data[f'T_y{extension}'].min()) / 1000)
    print('X')
    print((data[f'T_x{extension}'].max() - data[f'T_x{extension}'].min()) / 1000)

    print('LENGTHS:------------------')
    print(len(data[f'T_z{extension}']))

    ax.scatter(data[f'T_x{extension}'], data[f'T_y{extension}'], data[f'T_z{extension}'], label=f'{chess_size}mm',
               alpha=0.5, s=sizes[idx], marker=shapes[idx])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def visualise_poses(merged = True):

    # plot in 3D the x, y and z poses
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sizes = [15, 5, 2, 1]
    shapes = ['o', 's', 'p', 'P']
    for idx, chess_size in enumerate([20,30]):

        if merged:
            data_pth = f'/Users/aure/Documents/CARES/code/charuco_calibration_analysis/results/hand_eye/split_data/MC_None_PC_0.2/{chess_size}_merged_corner_data_reprojection_dataset.pkl'
            #data_pth = f'/Users/aure/Documents/CARES/code/charuco_calibration_analysis/results/hand_eye/split_data/MC_None_PC_0.2/{chess_size}_merged_corner_data_calibration_dataset.pkl'
            data = pd.read_pickle(data_pth)

            T_to_xyz(data, '_endo')
            print_and_show_data_3D(data, '_endo', ax, sizes, shapes, idx, chess_size)
            T_to_xyz(data, '_rs')
            #print_and_show_data_3D(data, '_rs', ax, sizes, shapes, idx, chess_size)


        else:
            data_pth_rs = f'/Users/aure/Documents/CARES/code/charuco_calibration_analysis/results/hand_eye/raw_corner_data/MC_None_PC_None/{chess_size}_realsense_corner_data.pkl'
            data_pth_endo = f'/Users/aure/Documents/CARES/code/charuco_calibration_analysis/results/hand_eye/raw_corner_data/MC_None_PC_None/{chess_size}_endo_corner_data.pkl'
            data_rs = pd.read_pickle(data_pth_rs)
            data_endo = pd.read_pickle(data_pth_endo)

            T_to_xyz(data_rs, '')
            T_to_xyz(data_endo, '')
            #print_and_show_data_3D(data_rs, '', ax, sizes, shapes, idx, chess_size)
            print_and_show_data_3D(data_endo, '', ax, sizes, shapes, idx, chess_size)


        ax.legend()
    plt.show()
    
    return 
 


def main(table_pth='results/hand_eye/raw_corner_data/MC_None_PC_None', 
         cameras=['endo', 'realsense'],
         chess_sizes=[15, 20, 25, 30], 
         poses = [0, 1, 2, 3, 4, 5, 6, 7, 8],
         n=50, 
         repeats=10,
         num_images_step=1,
         visualise_reprojection_error=False,
         waitTime=0):
    
    poses_lst = poses.copy()
    for camera in cameras:
        data_df = pd.concat([pd.read_pickle(pth) for pth in glob.glob(f'{table_pth}/*_{camera}_corner_data.pkl')], ignore_index=True)
        # add xyz distance from T
        T_to_xyz(data_df, extension='')
        
        for chess_size in chess_sizes:
            # check range of xyz distances
            data_df_chess_size = data_df[data_df['chess_size'] == chess_size]
            data_for_reprojection = data_df[data_df['chess_size'] != chess_size]

            """ z_min = data_df_filtered['T_z'].min()
            z_max = data_df_filtered['T_z'].max()
            z_step = (z_max - z_min) / 10
            z_range = np.arange(z_min, z_max, z_step)
            # plot histogram of xyz distances
            plt.figure(figsize=(12, 8))
            plt.hist(data_df_filtered['T_z'], bins=20)
            plt.xlabel('z Distance (mm)')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Z Distance distribution for {chess_size}mm chessboard for {camera} camera')
            plt.show() """
            reprojection_errors = []
            for pose in poses:
                # filter data_df_filtered by pose selected 
                data_df_filtered = data_df_chess_size[
                    (data_df_chess_size['pose'].isin([pose]))] #&
                    #(data_for_calibration['deg'].isin(angle)) ]

                # round T_z to nearest 10
                data_df_filtered['T_z'] = np.round(data_df_filtered['T_z'] / 10) * 10

                # only select rows with distance z of max occurrences
                z_max = data_df_filtered['T_z'].mode().values
                if len(z_max) > 0:
                    z_max = z_max[0]
                else:
                    poses_lst.remove(pose)
                    continue
                data_df_filtered[data_df_filtered['T_z']==z_max]


                # calculate reprojection error
                result = perform_analysis(camera,
                                        data_df_filtered, data_for_reprojection, repeats=repeats,
                                        num_images_start=n, num_images_end=n + 1,
                                        num_images_step=num_images_step,
                                        visualise_reprojection_error=visualise_reprojection_error,
                                          waitTime=waitTime,
                                        results_pth='', thread_num=f'{pose}')
                reprojection_errors.append(result['average_error'].values[0])
                """ results = {'num_images_lst': num_images_lst,
               'errors_lst': error_lst,
               'num_corners_detected_lst': num_corners_detected_lst,
               'intrinsics': all_intrinsics,
               'distortion': all_distortion,
               'average_error': average_error_lst,
               'std_error': std_error_lst} """
                
            # plot reprojection error vs pose
            plt.plot(poses_lst, reprojection_errors)
            plt.xlabel('Pose')
            plt.ylabel('Reprojection Error')
            plt.title(f'Reprojection Error vs Pose for {chess_size}mm chessboard for {camera} camera')
            plt.show()
                
                
            

            



if __name__=='__main__': 
    #visualise_poses(merged = False)
    main() 