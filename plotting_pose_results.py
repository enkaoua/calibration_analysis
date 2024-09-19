

import glob
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np


def main(pose_results_pth = 'results/hand_eye/pose_analysis/MC_6.0_PC_0.2_size_15_cam_realsense_repeats5_sample_combinations_10'): 

    # load data
    results_df = pd.concat([pd.read_pickle(pth) for pth in glob.glob(f'{pose_results_pth}/*.pkl')], ignore_index=True)

    # get median of each row from errors_lst and add it to new column called 'mean_reprojection_error
    results_df['median_reprojection_error'] = results_df.apply(lambda row: np.median(row['errors_lst']), axis=1)
    # get min of errors_lst
    results_df['min_reprojection_error'] = results_df.apply(lambda row: np.min(row['errors_lst']), axis=1)
    # get max of errors_lst
    results_df['max_reprojection_error'] = results_df.apply(lambda row: np.max(row['errors_lst']), axis=1)

    # do median over all num_poses that are the same
    #results_df['median_reprojection_error_num_poses'] = results_df.groupby('num_poses')['median_reprojection_error'].transform('median')

    # join all rows that are the same num_poses and num_angles. Reprojection error is the median of all the median_reprojection_error for each num_poses and num_angles in that combination
    results_df['median_reprojection_error_num_poses_num_angles'] = results_df.groupby(['num_poses', 'num_angles'])['median_reprojection_error'].transform('median')

    # create a new dataframe where each row is a unique combination of num_poses and num_angles
    results_df_unique = results_df.drop_duplicates(subset=['num_poses', 'num_angles'])
    

    # num poses in ascending order
    num_poses = results_df['num_poses'].unique()
    num_poses.sort()
    # num angles in ascending order
    num_angles = results_df['num_angles'].unique()
    num_angles.sort()

    results = []
    for num_pose in num_poses:
        for num_angle in num_angles:
            # get all rows that have the same num_poses and num_angles
            results_df_subset = results_df[(results_df['num_poses'] == num_pose) & (results_df['num_angles'] == num_angle)]
            # get the median of the median_reprojection_error
            median_reprojection_error = results_df_subset['median_reprojection_error'].median()
            results.append({'num_poses': num_pose, 'num_angles': num_angle, 'median_reprojection_error': median_reprojection_error})


    R = pd.DataFrame(results)
    # Visualize results as a heatmap
    heatmap_data = R.pivot_table(index='num_poses', columns='num_angles', values='median_reprojection_error')

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=vmin, vmax=vmax)
    plt.title('Mean Reprojection Error by Number of Poses and Angles')
    plt.xlabel('Number of Angles')
    plt.ylabel('Number of Poses')
    #plt.show()
    plt.savefig(f'{pose_results_pth}/heatmap_second_version.png')

    return 



if __name__ == '__main__':
    hand_eye = False

    if hand_eye == True:
        calibration_pth = 'results/hand_eye'

        min_num_corners = 6.0  # if none selected, the percentage of corners is used (with min 6 corners)
        percentage_of_corners = 0.2
        threshold = 30
        # analysis parameters
        R = None
        repeats = 5  # number of repeats per number of images analysis
        num_images_start = 50
        num_images_end = 51
        num_images_step = 1
        """ endo = False
        rs = True """
        cam = 'Ropt'
        shift = [0.3, 0.1]

        sample_combinations=10
        chess_size = 30
        vmin = 7
        vmax=12

    else:
        calibration_pth = 'results/intrinsics'
        min_num_corners = 6.0
        percentage_of_corners = 0.5
        threshold = 2

        # analysis parameters
        R = None
        num_images_start = 5
        num_images_end = 60
        num_images_step = 5
        repeats = 5  # number of repeats per number of images analysis
        endo = True
        rs = True
        shift = [0.3, 0.1]

        sample_combinations=20
        chess_size = 20
        cam = 'realsense'
        vmin = 0.2
        vmax=2

    rec_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    rec_analysis = f'R{R}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}_{rec_data}'

    main(pose_results_pth=f'{calibration_pth}/pose_analysis/{rec_data}_size_{chess_size}_cam_{cam}_repeats{repeats}_sample_combinations_{sample_combinations}')
    # plot_info_as_bar(info_pth=f'{calibration_pth}/raw_corner_data/{rec_data}')
