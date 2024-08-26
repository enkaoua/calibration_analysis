

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

    # Visualize results as a heatmap
    heatmap_data = results_df.pivot(index='num_poses', columns='num_angles', values='median_reprojection_error')

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
    plt.title('Mean Reprojection Error by Number of Poses and Angles')
    plt.xlabel('Number of Angles')
    plt.ylabel('Number of Poses')
    #plt.savefig(f'{pose_results_pth}/heatmap_second_version.png')
    plt.imshow()
    return 



if __name__ == '__main__':
    hand_eye = True

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
        cam = 'realsense'
        shift = [0.3, 0.1]

        sample_combinations=10
        chess_size = 30

    else:
        calibration_pth = 'results/intrinsics'
        min_num_corners = 6.0
        percentage_of_corners = 0.5
        threshold = 2

        # analysis parameters
        R = 100
        num_images_start = 5
        num_images_end = 60
        num_images_step = 5
        repeats = 10  # number of repeats per number of images analysis
        endo = True
        rs = True
        shift = [0.3, 0.1]

        sample_combinations=10
        chess_size = 15

    rec_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    rec_analysis = f'R{R}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}_{rec_data}'

    main(pose_results_pth=f'{calibration_pth}/pose_analysis/{rec_data}_size_{chess_size}_cam_{cam}_repeats{repeats}_sample_combinations_{sample_combinations}')
    # plot_info_as_bar(info_pth=f'{calibration_pth}/raw_corner_data/{rec_data}')
