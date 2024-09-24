import itertools
import random
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os

from tqdm import tqdm
from charuco_utils import perform_analysis, perform_hand_eye_calibration_analysis
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
         cameras=['realsense','endo'],
         chess_sizes=[20,15,25,30], 
         angles = [0, 1, 2, 3, 4, 5, 6, 7, 8],
         n=20, 
         repeats=5,
         num_images_step=1,
         visualise_reprojection_error=False,
         waitTime=0,
         HAND_EYE=False,
         sample_combinations = 5):
    
    if HAND_EYE:
        min_num_corners = 6.0
        percentage_corners = 0.3
        rec_name = f'MC_{min_num_corners}_PC_{percentage_corners}'

        table_pth = f'results/hand_eye/split_data/{rec_name}'
        file_pth = 'corner_data_calibration_dataset'
        extension = '_endo'
        cameras = ['*']

        distance_analysis = 'results/hand_eye/distance_analysis'

    else:
        min_num_corners = 6.0
        percentage_corners = 0.5
        rec_name = f'MC_{min_num_corners}_PC_{percentage_corners}'
        table_pth = f'results/hand_eye/filtered_data/{rec_name}'
        file_pth = 'corner_data'
        extension = ''
        distance_analysis = 'results/intrinsics/angle_distance_analysis'


    # create folder distance_analysis if it doesn't exist
    if not os.path.exists(distance_analysis):
        os.makedirs(distance_analysis)

    angles_lst = angles.copy()
    for camera in cameras:
        data_df = pd.concat([pd.read_pickle(pth) for pth in glob.glob(f'{table_pth}/*_{camera}_{file_pth}.pkl')], ignore_index=True)
        # add xyz distance from T
        T_to_xyz(data_df, extension=extension)

        data_df[f'T_z{extension}'] = np.round(data_df[f'T_z{extension}'] / 10) * 10

        
        for chess_size in chess_sizes:

            distance_analysis_chess = f'{distance_analysis}/{rec_name}_size_{chess_size}_cam_{camera}_repeats{repeats}_sample_combinations_{sample_combinations}'
            if not os.path.exists(distance_analysis_chess):
                os.makedirs(distance_analysis_chess)

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
            

            #angle_combinations = list(itertools.combinations(angles, num_angles))
            #distance_combinations = list(itertools.combinations(distances, num_distances))

            #for angle in angles:

            # only select rows with distance z of max occurrences
            """ z_max = data_df_filtered[f'T_z{extension}'].mode().values
            if len(z_max) > 0:
                z_max = z_max[0]
            else:
                angles_lst.remove(angle)
                continue
            data_df_filtered[data_df_filtered[f'T_z{extension}']==z_max] """
            distances = data_df_chess_size[f'T_z{extension}'].unique().sort
            # only grab middle 10 distances
            grouped_df = data_df_chess_size.groupby(f'T_z{extension}').count()
            grouped_df_filtered = grouped_df[grouped_df['frame_number']>100].reset_index()
            # grab distances
            distances = grouped_df_filtered[f'T_z{extension}'].values

            simple_results = []
            for num_angles in tqdm(range(1, len(angles) + 1), desc='Number of angles'):
                for num_distances in tqdm(range(1, len(distances) + 1), desc='Number of distances', leave=False):
                    
                    # check if pickle file for this already exists
                    save_pth = f'{distance_analysis_chess}/results_P{num_angles}_distance{num_distances}.pkl'
                    if os.path.exists(save_pth):
                        # load results and append to simple_results
                        result = pd.read_pickle(save_pth)
                        reprojection_errors = result['average_error']
                        overall_mean_error = np.mean(reprojection_errors)
                        # Append results
                        simple_results.append({
                            'num_angles': num_angles,
                            'num_distances': num_distances,
                            'mean_reprojection_error': overall_mean_error
                        })

                        print(f'loading results from pickle file for {num_angles} angles and {num_distances} distances')
                        continue
                    
                    
                    angle_combinations = list(itertools.combinations(angles, num_angles))
                    distance_combinations = list(itertools.combinations(distances, num_distances))

                    num_extra_comb = 50 # extra combinations in case some of sample_combinations not have enough images
                    if sample_combinations:
                        # pick a random set of n angle and distance combinations out of the above ones
                        if len(angle_combinations) > sample_combinations+num_extra_comb:
                            angle_combinations = np.array(random.sample(angle_combinations, sample_combinations+num_extra_comb))
                        if len(distance_combinations) > sample_combinations+num_extra_comb:
                            distance_combinations = np.array(random.sample(distance_combinations, sample_combinations+num_extra_comb))
                        
                        # order the selected angle and distance combinations in terms of descending order of the number of images they have
                        # angles to reject- add to list all combinations of angles and distances which don't have more than num_images in total
                        num_images_found = []
                        possible_combinations = []
                        for angle in angle_combinations:
                            for distance in distance_combinations:
                                num_images_for_combination = len(data_df_chess_size[
                                    (data_df_chess_size['deg'].isin(angle)) &
                                    (data_df_chess_size[f'T_z{extension}'].isin(distance))
                                ])
                                if num_images_for_combination<3:
                                    continue
                                num_images_found.append(num_images_for_combination)
                                possible_combinations.append((angle, distance))
                                
                        """ # order the selected angle and distance combinations in terms of descending order of the number of images they have
                        num_images_found = np.array(num_images_found)
                        # order the selected angle and distance combinations in terms of descending order of the number of images they have
                        sorted_indices = np.argsort(num_images_found)[::-1]
                        # angle and distance combinations is a list of inhomogeneous tuples so order list in terms of indeces
                        possible_combinations = [possible_combinations[i] for i in sorted_indices] """

                        # only select the top n combinations
                        possible_combinations = possible_combinations[:sample_combinations]
                    
                    # only select the top n combinations
                    possible_combinations = possible_combinations[:sample_combinations]
                    if len(possible_combinations) ==0:
                        print(f'{distance} {angle} not enough imgs')

                    results_iteration = []
                    reprojection_errors = []
                    for possible_combination in tqdm(possible_combinations, desc='angle Combinations', leave=False):
                        
                        angle = possible_combination[0]
                        distance = possible_combination[1]
                
                
                
                    
                        # filter data_df_filtered by angle selected 
                        """ data_df_filtered = data_df_chess_size[
                            (data_df_chess_size['deg'].isin(angle))] #&
                            #(data_for_calibration['deg'].isin(distance)) ] """

                        data_df_filtered_for_distance = data_df_chess_size[
                                    (data_df_chess_size['deg'].isin(angle)) &
                                    (data_df_chess_size[f'T_z{extension}'].isin(distance))
                                ]
                        #data_df_filtered_for_distance = data_df_filtered[data_df_filtered[f'T_z{extension}']==distance]


                        # skipping if not enough data
                        """ if len(data_df_filtered_for_distance) < num_images_start:
                            distances_lst.remove(distance)
                            print(f'Not enough data for distance {distance}')
                            continue """
                        if HAND_EYE:
                            result = perform_hand_eye_calibration_analysis(data_df_filtered_for_distance,
                                                                data_for_reprojection,
                                                                f'results/intrinsics/best_intrinsics',
                                                                chess_size,
                                                                repeats=repeats,
                                                                num_images_start=n,
                                                                num_images_end=n + 1,
                                                                num_images_step=num_images_step,
                                                                visualise_reprojection_error=visualise_reprojection_error,
                                                                waitTime=waitTime,
                                                                results_pth='')

                        else:
                            # calculate reprojection error
                            result = perform_analysis(camera,
                                                    data_df_filtered_for_distance, data_for_reprojection, repeats=repeats,
                                                    num_images_start=n, num_images_end=n + 1,
                                                    num_images_step=num_images_step,
                                                    visualise_reprojection_error=visualise_reprojection_error,
                                                    waitTime=waitTime,
                                                    results_pth='', thread_num=f'{angle}')
                        
                        # get median of errors_lst
                        """ median_err = np.median(result['errors_lst'].values[0])
                        Q1_err = np.quantile(result['errors_lst'].values[0], 0.25)
                        Q3_err = np.quantile(result['errors_lst'].values[0], 0.75)
                        reprojection_errors.append(median_err)
                        Q1s.append(Q1_err)
                        Q3s.append(Q3_err) """
                        results_iteration.append(result)
                        reprojection_errors.append(result['average_error'])

                    
                
                    # Calculate the overall mean reprojection error for the current combination
                    overall_mean_error = np.mean(reprojection_errors)
                    # Append results
                    simple_results.append({
                        'num_angles': num_angles,
                        'num_distances': num_distances,
                        'mean_reprojection_error': overall_mean_error
                    })
                    results_combined = pd.concat(results_iteration, axis=0)
                    # save results for this angle and distance
                    results_combined.to_pickle(
                        f'{distance_analysis_chess}/results_P{num_angles}_distance{num_distances}.pkl')

            #total_run_time_end = time.time()
            #print(f'Total run time: {(total_run_time_end - total_run_time_start) / 60} minutes')
            # Convert results to a dataframe
            simple_results_df = pd.DataFrame(simple_results)
            # load and merge all dataframes of all angles and distances
            # simple_results_df = pd.concat([pd.read_pickle(pth) for pth in glob.glob(f'{calibration_analysis_results_save_pth}/results_P*_A*.pkl') ], ignore_index=True)

            # Visualize results as a heatmap
            heatmap_data = simple_results_df.pivot(index='num_angles', columns='num_distances', values='mean_reprojection_error')

            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
            plt.title('Mean Reprojection Error by Number of angles and distances')
            plt.xlabel('Number of distances')
            plt.ylabel('Number of angles')
            plt.savefig(f'{distance_analysis_chess}/heatmap.png')
                

                
                
                
            

            



if __name__=='__main__': 
    #visualise_angles(merged = False)
    main() 