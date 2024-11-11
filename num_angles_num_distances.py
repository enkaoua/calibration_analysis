import argparse
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
 


def num_angles_num_distances_analysis(table_pth='results/hand_eye/raw_corner_data/MC_None_PC_None', 
         cameras=['realsense','endo'],
         chess_sizes=[20,15,25,30], 
         param_we_are_testing = 'angle',
         
         n=30, 
         repeats=10,
         
         R = None,
         visualise_reprojection_error=False,
         waitTime=0,
         HAND_EYE=True,
         sample_combinations = 20,
         optimise=True,
         
        min_num_corners = 6.0,
        percentage_corners = 0.4):
    
    rec_name = f'R{R}_MC_{min_num_corners}_PC_{percentage_corners}'

    if HAND_EYE:
        # for hand-eye calib, we need tables to be the merged tables
        table_pth = f'results/hand_eye/split_data/{rec_name}'
        file_pth = 'corner_data_calibration_dataset'
        extension = '_endo'
        cameras = ['*']

        distance_analysis = f'results/hand_eye/{param_we_are_testing}_distance_analysis'

    else:
        # intrinsic calibration- using hand-eye datset as it has the T column
        table_pth = f'results/hand_eye/filtered_data/MC_{min_num_corners}_PC_{percentage_corners}'
        file_pth = 'corner_data'
        extension = ''
        distance_analysis = f'results/intrinsics/{param_we_are_testing}_distance_analysis'

    if param_we_are_testing == 'angle':
        params = [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10]
        param_filter_name = 'deg'
    else:
        params = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        param_filter_name = 'pose'

    print(f'################## {param_we_are_testing.capitalize()} DISTANCE ANALYSIS Hand_eye => {HAND_EYE} ##################')
    print('')
    print(f'table_pth: {table_pth}') 
    print(f'cameras: {cameras}')
    print(f'chess_sizes: {chess_sizes}')
    print(f'param we are measuring: {param_we_are_testing} --> {params}')
    
    print(f'num images: {n}')
    print(f'repeats: {repeats}')
    
    print(f'reprojection sample size: {R}')
    print(f'visualise_reprojection_error: {visualise_reprojection_error}')
    print(f'waitTime: {waitTime}')
    print(f'HAND_EYE: {HAND_EYE}')
    print(f'sample_combinations: {sample_combinations}')
    print(f'optimise: {optimise}')
    
    print(f'min_num_corners: {min_num_corners} ')
    print(f'percentage_corners: {percentage_corners}')

    print(f'saving results in: {distance_analysis}')

    print('########################################')
    

    # create folder distance_analysis if it doesn't exist
    if not os.path.exists(distance_analysis):
        os.makedirs(distance_analysis)

    for camera in cameras:
        # grabbing all tables for this camera (all chessboards)
        data_df = pd.concat([pd.read_pickle(pth) for pth in glob.glob(f'{table_pth}/*_{camera}_{file_pth}.pkl')], ignore_index=True)
        # add xyz distance from T
        T_to_xyz(data_df, extension=extension)
        # rounding to nearest 10th
        data_df[f'T_z{extension}'] = np.round(data_df[f'T_z{extension}'] / 10) * 10

        # performing analysis for each chessboard size
        for chess_size in chess_sizes:
            # path where to save distance analysis results- create folder if doesnt exist
            distance_analysis_chess = f'{distance_analysis}/N{n}_{rec_name}_size_{chess_size}_cam_{camera}_repeats{repeats}_sample_combinations_{sample_combinations}'
            if not os.path.exists(distance_analysis_chess):
                os.makedirs(distance_analysis_chess)

            # grab data to be used for claibration and reprojection of chess size we are testing
            data_df_chess_size = data_df[data_df['chess_size'] == chess_size]
            data_for_reprojection = data_df[data_df['chess_size'] != chess_size]

            # check range of xyz distances
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
            plt.show() 
            #angle_combinations = list(itertools.combinations(angles, num_angles))
            #distance_combinations = list(itertools.combinations(distances, num_distances))
            """
            
            # grab all existing distances in dataframe
            distances = data_df_chess_size[f'T_z{extension}'].unique().sort
            # only use distances where there are more than 100 frames to test on
            grouped_df = data_df_chess_size.groupby(f'T_z{extension}').count() # group by distance and count how many occurences in each distance group
            grouped_df_filtered = grouped_df[grouped_df['frame_number']>100].reset_index() # filter those groups wiht not enough frames
            distances = grouped_df_filtered[f'T_z{extension}'].values # # grab distances 

            simple_results = []
            for num_params in tqdm(range(1, len(params) + 1), desc=f'Number of {param_we_are_testing}'):

                for num_distances in tqdm(range(1, len(distances) + 1), desc='Number of distances', leave=False):
                    
                    # check if pickle file for this already exists
                    save_pth = f'{distance_analysis_chess}/results_P{num_params}_distance{num_distances}.pkl'
                    if os.path.exists(save_pth):
                        # load results and append to simple_results
                        result = pd.read_pickle(save_pth)
                        reprojection_errors = result['average_error']
                        overall_mean_error = np.mean(reprojection_errors)
                        # Append results
                        simple_results.append({
                            f'num_{param_we_are_testing}s': num_params,
                            'num_distances': num_distances,
                            'mean_reprojection_error': overall_mean_error
                        })

                        print(f'loading results from pickle file for {num_params} params and {num_distances} distances')
                        continue
                    
                    # all possble combinations of the params and distance given how many of the params and distance to select
                    param_combinations = list(itertools.combinations(params, num_params))
                    distance_combinations = list(itertools.combinations(distances, num_distances))

                    # find combinations to test
                    num_extra_comb = 50 # extra combinations in case some of sample_combinations not have enough images
                    if sample_combinations:
                        # pick a random set of n params and distance combinations out of the above ones
                        if len(param_combinations) > sample_combinations+num_extra_comb:
                            param_combinations = np.array(random.sample(param_combinations, sample_combinations+num_extra_comb))
                        if len(distance_combinations) > sample_combinations+num_extra_comb:
                            distance_combinations = np.array(random.sample(distance_combinations, sample_combinations+num_extra_comb))
                        
                        # only select combinations which have more than 10 images- over all combinations selected above,
                        #  check how many images they have and only add it to selection if it's more than 10
                        num_images_found = []
                        possible_combinations = []
                        for p in param_combinations:
                            for distance in distance_combinations:
                                num_images_for_combination = len(data_df_chess_size[
                                    (data_df_chess_size[param_filter_name].isin(p)) &
                                    (data_df_chess_size[f'T_z{extension}'].isin(distance))
                                ])
                                if num_images_for_combination<10:
                                    continue
                                num_images_found.append(num_images_for_combination)
                                possible_combinations.append((p, distance))
                                
                        """ 
                        # order the selected params and distance combinations in terms of descending order of the number of images they have
                        # params to reject- add to list all combinations of params and distances which don't have more than num_images in total
                        # order the selected params and distance combinations in terms of descending order of the number of images they have
                        num_images_found = np.array(num_images_found)
                        # order the selected params and distance combinations in terms of descending order of the number of images they have
                        sorted_indices = np.argsort(num_images_found)[::-1]
                        # params and distance combinations is a list of inhomogeneous tuples so order list in terms of indeces
                        possible_combinations = [possible_combinations[i] for i in sorted_indices] 
                        # only select the top n combinations
                        """
                        # select only number of sample combinations entered by used
                        possible_combinations = possible_combinations[:sample_combinations]
                    
                    # if there weren't any combinations with more than 10 images, skip this combination
                    if len(possible_combinations) ==0:
                        print(f'{num_distances} {num_params} not enough imgs')
                        continue
                    # skipping if not enough data
                    """ if len(data_df_filtered_for_distance) < num_images_start:
                        distances_lst.remove(distance)
                        print(f'Not enough data for distance {distance}')
                        continue """

                    # analysis for each of the possible combinations
                    results_iteration = []
                    reprojection_errors = []
                    for possible_combination in tqdm(possible_combinations, desc=f'{param_we_are_testing} Combinations', leave=False):
                        
                        param = possible_combination[0]
                        distance = possible_combination[1]
                
                        # filter data_df_filtered by param selected 
                        data_df_filtered_for_distance = data_df_chess_size[
                                    (data_df_chess_size[param_filter_name].isin(param)) &
                                    (data_df_chess_size[f'T_z{extension}'].isin(distance))
                                ]

                        # hand-eye/intrinsic analysis
                        if HAND_EYE:
                            result = perform_hand_eye_calibration_analysis(data_df_filtered_for_distance,
                                                                data_for_reprojection,
                                                                f'results/intrinsics/best_intrinsics',
                                                                chess_size,
                                                                repeats=repeats,
                                                                num_images_start=n,
                                                                num_images_end=n + 1,
                                                                num_images_step=1,
                                                                visualise_reprojection_error=visualise_reprojection_error,
                                                                waitTime=waitTime,
                                                                results_pth='',
                                                                optimise=optimise)

                        else:
                            # calculate reprojection error
                            result = perform_analysis(camera,
                                                    data_df_filtered_for_distance, 
                                                    data_for_reprojection, 
                                                    repeats=repeats,
                                                    num_images_start=n, num_images_end=n + 1,
                                                    num_images_step=1,
                                                    visualise_reprojection_error=visualise_reprojection_error,
                                                    waitTime=waitTime,
                                                    results_pth='', thread_num=f'{param}')
                        
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
                        f'num_{param_we_are_testing}': num_params,
                        'num_distances': num_distances,
                        'mean_reprojection_error': overall_mean_error
                    })
                    results_combined = pd.concat(results_iteration, axis=0)
                    # save results for this param and distance
                    results_combined.to_pickle(
                        f'{distance_analysis_chess}/results_P{num_params}_distance{num_distances}.pkl')

            #total_run_time_end = time.time()
            #print(f'Total run time: {(total_run_time_end - total_run_time_start) / 60} minutes')
            # Convert results to a dataframe
            simple_results_df = pd.DataFrame(simple_results)
            # load and merge all dataframes of all angles and distances
            # simple_results_df = pd.concat([pd.read_pickle(pth) for pth in glob.glob(f'{calibration_analysis_results_save_pth}/results_P*_A*.pkl') ], ignore_index=True)

            # Visualize results as a heatmap
            heatmap_data = simple_results_df.pivot(index=f'num_{param_we_are_testing}s', columns='num_distances', values='mean_reprojection_error')

            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
            plt.title(f'Mean Reprojection Error for Number of {param_we_are_testing} and distances')
            plt.xlabel('Number of distances')
            plt.ylabel(f'Number of {param_we_are_testing}')
            plt.savefig(f'{distance_analysis_chess}/heatmap_{param_we_are_testing}_distance_chess_{chess_size}_PC_{percentage_corners}_sample_combinations_{sample_combinations}_N{n}_repeats{repeats}_camera_{camera}.png')
            
            # produce same but 3D surface heatmap
            plt.figure(figsize=(12, 8))
            ax = plt.axes(projection='3d')
            x = simple_results_df[f'num_{param_we_are_testing}s']
            y = simple_results_df['num_distances']
            z = simple_results_df['mean_reprojection_error']
            # smooth the surface
            from scipy.interpolate import griddata
            xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((x, y), z, (xi, yi), method='linear')
            # color map green to red
            cmap = plt.get_cmap('RdYlGn_r')
            ax.plot_surface(xi, yi, zi, cmap=cmap, edgecolor='none')
            # add color bar to represent reprojection error
            cbar = plt.colorbar(ax.plot_surface(xi, yi, zi, cmap=cmap, edgecolor='none'), ax=ax, shrink=0.5, aspect=5)
            cbar.set_label('Reprojection Error')
            ax.set_xlabel(f'Number of {param_we_are_testing}s')
            ax.set_ylabel('Number of distances')
            ax.set_zlabel('Mean Reprojection Error')
            plt.title(f'Mean Reprojection Error for Number of {param_we_are_testing} and distances')
            plt.savefig(f'{distance_analysis_chess}/3D_surface_{param_we_are_testing}_distance_chess_{chess_size}_PC_{percentage_corners}_sample_combinations_{sample_combinations}_N{n}_repeats{repeats}_camera_{camera}.png')

       
            
def add_distance_analysis_args_to_parser(parser):
    parser.add_argument('--table_path', type=str, default='results/hand_eye/raw_corner_data/MC_None_PC_None',
                        help='path to where images uesd for calibration are stored')
    parser.add_argument('--cameras', type=list, default=['realsense', 'endo'], help='cameras used for calibration')
    parser.add_argument('--chess_sizes', type=list, default=[20], #, 25, 30, 15
                        help='sizes of chessboard used for calibration')
    #parser.add_argument('-a','--angles', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='angles to analyse')
    parser.add_argument('-t_param','--param_we_are_testing', type=str, default='position', help='angle or position')

    parser.add_argument('--repeats', type=int, default=20, help='number of repeats per number of images analysis')
    
    parser.add_argument('--num_images', type=int, default=30, help='number of images to start analysis')
    parser.add_argument('--sample_combinations', type=int, default=30, help='number of combinations to be used when testing x num angles/distances')

    parser.add_argument('--reprojection_sample_size', type=int, default=None,
                        help='number of samples to use for reprojection error. If a number is selected, a random number of the same board dataset is selected. If None is selected, the dataset of all other boards is used for reprojection error. If ')
    parser.add_argument('--min_num_corners', type=str, default=6.0,
                        help='minimum number of corners to use for calibration')
    parser.add_argument('--percentage_of_corners', type=str, default=0.4,
                        help='percentage of corners to use for calibration')
    parser.add_argument('--visualise_corner_detection', type=bool, default=False,
                        help='if set to true, will visualise corner detection')

    parser.add_argument('--visualise_reprojection_error', type=bool, default=False,
                        help='if set to true, will visualise reprojection error')
    parser.add_argument('--waitTime', type=int, default=0, help='time to wait before capturing next image')


    """ parser.add_argument('--results_pth', type=str, default='results/hand_eye', help='path to save results')
    parser.add_argument('--intrinsics_for_he', type=str,
                        default='results/intrinsics/best_intrinsics', #results/intrinsics/best_intrinsics
                        help='path to intrinsics results for he') """
    
    """ parser.add_argument('--results_pth', type=str, default='results/intrinsics', help='path to save results')
    parser.add_argument('--intrinsics_for_he', type=str,
                        default='', #results/intrinsics/best_intrinsics
                        help='path to intrinsics results for he') """

    parser.add_argument('-he','--hand_eye', action='store_false', help='if set to true, will store intrinsics for hand eye') 
                    
    return parser

            



if __name__=='__main__': 
    parser = argparse.ArgumentParser(
        description='calibration analysis looking at num distances and num angle variety \
            \
            ')

    # adding all necessary args for cl app
    add_distance_analysis_args_to_parser(parser)
    # grabbing args selected
    args = parser.parse_args()

    table_pth = args.table_path
    reprojection_sample_size = args.reprojection_sample_size
    # convert string to int or none
    if reprojection_sample_size is not None:
        reprojection_sample_size = int(reprojection_sample_size)
    min_num_corners = args.min_num_corners
    # convert string to int or none
    if min_num_corners is not None:
        min_num_corners = float(min_num_corners)

    percentage_of_corners = args.percentage_of_corners
    # convert string to float or none
    if percentage_of_corners is not None:
        percentage_of_corners = float(percentage_of_corners)
    sample_combinations = int(args.sample_combinations)
    
    repeats = int(args.repeats)
    num_images = int(args.num_images)

    visualise_reprojection_error = bool(args.visualise_reprojection_error)
    waitTime = int(args.waitTime)
    
    cameras = args.cameras
    param_we_are_testing = args.param_we_are_testing
    chess_sizes = args.chess_sizes


    #visualise_angles(merged = False)
    num_angles_num_distances_analysis(table_pth=table_pth, 
         cameras=cameras,
         chess_sizes=chess_sizes, 
         param_we_are_testing = param_we_are_testing,
         n=num_images, 
         repeats=repeats,
         R= reprojection_sample_size,
         visualise_reprojection_error=visualise_reprojection_error,
         waitTime=waitTime,
         HAND_EYE=bool(args.hand_eye),
         sample_combinations = sample_combinations,
         optimise=True,
         min_num_corners = min_num_corners,
        percentage_corners = percentage_of_corners)
    

     