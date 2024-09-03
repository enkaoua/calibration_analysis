import argparse
import concurrent.futures
import itertools
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from charuco_utils import perform_analysis, perform_hand_eye_calibration_analysis





def process_possible_combinations(args):
    num_images_start, data_for_calibration, pose, angle, camera, data_for_reprojection, repeats, num_images_step, visualise_reprojection_error, waitTime, num_poses, num_angles, intrinsics_for_he, size_chess, optimise = args  # , results_iteration, reprojection_errors
    # for angle in tqdm(angle_combinations, desc='Angle Combinations', leave=False):
    import warnings
    warnings.filterwarnings("error")
    # num_images_start = num_images

    # filter out whatever is not the current pose and angle
    filtered_calibration_data = data_for_calibration[
        (data_for_calibration['pose'].isin(pose)) &
        (data_for_calibration['deg'].isin(angle))
        ]
    # if we have less than the number of images specified (eg 50, take that as the new start)
    if len(filtered_calibration_data) < num_images_start:
        num_images_start = len(filtered_calibration_data)
        # print(f'reducing sample size to {len(filtered_calibration_data)} as that is max images in this filtered data')
    # ignore iteration if there's no data corresponding to requirement
    if len(filtered_calibration_data) != 0:
        # print(f'angle {angle}, pose {pose} is empty')
        # calculate reprojection error
        if len(intrinsics_for_he) > 0:
            results = perform_hand_eye_calibration_analysis(filtered_calibration_data,
                                                          data_for_reprojection,
                                                          intrinsics_for_he,
                                                          size_chess,
                                                          repeats=repeats,
                                                          num_images_start=num_images_start,
                                                          num_images_end=num_images_start+1,
                                                          num_images_step=num_images_step,
                                                          visualise_reprojection_error=visualise_reprojection_error,
                                                          waitTime=waitTime,
                                                          results_pth='',
                                                          optimise= optimise)
        else:
            results = perform_analysis(camera,
                                    filtered_calibration_data, data_for_reprojection, repeats=repeats,
                                    num_images_start=num_images_start, num_images_end=num_images_start + 1,
                                    num_images_step=num_images_step,
                                    visualise_reprojection_error=visualise_reprojection_error, waitTime=waitTime,
                                    results_pth='', thread_num=f'{pose}_{angle}')

        # results['filter_pose'] = pose
        # results['filter_angle'] = angle
        results['num_poses'] = num_poses
        results['num_angles'] = num_angles
        results['sample size'] = num_images_start
        # results_iteration = pd.concat([results_iteration, results], axis=0)

        """ results_iteration.append(results)
        reprojection_errors.append(results['average_error']) """
        return results


def main_pose_analysis(
        size_chess=15,
        num_images=50,
        poses=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        angles=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        camera='realsense',
        data_pth=f'results/intrinsics',
        min_num_corners=6.0,
        percentage_corners=0.5,
        repeats=5,
        visualise_reprojection_error=False,
        waitTime=0,
        sample_combinations=10,
        intrinsics_for_he = '',
        optimise=True
):
    rec_name = f'MC_{min_num_corners}_PC_{percentage_corners}'
    split_data_pth = f'{data_pth}/split_data/{rec_name}'
    # pth of data to perform calibration
    data_for_calibration = pd.read_pickle(f'{split_data_pth}/{size_chess}_endo_corner_data_calibration_dataset.pkl')
    data_for_reprojection = pd.read_pickle(f'{split_data_pth}/{size_chess}_endo_corner_data_reprojection_dataset.pkl')

    num_images_step = 1

    calibration_analysis_results_save_pth = f'{data_pth}/pose_analysis/{rec_name}_size_{size_chess}_cam_{camera}_repeats{repeats}_sample_combinations_{sample_combinations}'
    # create calibration_analysis_results_save_pth if it does not exist
    if not os.path.exists(calibration_analysis_results_save_pth):
        os.makedirs(calibration_analysis_results_save_pth)

    total_run_time_start = time.time()
    # results_all = pd.DataFrame()
    simple_results = []
    for num_poses in tqdm(range(1, len(poses) + 1), desc='Number of Poses'):
        for num_angles in tqdm(range(1, len(angles) + 1), desc='Number of Angles', leave=False):

            # check if pickle file for this already exists
            save_pth = f'{calibration_analysis_results_save_pth}/results_P{num_poses}_A{num_angles}.pkl'
            if os.path.exists(save_pth):
                # load results and append to simple_results
                result = pd.read_pickle(save_pth)
                reprojection_errors = result['average_error']
                overall_mean_error = np.mean(reprojection_errors)
                # Append results
                simple_results.append({
                    'num_poses': num_poses,
                    'num_angles': num_angles,
                    'mean_reprojection_error': overall_mean_error
                })

                print(f'loading results from pickle file for {num_poses} poses and {num_angles} angles')
                continue

            # for this specific number of poses and number of angles, get all the possible combinations
            # of poses and angles
            pose_combinations = list(itertools.combinations(poses, num_poses))
            angle_combinations = list(itertools.combinations(angles, num_angles))

            if sample_combinations:
                # pick a random set of n pose and angle combinations out of the above ones
                if len(pose_combinations) > sample_combinations:
                    pose_combinations = random.sample(pose_combinations, sample_combinations)
                if len(angle_combinations) > sample_combinations:
                    angle_combinations = random.sample(angle_combinations, sample_combinations)

            # for each combination of poses and angles filter out the data and calculate error 

            """ results_iteration = []
            reprojection_errors = []
            for pose in tqdm(pose_combinations, desc='Pose Combinations', leave=False):
                
                for angle in angle_combinations: 
                    args = (num_images,data_for_calibration, pose,angle, camera, data_for_reprojection, repeats, num_images_step,visualise_reprojection_error, waitTime, num_poses, num_angles)
                    result = process_possible_combinations(args )
                    if result is None:
                        continue
                    results_iteration.append(result)
                    reprojection_errors.append(result['average_error']) """
            # lists un parallel processing

            # manager = multiprocessing.Manager().list()
            results_iteration = []
            reprojection_errors = []
            """ with concurrent.futures.ProcessPoolExecutor() as pool:
                args_list = [(num_images, data_for_calibration, pose, angle, camera, data_for_reprojection, repeats,
                              num_images_step, visualise_reprojection_error, waitTime, num_poses, num_angles, intrinsics_for_he, size_chess, optimise) for pose
                             in pose_combinations for angle in angle_combinations]
                results_all_combinations = tqdm(pool.map(process_possible_combinations, args_list),
                                                total=len(args_list), leave=False)
                # add to 
                for result in results_all_combinations:
                    if result is None:
                        continue
                    results_iteration.append(result)
                    reprojection_errors.append(result['average_error']) """
            
            # run non-parallel
            for pose in tqdm(pose_combinations, desc='Pose Combinations', leave=False):
                for angle in angle_combinations:
                    args = (num_images, data_for_calibration, pose, angle, camera, data_for_reprojection, repeats,
                            num_images_step, visualise_reprojection_error, waitTime, num_poses, num_angles, intrinsics_for_he, size_chess, optimise)
                    result = process_possible_combinations(args)
                    if result is None:
                        continue
                    results_iteration.append(result)
                    reprojection_errors.append(result['average_error'])
            
            # Calculate the overall mean reprojection error for the current combination
            overall_mean_error = np.mean(reprojection_errors)
            # Append results
            simple_results.append({
                'num_poses': num_poses,
                'num_angles': num_angles,
                'mean_reprojection_error': overall_mean_error
            })

            results_combined = pd.concat(results_iteration, axis=0)
            # save results for this pose and angle
            results_combined.to_pickle(
                f'{calibration_analysis_results_save_pth}/results_P{num_poses}_A{num_angles}.pkl')
            # simple_results.to_pickle(f'{calibration_analysis_results_save_pth}/simple_results_P{num_poses}_A{num_angles}.pkl')

    total_run_time_end = time.time()
    print(f'Total run time: {(total_run_time_end - total_run_time_start) / 60} minutes')
    # Convert results to a dataframe
    simple_results_df = pd.DataFrame(simple_results)
    # load and merge all dataframes of all poses and angles
    # simple_results_df = pd.concat([pd.read_pickle(pth) for pth in glob.glob(f'{calibration_analysis_results_save_pth}/results_P*_A*.pkl') ], ignore_index=True)

    # Visualize results as a heatmap
    heatmap_data = simple_results_df.pivot(index='num_poses', columns='num_angles', values='mean_reprojection_error')

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
    plt.title('Mean Reprojection Error by Number of Poses and Angles')
    plt.xlabel('Number of Angles')
    plt.ylabel('Number of Poses')
    plt.savefig(f'{calibration_analysis_results_save_pth}/heatmap.png')

    # save results
    # results_all.to_pickle(f'{calibration_analysis_results_save_pth}/results.pkl')
    # simple_results_df.to_pickle(f'{calibration_analysis_results_save_pth}/simple_results.pkl')
    """ print(results.describe())
    
    # plot as heatmap
    df_pivot = results_all.pivot(index='filter_pose', columns='filter_angle', values='average_error')
    print(df_pivot)

    ax = sns.heatmap(df_pivot, cmap="YlOrBr")
    ax.set_title("Heatmap of Z values")

    plt.show() """

    return


if __name__ == '__main__':
    # warnings.filterwarnings('ignore', message='RuntimeWarning: overflow encountered in square')
    # warnings.filterwarnings("error")
    parser = argparse.ArgumentParser(
        description='pose analysis ')

    parser.add_argument('-size','--size_chess', type=int, default=30, help='size of chessboard used for calibration')
    parser.add_argument('-n','--num_images', type=int, default=50, help='number of images to start analysis')
    parser.add_argument('-p','--poses', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8], help='poses to analyse')
    parser.add_argument('-a','--angles', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='angles to analyse')
    parser.add_argument('-cam','--camera', type=str, default='realsense', help='camera to analyse')
    #parser.add_argument('d','--data_pth', type=str, default='results/intrinsics', help='path to where data is found')
    parser.add_argument('-mc','--min_num_corners', type=float, default=6.0,
                        help='minimum number of corners to use for calibration')
    parser.add_argument('-pc','--percentage_corners', type=float, default=0.2,
                        help='percentage of corners to use for calibration')
    parser.add_argument('-r','--repeats', type=int, default=5, help='number of repeats per number of images analysis')
    parser.add_argument('-v','--visualise_reprojection_error', type=bool, default=False,
                        help='if set to true, will visualise reprojection error')
    parser.add_argument('-w', '--waitTime', type=int, default=0, help='time to wait before capturing next image')
    parser.add_argument('-s','--sample_combinations', type=int, default=10, help='number of combinations to sample')
    
    # hand eye -- if this is enabled, store as true 
    parser.add_argument('-he','--intrinsics_for_he', action='store_false', help='if set to true, will store intrinsics for hand eye') 
    # hand eye optimisation
    parser.add_argument('-opt','--hand_eye_optimisation', action='store_false', help='if set to true, will run hand eye optimisation')
    
    
    args = parser.parse_args()
    print(f'intrinsics_for_he {args.intrinsics_for_he}')

    if args.intrinsics_for_he:
        print('hand eye analysis')
        best_intrinsics_pth = f'results/intrinsics/best_intrinsics'
        data_pth = f'results/hand_eye'
        camera = args.camera
        percentage_corners = 0.2
        if args.hand_eye_optimisation:
            print('hand eye optimisation')
    else:
        print('intrinsics analysis')
        best_intrinsics_pth = f''
        data_pth = f'results/intrinsics'
        camera = args.camera
        percentage_corners = 0.5

    main_pose_analysis(
        size_chess=args.size_chess,
        num_images=args.num_images,
        poses=args.poses,
        angles=args.angles,
        camera=camera,
        data_pth=data_pth,
        min_num_corners=args.min_num_corners,
        percentage_corners=percentage_corners,
        repeats=args.repeats,
        visualise_reprojection_error=args.visualise_reprojection_error,
        waitTime=args.waitTime,
        sample_combinations=args.sample_combinations,
        intrinsics_for_he=best_intrinsics_pth,
        optimise = args.hand_eye_optimisation
    )
    warnings.resetwarnings()
