import argparse
import glob
import itertools
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm 
import concurrent.futures
import multiprocessing
from charuco_utils import generate_charuco_board, perform_analysis

def T_to_xyz(data, extension):
    data[f'T_x{extension}'] = data[f'T{extension}'].apply(lambda x: x[0,3])
    data[f'T_y{extension}'] = data[f'T{extension}'].apply(lambda x: x[1,3])
    data[f'T_z{extension}'] = data[f'T{extension}'].apply(lambda x: x[2,3])


def print_and_show_data_3D(data, extension, ax, sizes, shapes, idx, chess_size):
    print(f'{extension} DATA STATS')

    print('RANGES:------------------')
    print('Z')
    print((data[f'T_z{extension}'].max()-data[f'T_z{extension}'].min())/1000)
    print('Y')
    print((data[f'T_y{extension}'].max()-data[f'T_y{extension}'].min())/1000)
    print('X')
    print((data[f'T_x{extension}'].max()-data[f'T_x{extension}'].min())/1000)

    print('LENGTHS:------------------')
    print(len(data[f'T_z{extension}']))

    ax.scatter(data[f'T_x{extension}'], data[f'T_y{extension}'], data[f'T_z{extension}'], label=f'{chess_size}mm', alpha=0.5, s=sizes[idx], marker=shapes[idx])
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
            data_pth_rs = f'/Users/aure/Documents/CARES/code/charuco_calibration_analysis/results/hand_eye/raw_corner_data/MC_None_PC_0.2/{chess_size}_realsense_corner_data.pkl'
            data_pth_endo = f'/Users/aure/Documents/CARES/code/charuco_calibration_analysis/results/hand_eye/raw_corner_data/MC_None_PC_0.2/{chess_size}_endo_corner_data.pkl'
            data_rs = pd.read_pickle(data_pth_rs)
            data_endo = pd.read_pickle(data_pth_endo)

            T_to_xyz(data_rs, '')
            T_to_xyz(data_endo, '')
            #print_and_show_data_3D(data_rs, '', ax, sizes, shapes, idx, chess_size)
            print_and_show_data_3D(data_endo, '', ax, sizes, shapes, idx, chess_size)


        ax.legend()
    plt.show()
    
    return 


def process_angle_combinations(num_images_start,data_for_calibration, pose,angle, camera, data_for_reprojection, repeats, num_images_step,visualise_reprojection_error, waitTime, num_poses, num_angles, results_iteration, reprojection_errors  ):
    #for angle in tqdm(angle_combinations, desc='Angle Combinations', leave=False):
    import warnings
    warnings.filterwarnings("error")            
    #num_images_start = num_images

    # filter out whatever is not the current pose and angle
    filtered_calibration_data = data_for_calibration[
        (data_for_calibration['pose'].isin(pose)) &
        (data_for_calibration['deg'].isin(angle))
        ]
    # if we have less than the number of images specified (eg 50, take that as the new start)
    if len(filtered_calibration_data)<num_images_start:
        num_images_start=len(filtered_calibration_data)
        #print(f'reducing sample size to {len(filtered_calibration_data)} as that is max images in this filtered data')
    # ignore iteration if there's no data corresponding to requirement
    if len(filtered_calibration_data)!=0:
        #print(f'angle {angle}, pose {pose} is empty')
        # calculate reprojection error
        results = perform_analysis(camera,  
                                    filtered_calibration_data,data_for_reprojection, repeats=repeats, 
                                    num_images_start=num_images_start, num_images_end=num_images_start+1, num_images_step=num_images_step,
                                    visualise_reprojection_error=visualise_reprojection_error, waitTime=waitTime,
                                    results_pth = '')
        
        #results['filter_pose'] = pose
        # results['filter_angle'] = angle
        results['num_poses'] = num_poses
        results['num_angles'] = num_angles
        results['sample size'] = num_images_start
        #results_iteration = pd.concat([results_iteration, results], axis=0)

        results_iteration.append(results)
        reprojection_errors.append(results['average_error'])
        #return results
 

def main_pose_analysis(
    size_chess = 30,
    num_images = 50,
    poses = [ 0,1,2,3,4,5,6,7,8],
    angles = [ 0,1,2,3,4,5,6,7,8,9,10],    
    camera = 'endo' ,
    data_pth = f'/Users/aure/Documents/CARES/code/charuco_calibration_analysis/results/intrinsics/split_data/MC_None_PC_0.2',
    calibration_analysis_results_save_pth = f'/Users/aure/Documents/CARES/code/charuco_calibration_analysis/results/intrinsics/pose_analysis/',
    repeats = 1,
    visualise_reprojection_error = False,
    waitTime = 0
): 
    

    # pth of data to perform calibration
    data_for_calibration = pd.read_pickle(f'{data_pth}/{size_chess}_endo_corner_data_calibration_dataset.pkl')
    data_for_reprojection = pd.read_pickle(f'{data_pth}/{size_chess}_endo_corner_data_reprojection_dataset.pkl')

    num_images_step = 1

    # create calibration_analysis_results_save_pth if it does not exist
    if not os.path.exists(calibration_analysis_results_save_pth):
        os.makedirs(calibration_analysis_results_save_pth)

    total_run_time_start = time.time()
    #results_all = pd.DataFrame()
    simple_results = []
    for num_poses in tqdm(range(1, len(poses) + 1), desc='Number of Poses'):
        for num_angles in tqdm(range(1, len(angles) + 1), desc='Number of Angles', leave=False):

            # for this specific number of poses and number of angles, get all the possible combinations
            # of poses and angles
            reprojection_errors = []
            pose_combinations = list(itertools.combinations(poses, num_poses))
            angle_combinations = list(itertools.combinations(angles, num_angles))
            results_iteration = []
            # for each combination of poses and angles filter out the data and calculate error 
            for pose in tqdm(pose_combinations, desc='Pose Combinations', leave=False):
                
                
                """ for angle in tqdm(angle_combinations, desc='Angle Combinations', leave=False):
                
                    num_images_start = num_images

                    # filter out whatever is not the current pose and angle
                    filtered_calibration_data = data_for_calibration[
                        (data_for_calibration['pose'].isin(pose)) &
                        (data_for_calibration['deg'].isin(angle))
                        ]
                    # if we have less than the number of images specified (eg 50, take that as the new start)
                    if len(filtered_calibration_data)<num_images_start:
                        num_images_start=len(filtered_calibration_data)
                        #print(f'reducing sample size to {len(filtered_calibration_data)} as that is max images in this filtered data')
                    # ignore iteration if there's no data corresponding to requirement
                    if len(filtered_calibration_data)==0:
                        #print(f'angle {angle}, pose {pose} is empty')
                        continue
                    # calculate reprojection error
                    results = perform_analysis(camera,  
                                                filtered_calibration_data,data_for_reprojection, repeats=repeats, 
                                                num_images_start=num_images_start, num_images_end=num_images_start+1, num_images_step=num_images_step,
                                                visualise_reprojection_error=visualise_reprojection_error, waitTime=waitTime,
                                                results_pth = '')
                    
                    #results['filter_pose'] = pose
                    # results['filter_angle'] = angle
                    results['num_poses'] = num_poses
                    results['num_angles'] = num_angles
                    results['sample size'] = num_images_start
                    #results_iteration = pd.concat([results_iteration, results], axis=0)
                    results_iteration.append(results)

                    reprojection_errors.append(results['average_error'])
                 """
                
                
                """
            
                processes = []
                for angle in angle_combinations:
                    p = multiprocessing.Process(target=process_angle_combinations, args=(num_images,data_for_calibration, pose,angle, camera, data_for_reprojection, repeats, num_images_step,visualise_reprojection_error, waitTime, num_poses, num_angles, results_iteration, reprojection_errors))
                    p.start()
                    processes.append(p)
                for process in processes:
                    process.join()
                    """
                #manager = multiprocessing.Manager()
                #results_iteration = manager.list(results_iteration)
                #reprojection_errors = manager.list(reprojection_errors)
                processes = []
                for angle in angle_combinations:
                    """ p = multiprocessing.Process(
                        target = process_angle_combinations,
                        args = (num_images,data_for_calibration, pose,angle, camera, data_for_reprojection, repeats, num_images_step,visualise_reprojection_error, waitTime, num_poses, num_angles, results_iteration, reprojection_errors)
                    )
                    processes.append(p)
                    p.start() """
                    
                    """ for p in processes:
                        p.join() """
                
                    process_angle_combinations(num_images,data_for_calibration, pose,angle, camera, data_for_reprojection, repeats, num_images_step,visualise_reprojection_error, waitTime, num_poses, num_angles, results_iteration, reprojection_errors  )
                
                # Convert shared lists back to normal lists
                #results_iteration[:] = list(shared_results_iteration)
                #reprojection_errors[:] = list(shared_reprojection_errors)
            # Calculate the overall mean reprojection error for the current combination
            overall_mean_error = np.mean(reprojection_errors)
            # Append results
            simple_results.append({
                'num_poses': num_poses,
                'num_angles': num_angles,
                'mean_reprojection_error': overall_mean_error
            })


            # 
            results_combined = pd.concat(results_iteration, axis=0)
            # save results for this pose and angle
            results_combined.to_pickle(f'{calibration_analysis_results_save_pth}/results_P{num_poses}_A{num_angles}.pkl')
            #simple_results.to_pickle(f'{calibration_analysis_results_save_pth}/simple_results_P{num_poses}_A{num_angles}.pkl')

    
    total_run_time_end = time.time()
    print(f'Total run time: {(total_run_time_end - total_run_time_start)/60} minutes')
    # Convert results to a dataframe
    simple_results_df = pd.DataFrame(simple_results)
    # load and merge all dataframes of all poses and angles
    #simple_results_df = pd.concat([pd.read_pickle(pth) for pth in glob.glob(f'{calibration_analysis_results_save_pth}/results_P*_A*.pkl') ], ignore_index=True)

    # Visualize results as a heatmap
    heatmap_data = simple_results_df.pivot(index='num_poses', columns='num_angles', values='mean_reprojection_error')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
    plt.title('Mean Reprojection Error by Number of Poses and Angles')
    plt.xlabel('Number of Angles')
    plt.ylabel('Number of Poses')
    plt.show()

    # save results
    #results_all.to_pickle(f'{calibration_analysis_results_save_pth}/results.pkl')
    #simple_results_df.to_pickle(f'{calibration_analysis_results_save_pth}/simple_results.pkl')
    """ print(results.describe())
    
    # plot as heatmap
    df_pivot = results_all.pivot(index='filter_pose', columns='filter_angle', values='average_error')
    print(df_pivot)

    ax = sns.heatmap(df_pivot, cmap="YlOrBr")
    ax.set_title("Heatmap of Z values")

    plt.show() """

    return




if __name__=='__main__': 
    #warnings.filterwarnings('ignore', message='RuntimeWarning: overflow encountered in square') 
    import warnings
    warnings.filterwarnings("error")
    parser = argparse.ArgumentParser(
        description='pose analysis ')   
    
    parser.add_argument('--size_chess', type=int, default=30, help='size of chessboard used for calibration')
    parser.add_argument('--num_images', type=int, default=50, help='number of images to start analysis')
    parser.add_argument('--poses', type=list, default=[ 0,1], help='poses to analyse')
    parser.add_argument('--angles', type=list, default=[ 0,1], help='angles to analyse')
    parser.add_argument('--camera', type=str, default='endo', help='camera to analyse')
    parser.add_argument('--data_pth', type=str, default='results/intrinsics/split_data/MC_6.0_PC_0.5', help='path to where data is found')
    parser.add_argument('--calibration_analysis_results_save_pth', type=str, default='results/intrinsics/pose_analysis/', help='path to save results')
    parser.add_argument('--repeats', type=int, default=2, help='number of repeats per number of images analysis')
    parser.add_argument('--visualise_reprojection_error', type=bool, default=False, help='if set to true, will visualise reprojection error')
    parser.add_argument('--waitTime', type=int, default=0, help='time to wait before capturing next image')
    args = parser.parse_args()
    main_pose_analysis(
        size_chess = args.size_chess,
        num_images = args.num_images,
        poses = args.poses,
        angles = args.angles,
        camera = args.camera,
        data_pth = args.data_pth,
        calibration_analysis_results_save_pth = args.calibration_analysis_results_save_pth,
        repeats = args.repeats,
        visualise_reprojection_error = args.visualise_reprojection_error,
        waitTime = args.waitTime
    )
    warnings.resetwarnings()



