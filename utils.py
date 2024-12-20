import os

import cv2
import numpy as np
import pandas as pd
import sksurgerycore.transforms.matrix as skcm
from scipy.spatial.transform import Rotation

def T_to_xyz(data, extension=''):
    '''
    function to extract x,y,z from a 4x4 matrix
    '''
    data[f'T_x{extension}'] = data[f'T{extension}'].apply(lambda x: x[0,3])
    data[f'T_y{extension}'] = data[f'T{extension}'].apply(lambda x: x[1,3])
    data[f'T_z{extension}'] = data[f'T{extension}'].apply(lambda x: x[2,3])


def sort_and_filter_matched_corners(corners_endo, corners_realsense, ids_endo, ids_realsense, return_ids=False):
    '''
    function to sort and remove corners that dont match between two arrays given their IDs
    # TODO can extend this function to more than 2 sets of points
    '''

    # sort realsense ids and corners
    sorted_idx = np.argsort(ids_realsense.flatten())
    realsense_sorted_ids = ids_realsense[sorted_idx]
    corners_realsense_sorted = np.array(corners_realsense)[sorted_idx]

    sorted_idx = np.argsort(ids_endo.flatten())
    endo_sorted_ids = ids_endo[sorted_idx]
    corners_endo_sorted = np.array(corners_endo)[sorted_idx]

    # find common numbers in both lists
    # common_idx = np.intersect1d(idx_realsense_sorted,idx_endo_sorted)

    # IDs found in endo but not in realsense
    unique_endo_id = np.setdiff1d(endo_sorted_ids, realsense_sorted_ids)
    # remove unique_endo_id from endo_sorted_ids
    new_endo_idx = ~np.isin(endo_sorted_ids, unique_endo_id)  # (endo_sorted_ids != unique_endo_id).any(axis=1)
    # new_endo_idx = np.setdiff1d(endo_sorted_ids, unique_endo_id)

    if len(unique_endo_id) > 0:
        endo_sorted_ids = endo_sorted_ids[new_endo_idx]
        corners_endo_sorted = corners_endo_sorted[new_endo_idx]

    # remove unique IDs found in rs but not endo
    unique_rs_id = np.setdiff1d(realsense_sorted_ids, endo_sorted_ids)
    new_rs_idx = ~np.isin(realsense_sorted_ids, unique_rs_id)
    # new_rs_idx = np.setdiff1d(realsense_sorted_ids, unique_rs_id)
    if len(unique_rs_id) > 0:
        realsense_sorted_ids = realsense_sorted_ids[new_rs_idx]
        corners_realsense_sorted = corners_realsense_sorted[new_rs_idx]

    if return_ids:
        return corners_endo_sorted, corners_realsense_sorted, endo_sorted_ids, realsense_sorted_ids
    return corners_endo_sorted, corners_realsense_sorted


def extrinsic_vecs_to_matrix(rvec, tvec):
    """
    Method to convert rvec and tvec to a 4x4 matrix.

    :param rvec: [3x1] ndarray, Rodrigues rotation params
    :param rvec: [3x1] ndarray, translation params
    :return: [3x3] ndarray, Rotation Matrix
    """
    rotation_matrix = (cv2.Rodrigues(rvec))[0]
    transformation_matrix = \
        skcm.construct_rigid_transformation(rotation_matrix, tvec)
    return transformation_matrix


def extrinsic_matrix_to_vecs(transformation_matrix):
    """
    Method to convert a [4x4] rigid body matrix to an rvec and tvec.

    :param transformation_matrix: [4x4] rigid body matrix.
    :return [3x1] Rodrigues rotation vec, [3x1] translation vec
    """
    rmat = transformation_matrix[0:3, 0:3]
    rvec = (cv2.Rodrigues(rmat))[0]
    tvec = np.ones((3, 1))
    tvec[0:3, 0] = transformation_matrix[0:3, 3]
    return rvec, tvec


def sample_dataset(df, total_samples=100, groupby_cats=['position_category', 'angle_category']):
    
    # if the length of the dataframe is smaller than total_samples, return the entire dataframe and remaining samples is None
    if total_samples is None:
        return df, None
    if len(df) < total_samples:
        return df, None

    # Group by poses and angles
    grouped = df.groupby(groupby_cats)

    # Calculate how many samples we should pick per group (approximately)
    num_groups = len(grouped)
    samples_per_group = total_samples // num_groups

    # Collect samples from each group
    samples = []

    for i, group in grouped:
        # Determine the number of samples to take from this group
        n_samples = min(samples_per_group, len(group))

        # Randomly sample from the group
        sampled_group = group.sample(n_samples)

        samples.append(sampled_group)

    # Concatenate all samples into a single DataFrame
    selected_samples = pd.concat(samples)

    # If we have fewer than the samples we needed due to rounding, sample additional rows randomly from the result
    if len(selected_samples) < total_samples:
        additional_samples = df.sample(total_samples - len(selected_samples))
        selected_samples = pd.concat([selected_samples, additional_samples])

    # If we have more than the samples needed, randomly sample the required number of rows from the result
    if len(selected_samples) > total_samples:
        selected_samples = selected_samples.sample(total_samples)

    # Get the rest of the dataset by dropping the selected indices from the original DataFrame
    remaining_samples = df.drop(selected_samples.index)

    # Reset index for final result
    selected_samples = selected_samples.reset_index(drop=True)
    remaining_samples = remaining_samples.reset_index(drop=True)

    return selected_samples, remaining_samples


def create_folders(folders):
    """
    creates specified folders if they dont exist
    """
    for folder in folders:
        if not os.path.isdir(f'{folder}'):
            os.makedirs(f'{folder}')


def reprojection_error(imgpoints_detected, imgpoints_reprojected, image=None, IDs=None):
    """
    calculate reprojection error given the detected and reprojected points
    """
    diff = (imgpoints_detected - imgpoints_reprojected)
    if diff.any()>1000:
        # to avoid overflow
        error_np=np.inf
    else:
        squared_diffs = np.square(diff)
        error_np = np.sqrt(np.sum(squared_diffs) / len(imgpoints_reprojected))
        # round up to 5 decimal places
        error_np = round(error_np, 5)

    if image is not None:
        img_shape = image.shape
        for idx, corner_detected in enumerate(imgpoints_detected):
            # change dtypw of corner to int
            corner_detected = corner_detected.astype(int)
            corner_reprojected = imgpoints_reprojected[idx].astype(int)

            centre_detected = corner_detected.ravel()
            centre_reprojected = corner_reprojected.ravel()

            # check if points are within image
            if centre_detected[0] < 0 or centre_detected[0] > img_shape[1] or centre_detected[1] < 0 or centre_detected[
                1] > img_shape[0]:
                continue
            if centre_reprojected[0] < 0 or centre_reprojected[0] > img_shape[1] or centre_reprojected[1] < 0 or \
                    centre_reprojected[1] > img_shape[0]:
                continue
            cv2.circle(image, (int(centre_detected[0]), int(centre_detected[1])), 3, (0, 0, 255), -1)
            cv2.circle(image, (int(centre_reprojected[0]), int(centre_reprojected[1])), 3, (0, 255, 0), -1)
            # TODO ADD IDs to each tag
            if IDs is not None:
                # add ID of detected tag
                ID=IDs[idx]
                cv2.putText(image, f'{ID}', (int(centre_detected[0]), int(centre_detected[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, f'{ID}', (int(centre_reprojected[0]), int(centre_reprojected[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        return error_np, image
    
    return error_np


def remove_outliers(r_lst, t_lst, threshold=10):
    """
    removes outliers from the list of rotation and translation vectors
    Args:
        r_lst (list): list of rotation vectors
        t_lst (list): list of translation vectors
        threshold (int): threshold for outlier detection
    Returns:
        r_lst (list): list of rotation vectors
        t_lst (list): list of translation vectors
    """
    # convert to dataframe
    """ df_t = pd.DataFrame(np.array(t_lst).squeeze(), columns=['x', 'y', 'z'])
    df_r = pd.DataFrame(np.array(r_lst).squeeze(), columns=['rx', 'ry', 'rz'])

    # Calculate mean and standard deviation for each column in both DataFrames
    mean_t, std_t = df_t.mean(), df_t.std()
    mean_r, std_r = df_r.mean(), df_r.std()
    # Calculate z-scores for each column in both DataFrames
    z_scores_t = (df_t - mean_t) / std_t
    z_scores_r = (df_r - mean_r) / std_r
    # Identify outliers based on z-scores
    outliers_t = (z_scores_t > threshold).any(axis=1)
    outliers_r = (z_scores_r > threshold).any(axis=1)
    # Combine masks to find rows that meet criteria in both lists
    outliers_combined = outliers_t | outliers_r
    # Remove outliers from the lists
    r_lst = [r for i, r in enumerate(r_lst) if not outliers_combined[i]]
    t_lst = [t for i, t in enumerate(t_lst) if not outliers_combined[i]] """

    # Convert lists to DataFrames
    df_t = pd.DataFrame(np.array(t_lst).squeeze(), columns=['x', 'y', 'z'])
    df_r = pd.DataFrame(np.array(r_lst).squeeze(), columns=['rx', 'ry', 'rz'])
    from scipy.spatial.distance import cdist

    # Calculate pairwise distances within each DataFrame
    distances_t = cdist(df_t, df_t, metric='euclidean')
    distances_r = cdist(df_r, df_r, metric='euclidean')

    # Calculate the median of distances for each row
    median_distances_t = np.median(distances_t, axis=1)
    median_distances_r = np.median(distances_r, axis=1)

    # Determine threshold as 75th percentile for outlier detection
    threshold_t = np.percentile(median_distances_t, 75)
    threshold_r = np.percentile(median_distances_r, 75)

    # Identify non-outliers within each list
    mask_t = median_distances_t <= threshold_t
    mask_r = median_distances_r <= threshold_r

    # Combine masks to keep rows that meet criteria in both lists
    mask_combined = mask_t & mask_r

    # Filter rows and convert back to lists
    t_lst = df_t[mask_combined].values.tolist()
    r_lst = df_r[mask_combined].values.tolist()

    #r_lst = [r for i, r in enumerate(r_lst) if not filtered_r_lst[i]]
    #t_lst = [t for i, t in enumerate(t_lst) if not filtered_t_lst[i]]
    return r_lst, t_lst, mask_combined


def calculate_transform_average(r_lst, t_lst):
    """
    calculates mean of rotation and translation vectors using scipy
    Args:
        r_lst (list): list of rotation vectors
        t_lst (list): list of translation vectors
    Returns:
        mean_he (np.array): 4x4 transformation matrix
    """
    """ # remove any outlier that is too different from the rest
    # convert to dataframe
    df_t = pd.DataFrame(np.array(t_lst).squeeze(), columns=['x', 'y', 'z'])
    df_r = pd.DataFrame(np.array(r_lst).squeeze(), columns=['rx', 'ry', 'rz'])

    # Calculate mean and standard deviation for each column in both DataFrames
    mean_t, std_t = df_t.mean(), df_t.std()
    mean_r, std_r = df_r.mean(), df_r.std()

    # Identify rows within one standard deviation for both DataFrames
    mask_t = ((df_t >= mean_t - std_t) & (df_t <= mean_t + std_t)).all(axis=1)
    mask_r = ((df_r >= mean_r - std_r) & (df_r <= mean_r + std_r)).all(axis=1)

    # Combine masks to find rows that meet criteria in both lists
    mask_combined = mask_t & mask_r

    # Filter rows and convert back to lists
    filtered_t_lst = df_t[mask_combined].values.tolist()
    filtered_r_lst = df_r[mask_combined].values.tolist() """
    r_lst, t_lst, mask = remove_outliers(r_lst, t_lst, threshold=10)

    # average hand eye
    scipy_rot = Rotation.from_rotvec(np.array(r_lst).reshape((-1, 3)), degrees=False)
    mean_rot = scipy_rot.mean().as_rotvec()

    mean_t = np.mean(np.asarray(t_lst).reshape(-1, 3), axis=0)

    mean_he = extrinsic_vecs_to_matrix(mean_rot, mean_t)
    return mean_he, mask


def find_best_intrinsics(intrinsics_pth, size_chess, camera, save_path=''):
    intrinsics_all_data = pd.read_pickle(f'{intrinsics_pth}/{size_chess}_{camera}_calibration_data.pkl')
    # find where average_error is smallest
    intrinsics_all_data = intrinsics_all_data[
        intrinsics_all_data.average_error == intrinsics_all_data.average_error.min()]
    errors_all = intrinsics_all_data['errors_lst'].values[0]
    min_error = min(errors_all)
    intrinsics = intrinsics_all_data['intrinsics'].values[0][errors_all.index(min_error)]
    distortion = intrinsics_all_data['distortion'].values[0][errors_all.index(min_error)]

    if len(save_path) > 0:
        # save as txt file
        np.savetxt(f'{save_path}/{size_chess}_{camera}_intrinsics.txt', intrinsics)
        np.savetxt(f'{save_path}/{size_chess}_{camera}_distortion.txt', distortion)

    return intrinsics, distortion, min_error


def filter_and_merge_hand_eye_df(data_df_endo, data_df_realsense, min_num_corners, main_run = False):
    # combine information of paths for filtering those that don't match between endo and rs
    # TODO remove warning
    if main_run:
        common_columns = ['frame_number']
    else:
        common_columns = ['chess_size', 'pose', 'deg', 'direction', 'frame_number']
    data_df_endo['combined_info'] = data_df_endo[common_columns].astype(
        str).agg('_'.join, axis=1)
    data_df_realsense['combined_info'] = data_df_realsense[
        common_columns].astype(str).agg('_'.join, axis=1)
        
    # find common images between endo and realsense
    common_keys = set(data_df_endo['combined_info']).intersection(set(data_df_realsense['combined_info']))
    # take out file names that don't match and reset index to ensure they're matching
    data_df_endo = data_df_endo[data_df_endo['combined_info'].isin(common_keys)].reset_index(drop=True)
    data_df_realsense = data_df_realsense[data_df_realsense['combined_info'].isin(common_keys)].reset_index(drop=True)

    # Drop the info key column 
    data_df_endo.drop(columns=['combined_info', 'num_detected_corners'], inplace=True)
    # data_df_realsense.drop(columns=['combined_info'], inplace=True)
    data_df_realsense.drop(columns=['combined_info', 'num_detected_corners'], inplace=True) #inplace used to do operation in place (instead of returning copy) and return None.

    # merge endo and rs into one dataframe and add suffixes (_endo or _rs) to 
    data_df_combined = pd.merge(
        data_df_endo,
        data_df_realsense,
        on=common_columns,
        suffixes=('_endo', '_rs')
    )

    # add empty column num_corners_detected
    data_df_combined['num_corners_detected'] = np.nan

    #### HAND-EYE CALIBRATION ####
    # filter out any unmatched points        
    removed_ids = []
    for row_idx, row in data_df_combined.iterrows():
        pnts_endo = row['imgPoints_endo']
        pnts_3d_rs = row['objPoints_rs']
        pnts_rs = row['imgPoints_rs']
        pnts_3d_endo = row['objPoints_endo']
        ids_e = row['ids_endo']
        ids_r = row['ids_rs']

        # sort and filter matched corners
        imgPoints_matched_endo, objPoints_matched_rs, ids_endo, ids_rs = sort_and_filter_matched_corners(pnts_endo, pnts_3d_rs,
                                                                                                 ids_e, ids_r,
                                                                                                 return_ids=True)
        # filter also realsense pnts and endo object points
        imgPoints_matched_rs, objPoints_matched_endo, ids_rs_2, ids_endo_2 = sort_and_filter_matched_corners(pnts_rs, pnts_3d_endo,
                                                                                                         ids_r, ids_e,
                                                                                                         return_ids=True)
        if len(imgPoints_matched_endo) < min_num_corners:
            # remove row from dataframe if the number of points is less than the minimum number of corners
            data_df_combined.drop(row_idx, inplace=True)
            removed_ids.append(row_idx)
        else:
            # update the dataframe with matched corners and their ids
            data_df_combined.at[row_idx, 'imgPoints_endo'] = imgPoints_matched_endo
            data_df_combined.at[row_idx, 'imgPoints_rs'] = imgPoints_matched_rs
            data_df_combined.at[row_idx, 'objPoints_rs'] = objPoints_matched_rs
            data_df_combined.at[row_idx, 'objPoints_endo'] = objPoints_matched_endo
            data_df_combined.at[row_idx, 'ids_endo'] = ids_endo
            data_df_combined.at[row_idx, 'ids_rs'] = ids_rs
            data_df_combined.at[row_idx, 'num_corners_detected'] = len(ids_endo)
    return data_df_combined


def select_min_num_corners(min_num_corners, percentage_of_corners, num_chess_corners):
    """
    Selects the minimum number of corners based on the provided criteria.
    This function determines the minimum number of corners to be used based on either a fixed minimum number,
    a percentage of the total number of chessboard corners, or both. The function prioritizes the larger value
    when both criteria are provided.
    Parameters:
        min_num_corners (int or None): The minimum number of corners specified. 
        If None, the function will use the percentage_of_corners.
    percentage_of_corners (float or None): 
        The percentage of the total number of chessboard corners to be used. 
        If None, the function will use min_num_corners.
    num_chess_corners (int): 
        The total number of chessboard corners.
    Returns:
    int: The determined minimum number of corners based on the provided criteria.
    If both are none, the min number of corners is 1. 
    If only the percentage corners is specified, we take the percentage_of_corners.
    If only the min corners is specified, we ake that as the number
    If both are specified, we take the larger one.
    """
    
    
    
    # None for both, min num corners as 1
    if min_num_corners is None and percentage_of_corners is None:
        min_num_corners = 1
    # only percentage of corners specified, we take percentage_of_corners
    elif min_num_corners is None and percentage_of_corners is not None:
        min_num_corners = int(percentage_of_corners * num_chess_corners)
    # Only min_num_corners specified
    elif min_num_corners is not None and percentage_of_corners is None:
        pass
    # both specified- we take whatever is largest
    else:
        min_num_corners_by_percentage = int(percentage_of_corners * num_chess_corners)
        if min_num_corners_by_percentage > min_num_corners:
            min_num_corners = min_num_corners_by_percentage
    return min_num_corners


def get_average_std(data, threshold=100, return_mean=False):
    """
    Computes the average, standard deviation, median, and quartiles of a list of data.
    Parameters:
        data (list): A list of data points. Each element in the list is a list of errors. 
        threshold (float): A threshold value to filter out outliers.
        return_mean (bool): If True, the function will return the mean and std of the data. If False, the function will return the median and IQR
    Returns:
        tuple: A tuple containing the average, standard deviation OR the median, and quartiles of the data.
    """
    avg_lst = []
    std_lst = []
    median_lst = []
    Q1_lst = []
    Q3_lst = []
    # filter out errors above threshold/ extreme errors outside IQR
    if threshold is None:
        threshold = np.median(np.array(data))+2*np.std(np.array(data))
        #threshold=20
    for errors in data:
        errors_np = np.array(errors)

        e = errors_np[errors_np < threshold]
        if e.size == 0:
            avg_lst.append(threshold)
            median_lst.append(threshold)
            std_lst.append(threshold)
            Q1_lst.append(threshold)
            Q3_lst.append(threshold)
            continue
        median_lst.append(np.percentile(e, 50))  # np.mean(e)
        avg_lst.append(np.mean(e))  # np.mean(e)
        std_lst.append(np.std(e))
        Q1_lst.append(np.percentile(e, 25))
        Q3_lst.append(np.percentile(e, 75))

    if return_mean:
        return np.array(avg_lst), np.array(avg_lst)-np.array(std_lst), np.array(avg_lst)+np.array(std_lst)
    else:
        return np.array(median_lst), Q1_lst, Q3_lst
