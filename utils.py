import os
import pandas as pd
import cv2
import sksurgerycore.transforms.matrix as skcm
import numpy as np
from scipy.spatial.transform import Rotation



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
    #common_idx = np.intersect1d(idx_realsense_sorted,idx_endo_sorted)
    
    # IDs found in endo but not in realsense
    unique_endo_id = np.setdiff1d(endo_sorted_ids, realsense_sorted_ids)
    # remove unique_endo_id from endo_sorted_ids
    new_endo_idx = ~np.isin(endo_sorted_ids, unique_endo_id)#(endo_sorted_ids != unique_endo_id).any(axis=1)
    #new_endo_idx = np.setdiff1d(endo_sorted_ids, unique_endo_id)

    if len(unique_endo_id)>0:
        endo_sorted_ids = endo_sorted_ids[new_endo_idx]
        corners_endo_sorted = corners_endo_sorted[new_endo_idx]

    # remove unique IDs found in rs but not endo
    unique_rs_id = np.setdiff1d(realsense_sorted_ids, endo_sorted_ids)
    new_rs_idx = ~np.isin(realsense_sorted_ids, unique_rs_id)
    #new_rs_idx = np.setdiff1d(realsense_sorted_ids, unique_rs_id)
    if len(unique_rs_id)>0:
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



def sample_dataset(df, total_samples=100):
    # Total samples required
    #total_samples = 100

    # Group by poses and angles
    grouped = df.groupby(['pose', 'deg'])

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

    for folder in folders:
        if not os.path.isdir(f'{folder}'):
            os.makedirs(f'{folder}')

def reprojection_error(imgpoints_detected, imgpoints_reprojected, image=None):
    """
    calculate reprojection error given the detected and reprojected points
    """
    squared_diffs = np.square(imgpoints_detected- imgpoints_reprojected)
    error_np = np.sqrt(np.sum(squared_diffs)/len(imgpoints_reprojected))

    # different way of calculating
    error = np.sqrt((np.square(cv2.norm(imgpoints_detected, imgpoints_reprojected, cv2.NORM_L2))/len(imgpoints_reprojected)))
    if image is not None:
        img_shape = image.shape
        for corner_reprojected, corner_detected in zip(imgpoints_reprojected,imgpoints_detected):
            # change dtypw of corner to int
            corner_detected = corner_detected.astype(int)
            corner_reprojected = corner_reprojected.astype(int)
            centre_detected = corner_detected.ravel()
            centre_reprojected = corner_reprojected.ravel()
            # check if points are within image
            if centre_detected[0] < 0 or centre_detected[0] > img_shape[1] or centre_detected[1] < 0 or centre_detected[1] > img_shape[0]:
                continue
            if centre_reprojected[0] < 0 or centre_reprojected[0] > img_shape[1] or centre_reprojected[1] < 0 or centre_reprojected[1] > img_shape[0]:
                continue
            cv2.circle(image, (int(centre_detected[0]),    int(centre_detected[1])),    3, (0, 0, 255), -1)
            cv2.circle(image, (int(centre_reprojected[0]), int(centre_reprojected[1])), 3, (0, 255, 0), -1)

        return error_np, error, image
    
    return error_np, error


def calculate_transform_average(r_lst, t_lst):
    """
    calculates mean of rotation and translation vectors using scipy
    Args:
        r_lst (list): list of rotation vectors
        t_lst (list): list of translation vectors
    Returns:
        mean_he (np.array): 4x4 transformation matrix
    """
    # average hand eye
    scipy_rot = Rotation.from_rotvec(np.array(r_lst).reshape((-1,3)), degrees=False)
    mean_rot = scipy_rot.mean().as_rotvec()

    mean_t = np.mean(np.asarray(t_lst).reshape(-1,3), axis=0)

    mean_he = extrinsic_vecs_to_matrix(mean_rot, mean_t)
    return mean_he


def find_best_intrinsics(intrinsics_pth, size_chess, camera, save_path=''):
    intrinsics_all_data = pd.read_pickle(f'{intrinsics_pth}/{size_chess}_{camera}_calibration_data.pkl')
    # find where average_error is smallest
    intrinsics_all_data = intrinsics_all_data[intrinsics_all_data.average_error == intrinsics_all_data.average_error.min()]
    errors_all = intrinsics_all_data['errors'].values[0]
    intrinsics = intrinsics_all_data['intrinsics'].values[0][errors_all.index(min(errors_all))]
    distortion = intrinsics_all_data['distortion'].values[0][errors_all.index(min(errors_all))]

    if len(save_path)>0:
        # save as txt file
        np.savetxt(f'{save_path}/{size_chess}_{camera}_intrinsics.txt', intrinsics)
        np.savetxt(f'{save_path}/{size_chess}_{camera}_distortion.txt', distortion)

    return intrinsics, distortion


def filter_and_merge_hand_eye_df(data_df_endo, data_df_realsense, info_df_endo):
    # combine information of paths for filtering those that don't match between endo and rs
    data_df_endo['combined_info'] = data_df_endo[['chess_size', 'pose', 'deg', 'direction', 'frame_number']].astype(str).agg('_'.join, axis=1)
    data_df_realsense['combined_info'] = data_df_realsense[['chess_size', 'pose', 'deg', 'direction', 'frame_number']].astype(str).agg('_'.join, axis=1)

    # find common images between endo and realsense
    common_keys = set(data_df_endo['combined_info']).intersection(set(data_df_realsense['combined_info']))
    # take out file names that don't match and reset index to ensure they're matching
    data_df_endo = data_df_endo[data_df_endo['combined_info'].isin(common_keys)].reset_index(drop=True)
    data_df_realsense = data_df_realsense[data_df_realsense['combined_info'].isin(common_keys)].reset_index(drop=True)

    # Drop the info key column 
    data_df_endo.drop(columns=['combined_info'], inplace=True)
    data_df_realsense.drop(columns=['combined_info'], inplace=True)

    # merge endo and rs into one dataframe
    common_columns = ['chess_size', 'pose', 'deg', 'direction', 'frame_number']
    data_df_combined = pd.merge(
        data_df_endo,
        data_df_realsense,
        on=common_columns,
        suffixes=('_endo', '_rs')
    )
    
    
    #### HAND-EYE CALIBRATION ####
    # filter out any unmatched points        
    removed_ids = []
    for row_idx, row in data_df_combined.iterrows():
        pnts_endo = row['imgPoints_endo']
        pnts_3d_rs = row['objPoints_rs']
        ids_e = row['ids_endo']
        ids_r = row['ids_rs']
        imgPoints_matched, objPoints_matched, img_ids, obj_ids = sort_and_filter_matched_corners(pnts_endo, pnts_3d_rs, ids_e, ids_r, return_ids=True)
        if len(imgPoints_matched)<info_df_endo.data[3]:
            # remove row from dataframe
            data_df_combined.drop(row_idx, inplace=True)
            removed_ids.append(row_idx)
        else:
            # update the dataframe
            data_df_combined.at[row_idx, 'imgPoints_endo'] = imgPoints_matched
            data_df_combined.at[row_idx, 'objPoints_rs'] = objPoints_matched
            data_df_combined.at[row_idx, 'ids_endo'] = img_ids
            data_df_combined.at[row_idx, 'ids_rs'] = obj_ids
    return data_df_combined
