import os
import pandas as pd
import cv2
import sksurgerycore.transforms.matrix as skcm
import numpy as np

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

