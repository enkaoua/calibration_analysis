


import argparse
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
import scipy

from charuco_utils import calculate_hand_eye_reprojection_error, calibrate_and_evaluate, calibrate_hand_eye_pnp_reprojection, detect_charuco_board_pose_images, detect_corners_charuco_cube_images
from record_utils import RealsenseVideoSourceAPI
from utils import extrinsic_matrix_to_vecs, extrinsic_vecs_to_matrix, sort_and_filter_matched_corners



def detect_board_position_and_corners(image, board, num_chess_corners,min_num_corners, intrinsics=None, distortion=None, cam='endo'):
    """   
    """
    board.setLegacyPattern(True) # set to True to use the old pattern
    if intrinsics is not None and distortion is not None:
        # undistort image
        #image = cv2.undistort(image, intrinsics, distortion, None, intrinsics)
        pass
    else:
        # load estimated intrinsics and distortion
        intrinsics_pth = f'calibration_estimates/intrinsics_{cam}.txt'
        intrinsics = np.loadtxt(intrinsics_pth)
        distortion = np.zeros(5)


    parameters = cv2.aruco.DetectorParameters()
    dictionary = board.getDictionary()

    # chessboard corners in 3D and corresponding ids
    charuco_corners_3D = board.getChessboardCorners()  # allCorners3D_np
    #num_chess_corners = len(charuco_corners_3D)  # number_of_corners_per_face
    charuco_ids_3D = np.arange(0, num_chess_corners)  # all3DIDs_np

    #min_num_corners = int(percentage_of_corners * num_chess_corners)

    # detect aruco tags
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    if corners:
        # get markers corresponding to said board
        # Refine not detected markers based on the already detected and the board layout.
        
        #corners, ids, rejectedImgPoints, recoveredIdxs = cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)
        
        # interpolate to get corners of charuco board
        ret, charuco_detected_corners, charuco_detected_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

        #print('detected corners: ', len(corners))
        if ret<min_num_corners:
            #cv2.aruco.drawDetectedCornersCharuco(image, corners, ids)
            output_image = cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 5, 0))

            return output_image, None, None, None, None, None, None
        
        # Estimate the pose of the ChArUco board
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_detected_corners, charuco_detected_ids, board,
                                                                intrinsics, distortion, None, None)
        
        if retval:
            # Draw the ChArUco board axis
            image = cv2.drawFrameAxes(image, intrinsics, None, rvec, tvec, length=37)
        # by this point, we have filtered out any images that don't have enough charuco 
        # board corners detected so we can add them to the list of images to save
        # draw the detected corners on the image
        cv2.aruco.drawDetectedCornersCharuco(image, charuco_detected_corners, charuco_detected_ids)
        #cv2.imshow('charuco board', image)
        #cv2.waitKey(waiting_time)
    else:
        return image, None, None, None, None, None, None

    # find the corresponding 3D pnts
    _, allCorners3D_np_sorted_filtered, _, allIDs3D_np_sorted_filtered = sort_and_filter_matched_corners(
                                                                                charuco_detected_corners.squeeze(), 
                                                                                charuco_corners_3D, charuco_detected_ids, 
                                                                                charuco_ids_3D,
                                                                                return_ids=True)

    # add the detected charuco corners to the list of all charuco corners
    # add the pose to the list of all poses
    #tag2cam = extrinsic_vecs_to_matrix(rvec, tvec)

    #return imgPoints, objPoints, num_detected_corners, ids_all  # rvec, tvec,
    num_detected_corners = len(charuco_detected_corners)
    all_3D_corners = allCorners3D_np_sorted_filtered.reshape(-1, 1, 3)
    return image, rvec, tvec, charuco_detected_corners, all_3D_corners, num_detected_corners, allIDs3D_np_sorted_filtered 


def initialise_cameras(endo_port, realsense_save_path, rs_port = 1):
    # initialising first cam (endoscope)
    endo_cam = cv2.VideoCapture(endo_port)
    if realsense_save_path:
        rs_cam = cv2.VideoCapture(rs_port)
        return endo_cam, rs_cam
    # initialising second camera (realsense)
    if realsense_save_path:
        if not os.path.exists(realsense_save_path):        
            os.makedirs(f'{realsense_save_path}/bag')
        # initialise realsense config
        realsense_cam = RealsenseVideoSourceAPI()
        realsense_cam.initialise_stereo_live_stream()
        return endo_cam, realsense_cam
    return endo_cam, None

def read_images(endo_cam, realsense_cam, realsense_save_path, count):
    ret_endo, endo_img = endo_cam.read() # capture image from first camera (endoscope)       
    if realsense_save_path:
        ret_rs, rs_img = realsense_cam.read()
        return ret_endo, ret_rs, endo_img, rs_img
    # read realsense
    if realsense_save_path:
        #ret, frames, color_image = read_realsense_image(pipeline)
        ret_rs, right_frame, left_frame, right_image, left_image, rs_img, depth_image = realsense_cam.read_stereo_image()
        return ret_endo, ret_rs, endo_img, rs_img
    return ret_endo, None, endo_img, None

def save_images(endo_save_pth, endo_img, realsense_save_path, realsense_cam, rs_img):
    # save img1
    #print(name)
    cv2.imwrite(f'{endo_save_pth}', endo_img) 

    # save realsense image
    if realsense_save_path:
        # save captured frame from cam 2 (realsense)
        cv2.imwrite(realsense_save_path, rs_img) 

        # SAVE REALSENSE CAMERA INFO TO BAG
        #realsense_cam.save_frameset(f'{realsense_save_path}')

def end_recording(endo_cam, realsense_save_path,realsense_cam ):
    print('stopped recording')

    endo_cam.release()
    
    # Stop streaming
    if realsense_save_path:
        realsense_cam.release() #.stop()

    cv2.destroyAllWindows()

    return
    



def rodrigues_to_euler(rvec):
    """Converts a rotation vector (rvec) to Euler angles in degrees (rx, ry, rz)."""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.degrees(np.array([x, y, z]))


""" def calib_frames_to_dataframe(calib_frames):

    data = []
    poses = calib_frames['poses']
    
    for idx, (rvec, tvec) in enumerate(poses):
        # Extract translation (x, y, z)
        tx, ty, tz = tvec.flatten()
        
        # Convert rotation vector to Euler angles (degrees)
        rx, ry, rz = rodrigues_to_euler(rvec)
        
        # Collect all the fields for each frame
        img_points = calib_frames['imgPoints'][idx]
        obj_points = calib_frames['objPoints'][idx]
        path = calib_frames['paths'][idx]
        num_corners = calib_frames['num_detected_corners'][idx]
        center_point = calib_frames['centre_point'][idx]
        
        # Append everything to the list
        data.append([tx, ty, tz, rx, ry, rz, img_points, obj_points, path, num_corners, center_point])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'rx', 'ry', 'rz', 'imgPoints', 'objPoints', 'paths', 'num_detected_corners', 'centre_point'])
    
    return df """

def calib_frames_to_dataframe(calib_frames, extension = '', T=False):
    """Adds x, y, z and Euler angle columns rx, ry, rz to an existing DataFrame of calibration frames."""
    if T:
        # convert 4x4 matrices T  to new column "poses" with (rvec, tvec) using rvec, tvec = extrinsic_matric_to_vecs(T)
        calib_frames['poses'] = calib_frames[f'T{extension}'].apply(lambda pose: extrinsic_matrix_to_vecs(pose))

    # Extract x, y, z from tvec and Euler angles from rvec using apply
    calib_frames[['x', 'y', 'z']] = calib_frames[f'poses{extension}'].apply(lambda pose: pd.Series(pose[1].flatten()))
    
    calib_frames[['rx', 'ry', 'rz']] = calib_frames[f'poses{extension}'].apply(
        lambda pose: pd.Series(rodrigues_to_euler(pose[0]))
    )
    
    return calib_frames



def assign_category(param_rows, grid_size=None, minimum=None, maximum=None, bins=None, spacing=10):
    if bins is None :
        if minimum is None :
            minimum = param_rows.min()
        if maximum is None :
            maximum = param_rows.max()
        
        if grid_size:
            bins = np.linspace(minimum, maximum, grid_size+1)
        else:
            # however many bins can be created with spacing defined (but include min and max)
            bins = np.arange(minimum, maximum, spacing)

    # assign each row to a category
    categories = np.digitize(param_rows, bins) - 1 
    # rows with category assigned to -1 should be assigned 0 and rows with category assigned to larger than grid_size-1 should be assigned grid_size-1
    categories[categories == -1] = 0
    categories[categories > len(bins)-2] = len(bins)-2
    return categories

def select_diverse_samples(df, num_images_for_calibration, min_positions=1, min_distances=1, min_angles=1, grid_size=4):
    """
    Selects samples from a DataFrame while maximizing diversity across given categories and spatial spread.

    Parameters:
    - df: The input DataFrame.
    - num_images_for_calibration: The number of samples to select.
    - min_positions, min_distances, min_angles: Minimum unique values required for each category.

    Returns:
    - A DataFrame containing the selected samples.
    """
    ########## DISTANCES ##########

    # Step 1: Select Categories that are Maximally Separated
    # -------------------------------------
    # Select categories for positions, distances, and angles
    unique_distances = sorted(df['distance_category'].unique())
    selected_distances = np.linspace(0, len(unique_distances)-1, min(min_distances, len(unique_distances)), dtype=int)
    selected_distances = [unique_distances[i] for i in selected_distances]

    # filter df to include rows with selected categories (when row meets either of categories)
    #filtered_df = df[(df['position_category'].isin(selected_positions)) | (df['distance_category'].isin(selected_distances)) | (df['angle_category'].isin(selected_angles))]
    filtered_df_dist = df[(df['distance_category'].isin(selected_distances))] #.reset_index(drop=True)
    
    # Stage 2: Maximize Spread Within Selected Categories
    # -------------------------------------
    # select one sample from each category which is the most high spread from the rest of the samples in the category
    final_samples_dist = []
    # for distances, select minimum of each category
    for distance in selected_distances:
        # filter df to include rows with selected category
        dist = filtered_df_dist[filtered_df_dist['distance_category'] == distance]
        # select onr row of minimum distance of this category
        min_distance_row = dist.loc[dist['z'].idxmin()]
        final_samples_dist.append(min_distance_row)

    # drop selected samples from original df
    df_dist = pd.DataFrame(final_samples_dist)
    df = df.drop(df_dist.index).reset_index(drop=True)

    ########## POSITIONS ##########

    # Determine maximum unique values we can select based on constraints
    unique_positions = sorted(df['position_category'].unique())
    selected_positions = np.linspace(0, len(unique_positions)-1, min(min_positions, len(unique_positions)), dtype=int)
    selected_positions = [unique_positions[i] for i in selected_positions]

    filtered_df_pos = df[(df['position_category'].isin(selected_positions))]
    final_samples_pos = []
    for idx, position in enumerate(selected_positions):
        pos = filtered_df_pos[filtered_df_pos['position_category'] == position]
        # sort by x and y (x takes precedence as img larger on x axis)
        pos = pos.sort_values(by=['x', 'y'])
        if idx % (grid_size-1) == 0:
            # select min
            sample = pos.iloc[0]
        if idx % (grid_size-1) == grid_size-2:
            # select max
            sample = pos.iloc[-1]
        else:
            # select centre row
            sample = pos.iloc[int(len(pos)/2)]

        final_samples_pos.append(sample)

    # drop selected samples from original df
    df_pos = pd.DataFrame(final_samples_pos)
    df = df.drop(df_pos.index)

    ########## ANGLES ##########
    unique_angles = sorted(df['angle_category'].unique())
    selected_angles = np.linspace(0, len(unique_angles)-1, min(min_angles, len(unique_angles)), dtype=int)
    selected_angles = [unique_angles[i] for i in selected_angles]

    filtered_df_angle = df[(df['angle_category'].isin(selected_angles))]
    final_samples_angle = []

    for angle in selected_angles:
        # filter df to include rows with selected category
        deg = filtered_df_angle[filtered_df_angle['angle_category'] == angle]
        # sort rows by rx ry rz
        deg = deg.sort_values(by=['rx', 'ry', 'rz'])
        # select centre row
        sample = deg.iloc[int(len(deg)/2)]
        final_samples_angle.append(sample)

    # drop selected samples from original df
    df_angle = pd.DataFrame(final_samples_angle)
    remaining_df = df.drop(df_angle.index).reset_index(drop=True)
    
    # final selection
    final_selected_df = pd.concat([df_dist, df_pos, df_angle])

    

    return final_selected_df, remaining_df
    



def select_max_spread_rows(df_subset, coords=['x', 'y']):
    # Calculate pairwise distances for the given coordinates
    coordinates = df_subset[coords].values
    pairwise_distances = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(coordinates)
        )

    # Use a greedy selection to maximize spread
    selected_indices = [0]  # Start with an initial random point (index 0 of the subset)
    for _ in range(1, min(len(df_subset), 1)):  # Only select one row to ensure spread
        remaining_indices = [i for i in range(len(df_subset)) if i not in selected_indices]
        # Find the index that is maximally distant from current selections
        max_distance_idx = max(remaining_indices, key=lambda idx: np.min(pairwise_distances[idx, selected_indices]))
        selected_indices.append(max_distance_idx)
    
    # Return the maximally spread row
    return df_subset.iloc[selected_indices].head(1)



def bin_and_sample(df, num_images_for_calibration=30, grid_size_x=3, grid_size_y=3, min_positions=4, min_distances=1, min_angles=5, max_distance_threshold=1000,min_distance_threshold=10, min_angle_threshold=-40, max_angle_threshold=40):
    """Bins the data and selects evenly distributed samples ensuring 4 different poses and angles."""
    # save dataframe as pkl
    #df.to_pickle('unprocessed_data.pkl')
    # first get rid of any rows smaller than min_angle_threshold deg or larger than max_angle_threshold
    #df = df[(df['rx'] >= min_angle_threshold) & (df['rx'] <= max_angle_threshold) & (df['ry'] >= min_angle_threshold) & (df['ry'] <= max_angle_threshold) & (df['rz'] >= min_angle_threshold) & (df['rz'] <= max_angle_threshold)]

    # create bins/category for distances
    # filter images with distances larger than max distance threshold and smaler than min distance threshold
    df = df[(df['z'] < max_distance_threshold) & (df['z'] > min_distance_threshold)]

    # 1) split distances into 10 bins depending on min and max- equal separation
    """ min_z, max_z = df['z'].min(), df['z'].max()
    z_bins = np.linspace(min_z, max_z, 10) """
    # 2) assign distance category to each row
    # Function to assign each row to a distance category based on z
    """     def get_distance_category(row):
        z_bin = np.digitize(row['z'], z_bins) - 1  # Subtract 1 for 0-based indexing
        return z_bin """
    #df['distance_category'] = df.apply(get_distance_category, axis=1)
    df['distance_category'] = assign_category(df['z'], 10)

    # create new column category for poses where we have a grid 1-9 ( top left x-y combo should be pose 1 with grid of 9- where 9th is bottom right)
    # 1) define coordinates for grid (split x into 3 and y into 3 from min to max)
    # filter df so that only the most common distance is used
    df_filtered = df[df['distance_category'] == df['distance_category'].mode()[0]]

    
    min_x, max_x = df_filtered['x'].min(), df_filtered['x'].max()
    x_cats = assign_category(df['x'], grid_size=grid_size_x, minimum=min_x, maximum=max_x)
    min_y, max_y = df_filtered['y'].min(), df_filtered['y'].max()
    y_cats = assign_category(df['y'], grid_size=grid_size_y, minimum=min_y, maximum=max_y)

    # combine to create position category
    df['position_category'] = 3 * y_cats + x_cats
    
    
    # Define angle bins for rx, ry, and rz
    angle_bins = [min_angle_threshold, -30, 0, 30, max_angle_threshold]  # Corresponds to ranges: (-inf, -30], (-30, 0], (0, 30]
    """ # create bins for angles (-30,-30,-30 is bin 1, 0,-30,-30 is bin 2, 0,0,-30 3 etc until 30,30,30)
    
    # Define function to determine angle category based on rx, ry, rz
    def get_angle_category(row):
        # Determine bins for each angle component
        rx_bin = np.digitize(row['rx'], angle_bins) - 1  # -1 for 0-based indexing
        ry_bin = np.digitize(row['ry'], angle_bins) - 1
        rz_bin = np.digitize(row['rz'], angle_bins) - 1
        
        # Compute a unique bin index for (rx, ry, rz)
        angle_category = 9 * rx_bin + 3 * ry_bin + rz_bin
        return angle_category

    # Apply function to each row to create 'angle_category'
    df['angle_category'] = df.apply(get_angle_category, axis=1) """
    
    angle_cats_x = assign_category(df['rx'], spacing=10)
    angle_cats_y = assign_category(df['ry'], spacing=10)
    angle_cats_z = assign_category(df['rz'], spacing=10)

    # combine to create angle category
    angle_category = 9 * angle_cats_z + 3 * angle_cats_y + angle_cats_x
    df['angle_category'] = angle_category
    
    """ # Sample evenly distributed frames across all groups

    # Group by bins
    grouped_df = df.groupby(['angle_category', 'position_category', 'distance_category'])
    
    # Ensure we have at least 4 distinct position groups
    position_groups = grouped_df['x', 'y', 'z'].ngroups

    # get rid of rows of grouped which dont have any in the counts
    if position_groups < 4:
        raise ValueError("Not enough distinct 3D positions available.")
    
    # Ensure we have at least 4 distinct angle groups
    angle_groups = grouped_df['rx', 'ry', 'rz'].ngroups
    if angle_groups < 4:
        raise ValueError("Not enough distinct angles available.")

    sampled_df = grouped_df.apply(lambda group: group.sample(min(1, len(group))))  # Sample 1 frame per group
    sampled_df = sampled_df.sample(n=min(num_images_for_calibration, len(sampled_df)))  # Select the final frames
    
    rest_of_df = df.drop(sampled_df.index)

    # Reset index for final result
    sampled_df = sampled_df.reset_index(drop=True)
    rest_of_df = rest_of_df.reset_index(drop=True) """

    sampled_df, rest_of_df = select_diverse_samples(df, num_images_for_calibration, min_positions=min_positions, min_distances=min_distances, min_angles=min_angles, grid_size=grid_size_x)

    return sampled_df, rest_of_df



def show_frame(img, count, max_frames_recorded_for_calibration, endo_port, correct=True, extra_text=None):
    if correct:
        col = (0, 255, 0)
    else:
        col = (0, 0, 255)
    # percentage of frames recorded for calib
    cv2.putText(img, f'percentage frames: {round(100*(count/max_frames_recorded_for_calibration),1)}%', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    # if frame recorded or not
    position_to_place_text = (20, img.shape[0])
    cv2.putText(img, f'Frame recorded: {correct}', position_to_place_text, cv2.FONT_HERSHEY_SIMPLEX, 5, col, 2)
    # distance vector from cam
    if extra_text is not None:
        cv2.putText(img, f'{extra_text}', (100, int(img.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, col, 2)
    cv2.namedWindow(f'endoscope port {endo_port}', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(f'endoscope port {endo_port}', img)


def record_board_live(save_path, 
                      realsense_save_path='', 
                      endo_port = 0, 
                      rs_port=1,
                      board=None, 
                      #calibration_estimates_pth = 'calibration_estimates/intrinsics_endo.txt', 
                      calibration_estimates_pth_endo = 'calibration_estimates/intrinsics_mac.txt', 
                      calibration_estimates_pth_rs = 'calibration_estimates/intrinsics_realsense.txt',
                      max_frames_recorded_for_calibration=np.inf, 
                      save_calibration_images = True,
                      num_images_for_calibration=10,
                      percentage_of_corners = 0.3,
                      HAND_EYE = False,
                      too_far_distance = 1000
                      #grid_size_x=3,  grid_size_y=3, 
                      #min_positions=4,  min_distances=1,  min_angles=5, 
                      #max_distance_threshold=1000,  min_distance_threshold=10, 
                      #min_angle_threshold=-40,  max_angle_threshold=40,
                      #calibrate_at_the_end_of_recording = False,

                      ):
    """
    records video with realsense and endo cams

    Press 'q' or esc to quit.
    parames:
        - endo_save_path (str): path where to save images from endoscope camera
        - realsense_save_path (str, optional): path where to save images for calibration of second camera (if you have a second camera set up)
        - endo_port (int, optional): specify port of first camera for reading opencv images
    """
    RECORD_RS = False
    # Creating folders where vid and images for first cam will be saved 
    endo_save_path = os.path.join(save_path, 'endo_images')
    if not os.path.exists(endo_save_path):
        os.makedirs(endo_save_path)
    if len(realsense_save_path ) > 0:
        realsense_save_path = os.path.join(save_path, 'realsense_images')
        if not os.path.exists(realsense_save_path):
            os.makedirs(realsense_save_path)
        RECORD_RS = True
        

    # ----- CAMERA INIIALISATIONS AND CONFIGURATIONS -----
    # check if endo_port is a string or int
    image_file =  isinstance(endo_port, str)

    if not image_file:
        # initialise cameras if from port
        endo_cam, realsense_cam = initialise_cameras(endo_port, realsense_save_path, rs_port=rs_port)
    else:
        # to load existing frames (and we dont need to save them)
        save_calibration_images =  False
        image_files_endo = glob.glob(endo_save_path + '/*.png')
        image_files_rs = glob.glob(realsense_save_path + '/*.png')

    count = 0 # count number of frames
    calib_frames_endo = {
        'imgPoints': [], 'objPoints': [],
        'paths': [], 'poses': [],
        'num_detected_corners': [],
        'centre_point': [],
        'ids': []
    } 
    # if realsense recorded copy empty df same as endo
    if RECORD_RS:
        calib_frames_rs = {
        'imgPoints': [], 'objPoints': [],
        'paths': [], 'poses': [],
        'num_detected_corners': [],
        'centre_point': [],
        'ids': []
    } 

    
    # if we do hand-eye create df with empty columns
    if HAND_EYE:
        calib_frames_combined = {
            'imgPoints_endo': [], 'imgPoints_rs': [], 'objPoints_endo': [], 'objPoints_rs': [],
            'paths_endo': [], 'paths_rs': [], 'poses_endo': [], 'poses_rs': [],'T_endo':[], 'T_rs':[],
            'num_detected_corners': [],
            'ids_endo': [], 'ids_rs': [], 'centre_point_endo': [], 'centre_point_rs': [],
        }



        
    # -------------- RECORDING ------------------
    # load `poses to display for user to follow
    intrinsics_endo = np.loadtxt(calibration_estimates_pth_endo)
    distortion_endo = np.zeros(5)
    intrinsics_rs = np.loadtxt(calibration_estimates_pth_rs)
    distortion_rs = np.zeros(5)
    #err = np.inf
    # record_calib = False
    charuco_corners_3D = board.getChessboardCorners()  # allCorners3D_np
    num_chess_corners = len(charuco_corners_3D)  # number_of_corners_per_face 
    min_num_corners = int(percentage_of_corners * num_chess_corners)
    while True:
        key = 0xFF & cv2.waitKey(1)
        # -------> READ IMAGES
        if not image_file:
            # read images from both cameras
            ret_endo, ret_rs, endo_img, rs_img = read_images(endo_cam, realsense_cam, realsense_save_path, count)
            if RECORD_RS:
                returned = ret_endo and ret_rs
            else:
                returned = ret_endo
        else:
            # read images from files
            endo_img = cv2.imread(image_files_endo[count])
            if RECORD_RS:
                rs_img = cv2.imread(image_files_rs[count])
                # check if neither rs_img or endo_img is None
                returned = rs_img is not None and endo_img is not None
            else:
                returned = endo_img is not None

        if not returned:
            continue

        # name and show frame from first cam (endoscope)
        endo_frame_name = '{}/{:08d}.png'.format(endo_save_path, count)
        annotated_endo_img = endo_img.copy()

        if RECORD_RS:
            rs_frame_name = '{}/{:08d}.png'.format(realsense_save_path, count)
            annotated_rs_img = rs_img.copy()

        # display all centre points (green for calibration frames, red for reprojection frames)
        """ for centre_point in calib_frames['imgPoints']:
            cv2.circle(annotated_endo_img, centre_point, 5, (0, 255, 0), -1)
        for centre_point in reproj_frames['imgPoints']:
            cv2.circle(annotated_endo_img, centre_point, 5, (0, 0, 255), -1) """

        # -------> DETECT BOARD AND CORNERS
        # detect corners
        annotated_endo_img, tag2cam_rvec_endo, tag2cam_tvec_endo, charuco_detected_corners_endo, pnts_3d_endo, num_detected_corners_endo, ids_endo  = detect_board_position_and_corners(annotated_endo_img, board, num_chess_corners, min_num_corners, intrinsics=intrinsics_endo, distortion=distortion_endo, cam='endo')
        #num_detected_corners = num_detected_corners_endo
        merged_img = annotated_endo_img
        num_corners_assert = not num_detected_corners_endo or num_detected_corners_endo < min_num_corners

            
        if RECORD_RS:
            annotated_rs_img, tag2cam_rvec_rs, tag2cam_tvec_rs, charuco_detected_corners_rs, pnts_3d_rs, num_detected_corners_rs, ids_rs = detect_board_position_and_corners(annotated_rs_img, board,num_chess_corners, min_num_corners , intrinsics=intrinsics_rs, distortion=distortion_rs, cam='realsense')
            # downsample rs img so it matches endo img
            annotated_rs_img = cv2.resize(annotated_rs_img, (annotated_endo_img.shape[1], annotated_endo_img.shape[0]))
            merged_img = np.hstack((annotated_endo_img, annotated_rs_img)) 
            num_corners_assert = num_corners_assert or not num_detected_corners_rs or num_detected_corners_rs < min_num_corners

        # continue if not enough corners detected on either camera
        if num_corners_assert or tag2cam_tvec_endo is None or tag2cam_tvec_endo[2] > too_far_distance:
            extra_text = 'Not enough corners detected on either camera'
            if tag2cam_tvec_endo is not None:
                    if tag2cam_tvec_endo[2] > too_far_distance:
                        extra_text = 'TOO FAR'
            show_frame(merged_img, count, max_frames_recorded_for_calibration, endo_port, correct=False, extra_text=extra_text)
            continue
        
        
        if HAND_EYE:    
            # sort and filter matched corners
            imgPoints_matched_endo, objPoints_matched_rs, ids_endo, ids_rs = sort_and_filter_matched_corners(charuco_detected_corners_endo, pnts_3d_rs,
                                                                                                    ids_endo, ids_rs,
                                                                                                    return_ids=True)
            # filter also realsense pnts and endo object points
            imgPoints_matched_rs, objPoints_matched_endo, ids_rs_2, ids_endo_2 = sort_and_filter_matched_corners(charuco_detected_corners_rs, pnts_3d_endo,
                                                                                                            ids_rs, ids_endo,
                                                                                                            return_ids=True)
            num_detected_corners = len(imgPoints_matched_endo)
            #merged_img = np.hstack((annotated_endo_img, annotated_rs_img)) 


            # continue if not enough corners detected
            if not num_detected_corners or num_detected_corners < min_num_corners or tag2cam_tvec_endo is None or tag2cam_tvec_endo[2] > too_far_distance:
                extra_text = 'Not enough MATCHING corners detected on either camera'
                if tag2cam_tvec_endo is not None:
                    if tag2cam_tvec_endo[2] > too_far_distance:
                        extra_text = 'TOO FAR'
                show_frame(merged_img, count, max_frames_recorded_for_calibration, endo_port, correct=False, extra_text=extra_text)

                continue     


        # ADD DATA TO calibration frames dictionary
        # add endo data
        # create 4x4 matrix from rvec and tvec
        center_point_endo = tuple(charuco_detected_corners_endo.mean(axis=0).astype(int).squeeze())
        calib_frames_endo['poses'].append((tag2cam_rvec_endo, tag2cam_tvec_endo))
        calib_frames_endo['paths'].append(endo_frame_name)
        calib_frames_endo['imgPoints'].append(charuco_detected_corners_endo)
        calib_frames_endo['objPoints'].append(pnts_3d_endo)
        calib_frames_endo['num_detected_corners'].append(num_detected_corners_endo)
        calib_frames_endo['centre_point'].append(center_point_endo)
        calib_frames_endo['ids'].append(ids_endo)

        # add rs data if rs pth
        if RECORD_RS:
            # create 4x4 matrix from rvec and tvec
            center_point_rs = tuple(charuco_detected_corners_rs.mean(axis=0).astype(int).squeeze())
            calib_frames_rs['poses'].append((tag2cam_rvec_rs, tag2cam_tvec_rs))
            calib_frames_rs['paths'].append(rs_frame_name)
            calib_frames_rs['imgPoints'].append(charuco_detected_corners_rs)
            calib_frames_rs['objPoints'].append(pnts_3d_rs)
            calib_frames_rs['num_detected_corners'].append(num_detected_corners_rs)
            calib_frames_rs['centre_point'].append(center_point_rs)
            calib_frames_rs['ids'].append(ids_rs)

        if HAND_EYE:
            pose_endo = extrinsic_vecs_to_matrix(tag2cam_rvec_endo, tag2cam_tvec_endo)
            pose_rs = extrinsic_vecs_to_matrix(tag2cam_rvec_rs, tag2cam_tvec_rs)

            calib_frames_combined['poses_endo'].append((tag2cam_rvec_endo, tag2cam_tvec_endo))
            calib_frames_combined['poses_rs'].append((tag2cam_rvec_rs, tag2cam_tvec_rs))
            calib_frames_combined['T_endo'].append(pose_endo)
            calib_frames_combined['T_rs'].append(pose_rs)
            calib_frames_combined['paths_endo'].append(endo_frame_name)
            calib_frames_combined['paths_rs'].append(rs_frame_name)
            calib_frames_combined['imgPoints_endo'].append(imgPoints_matched_endo)
            calib_frames_combined['imgPoints_rs'].append(imgPoints_matched_rs)
            calib_frames_combined['objPoints_endo'].append(objPoints_matched_endo)
            calib_frames_combined['objPoints_rs'].append(objPoints_matched_rs)
            calib_frames_combined['ids_endo'].append(ids_endo)
            calib_frames_combined['ids_rs'].append(ids_rs)
            calib_frames_combined['num_detected_corners'].append(num_detected_corners)
            calib_frames_combined['centre_point_endo'].append(center_point_endo)
            calib_frames_combined['centre_point_rs'].append(center_point_rs)

        #cv2.namedWindow(f'endoscope port {endo_port}', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow(f'endoscope port {endo_port}', merged_img)
        show_frame(merged_img, count, max_frames_recorded_for_calibration, endo_port, correct=True)
        # ----------- SAVING IMAGES ---------------------
        if save_calibration_images:
            save_images(f'{endo_frame_name}', endo_img, rs_frame_name, realsense_cam, rs_img)

        count += 1

        # -------> CHECK IF GOAL ACHIEVED THAT AT LEAST X DIFFERENT ANGLES (30 DEGREE DIFFERENCES) AND X POSES (10 CM DIFF) HAVE BEEN DETECTED
        
        # when finished recording, save video and release streams
        if key == 27 or key==ord('q') or count == max_frames_recorded_for_calibration :
            total_images = len(calib_frames_combined['paths_endo']) if HAND_EYE else len(calib_frames_endo['paths'])
            if total_images < num_images_for_calibration:
                print('Not enough frames, continuing')
                continue

            # save endo dataframe
            #df_endo = calib_frames_to_dataframe(calib_frames_endo)
            df_endo = pd.DataFrame(calib_frames_endo)
            df_endo.to_pickle(f'{save_path}/data_endo.pkl')

            if RECORD_RS:
                # save rs dataframe
                #df_rs = calib_frames_to_dataframe(calib_frames_rs)
                df_rs = pd.DataFrame(calib_frames_rs)
                df_rs.to_pickle(f'{save_path}/data_rs.pkl')

            if HAND_EYE:
                # save combined dataframe
                #df_combined = calib_frames_to_dataframe(calib_frames_combined)
                df_combined = pd.DataFrame(calib_frames_combined)
                df_combined.to_pickle(f'{save_path}/data_combined.pkl')

            end_recording(endo_cam, realsense_save_path,realsense_cam )
            """ # Bin and sample frames
            if calibrate_at_the_end_of_recording:
                frames_for_calibration,frames_for_reprojection  =  bin_and_sample(df, num_images_for_calibration=num_images_for_calibration, grid_size_x=grid_size_x, grid_size_y=grid_size_y, min_positions=min_positions, min_distances=min_distances, min_angles=min_angles, max_distance_threshold=max_distance_threshold,min_distance_threshold=min_distance_threshold, min_angle_threshold=min_angle_threshold, max_angle_threshold=max_angle_threshold)

                intrinsics, dist, err, num_corners_detected = perform_calibration(frames_for_calibration, frames_for_reprojection, calibration_estimates_pth_endo, visualise_reprojection_error = False,     waitTime = 1)
                                                                             
                print('Calibration successful')
                print('intrinsics: ', intrinsics)
                print('dist: ', dist)
                print('err: ', err)
                print('num_corners_detected: ', num_corners_detected) """
            
            
            """ if err < 0.5:
                print('Calibration successful')
                print('mtx: ', mtx)
                print('dist: ', dist)
                print('err: ', err)
                print('num_corners_detected: ', num_corners_detected)
                end_recording(endo_cam, realsense_save_path,realsense_cam )
                break
            else:
                print('30 frames unsuccessful, continuing')
                continue """
            
            # end_recording(endo_cam, realsense_save_path,realsense_cam )
            break  
    if RECORD_RS:
        return df_endo, df_rs, None
    if HAND_EYE:
        return df_endo, df_rs, df_combined
    return df_endo, None, None



def perform_hand_eye_calibration(calibration_data, data_for_optimisation,reprojection_data_df, intrinsics_endo, distortion_endo, optimise=True, error_threshold=1, num_samples_for_optimisation=100, waitTime=1, visualise_reprojection_error=False):
        """ intrinsics_endo = np.loadtxt(f'{intrinsics_endo_pth}')
        distortion_endo = np.loadtxt(f'{distortion_endo_pth}') """
        hand_eye = calibrate_hand_eye_pnp_reprojection(calibration_data,data_for_optimisation, intrinsics_endo=intrinsics_endo, distortion_endo=distortion_endo, optimise=optimise, error_threshold=error_threshold, num_samples_for_optimisation=num_samples_for_optimisation, groupby_cats=['position_category', 'angle_category'])
        
        world2realsense = reprojection_data_df['T_rs'].values
        objPoints = reprojection_data_df['objPoints_rs'].values
        imgPoints = reprojection_data_df['imgPoints_endo'].values

        # calculate reprojection error
        if visualise_reprojection_error:
            endo_reprojection_images_pth = reprojection_data_df['paths_endo'].values
        else:
            endo_reprojection_images_pth = []

        err_np = calculate_hand_eye_reprojection_error(hand_eye, world2realsense,
                                                       objPoints, imgPoints,
                                                       intrinsics_endo, distortion_endo,
                                                       waitTime=waitTime,
                                                       endo_reprojection_images_pth=endo_reprojection_images_pth)
        
        reprojection_error_mean_final = pd.DataFrame(err_np).median()[0]
        """ mean_err = pd.DataFrame(err_np).mean()[0]
        if mean_err - median_err > 0.5:
            reprojection_error_mean_final = median_err
        else:
            reprojection_error_mean_final = mean_err """
        return hand_eye, reprojection_error_mean_final


def perform_calibration(calib_frames, reproj_frames, calibration_estimates_pth, visualise_reprojection_error = False,     waitTime = 1):
    calibration_data_df = pd.DataFrame(calib_frames)
    reproj_data_df = pd.DataFrame(reproj_frames)
    num_images_to_sample_for_calibration = None

    intrinsics_initial_guess_pth = f'{calibration_estimates_pth}'
    endo_img = cv2.imread(calib_frames['paths'].values[0])
    image_shape = endo_img.shape[:2]

    # perform calibration on selected frames
    args = calibration_data_df, num_images_to_sample_for_calibration, reproj_data_df, intrinsics_initial_guess_pth, image_shape, visualise_reprojection_error, waitTime
    mtx, dist, err, num_corners_detected = calibrate_and_evaluate(args)

    return mtx, dist, err, num_corners_detected



def update_Ts(df_combined, board, intrinsics_endo, distortion_endo, intrinsics_rs, distortion_rs):
    " update T_endo and T_rs in df_combined with new T_endo and T_rs by estimating poses using new intrinsics and distortion matrices"
    img_points_endo = df_combined['imgPoints_endo'].values
    detected_ids_endo = df_combined['ids_endo'].values
    img_points_rs = df_combined['imgPoints_rs'].values
    detected_ids_rs = df_combined['ids_rs'].values
    # estimate charuco board pose for each row

    T_rs_list = []
    T_endo_list = []
    for i in range(len(img_points_endo)):
        # undistort points
        img_points_endo[i] = cv2.undistortPoints(img_points_endo[i], intrinsics_endo, distortion_endo, P=intrinsics_endo)
        img_points_rs[i] = cv2.undistortPoints(img_points_rs[i], intrinsics_rs, distortion_rs, P=intrinsics_rs)
        retval, rvec_endo, tvec_endo = cv2.aruco.estimatePoseCharucoBoard(img_points_endo[i], detected_ids_endo[i], board,
                                                                intrinsics_endo, distortion_endo, None, None)
        retval, rvec_rs, tvec_rs = cv2.aruco.estimatePoseCharucoBoard(img_points_rs[i], detected_ids_rs[i], board, 
                                                                      intrinsics_rs, distortion_rs, None, None)
        # replace value of row
        df_combined.at[i, 'T_endo'] = extrinsic_vecs_to_matrix(rvec_endo, tvec_endo)
        df_combined.at[i, 'T_rs'] = extrinsic_vecs_to_matrix(rvec_rs, tvec_rs)
        #T_rs_list.append(extrinsic_vecs_to_matrix(rvec_rs, tvec_rs))
        #T_endo_list.append(extrinsic_vecs_to_matrix(rvec_endo, tvec_endo))

    #df_combined['T_rs'] = T_rs_list
    #df_combined['T_endo'] = T_endo_list
    return df_combined



def main(aruco_w=7,
         aruco_h=11,
         size_of_checkerboard=15,
         aruco_size=11,
                      calib_save_path='results/user_study/mac/aure/rs',
                      realsense_save_path='results/user_study/mac/aure/rs', 
                      endo_port = 1, 
                      rs_port = 0,
                      calibration_estimates_pth_endo = 'calibration_estimates/intrinsics_mac.txt', 
                      calibration_estimates_pth_rs = 'calibration_estimates/intrinsics_realsense.txt', 
                      max_frames_recorded_for_calibration=200, #np.inf, 
                    too_far_distance = 1500,
                      save_calibration_images = True,
                      

                      num_images_for_calibration=10,
                      grid_size_x=3, 
                      grid_size_y=3, 
                      min_positions=9, 
                      min_distances=1, 
                      min_angles=10, 
                      max_distance_threshold=1000,
                      min_distance_threshold=10, 
                      min_angle_threshold=-40, 
                      max_angle_threshold=40, 
                      visualise_reprojection_error = False,
                      waitTime=0,
                      reprojection_df_pth = 'results/user_study/mac/aure/reprojection_dataset',#'', # pth where the reprojection df is saved,
                      percentage_of_corners = 0.5,
                      HAND_EYE = True,
                      num_images_for_he_calibration=30
                      ): 
    



    # 0) initialise params
    aruco_dict = cv2.aruco.DICT_4X4_1000
    board = cv2.aruco.CharucoBoard(
        (aruco_w, aruco_h),  # 7x5
        size_of_checkerboard,
        aruco_size,
        cv2.aruco.getPredefinedDictionary(aruco_dict)
    )


    # RECORD REPROJECTION IMAGES:
    data_pth = glob.glob(f'{reprojection_df_pth}/data*.pkl')
    if len(data_pth) >= 1:
        # read data
        df_endo_reproj = pd.read_pickle(f'{reprojection_df_pth}/data_endo.pkl')
        if len(realsense_save_path)>0:
            df_rs_reproj = pd.read_pickle(f'{reprojection_df_pth}/data_rs.pkl')
        if HAND_EYE:
            df_combined_reproj = pd.read_pickle(f'{reprojection_df_pth}/data_combined.pkl')
    else:
        # record images
        df_endo_reproj, df_rs_reproj, df_combined_reproj = record_board_live(reprojection_df_pth, 
                        realsense_save_path=realsense_save_path, 
                        endo_port = endo_port, 
                        rs_port=rs_port,
                        board=board, 
                        calibration_estimates_pth_endo = calibration_estimates_pth_endo, 
                        calibration_estimates_pth_rs = calibration_estimates_pth_rs, 
                        max_frames_recorded_for_calibration=300, 
                        save_calibration_images = True,
                        num_images_for_calibration=num_images_for_calibration,
                        percentage_of_corners=percentage_of_corners,
                        HAND_EYE = HAND_EYE,
                        too_far_distance = too_far_distance

                        )

    # RECORD DATA FOR CALIBRATION
    # check if theres a csv file in calib_save_path
    data_pth = glob.glob(f'{calib_save_path}/data*.pkl')
    if len(data_pth) >= 1:
        # read data
        df_endo = pd.read_pickle(f'{calib_save_path}/data_endo.pkl')
        if len(realsense_save_path)>0:
            df_rs = pd.read_pickle(f'{calib_save_path}/data_rs.pkl')
        if HAND_EYE:
            df_combined = pd.read_pickle(f'{calib_save_path}/data_combined.pkl')
        
    else:
        # 1) record video of board with different poses and angles- tell user if they're too close
        # record images
        # time how long it takes in minumts
        start_time = time.time()
        df_endo, df_rs, df_combined = record_board_live(calib_save_path, 
                        realsense_save_path=realsense_save_path, 
                        endo_port = endo_port, 
                        rs_port=rs_port,
                        board=board, 
                        calibration_estimates_pth_endo = calibration_estimates_pth_endo, 
                        calibration_estimates_pth_rs = calibration_estimates_pth_rs, 
                        max_frames_recorded_for_calibration=max_frames_recorded_for_calibration, 
                        save_calibration_images = save_calibration_images,
                        num_images_for_calibration=num_images_for_calibration,
                        percentage_of_corners=percentage_of_corners,
                        HAND_EYE = HAND_EYE,
                        too_far_distance = too_far_distance

                        
                        )
        total_time = (time.time() - start_time)/60
        print(f'Time to record calibration images: {total_time:.2f} minutes')
        # save time to record calibration images in folder 
        with open(f'{calib_save_path}/time_to_record_calibration_images.txt', 'w') as f:
            f.write(f'Time to record calibration images: {total_time:.2f} minutes')

        """    
        calibrate_at_the_end_of_recording = False,
     
        grid_size_x=grid_size_x, 
                        grid_size_y=grid_size_y, 
                        min_positions=min_positions, 
                        min_distances=min_distances, 
                        min_angles=min_angles, 
                        max_distance_threshold=max_distance_threshold,
                        min_distance_threshold=min_distance_threshold, 
                        min_angle_threshold=min_angle_threshold, 
                        max_angle_threshold=max_angle_threshold,
                        calibrate_at_the_end_of_recording = calibrate_at_the_end_of_recording """
        
    
    # ENDO INTRINSIC CALIBRATION
    # GENERATE TABLE DATA
    df_endo = calib_frames_to_dataframe(df_endo, extension = '')
    """ image_pths = df_endo['paths'].values
    updated_image_pths, min_corners, imgPoints, objPoints, num_detected_corners, ids,  = detect_corners_charuco_cube_images(
            board, image_pths, return_corners=True,
            #min_num_corners=min_num_corners, percentage_of_corners=percentage_of_corners,
            waiting_time=waitTime, visualise_corner_detection=True) """
    # select frames for calibration
    frames_for_calibration_endo,remaining_frames  = bin_and_sample(df_endo, num_images_for_calibration=num_images_for_calibration, 
                                                              grid_size_x=grid_size_x, grid_size_y=grid_size_y, 
                                                              min_positions=min_positions, min_distances=min_distances, min_angles=min_angles, 
                                                              max_distance_threshold=max_distance_threshold,min_distance_threshold=min_distance_threshold, 
                                                              min_angle_threshold=min_angle_threshold, max_angle_threshold=max_angle_threshold)
    if len(reprojection_df_pth) == 0:
        df_endo_reproj = remaining_frames
    else:
        df_endo_reproj = pd.read_pickle(f'{reprojection_df_pth}/data_endo.pkl') 
    intrinsics_endo, distortion_endo, err, num_corners_detected_endo = perform_calibration(frames_for_calibration_endo, df_endo_reproj, 
                                                                                           calibration_estimates_pth_endo, visualise_reprojection_error = visualise_reprojection_error, waitTime = waitTime)
           
    
    print('ENDO Calibration successful')
    print('intrinsics_endo: ', intrinsics_endo)
    print('distortion_endo: ', distortion_endo)
    print('err: ', err)
    print('num_corners_detected_endo: ', num_corners_detected_endo)
    # create calibration folder
    if not os.path.exists(f'{calib_save_path}/calibration'):
        os.makedirs(f'{calib_save_path}/calibration')
    # save endo calibration as txt file
    np.savetxt(f'{calib_save_path}/calibration/intrinsics_endo.txt', intrinsics_endo)
    np.savetxt(f'{calib_save_path}/calibration/distortion_endo.txt', distortion_endo)
    np.savetxt(f'{calib_save_path}/calibration/err_endo.txt', [err])

    # INTRINSIC CALIBRATION FOR REALSENSE
    if len(realsense_save_path) > 0:
        # add column with x,y,z,rz,rx,ry,rz
        df_rs = calib_frames_to_dataframe(df_rs, extension = '')
        frames_for_calibration_rs, remaining_frames  = bin_and_sample(df_rs, num_images_for_calibration=num_images_for_calibration, 
                                                              grid_size_x=grid_size_x, grid_size_y=grid_size_y, 
                                                              min_positions=min_positions, min_distances=min_distances, min_angles=min_angles, 
                                                              max_distance_threshold=max_distance_threshold,min_distance_threshold=min_distance_threshold, 
                                                              min_angle_threshold=min_angle_threshold, max_angle_threshold=max_angle_threshold)
        if len(reprojection_df_pth) == 0:
            df_rs_reproj = remaining_frames
        else:
            df_rs_reproj = pd.read_pickle(f'{reprojection_df_pth}/data_rs.pkl') 
        intrinsics_rs, distortion_rs, err_rs, num_corners_detected_rs = perform_calibration(frames_for_calibration_rs, df_rs_reproj, 
                                                                                            calibration_estimates_pth_rs, visualise_reprojection_error = visualise_reprojection_error, waitTime = waitTime)
            

        print('RS Calibration successful')
        print('intrinsics_rs: ', intrinsics_rs)
        print('distortion_rs: ', distortion_rs)
        print('err_rs: ', err_rs)
        print('num_corners_detected_rs: ', num_corners_detected_rs)
        # save intrinsics and distortion as txt file 
        np.savetxt(f'{realsense_save_path}/calibration/intrinsics_rs.txt', intrinsics_rs)
        np.savetxt(f'{realsense_save_path}/calibration/distortion_rs.txt', distortion_rs)
        np.savetxt(f'{realsense_save_path}/calibration/err_rs.txt', [err_rs])

    if True:
        if HAND_EYE:
            min_angles = 10
            min_distances = 2
            min_positions = 5
            
            ###############
            # load pkl file if it exists
            merged_data_corrected_pth = f'{calib_save_path}/merged_corrected_data.pkl'
            if os.path.exists(merged_data_corrected_pth):
                df_combined = pd.read_pickle(merged_data_corrected_pth)
            else:
                df_combined = main_with_board(board, intrinsics_endo, distortion_endo, intrinsics_rs, distortion_rs, calib_save_path, realsense_save_path)

            df_combined = calib_frames_to_dataframe(df_combined, extension = '_endo')

            #df_combined = update_Ts(df_combined,board, intrinsics_endo, distortion_endo, intrinsics_rs, distortion_rs)

            frames_for_he_calibration, remaining_frames  = bin_and_sample(df_combined, num_images_for_calibration=num_images_for_he_calibration, 
                                                                grid_size_x=grid_size_x, grid_size_y=grid_size_y, 
                                                                min_positions=min_positions, min_distances=min_distances, min_angles=min_angles, 
                                                                max_distance_threshold=max_distance_threshold,min_distance_threshold=min_distance_threshold, 
                                                                min_angle_threshold=min_angle_threshold, max_angle_threshold=max_angle_threshold)
            
            # update all values of T_endo and T_rs using new intrinsics and dist
            data_for_optimisation = remaining_frames

            if len(reprojection_df_pth) == 0:
                df_combined_reproj = remaining_frames
            else:
                merged_data_corrected_pth_reprojection = f'{reprojection_df_pth}/merged_corrected_data.pkl'
                if os.path.exists(merged_data_corrected_pth_reprojection):
                    df_combined_reproj = pd.read_pickle(merged_data_corrected_pth_reprojection)
                else:
                    df_combined_reproj = main_with_board(board, intrinsics_endo, distortion_endo, intrinsics_rs, distortion_rs, reprojection_df_pth, reprojection_df_pth)
                """ df_combined_reproj = pd.read_pickle(f'{reprojection_df_pth}/data_combined.pkl') 
                df_combined_reproj = update_Ts(df_combined_reproj,board, intrinsics_endo, distortion_endo, intrinsics_rs, distortion_rs) """
            
            #reprojection_data_df = frames_for_he_calibration # pd.read_pickle(f'{reprojection_df_pth}/data_combined.pkl') 
            hand_eye, err_he = perform_hand_eye_calibration(frames_for_he_calibration, data_for_optimisation,df_combined_reproj, intrinsics_endo, distortion_endo, 
                                                            optimise=True, error_threshold=1, 
                                                            num_samples_for_optimisation=100, waitTime=waitTime, 
                                                            visualise_reprojection_error=False)
            
            print('Hand eye calibration successful')
            print('hand_eye: ', hand_eye)
            print('err_he: ', err_he)
            # save hand eye calibration as txt file in calibration folder
            np.savetxt(f'{calib_save_path}/calibration/hand_eye.txt', hand_eye)
            np.savetxt(f'{calib_save_path}/calibration/err_he.txt', [err_he])

            ##### OWN DF
            hand_eye, err_he = perform_hand_eye_calibration(frames_for_he_calibration, data_for_optimisation,remaining_frames, intrinsics_endo, distortion_endo, 
                                                            optimise=True, error_threshold=1, 
                                                            num_samples_for_optimisation=100, waitTime=waitTime, 
                                                            visualise_reprojection_error=True)
            
            print('Hand eye calibration successful')
            print('hand_eye: ', hand_eye)
            print('err_he: ', err_he)
            # save hand eye calibration as txt file in calibration folder
            #np.savetxt(f'{calib_save_path}/calibration/hand_eye.txt', hand_eye)
            #np.savetxt(f'{calib_save_path}/calibration/err_he.txt', [err_he])


    

def generate_data_from_images(image_pths, board, table_data_pth, 
                         waiting_time=1,
                         intrinsics=None, distortion=None, visualise_corner_detection=False):

    if intrinsics is not None and distortion is not None:
        # this will also return board pose information (for hand-eye calibration)
        updated_image_pths, min_corners, T, imgPoints, objPoints, num_detected_corners, ids = detect_charuco_board_pose_images(
            board, image_pths, intrinsics, distortion, return_corners=True,
             # min_num_corners=1, # min_num_corners set to 1 for raw images percentage_of_corners=percentage_of_corners, 
            waiting_time=waiting_time, visualise_corner_detection=visualise_corner_detection)
    else:
        # for intrinsics calibration
        updated_image_pths, min_corners,   imgPoints, objPoints, num_detected_corners, ids,  = detect_corners_charuco_cube_images(
            board, image_pths, return_corners=True,
            #min_num_corners=min_num_corners, percentage_of_corners=percentage_of_corners,
            waiting_time=waiting_time, visualise_corner_detection=visualise_corner_detection)

    # convert updated image paths- split path to get: [image number, pose number, degree number, going forward or backward]
    # convert to pandas dataframe
    data = {'paths': updated_image_pths, 'imgPoints': imgPoints, 'objPoints': objPoints,
            'num_detected_corners': num_detected_corners, 'ids':ids}
    data_df = pd.DataFrame(data=data)

    # also adding frame number
    data_df['frame_number'] = data_df['paths'].str.extract('(\d+).png')
    
    # if intrinsics path, we want to also add the board pose
    if intrinsics is not None and distortion is not None:
        data_df['T'] = T

    # add x,y,z,rxryrz
    data_df = calib_frames_to_dataframe(data_df, extension = '', T=True)

    # convert to integers
    """ data_df[["pose", "deg", "distance"]] = data_df[["pose", "chess_size", "deg", "direction"]].apply(
        pd.to_numeric) """
    

    
    data_df.to_pickle(table_data_pth)

    return data_df


def main_with_board(board, 
                    intrinsics_endo, distortion_endo, intrinsics_rs, distortion_rs,
                    calib_save_path='results/user_study/mac/aure/rs',
                    realsense_save_path='results/user_study/mac/aure/rs', 
                    
                    
                    ):

    # INTRINSICS CALIBRATION
    # endo
    endo_df = pd.read_pickle(f'{calib_save_path}/data_endo.pkl')
    image_pths_endo = endo_df['paths'].values.tolist()
    table_data_pth_endo = f'{calib_save_path}/endo_corrected_data.pkl'
    data_df_endo = generate_data_from_images(image_pths_endo, board, table_data_pth_endo, 
                         waiting_time=1,
                         intrinsics=intrinsics_endo, distortion=distortion_endo, visualise_corner_detection=False)
    
    # realsense
    realsense_df = pd.read_pickle(f'{realsense_save_path}/data_rs.pkl')
    image_pths_realsense = realsense_df['paths'].values.tolist()
    table_data_pth_rs = f'{realsense_save_path}/rs_corrected_data.pkl'

    data_df_rs = generate_data_from_images(image_pths_realsense, board, table_data_pth_rs,
                                           waiting_time=1,
                                           intrinsics=intrinsics_rs, distortion=distortion_rs, visualise_corner_detection=False)
    
    # merge dataframes
    df_merged = filter_and_merge_hand_eye_df(data_df_endo, data_df_rs, min_num_corners=6)
    table_data_pth_merged = f'{calib_save_path}/merged_corrected_data.pkl'
    df_merged.to_pickle(table_data_pth_merged)

    return df_merged
    




def filter_and_merge_hand_eye_df(data_df_endo, data_df_realsense, min_num_corners):
    # combine information of paths for filtering those that don't match between endo and rs
    # TODO remove warning
    data_df_endo['combined_info'] = data_df_endo[['frame_number']].astype(
        str).agg('_'.join, axis=1)
    data_df_realsense['combined_info'] = data_df_realsense[
        ['frame_number']].astype(str).agg('_'.join, axis=1)

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
    common_columns = ['frame_number']
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
        imgPoints_matched_rs, objPoints_matched_endo, ids_endo_2, ids_rs_2 = sort_and_filter_matched_corners(pnts_rs, pnts_3d_endo,
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


    



if __name__=='__main__': 

    parser = argparse.ArgumentParser(
        description='user study calibration') 
    
    # adding all necessary args for cl app
    """ parser.add_argument('--save_path', type=str, default='results/user_study/mac/aure/rs2', 
                        help='path to where images uesd for calibration are stored') """
    parser.add_argument('--aruco_w', type=int, default=13,
                        help='')
    parser.add_argument('--aruco_h', type=int, default=9,
                    help='')
    parser.add_argument('--size_of_checkerboard', type=int, default=20,
                    help='')
    parser.add_argument('--aruco_size', type=int, default=15,
                    help='')
    # grabbing args selected
    args = parser.parse_args()


    main(aruco_w=int(args.aruco_w),
         aruco_h=int(args.aruco_h),
         size_of_checkerboard=int(args.size_of_checkerboard),
         aruco_size=int(args.aruco_size),
         #calib_save_path=args.save_path,
         calib_save_path='results/user_study/mac/matt/4',
         realsense_save_path='results/user_study/mac/matt/4', 
         endo_port = 1, 
        rs_port = 0,
        calibration_estimates_pth_endo = 'calibration_estimates/intrinsics_endo.txt', 
        calibration_estimates_pth_rs = 'calibration_estimates/intrinsics_realsense.txt', 
        max_frames_recorded_for_calibration=300, #np.inf, 
        too_far_distance = 300,
        save_calibration_images = True,

        num_images_for_calibration=10,
        grid_size_x=3, 
        grid_size_y=3, 
        min_positions=9, 
        min_distances=1, 
        min_angles=10, 
        max_distance_threshold=1500,
        min_distance_threshold=10, 
        min_angle_threshold=-40, 
        max_angle_threshold=40, 
        visualise_reprojection_error = False,
        waitTime=0,
        reprojection_df_pth = 'results/user_study/mac/aure/reprojection_dataset_endo_distance', #'results/user_study/mac/aure/reprojection_dataset_endo_distance3' ,#'', # pth where the reprojection df is saved,
        percentage_of_corners = 0.5,
        HAND_EYE = True,
        num_images_for_he_calibration=30
         )
     
    

                      