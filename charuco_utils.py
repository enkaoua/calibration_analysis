import multiprocessing
import random
import cv2
import numpy as np
import os
import glob
from sksurgerycore.algorithms.procrustes import orthogonal_procrustes
import scipy
from tqdm import tqdm
from utils import calculate_transform_average, extrinsic_matrix_to_vecs, extrinsic_vecs_to_matrix, find_best_intrinsics, get_average_std, \
    reprojection_error, sample_dataset, select_min_num_corners, sort_and_filter_matched_corners
import pandas as pd
import concurrent.futures


######################################################
######## HAND-EYE ##################################
######################################################


""" def add_T_column_to_table(data_pkl_pth):
    
    # load data
    data = pd.read_pickle(data_pkl_pth)

    # generate board
    chess_size = int(data_pkl_pth.split('/')[-1].split('_')[0])
    board = generate_charuco_board(chess_size)
    
    corners = data['imgPoints']
    ids = 
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_detected_corners, charuco_detected_ids, board,
                                                                intrinsics, distortion, None, None)
    if retval:
        # add the pose to the list of all poses
        tag2cam = extrinsic_vecs_to_matrix(rvec, tvec)
        T_all.append(tag2cam) """


def detect_charuco_board_pose_images(board, image_pths, intrinsics, distortion, return_corners=True, waiting_time=0, visualise_corner_detection=False
                                     #  min_num_corners=6,
                                     #percentage_of_corners=0.2, 
                                     ):
    """
    function to detect corners in a list of images given a board
    Parameters
    ----------
    board : cv2.aruco.CharucoBoard
        board to use for detection
    image_pths : list of strings
        list of paths to images to detect corners in 
    intrinsics: 3x3 np with 
        camera intrinsics
    distortion: 1x5 np array 
        cam dist params

    """

    if len(image_pths) == 0:
        raise (f'no images found')

    imgPoints = []
    objPoints = []
    num_detected_corners = []
    T_all = []
    ids_all = []
    updated_image_pths = image_pths.copy()
    parameters = cv2.aruco.DetectorParameters()
    dictionary = board.getDictionary()

    # chessboard corners in 3D and corresponding ids
    charuco_corners_3D = board.getChessboardCorners()  # allCorners3D_np
    num_chess_corners = len(charuco_corners_3D)  # number_of_corners_per_face
    charuco_ids_3D = np.arange(0, num_chess_corners)  # all3DIDs_np

    # select minimum number of corners to be detected 
    #min_num_corners = select_min_num_corners(min_num_corners, percentage_of_corners, num_chess_corners)
    min_num_corners = 6 # has to be 6 minimum as DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences.

    # detect corners in images
    for image_pth in image_pths:
        image = cv2.imread(image_pth)
        # if image is None (currupt): remove it from the list of images
        if image is None:
            updated_image_pths.remove(image_pth)
            continue
        # undistort image
        undistorted_image = cv2.undistort(image, intrinsics, distortion, None, intrinsics)
        # convert img to gray
        gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

        # detect aruco tags
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(undistorted_image, dictionary, parameters=parameters)
        if corners:
            # get markers corresponding to said board
            cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)
            # interpolate to get corners of charuco board
            ret, charuco_detected_corners, charuco_detected_ids = cv2.aruco.interpolateCornersCharuco(corners, ids,
                                                                                                      gray, board)
        else:
            ret = 0

        # if there are less than 6 points, remove img from paths and skip this image so we don't save it
        if ret < min_num_corners:
            updated_image_pths.remove(image_pth)
            cv2.imshow('charuco board', image)
            # print(f'skipping image {image_pth} because it has less than 6 corners')
            continue

        # by this point, we have filtered out any images that don't have enough charuco board corners detected so we can move onto detecting the board pose
        # draw the detected corners on the image
        # Estimate the pose of the ChArUco board
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_detected_corners, charuco_detected_ids, board,
                                                                intrinsics, distortion, None, None)

        if retval:
            if visualise_corner_detection:
                # draw markers
                annotated_image = cv2.aruco.drawDetectedCornersCharuco(undistorted_image, charuco_detected_corners,
                                                                       charuco_detected_ids, cornerColor= (0,255,0))
                # Draw the ChArUco board axis
                annotated_image = cv2.drawFrameAxes(annotated_image, intrinsics, None, rvec, tvec, length=37)
                # display annotated img
                cv2.imshow('charuco board', annotated_image)
                cv2.waitKey(waiting_time)

            # add the detected charuco corners to the list of all charuco corners
            imgPoints.append(charuco_detected_corners)
            # find the corresponding 3D pnts
            _, allCorners3D_np_sorted_filtered, _, allIDs3D_np_sorted_filtered = sort_and_filter_matched_corners(
                charuco_detected_corners.squeeze(), charuco_corners_3D, charuco_detected_ids, charuco_ids_3D,
                return_ids=True)
            objPoints.append(allCorners3D_np_sorted_filtered.reshape(-1, 1, 3))
            num_detected_corners.append(len(charuco_detected_corners))
            # add the pose to the list of all poses
            tag2cam = extrinsic_vecs_to_matrix(rvec, tvec)
            T_all.append(tag2cam)
            ids_all.append(allIDs3D_np_sorted_filtered)

        else:
            # if no pose was detected, remove the image from paths and go to next frame
            updated_image_pths.remove(image_pth)
            if visualise_corner_detection:
                cv2.imshow('charuco board', image)
            # print(f'skipping image {image_pth} because pose was not detected properly')
            continue

    if return_corners:
        return updated_image_pths, min_num_corners, T_all, imgPoints, objPoints, num_detected_corners, ids_all  # rvec, tvec,
    return updated_image_pths, min_num_corners, T_all, ids_all  # image.shape[0:-1]


def calibrate_hand_eye(T_endo_lst, T_realsense_lst):
    hand_eye_r_lst = []
    hand_eye_t_lst = []
    hand_eye_lst = []
    i=0
    for tag2endo, tag2realsense in zip(T_endo_lst, T_realsense_lst):
        # calculate hand-eye        
        hand_eye = tag2endo @ np.linalg.inv(tag2realsense)

        print(f'------------ hand_eye individual {i}')
        #print(f'tag2endo: \n {np.round(tag2endo)}')
        #print(f'tag2realsense: \n {np.round(tag2realsense)}')
        # print hand eye rounded to 1 decimal place
        print(f'hand_eye rounded: \n {np.round(hand_eye)}')
        print(f'-------------------------')
        i+=1

        r, t = extrinsic_matrix_to_vecs(hand_eye)
        hand_eye_r_lst.append(r)
        hand_eye_t_lst.append(t)
        hand_eye_lst.append(hand_eye)

    mean_he, mask = calculate_transform_average(hand_eye_r_lst, hand_eye_t_lst)
    # filter out outliers
    T_endo_lst_filtered = T_endo_lst[mask]
    T_realsense_lst_filtered = T_realsense_lst[mask]
    return mean_he, T_endo_lst_filtered, T_realsense_lst_filtered


def calculate_hand_eye_reprojection_error(hand_eye, world2realsense, objPoints, imgPoints, intrinsics_endo,
                                          distortion_endo, waitTime=1,
                                          endo_reprojection_images_pth=[], return_residuals=False, return_projected_pnts=False, 
                                          rs_reprojection_images_pth=[], intrinsics_rs=None, distortion_rs=None, imgPoints_rs=None,
                                          IDs_endo=None, IDs_rs=None):  # rs_reprojection_images_pth, board,detect_aruco=False,

    mean_errors_np = []
    residuals = []

    for i in range(len(objPoints)):

        if len(objPoints[i]) < 4 or len(imgPoints[i]) < 4:
            continue

        board_points_world = objPoints[i].reshape(-1, 3)
        # filter points by whatever was detected from endo side
        board_points_world_hom = cv2.convertPointsToHomogeneous(board_points_world).squeeze()
        # convert to realsense coord system
        points3D_realsense = (world2realsense[i] @ board_points_world_hom.T).T
        # hand-eye to convert to endo frame
        points3D_endo = (hand_eye @ points3D_realsense.T).T
        # project 3D points to 2D endo image plane
        proj_points_2d_endo, _ = cv2.projectPoints(cv2.convertPointsFromHomogeneous(points3D_endo), np.zeros((1, 3)),
                                                   np.zeros((1, 3)), intrinsics_endo, distortion_endo)
        proj_points_2d_endo = proj_points_2d_endo.squeeze().astype(np.float32)

        # compare those 2D points to the detected ones considered as ground truth
        img_points_endo_detected = np.array(imgPoints[i]).reshape(-1, 2).astype(np.float32)
        if IDs_endo is None:
                ID_endo = None
        else:
            ID_endo = IDs_endo[i]
        if len(endo_reprojection_images_pth) > 0:
            endo_im = cv2.imread(endo_reprojection_images_pth[i])
            undistorted_img_endo = cv2.undistort(endo_im, intrinsics_endo, distortion_endo)
            # calculate error
            
            error_np, annotated_image_endo_board = reprojection_error(img_points_endo_detected, proj_points_2d_endo,
                                                                      undistorted_img_endo, IDs=ID_endo)
            cv2.putText(annotated_image_endo_board, f'mean error: {error_np:.2f}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            
            
            # print(error)

            
            if len(rs_reprojection_images_pth)>0:
                # TODO add IDS of tags
                proj_points_2d_rs, _ = cv2.projectPoints(cv2.convertPointsFromHomogeneous(points3D_realsense), np.zeros((1, 3)),
                                                   np.zeros((1, 3)), intrinsics_rs, distortion_rs)
                proj_points_2d_rs = proj_points_2d_rs.squeeze().astype(np.float32)
                rs_im = cv2.imread(rs_reprojection_images_pth[i])
                undistorted_img_rs = cv2.undistort(rs_im, intrinsics_rs, distortion_rs)
                
                img_points_rs_detected = np.array(imgPoints_rs[i]).reshape(-1, 2).astype(np.float32)
                if IDs_rs is None:
                    ID_rs = None
                else:
                    ID_rs = IDs_rs[i]
                error_np_rs, annotated_image_endo_board_rs = reprojection_error(img_points_rs_detected, proj_points_2d_rs,
                                                                      undistorted_img_rs,IDs=ID_rs)
                
                # add pose axis to image
                rvec, tvec = extrinsic_matrix_to_vecs(world2realsense[i])
                annotated_image_endo_board_rs = cv2.drawFrameAxes(annotated_image_endo_board_rs, intrinsics_rs, None, rvec, tvec, length=37)

                # resize image of endoscope and rs to match
                rs_im = cv2.resize(annotated_image_endo_board_rs, (endo_im.shape[1], endo_im.shape[0]))
                # combined both images
                combined_img = np.hstack((rs_im, undistorted_img_endo))
                cv2.imshow('charuco board', combined_img)
            else:
                cv2.imshow('charuco board', annotated_image_endo_board)

            cv2.waitKey(waitTime)
            # add error as text to image
                        
            #cv2.waitKey(waitTime)
        else:
            error_np = reprojection_error(img_points_endo_detected, proj_points_2d_endo,IDs=ID_endo)

        mean_errors_np.append(error_np)
        if return_residuals:
            diff_points  = abs((img_points_endo_detected - proj_points_2d_endo)).reshape((-1))
            residuals.append(diff_points)
            #total_residual_err += np.sum(diff_points.reshape((-1))) / len(diff_points.reshape((-1)))

    if return_residuals:
        return np.hstack(residuals) # total_residual_err #
    if return_projected_pnts:
        return mean_errors_np, points3D_endo ,points3D_realsense #img_points_endo_detected, proj_points_2d_endo
    return mean_errors_np


def calibrate_hand_eye_pnp(calibration_data, intrinsics_endo=None, intrinsics_rs=None, distortion_endo=None, distortion_rs=None):
    all_imgPoints_endo = calibration_data['imgPoints_endo'].values[1]
    all_imgPoints_rs = calibration_data['imgPoints_rs'].values[1]
    all_objPoints_endo  = calibration_data['objPoints_endo'].values[1]
    all_objPoints_rs  = calibration_data['objPoints_rs'].values[1]

    """ all_objPoints_endo = []
    all_objPoints_rs = []
    all_imgPoints_endo = []
    all_imgPoints_rs = []

    # concatenate all repeats
    for i in range(len(imgPoints_endo)):
        all_objPoints_endo.append(objPoints_endo[i].reshape(-1, 3))  # Flatten (n, 1, 3) to (n, 3)
        all_objPoints_rs.append(obj_points_rs[i].reshape(-1, 3))      # Flatten (n, 1, 3) to (n, 3)
        all_imgPoints_endo.append(imgPoints_endo[i].reshape(-1, 2))  # Flatten (n, 1, 2) to (n, 2)
        all_imgPoints_rs.append(img_points_rs[i].reshape(-1, 2))  # Flatten (n, 1, 2) to (n, 2)
    
    # Concatenate all points into large arrays
    all_objPoints_endo = np.vstack(all_objPoints_endo)  # Shape (total_points, 3)
    all_objPoints_rs = np.vstack(all_objPoints_rs) 
    all_imgPoints_endo = np.vstack(all_imgPoints_endo)  # Shape (total_points, 2)
    all_imgPoints_rs = np.vstack(all_imgPoints_rs)  # Shape (total_points, 2)
    """
        
    # Estimate camera poses using solvePnP
    _, rvec_endo, tvec_endo = cv2.solvePnP(all_objPoints_endo, all_imgPoints_endo, intrinsics_endo, distortion_endo)
    _, rvec_rs, tvec_rs = cv2.solvePnP(all_objPoints_rs, all_imgPoints_rs, intrinsics_rs, distortion_rs)

    tag2_rs = extrinsic_vecs_to_matrix(rvec_rs, tvec_rs)
    tag2_endo = extrinsic_vecs_to_matrix(rvec_endo, tvec_endo)
    hand_eye = tag2_endo @ np.linalg.inv(tag2_rs)

    """ # Compute the essential matrix
    E, mask = cv2.findEssentialMat(all_imgPoints_rs, all_imgPoints_endo, method=cv2.RANSAC, prob=0.999, threshold=0.1)
    # Decompose the essential matrix to obtain rotation and translation
    _, R, T, mask = cv2.recoverPose(E,all_imgPoints_rs, all_imgPoints_endo, mask=mask)
    
    # Convert rotation matrix to a 4x4 homogeneous matrix
    hand_eye = np.eye(4)
    hand_eye[:3, :3] = R  # R should be a 3x3 matrix
    hand_eye[:3, 3] = T.flatten()  # T should be a 3x1 vector """
    return hand_eye
    

def calibrate_he_opencv(calibration_data):
    # 
    T_endo = calibration_data['T_endo'].values
    T_rs = calibration_data['T_rs'].values

    rvecs_rs_lst = []
    tvecs_rs_lst = []
    rvecs_endo_lst = []
    tvecs_endo_lst = []
    for i in range(len(T_endo)):
        rvec_endo, tvec_endo = extrinsic_matrix_to_vecs(np.linalg.inv(T_endo[i]))
        rvec_rs, tvec_rs = extrinsic_matrix_to_vecs(T_rs[i])
        rvecs_endo_lst.append(rvec_endo)
        tvecs_endo_lst.append(tvec_endo)
        rvecs_rs_lst.append(rvec_rs)
        tvecs_rs_lst.append(tvec_rs)

    # convert to r and t
    #rvec_rs, tvec_rs = extrinsic_matrix_to_vecs(T_rs)
    #rvec_endo, tvec_endo = extrinsic_matrix_to_vecs(T_endo)
    
    rvec, tvec = cv2.calibrateHandEye(rvecs_endo_lst,        #R_gripper2base, endo to target
                                      tvecs_endo_lst,	        #t_gripper2base,
                                       rvecs_rs_lst,	        #R_target2cam, target to realsense
                                       tvecs_rs_lst,	        #t_target2cam,
                                     	#R_cam2gripper,
                                     	#t_cam2gripper,
                                     	#method = CALIB_HAND_EYE_TSAI 
                                    )	
    
    # convert to matrix
    #hand_eye = extrinsic_vecs_to_matrix(rvec, tvec)
    hand_eye = np.eye(4)
    hand_eye[:3, :3] = cv2.Rodrigues(rvec)[0]
    hand_eye[:3, 3] = tvec.flatten()
    return hand_eye


def hand_eye_objective(params, world2realsense, objPoints, imgPoints, intrinsics_endo, distortion_endo):
    
    # Decompose params into rotation vector (rvec) and translation vector (tvec)
    rvec = params[:3]
    tvec = params[3:6]

    #rvec = np.zeros((1,3))
    
    # Convert rvec to rotation matrix
    R = cv2.Rodrigues(rvec)[0]
    
    # Construct the 4x4 hand-eye matrix
    hand_eye = np.eye(4)
    hand_eye[:3, :3] = R
    hand_eye[:3, 3] = tvec
    

    # Calculate reprojection errors using the hand-eye matrix
    reprojection_errors = calculate_hand_eye_reprojection_error(
        hand_eye, world2realsense, objPoints, imgPoints, intrinsics_endo, distortion_endo, return_residuals=True
    )
    
    # Return the reprojection errors as a flattened array
    return reprojection_errors


def calibrate_hand_eye_pnp_reprojection(calibration_data,data_for_optimisation, intrinsics_endo=None, distortion_endo=None, optimise = True, error_threshold = 1, num_samples_for_optimisation=100, groupby_cats=['pose', 'deg']):

    # PERFORM HAND-EYE
    T_endo_lst = calibration_data['T_endo'].values
    T_realsense_lst = calibration_data['T_rs'].values
    hand_eye, T_endo_lst, T_realsense_lst = calibrate_hand_eye(T_endo_lst, T_realsense_lst)
    #hand_eye = registration_hand_eye(calibration_data)

    
    
    # OPTIMISE HAND-EYE
    data_used_for_optimisation , _ = sample_dataset(data_for_optimisation, total_samples=num_samples_for_optimisation, groupby_cats=groupby_cats)
    world2realsense = data_used_for_optimisation['T_rs'].values
    objPoints = data_used_for_optimisation['objPoints_rs'].values
    imgPoints = data_used_for_optimisation['imgPoints_endo'].values
    #endo_reprojection_images_pth = calibration_data['paths_endo'].values
    
    # convert all arrays in world2realsense to float 32
    world2realsense_d32 = []
    for i in range(len(world2realsense)):
        world2realsense_d32.append(world2realsense[i].astype(np.float32))

    # Calculate the initial mean reprojection error
    initial_reprojection_errors = calculate_hand_eye_reprojection_error(
        hand_eye, world2realsense, objPoints, imgPoints, intrinsics_endo, distortion_endo
    )

    # Calculate the initial mean reprojection error
    initial_mean_reprojection_error = np.mean(initial_reprojection_errors)
    # Calculate the median reprojection error
    #initial_median_reprojection_error = np.median(initial_reprojection_errors)
    #print(f'Initial mean reprojection error: {initial_mean_reprojection_error}')

    if optimise:
       
        # Extract rotation (rvec) and translation (tvec) from the initial hand-eye matrix
        rvec, tvec = extrinsic_matrix_to_vecs(hand_eye)
        initial_params = np.vstack((rvec, tvec)).flatten()  # Combine rvec and tvec into a single parameter vector

        # Custom optimization loop
        current_error = initial_mean_reprojection_error
        iteration = 0
        diff = 100

        while current_error > error_threshold and diff > 1e-1:

            # ensure all inputs are .astype(dtype=np.float32)
            intrinsics_endo = intrinsics_endo.astype(dtype=np.float32)
            distortion_endo = distortion_endo.astype(dtype=np.float32)
            
            # Optimize the hand-eye transformation      
            result = scipy.optimize.least_squares(
                fun=hand_eye_objective,
                x0=initial_params,
                args=(world2realsense_d32, objPoints, imgPoints, intrinsics_endo, distortion_endo),
                method='lm',  # Levenberg-Marquardt optimization
                x_scale=0.1,
                diff_step=1e-6,
                ftol = 1e-9,
                max_nfev = 1000,
            )
            
            # get the optimised parameters
            optimised_params = result.x
            optimised_rvec = optimised_params[:3]
            optimised_tvec = optimised_params[3:6]
            # Convert back to 4x4 matrix
            optimised_hand_eye = extrinsic_vecs_to_matrix(optimised_rvec, optimised_tvec)
            
            #  Update the initial parameters for the next iteration
            initial_params = optimised_params

            # Calculate the current reprojection error
            reprojection_errors = calculate_hand_eye_reprojection_error(
                optimised_hand_eye, world2realsense, objPoints, imgPoints, intrinsics_endo, distortion_endo
            )

            iteration += 1
            diff = abs(current_error - np.mean(reprojection_errors))

            # Calculate the mean reprojection error
            current_error = np.mean(reprojection_errors)
            
            # Optional: add a stopping criterion based on the number of iterations
            if iteration > 1000:  # Example: stop after 1000 iterations to avoid infinite loop
                print("Reached maximum iterations without achieving desired error.")
                break

        
        hand_eye = optimised_hand_eye
    """ # Calculate the final reprojection error
    final_reprojection_errors = calculate_hand_eye_reprojection_error(
        hand_eye, world2realsense, objPoints, imgPoints, intrinsics_endo, distortion_endo
    )

    # Calculate the mean reprojection error
    final_mean_reprojection_error = np.mean(final_reprojection_errors)
    # Calculate the median reprojection error
    final_median_reprojection_error = np.median(final_reprojection_errors) """
    
    # check if the mean reprojection error is less than the median reprojection error
    """ print(f'Initial mean reprojection error: {initial_mean_reprojection_error}')
    print(f'Final mean reprojection error: {final_mean_reprojection_error}')

    print(f'Initial median reprojection error: {initial_median_reprojection_error}')
    print(f'Median reprojection error: {final_median_reprojection_error}') """
    
    if optimise:
        return optimised_hand_eye
    return hand_eye



def registration_hand_eye(calibration_data):

    world2realsense = calibration_data['T_rs'].values
    world2endo = calibration_data['T_endo'].values
    objPoints = calibration_data['objPoints_rs'].values
    imgPoints = calibration_data['imgPoints_endo'].values

    rs_pnts = []
    endo_pnts = []
    for i in range(len(objPoints)):

        if len(objPoints[i]) < 4 or len(imgPoints[i]) < 4:
            continue

        board_points_world = objPoints[i].reshape(-1, 3)
        # filter points by whatever was detected from endo side
        board_points_world_hom = cv2.convertPointsToHomogeneous(board_points_world).squeeze()
        # convert to realsense coord system
        points3D_realsense = (world2realsense[i] @ board_points_world_hom.T).T
        points_3D_endo = (world2endo[i] @ board_points_world_hom.T).T

        # convert back from homogeneous to 3D
        points3D_realsense = cv2.convertPointsFromHomogeneous(points3D_realsense).squeeze()
        points_3D_endo = cv2.convertPointsFromHomogeneous(points_3D_endo).squeeze()

        rs_pnts.append(points3D_realsense)
        endo_pnts.append(points_3D_endo)

    rs_pnts = np.vstack(rs_pnts)
    endo_pnts = np.vstack(endo_pnts)
    #optimised_hand_eye, sca = scipy.linalg.orthogonal_procrustes(rs_pnts, endo_pnts)                               
    R, t, e = orthogonal_procrustes(endo_pnts, rs_pnts)

    optimised_hand_eye = np.eye(4)
    optimised_hand_eye[:3, :3] = R
    optimised_hand_eye[:3, 3] = t.T

    return optimised_hand_eye

def calibrate_hand_eye_registration(calibration_data, intrinsics_endo=None, distortion_endo=None, optimise=False):
    T_endo_lst = calibration_data['T_endo']
    T_realsense_lst = calibration_data['T_rs']

    hand_eye = calibrate_hand_eye(T_endo_lst, T_realsense_lst)
    
    world2realsense = calibration_data['T_rs'].values
    world2endo = calibration_data['T_endo'].values
    objPoints = calibration_data['objPoints_rs'].values
    imgPoints = calibration_data['imgPoints_endo'].values
    #endo_reprojection_images_pth = calibration_data['paths_endo'].values
    
    # convert all arrays in world2realsense to float 32
    #world2realsense = world2realsense.astype(np.float32)
    """ world2realsense_d32 = []
    for i in range(len(world2realsense)):
        world2realsense_d32.append(world2realsense[i].astype(np.float32))
    """
    # Calculate the initial mean reprojection error
    initial_reprojection_errors, endo_detected_pnts, rs_projected_pnts = calculate_hand_eye_reprojection_error(
        hand_eye, world2realsense, objPoints, imgPoints, intrinsics_endo, distortion_endo, return_projected_pnts=True
    )

    # Calculate the initial mean reprojection error
    initial_mean_reprojection_error = np.mean(initial_reprojection_errors)
    # Calculate the median reprojection error
    initial_median_reprojection_error = np.median(initial_reprojection_errors)

    if optimise:
        # find transform matrix to register projected points from realsense to endo detected
        #T, sca = scipy.linalg.orthogonal_procrustes(endo_detected_pnts, rs_projected_pnts)

        rs_pnts = []
        endo_pnts = []
        for i in range(len(objPoints)):

            if len(objPoints[i]) < 4 or len(imgPoints[i]) < 4:
                continue

            board_points_world = objPoints[i].reshape(-1, 3)
            # filter points by whatever was detected from endo side
            board_points_world_hom = cv2.convertPointsToHomogeneous(board_points_world).squeeze()
            # convert to realsense coord system
            points3D_realsense = (world2realsense[i] @ board_points_world_hom.T).T
            points_3D_endo = (world2endo[i] @ board_points_world_hom.T).T

            # convert back from homogeneous to 3D
            points3D_realsense = cv2.convertPointsFromHomogeneous(points3D_realsense).squeeze()
            points_3D_endo = cv2.convertPointsFromHomogeneous(points_3D_endo).squeeze()

            rs_pnts.append(points3D_realsense)
            endo_pnts.append(points_3D_endo)

        rs_pnts = np.vstack(rs_pnts)
        endo_pnts = np.vstack(endo_pnts)
        #optimised_hand_eye, sca = scipy.linalg.orthogonal_procrustes(rs_pnts, endo_pnts)                               
        R, t, e = orthogonal_procrustes(endo_pnts, rs_pnts)

        optimised_hand_eye = np.eye(4)
        optimised_hand_eye[:3, :3] = R
        optimised_hand_eye[:3, 3] = t.T
        
        #optimised_hand_eye = T@hand_eye

        # calculate reprojection error of optimised hand eye
        final_reprojection_errors = calculate_hand_eye_reprojection_error(
            optimised_hand_eye, world2realsense, objPoints, imgPoints, intrinsics_endo, distortion_endo
        )

        # Calculate the initial mean reprojection error
        final_mean_reprojection_error = np.mean(final_reprojection_errors)
        # Calculate the median reprojection error
        final_median_reprojection_error = np.median(final_reprojection_errors)


    # check if the mean reprojection error is less than the median reprojection error
    #print(f'Initial mean reprojection error: {initial_mean_reprojection_error}')
    #print(f'Final mean reprojection error: {final_mean_reprojection_error}')

    print(f'Initial median reprojection error: {initial_median_reprojection_error}')
    print(f'Median reprojection error: {final_median_reprojection_error}')
    

    return hand_eye

def he_analysis(data_df, reprojection_data_df, intrinsics_pth, size_chess, waitTime=1, n=10, repeats=1000,
                visualise_reprojection_error=False, optimise=True, return_calibration_data=True):
    
    intrinsics_endo = np.loadtxt(f'{intrinsics_pth}/{size_chess}_endo_intrinsics.txt')
    distortion_endo = np.loadtxt(f'{intrinsics_pth}/{size_chess}_endo_distortion.txt')
    
    hand_eye_lst = []
    errors = []
    all_calibration_data = []
    for i in tqdm(range(repeats), desc='hand-eye calibration', leave=False):
        
        # SELECT DATASET/IMAGES FOR CALIBRATION AND REPROJECTION
        # sample n images from the dataset (number of images we are testing)
        calibration_data, _ = sample_dataset(data_df, total_samples=n, groupby_cats=['pose', 'deg'])
        # if reprojection_data_df is None, use calibration data as reprojection dataset
        if reprojection_data_df is None:
            reprojection_data_df = calibration_data
        
        # CALIBRATE
        # hand-eye calibration
        hand_eye = calibrate_hand_eye_pnp_reprojection(calibration_data,reprojection_data_df, intrinsics_endo=intrinsics_endo, distortion_endo=distortion_endo, optimise=optimise, error_threshold=1)

        # EVALUATE REPROJECTION ERROR
        # reprojection error
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
        
        # calculating mean reprojection error between all images.
        reprojection_error_mean_final = pd.DataFrame(err_np).median()[0]
        """ mean_err = pd.DataFrame(err_np).mean()[0]
        if mean_err - median_err > 0.5:
            reprojection_error_mean_final = median_err
        else:
            reprojection_error_mean_final = mean_err """
        
        errors.append(reprojection_error_mean_final)
        hand_eye_lst.append(hand_eye)
        all_calibration_data.append(calibration_data)

    if return_calibration_data:
        return errors, hand_eye_lst, all_calibration_data
    return errors, hand_eye_lst


def perform_hand_eye_calibration_analysis(data_df, reprojection_data_df, intrinsics_pth, size_chess, repeats=1000,
                                          num_images_start=5, num_images_end=60, num_images_step=2, waitTime=1,
                                          visualise_reprojection_error=False, results_pth='', optimise=True):
    """
    Perform hand-eye calibration analysis to evaluate reprojection error for different number of image samples and board types
    :param data_df: dataframe containing calibration data
    :param reprojection_data_df: dataframe containing reprojection data
    :param intrinsics_pth: path to intrinsics folder
    :param size_chess: size of chessboard
    :param repeats: number of repeats for each number of images
    :param num_images_start: start number of images 
    :param num_images_end: end number of images
    :param num_images_step: step size for number of images
    :param waitTime: time to wait for user input
    :param visualise_reprojection_error: whether to visualise reprojection error
    :param results_pth: path to save results
    :param optimise: whether to optimise hand-eye calibration

    :return: average reprojection error for each number of images
    
    """
    # create results folder if it doesn't exist
    """ if not os.path.exists(results_pth):
        os.makedirs(results_pth) """
    num_images_lst = np.arange(num_images_start, num_images_end, num_images_step)

    error_lst = []
    average_error_lst = []
    std_error_lst = []
    all_hand_eye = []
    num_corners_detected_lst = []
    calibration_data_lst = []

    for num_images in tqdm(num_images_lst, desc='number of images', leave=False):
        errors_lst, hand_eye_lst, calibration_data = he_analysis(data_df, reprojection_data_df,
                                               intrinsics_pth, size_chess,
                                               waitTime=waitTime, n=num_images, repeats=repeats,
                                               visualise_reprojection_error=visualise_reprojection_error,
                                               optimise=optimise, return_calibration_data=True)

        num_corners_detected = data_df['num_corners_detected'].sum()

        error_lst.append(errors_lst)
        average_error_lst.append(np.mean(errors_lst))
        std_error_lst.append(np.std(errors_lst))
        all_hand_eye.append(hand_eye_lst)
        num_corners_detected_lst.append(num_corners_detected)
        calibration_data_lst.append(calibration_data)



    # save intrinsics, distortion and errors
    data = {'num_images_lst': num_images_lst,
            'errors_lst': error_lst,
            'num_corners_detected_lst': num_corners_detected_lst,
            'hand_eye': all_hand_eye,
            'average_error': average_error_lst,
            'std_error': std_error_lst,
            'calibration_data': calibration_data_lst}
    data_df = pd.DataFrame(data=data)

    # save dataframe
    if len(results_pth) > 0:
        data_df.to_pickle(results_pth)
    return data_df


######################################################
######## INTRINSICS ##################################
######################################################

def calibrate_and_evaluate(args):
    board_data, n, R, intrinsics_initial_guess_pth, image_shape, visualise_reprojection_error, waitTime = args

    # sample from calibration dataset however many number of samples we're investigating
    calibration_data, _ = sample_dataset(board_data, total_samples=n, groupby_cats=['pose', 'deg'])
    
    if R is None:
        reprojection_data = calibration_data
    else:
        # use all reprojection dataset
        reprojection_data = R

    # select contents of object points and image points
    imgPoints = calibration_data.imgPoints.values
    objPoints = calibration_data.objPoints.values

    mtx, dist = calibrate_charuco_board(
        intrinsics_initial_guess_pth=intrinsics_initial_guess_pth,
        calibration_save_pth='',
        image_shape=image_shape,
        imgPoints=imgPoints, objPoints=objPoints,
        # if we want to calibrate from images
        # image_pths=[] ,
        # board = None
    )

    # calculate reprojection error with these values
    objPoints_reprojection = reprojection_data.objPoints.values
    imgPoints_reprojection = reprojection_data.imgPoints.values
    if visualise_reprojection_error:
        image_paths = reprojection_data.paths.values
    else:
        image_paths = None
    err = calculate_reprojection_error(mtx, dist, objPoints_reprojection, imgPoints_reprojection,
                                       image_pths=image_paths, waitTime=waitTime, IDs=reprojection_data.ids.values)

    num_corners_detected = calibration_data['num_detected_corners'].sum()
    return mtx, dist, err, num_corners_detected


def perform_analysis(camera, data_df, reprojection_data_df, repeats=1000, num_images_start=5, num_images_end=60,
                     num_images_step=2,
                     visualise_reprojection_error=False, waitTime=1, results_pth='',
                     thread_num=0):  # , info_df,board,image_pths
    """
    performs calibration analysis to evaluate reprojection error for different number of image samples and board types

    camera: 'endo' or 'realsense', specifies which camera so that rough intrinsics can be loaded
    data_df: corner detection data that was generated in previous step with all 2D and 3D corner detection information
    reprojection_data_df: same corner data detection generated but data to be used for reprojection error
    waitTime: time to wait between frames if we are visualising the reprojection error
    visualise_reprojection_error: whether to visualise reprojection error as an image with projected corners

    returns a dataframe with the results of the analysis
    {'num_images_lst': number of images used for calibration in each iteration
    'errors': list of reprojection errors of each iteration (each iteration will have a list of errors of the length of repeats)
    'intrinsics': list of intrinsics matrices for each iteration
    'distortion': list of distortion coefficients for each iteration
    'average_error': average reprojection error for each iteration
    'std_error': standard deviation of reprojection error for each iteration
    'num_corners_detected_lst': number of corners detected in each iteration

    }
    """
    # perform calibration analysis
    if camera == 'realsense':
        intrinsics_initial_guess_pth = f'calibration_estimates/intrinsics_realsense.txt'
        image_shape = (480, 848)
    else:
        intrinsics_initial_guess_pth = f'calibration_estimates/intrinsics_endo.txt'
        image_shape = (1080, 1920)

    num_images_lst = np.arange(num_images_start, num_images_end, num_images_step)
    average_error_lst = []
    error_lst = []
    all_intrinsics = []
    all_distortion = []
    std_error_lst = []
    num_corners_detected_lst = []
    # for num_images in tqdm(num_images_lst, desc='num_images', leave=False):
    for num_images in num_images_lst:
        """ errors, intrinsics, distortion, num_corners_detected = analyse_calibration_data(data_df,
                             reprojection_data_df, # number of frames to use for calculating reprojection loss
                             n = num_images, # number of frames to use for calibration
                             repeats = repeats, # number of repeats for the calibration
                             intrinsics_initial_guess_pth=intrinsics_initial_guess_pth,
                             visualise_reprojection_error=visualise_reprojection_error,
                             waitTime=waitTime, # time to display each image for (in seconds) when showing reprojection
                             image_shape=image_shape,
                             thread_num=thread_num) """

        intrinsics = []
        distortion = []
        errors = []
        num_corners_detected = []
        # load one of the images to get the shape of the image

        """ for i in tqdm(range(repeats), desc = f'thread {thread_num}'):  

            args = (data_df,num_images,reprojection_data_df, intrinsics_initial_guess_pth, image_shape, visualise_reprojection_error, waitTime)
            mtx, dist, err, num = calibrate_and_evaluate(args)
            intrinsics.append(mtx)
            distortion.append(dist)
            errors.append(err)
            num_corners_detected.append( num)  """

        with concurrent.futures.ProcessPoolExecutor() as pool:
            args_list = [(data_df, num_images, reprojection_data_df, intrinsics_initial_guess_pth, image_shape,
                          visualise_reprojection_error, waitTime) for i in range(repeats)]
            results_all = pool.map(calibrate_and_evaluate, args_list)

            # add to 
            for result in results_all:
                mtx, dist, err, num = result
                intrinsics.append(mtx)
                distortion.append(dist)
                errors.append(err)
                num_corners_detected.append(num)

                # ignore infinite errors
        errors_filtered = [e for e in errors if not np.isinf(e)]
        # ignore anything larger than 20
        # errors_filtered = [e for e in errors if e < 20]
        # print how many were infinite or larger than 20
        # print(f'Number of larger errors: {len(errors)-len(errors_filtered)}')
        # if errors filtered is empty, add a nan
        if len(errors_filtered) == 0:
            average_error_lst.append(np.nan)
            std_error_lst.append(np.nan)
        else:
            average_error_lst.append(np.median(errors_filtered))
            std_error_lst.append(np.std(errors_filtered))

        error_lst.append(errors)
        all_intrinsics.append(intrinsics)
        all_distortion.append(distortion)
        num_corners_detected_lst.append(num_corners_detected)

    # save intrinsics, distortion and errors
    results = {'num_images_lst': num_images_lst,
               'errors_lst': error_lst,
               'num_corners_detected_lst': num_corners_detected_lst,
               'intrinsics': all_intrinsics,
               'distortion': all_distortion,
               'average_error': average_error_lst,
               'std_error': std_error_lst}
    results_df = pd.DataFrame(data=results)

    # save dataframe
    if len(results_pth) > 0:
        results_df.to_pickle(results_pth)
    return results_df


def calculate_reprojection_error(mtx, dist, objPoints, imgPoints, image_pths=None, waitTime=1, IDs=None):
    """
    calculate reprojection error on a set of points from images given the intrinsics and distortion coefficients

    Parameters
    ----------
    mtx : ndarray
    camera intrinsics matrix
    dist : ndarray
    distortion coefficients
    objPoints : ndarray
    3D points of the chessboard
    imgPoints : ndarray
    2D points of the chessboard

    image_pths : list of strings
    list of image paths to display the reprojection error, by default None
    waitTime : int, optional
    time to wait for key press to continue, by default 1
    """

    mean_errors = []
    for i in range(len(objPoints)):

        if len(objPoints[i]) < 4 or len(imgPoints[i]) < 4:
            continue

        # Estimate rvec and tvec using solvePnP
        retval, rvec, tvec = cv2.solvePnP(objPoints[i], imgPoints[i], mtx, dist)
        # Project 3D points to image plane
        imgpoints_reprojected, _ = cv2.projectPoints(objPoints[i], rvec, tvec, mtx, dist)
        imgpoints_detected = imgPoints[i]
        # calculate error
        if IDs is None:
                ID = None
        else:
            ID = IDs[i]
        if image_pths is not None:
            image = cv2.imread(image_pths[i])
            
            error_np, annotated_image = reprojection_error(imgpoints_detected, imgpoints_reprojected, image=image, IDs=ID)
            cv2.imshow('charuco board', annotated_image)
            cv2.waitKey(waitTime)
        else:
            error_np = reprojection_error(imgpoints_detected, imgpoints_reprojected, IDs=ID)
        mean_errors.append(error_np)
    
    mean = pd.DataFrame(mean_errors).mean()[0]
    median = pd.DataFrame(mean_errors).median()[0]
    if mean - median > 0.5:
        reprojection_error_mean_final = median
    else:
        reprojection_error_mean_final = mean

    return reprojection_error_mean_final


def detect_corners_charuco_cube_images(board, image_pths,  return_corners=True, waiting_time=0,
                                       visualise_corner_detection=False
                                       # min_num_corners=6, percentage_of_corners=0.2, 
                                       ):
   
    """
    function to detect corners in a list of images given a board
    Parameters
    ----------
    board : cv2.aruco.CharucoBoard
        board to use for detection
    image_pths : list of strings
        list of paths to images to detect corners in 
    
    """
    # min_num_corners : int (minimum number of corners to detect in an image for it to be saved (default=6)
    #waiting_time : int (time to wait for a new frame in ms

    if len(image_pths) == 0:
        raise (f'no images found')

    imgPoints = []
    objPoints = []
    num_detected_corners = []
    ids_all = []
    
    updated_image_pths = image_pths.copy()
    parameters = cv2.aruco.DetectorParameters()
    dictionary = board.getDictionary()

    # chessboard corners in 3D and corresponding ids
    charuco_corners_3D = board.getChessboardCorners()  # allCorners3D_np
    num_chess_corners = len(charuco_corners_3D)  # number_of_corners_per_face
    charuco_ids_3D = np.arange(0, num_chess_corners)  # all3DIDs_np

    # select minimum number of corners to be detected 
    #min_num_corners = select_min_num_corners(min_num_corners, percentage_of_corners, num_chess_corners)
    min_num_corners = 1

    # detect corners in images
    for image_pth in tqdm(image_pths, desc='image paths', leave=False):
        image = cv2.imread(image_pth)
        # if image is None (currupt): remove it from the list of images
        if image is None:
            updated_image_pths.remove(image_pth)
            continue
        # detect aruco tags
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        if corners:
            # interpolate to get corners of charuco board
            ret, charuco_detected_corners, charuco_detected_ids = cv2.aruco.interpolateCornersCharuco(corners, ids,
                                                                                                      gray, board)
        else:
            ret = 0

        # if there are less than minimum num of points selected, remove img from paths and skip this image so we don't save it
        if ret < min_num_corners:
            updated_image_pths.remove(image_pth)
            if visualise_corner_detection:
                cv2.imshow('charuco board', image)
            # print(f'skipping image {image_pth} because it has less than 6 corners')
            continue

        # by this point, we have filtered out any images that don't have enough charuco board corners detected so we can add them to the list of images to save
        # draw the detected corners on the image
        if visualise_corner_detection:
            cv2.aruco.drawDetectedCornersCharuco(image, charuco_detected_corners, charuco_detected_ids)
            cv2.imshow('charuco board', image)
            cv2.waitKey(waiting_time)
        # add the detected charuco corners to the list of all charuco corners
        imgPoints.append(charuco_detected_corners)
        num_detected_corners.append(len(charuco_detected_corners))
        ids_all.append(charuco_detected_ids)
        # find the corresponding 3D pnts
        _, allCorners3D_np_sorted_filtered = sort_and_filter_matched_corners(charuco_detected_corners.squeeze(),
                                                                             charuco_corners_3D, charuco_detected_ids,
                                                                             charuco_ids_3D)
        objPoints.append(allCorners3D_np_sorted_filtered.reshape(-1, 1, 3))

    
    if return_corners:
        return updated_image_pths, min_num_corners, imgPoints, objPoints, num_detected_corners, ids_all  # rvec, tvec,
    return updated_image_pths, min_num_corners, ids_all  # image.shape[0:-1]



def calibrate_charuco_board(
        intrinsics_initial_guess_pth='',
        calibration_save_pth='',
        image_shape=(1280, 720),
        imgPoints=None, objPoints=None,
        # if we want to calibrate from images
        image_pths=[],
        board=None
):
    '''
    calibrates a charuco board using a list of images or a list of 2D points and 3D points
    '''

    # all3DIDs_np, allCorners3D_np, boards, number_of_corners_per_face = boards_3D_points(faces, number_horizontally, number_vertically, size_chess, aruco_size, dictionary, charuco_board_save_pth=charuco_board_save_pth)
    if image_pths:
        _, imgPoints, objPoints, num_detected_corners, image_shape = detect_corners_charuco_cube_images(board,
                                                                                                        image_pths,
                                                                                                        waiting_time=0,
                                                                                                        min_num_corners=6,
                                                                                                        percentage_of_corners=0.2)

    # calibrate camera with allCorners and allIds (2D) of all cube charuco faces (allCorners3D, all3DIDs)
    if len(imgPoints) > 0:
        # to perform camera calibration with a non-planar target, we need to 
        # provide a rough initial estimate of the camera calibration parameters
        if len(intrinsics_initial_guess_pth) > 0:
            initial_camera_matrix = np.loadtxt(intrinsics_initial_guess_pth)
        else:
            fx = 420  # Focal length in x direction 800, 640
            fy = 420  # Focal length in y direction 800
            cx = image_shape[0] / 2  # Principal point x-coordinate (usually image width / 2) 320
            cy = image_shape[1] / 2  # Principal point y-coordinate (usually image height / 2) 240
            # Create the initial intrinsic & dist matrix guesses
            initial_camera_matrix = np.array([[fx, 0, cx],
                                              [0, fy, cy],
                                              [0, 0, 1]], dtype=np.float64)
        # set initial guess of distortion coefficients to zero
        initial_dist_coeffs = np.zeros(5, dtype=np.float64)
        # CALIB_USE_INTRINSIC_GUESS needs to be used when we provide a non-planar target
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS)  # cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO
        criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)
        # perform camera calibration using 3D object points and corresponding detected points in image 
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, image_shape, initial_camera_matrix,
                                                           initial_dist_coeffs, flags=flags, criteria=criteria)
        # save results
        if ret:

            if len(calibration_save_pth) > 0:

                # create folder to save calibration results
                if not os.path.exists(calibration_save_pth):
                    os.makedirs(calibration_save_pth)
                print("Calibration successful")
                print("Camera matrix:\n", mtx)
                print("Distortion coefficients:\n", dist)
                print("Rotation vectors:\n", rvecs)
                print("Translation vectors:\n", tvecs)
                # Save the camera calibration result in txt files
                np.savetxt(f'{calibration_save_pth}/camera_matrix.txt', mtx, delimiter=',')
                np.savetxt(f'{calibration_save_pth}/dist_coeffs.txt', dist, delimiter=',')

            return mtx, dist
    return None, None


######################################################

def generate_charuco_board(size_of_checkerboard, return_all_params=False):
    '''
    generate a charuco board object with the given size of checkerboards from the boards used in the experiments
    '''

    if size_of_checkerboard == 5:
        aruco_h = 25
        aruco_w = 35
        aruco_size = 3
    elif size_of_checkerboard == 10:
        aruco_h = 11
        aruco_w = 17
        aruco_size = 7
    elif size_of_checkerboard == 15:
        aruco_h = 7
        aruco_w = 11
        aruco_size = 11
    elif size_of_checkerboard == 20:
        aruco_h = 5
        aruco_w = 9
        aruco_size = 15
    elif size_of_checkerboard == 25:
        aruco_h = 5
        aruco_w = 7
        aruco_size = 18
    elif size_of_checkerboard == 30:
        aruco_h = 3
        aruco_w = 5
        aruco_size = 22
    else:
        aruco_h = 5
        aruco_w = 7
        aruco_size = 16
        size_of_checkerboard = 18

    aruco_dict = cv2.aruco.DICT_4X4_1000
    board = cv2.aruco.CharucoBoard(
        (aruco_w, aruco_h),  # 7x5
        size_of_checkerboard,
        aruco_size,
        cv2.aruco.getPredefinedDictionary(aruco_dict)
    )
    if return_all_params:
        return board, aruco_h, aruco_w, aruco_size, aruco_dict
    return board
