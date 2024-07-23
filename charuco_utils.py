import random
import cv2
import numpy as np
import os
import glob

from tqdm import tqdm
from utils import extrinsic_vecs_to_matrix, sample_dataset






""" def detect_charuco_board_pose(image, intrinsics, distortion, board, return_corners=False):
    parameters=cv2.aruco.DetectorParameters()
    #dictionary =cv2.aruco.getPredefinedDictionary(dict_id)
    dictionary = board.getDictionary()

    # load intrinsics
    #intrinsics = np.loadtxt(intrinsics)
    #distortion = np.loadtxt(distortion)
    # undistort image
    undistorted_image = cv2.undistort(image, intrinsics, distortion, None, intrinsics)
    # convert img to gray
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

    # create charuco board detector object
    
    # detect all charuco marker corners
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(undistorted_image, dictionary, parameters=parameters)
    if corners:
        #for i, board in enumerate(boards):
        # get markers corresponding to said board
        cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)
        # interpolate charuco corners
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        #cv2.aruco.Board.matchImagePoints(	charuco_corners,  charuco_ids, objPoints, imgPoints	)
        # save and show corners and ids
        if ret and len(charuco_corners)>4:
            #cv2.aruco.drawDetectedCornersCharuco(image, all_charuco_corners_np, all_charuco_ids_np)
            # Estimate the pose of the ChArUco board
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, intrinsics, distortion, None, None)

            if retval:
                #annotated_image = cv2.aruco.drawAxis(gray, intrinsics, distortion, rvec, tvec, 0.1)  # 0.1 is the length of the axis
                # draw markers
                annotated_image = cv2.aruco.drawDetectedCornersCharuco(undistorted_image, charuco_corners, charuco_ids)
                # Draw the ChArUco board axis
                annotated_image = cv2.drawFrameAxes(annotated_image, intrinsics, None, rvec, tvec, length=37)
                if return_corners:
                    return True, annotated_image, rvec, tvec, corners, ids

                return True,annotated_image, rvec, tvec
    if return_corners:
        return False, [], False, False, [], False
    return False, None, None, image """



def detect_charuco_board_pose_images( board, image_pths,intrinsics_pth, distortion_pth,return_corners=True, min_num_corners=6, waiting_time=0):
    """
    function to detect corners in a list of images given a board
    Parameters
    ----------
    board : cv2.aruco.CharucoBoard
        board to use for detection
    image_pths : list of strings
        list of paths to images to detect corners in 
    min_num_corners : int
        minimum number of corners to detect in an image for it to be saved (default=6)
    waiting_time : int
        time to wait for a new frame in ms

    """
    
    if len(image_pths) == 0:
        raise(f'no images found')
    # load intrinsics
    intrinsics = np.loadtxt(intrinsics_pth)
    distortion = np.loadtxt(distortion_pth)
    imgPoints = []
    objPoints = []
    T_all = []
    updated_image_pths = image_pths.copy()
    parameters=cv2.aruco.DetectorParameters()
    dictionary = board.getDictionary()

    # chessboard corners in 3D and corresponding ids
    charuco_corners_3D = board.getChessboardCorners() # allCorners3D_np
    num_chess_corners = len(charuco_corners_3D) # number_of_corners_per_face
    charuco_ids_3D = np.arange(0, num_chess_corners) # all3DIDs_np
    
    if min_num_corners is None:
        min_num_corners = int(0.2*num_chess_corners)
        if min_num_corners<6:
            min_num_corners = 6
    # detect corners in images
    for image_pth in image_pths:

        image = cv2.imread(image_pth)

        # undistort image
        undistorted_image = cv2.undistort(image, intrinsics, distortion, None, intrinsics)
        # convert img to gray
        gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

        
        # if image is None (currupt): remove it from the list of images
        if image is None:
            updated_image_pths.remove(image_pth)
            continue
        # detect aruco tags
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(undistorted_image, dictionary, parameters=parameters)
        if corners:
            # get markers corresponding to said board
            cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)
            # interpolate to get corners of charuco board
            ret, charuco_detected_corners, charuco_detected_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        else:
            ret = 0

        # if there are less than 6 points, remove img from paths and skip this image so we don't save it
        if ret<min_num_corners:
            updated_image_pths.remove(image_pth)
            cv2.imshow('charuco board', image)
            print(f'skipping image {image_pth} because it has less than 6 corners')
            continue
            
        # by this point, we have filtered out any images that don't have enough charuco board corners detected so we can move onto detecting the board pose
        # draw the detected corners on the image
        # Estimate the pose of the ChArUco board
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_detected_corners, charuco_detected_ids, board, intrinsics, distortion, None, None)

        if retval:
            #annotated_image = cv2.aruco.drawAxis(gray, intrinsics, distortion, rvec, tvec, 0.1)  # 0.1 is the length of the axis
            # draw markers
            annotated_image = cv2.aruco.drawDetectedCornersCharuco(undistorted_image, charuco_detected_corners, charuco_detected_ids)
            # Draw the ChArUco board axis
            annotated_image = cv2.drawFrameAxes(annotated_image, intrinsics, None, rvec, tvec, length=37)
            # display annotated img
            cv2.imshow('charuco board', image)
            cv2.waitKey(waiting_time)
            
            # add the detected charuco corners to the list of all charuco corners
            imgPoints.append(charuco_detected_corners)
            # find the corresponding 3D pnts
            _, allCorners3D_np_sorted_filtered = sort_and_filter_matched_corners(charuco_detected_corners.squeeze(), charuco_corners_3D, charuco_detected_ids, charuco_ids_3D)
            objPoints.append(allCorners3D_np_sorted_filtered.reshape(-1,1,3))
            # add the pose to the list of all poses
            tag2cam = extrinsic_vecs_to_matrix(rvec,tvec)
            T_all.append(tag2cam)
            #tag2endo = extrinsic_vecs_to_matrix(rvecs_endo,tvecs_endo)

        else:
            updated_image_pths.remove(image_pth)
            cv2.imshow('charuco board', image)
            print(f'skipping image {image_pth} because pose was not detected properly')
            continue

        
    if return_corners:
        return updated_image_pths, min_num_corners, T_all, imgPoints, objPoints #rvec, tvec, 
    return updated_image_pths, min_num_corners, T_all #image.shape[0:-1]




def analyse_calibration_data(board_data, 
                             R, # reprojection df
                             n = 1, # number of frames to use for calibration
                             repeats = 100, # number of times to repeat the calibration process
                             #size_chess = 15, 
                             #calibration_save_pth = '',
                             intrinsics_initial_guess_pth='',
                             visualise_reprojection_error = False,
                             waitTime=1 # wait time for when showing reprojection image
                             ): 
    """
    Function performs calibration on random set of n images and calculates the reprojection error.
    This is repeated "repeats" times and the reprojection errors are returned.
    
    """
    #path_to_endo_images = glob.glob(f'{data_path}/{size_chess}_charuco/pose*/acc_*_pos*_deg*_*/raw/he_calibration_images/hand_eye_endo/*.{img_ext}')
    #path_to_rs_images = glob.glob(f'{data_path}/{size_chess}_charuco/pose*/acc_*_pos*_deg*_*/raw/he_calibration_images/hand_eye_realsense/*.{img_ext}')
    # where to save information of aruco board
    #charuco_board_save_pth = f'{data_path}/{size_chess}_charuco'
    intrinsics = []
    distortion = []
    errors = []
    for i in tqdm(range(repeats)):
        # generate n+R random numbers that are in the range of the above loaded images
        #random_indices = random.sample(range(len(board_data)), R+n)
        #img_indeces_for_calibration = random_indices[R:]
        #img_indeces_for_reprojection = random_indices[:R]

        # load contents data from the table with these indeces 
        #calibration_data = board_data.iloc[img_indeces_for_calibration]
        #reprojection_data = board_data.iloc[img_indeces_for_reprojection]
        calibration_data, _  = sample_dataset(board_data, total_samples=n)
        reprojection_data = R
        
        # select contents of object points and image points
        imgPoints = calibration_data.imgPoints.values
        objPoints = calibration_data.objPoints.values
        # load one of the images to get the shape of the image
        image_shape = cv2.imread(calibration_data.paths.values[0]).shape[0:-1]
        mtx, dist = calibrate_charuco_board( 
                            intrinsics_initial_guess_pth=intrinsics_initial_guess_pth, 
                            calibration_save_pth='',
                            image_shape = image_shape,
                            imgPoints=imgPoints, objPoints=objPoints,
                            # if we want to calibrate from images
                            #image_pths=[] ,
                            #board = None 
                            )
        
        # calculate reprojection error with these values
        objPoints_reprojection = reprojection_data.objPoints.values
        imgPoints_reprojection = reprojection_data.imgPoints.values
        if visualise_reprojection_error:
            image_paths = reprojection_data.paths.values
        else: 
            image_paths = None
        err = calculate_reprojection_error(mtx, dist, objPoints_reprojection, imgPoints_reprojection, image_pths=image_paths, waitTime=waitTime)
        #print(err)
        intrinsics.append(mtx)
        distortion.append(dist)
        errors.append(err)

        # calculate reprojection error with these values
        
    #print(f'intrinsics: {intrinsics}')
    #print(f'distortion: {distortion}')
    

    """ # split the images into the ones used for calculating calibration and ones used for reprojection


    # load the images with these indeces from the list of images of each camera
    img_paths_for_calibration_endo = [path_to_endo_images[i] for i in img_indeces_for_calibration]
    img_paths_for_reprojection_rs = [path_to_rs_images[i] for i in img_indeces_for_reprojection]
    img_paths_for_reprojection_endo = [path_to_endo_images[i] for i in img_indeces_for_reprojection]
    img_paths_for_calibration_rs = [path_to_rs_images[i] for i in img_indeces_for_calibration]
    
    # create boards and 3D cube coords
    board, number_horizontally, number_vertically, aruco_size, dictionary_id = generate_charuco_board(size_chess)
    all3DIDs_np, allCorners3D_np, boards, number_of_corners_per_face = boards_3D_points([0], 
                                                                                        number_horizontally,
                                                                                        number_vertically,
                                                                                        size_chess,
                                                                                        aruco_size,
                                                                                        dictionary_id, 
                                                                                        charuco_board_save_pth=charuco_board_save_pth)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    
    camera_intrinsics_initial_guess_endo = np.loadtxt(f'{intrinsics_initial_guess_endo_pth}')
    intrinsics_endo, distortion_endo  = calibrate_charuco_cube(boards,  dictionary,
                                                                    number_of_corners_per_face, 
                                                                    allCorners3D_np, 
                                                                    all3DIDs_np, 
                                                                    camera_intrinsics_initial_guess=camera_intrinsics_initial_guess_endo, 
                                                                    calibration_save_pth=calibration_save_pth,
                                                                    image_pths=img_paths_for_calibration_endo, capture_all=False )
    

    camera_intrinsics_initial_guess_rs = np.loadtxt(f'{intrinsics_initial_guess_rs_pth}')
    intrinsics_rs, distortion_rs  = calibrate_charuco_cube(boards,
                                                               dictionary,
                                                                number_of_corners_per_face, 
                                                                allCorners3D_np, 
                                                                all3DIDs_np, 
                                                                camera_intrinsics_initial_guess=camera_intrinsics_initial_guess_rs, 
                                                                calibration_save_pth=calibration_save_pth,
                                                                image_pths=img_paths_for_calibration_rs, capture_all=False )
    
    """
    return errors, intrinsics, distortion



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


def calculate_reprojection_error(mtx, dist, objPoints, imgPoints, image_pths=None, waitTime=1):
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
    #faces, number_horizontally, number_vertically, size_chess, aruco_size, reprojection_images_pth,img_ext, charuco_board_save_pth, dictionary,wait_time=1):
   
    # 3D points of test board
    #all3DIDs_np, allCorners3D_np, boards, num_chess_corners = boards_3D_points(faces, number_horizontally, number_vertically, size_chess, aruco_size, dictionary, charuco_board_save_pth=charuco_board_save_pth)

    #image_pths = glob.glob(f'{reprojection_images_pth}/*.{img_ext}')
    #image_pths, imgPoints, objPoints, img_shape = detect_corners_charuco_cube_images( boards, dictionary,num_chess_corners, allCorners3D_np, all3DIDs_np, image_pths)
    
    # update image paths with non-rejected paths
    #image_pths = [image_pths[i] for i in range(len(image_pths)) if i not in rejected_ids]
    # Calculate reprojection error on new images
    mean_errors = []
    mean_errors_np = []
    for i in range(len(objPoints)):
        
        if len(objPoints[i])<4 or len(imgPoints[i])<4:
            continue
        # load image
        #image = cv2.imread(image_pths[i])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Estimate rvec and tvec using solvePnP
        retval, rvec, tvec = cv2.solvePnP(objPoints[i], imgPoints[i], mtx, dist)
        # Project 3D points to image plane
        imgpoints_reprojected, _ = cv2.projectPoints(objPoints[i], rvec, tvec, mtx, dist)
        imgpoints_detected = imgPoints[i]
        # calculate error
        if image_pths is not None:
            image = cv2.imread(image_pths[i])
            error_np, error, annotated_image = reprojection_error(imgpoints_detected, imgpoints_reprojected, image=image)
            cv2.imshow('charuco board', annotated_image)
            cv2.waitKey(waitTime) 
        else:
            error_np, error = reprojection_error(imgpoints_detected, imgpoints_reprojected)
        mean_errors_np.append(error_np)
        mean_errors.append(error)


    
    reprojection_error_mean_final = np.array(mean_errors).mean()
    #print(f'Reprojection error: {np.array(mean_errors).mean()}')
    #print(f'Reprojection error np: {np.array(mean_errors_np).mean()}')
    #path_name = os.path.basename(reprojection_images_pth).split('_')[-1]
    #print(f'{path_name} reprojection error: {reprojection_error_mean_final}')
    return reprojection_error_mean_final



def sort_and_filter_matched_corners(corners_endo, corners_realsense, ids_endo, ids_realsense):
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

    return corners_endo_sorted, corners_realsense_sorted




def detect_corners_charuco_cube_images( board, image_pths, min_num_corners=6, percentage_of_corners=0.2, waiting_time=0):
    """
    function to detect corners in a list of images given a board
    Parameters
    ----------
    board : cv2.aruco.CharucoBoard
        board to use for detection
    image_pths : list of strings
        list of paths to images to detect corners in 
    min_num_corners : int
        minimum number of corners to detect in an image for it to be saved (default=6)
    waiting_time : int
        time to wait for a new frame in ms

    """
    
    if len(image_pths) == 0:
        raise(f'no images found')
    
    imgPoints = []
    objPoints = []
    updated_image_pths = image_pths.copy()
    parameters=cv2.aruco.DetectorParameters()
    dictionary = board.getDictionary()

    # chessboard corners in 3D and corresponding ids
    charuco_corners_3D = board.getChessboardCorners() # allCorners3D_np
    num_chess_corners = len(charuco_corners_3D) # number_of_corners_per_face
    charuco_ids_3D = np.arange(0, num_chess_corners) # all3DIDs_np
    
    if min_num_corners is None:
        min_num_corners = int(percentage_of_corners*num_chess_corners)
        if min_num_corners<6:
            min_num_corners = 6
    # detect corners in images
    for image_pth in image_pths:

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
            ret, charuco_detected_corners, charuco_detected_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        else:
            ret = 0

        # if there are less than 6 points, remove img from paths and skip this image so we don't save it
        if ret<min_num_corners:
            updated_image_pths.remove(image_pth)
            cv2.imshow('charuco board', image)
            print(f'skipping image {image_pth} because it has less than 6 corners')
            continue
            
        # by this point, we have filtered out any images that don't have enough charuco board corners detected so we can add them to the list of images to save
        # draw the detected corners on the image
        cv2.aruco.drawDetectedCornersCharuco(image, charuco_detected_corners, charuco_detected_ids)
        cv2.imshow('charuco board', image)
        cv2.waitKey(waiting_time)
        # add the detected charuco corners to the list of all charuco corners
        imgPoints.append(charuco_detected_corners)
        # find the corresponding 3D pnts
        _, allCorners3D_np_sorted_filtered = sort_and_filter_matched_corners(charuco_detected_corners.squeeze(), charuco_corners_3D, charuco_detected_ids, charuco_ids_3D)
        objPoints.append(allCorners3D_np_sorted_filtered.reshape(-1,1,3))

    return updated_image_pths, imgPoints, objPoints, min_num_corners #image.shape[0:-1]


def calibrate_charuco_board( 
                           intrinsics_initial_guess_pth='', 
                           calibration_save_pth='',
                           image_shape = (1280, 720), 
                           imgPoints=None, objPoints=None,
                           # if we want to calibrate from images
                           image_pths=[] ,
                           board = None 
                           ): 
    '''
    calibrates a charuco board using a list of images or a list of 2D points and 3D points
    '''



    # all3DIDs_np, allCorners3D_np, boards, number_of_corners_per_face = boards_3D_points(faces, number_horizontally, number_vertically, size_chess, aruco_size, dictionary, charuco_board_save_pth=charuco_board_save_pth)
    if image_pths:
        _, imgPoints, objPoints, image_shape = detect_corners_charuco_cube_images( board, image_pths, waiting_time=0, min_num_corners=6, percentage_of_corners=0.2)

    # calibrate camera with allCorners and allIds (2D) of all cube charuco faces (allCorners3D, all3DIDs)
    if len(imgPoints) > 0:
        # to perform camera calibration with a non-planar target, we need to 
        # provide a rough initial estimate of the camera calibration parameters
        if len(intrinsics_initial_guess_pth) > 0:
            initial_camera_matrix = np.loadtxt(intrinsics_initial_guess_pth)
        else:
            fx = 420  # Focal length in x direction 800, 640
            fy = 420   # Focal length in y direction 800
            cx = image_shape[0]/2  # Principal point x-coordinate (usually image width / 2) 320
            cy = image_shape[1]/2  # Principal point y-coordinate (usually image height / 2) 240
            # Create the initial intrinsic & dist matrix guesses
            initial_camera_matrix = np.array([[fx, 0, cx],
                                            [0, fy, cy],
                                            [0, 0, 1]], dtype=np.float64)
        # set initial guess of distortion coefficients to zero
        initial_dist_coeffs = np.zeros(5, dtype=np.float64)
        # CALIB_USE_INTRINSIC_GUESS needs to be used when we provide a non-planar target
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS ) # cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO
        criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)
        # perform camera calibration using 3D object points and corresponding detected points in image 
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints,image_shape , initial_camera_matrix, initial_dist_coeffs, flags=flags, criteria=criteria)
        # save results
        if ret:

            if len(calibration_save_pth)>0:

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
            (aruco_w, aruco_h), # 7x5
            size_of_checkerboard,
            aruco_size,
            cv2.aruco.getPredefinedDictionary(aruco_dict)
        )
    if return_all_params:
        return board, aruco_h, aruco_w, aruco_size, aruco_dict
    return board



