import cv2
import numpy as np
import os





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




def detect_corners_charuco_cube_images( board, image_pths, min_num_corners=6, waiting_time=0):
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

    return updated_image_pths, imgPoints, objPoints, image.shape[0:-1]


def calibrate_charuco_board( 
                           camera_intrinsics_initial_guess='', 
                           calibration_save_pth='',
                           image_shape = (1280, 720),
                           # if we want to calibrate from images
                           image_pths=[] ,
                           board = None 
                           ): 
    '''
    faces = [0, 1, 2, 3], number_horizontally=5, number_vertically=5, size_chess=10, aruco_size=9,  charuco_board_save_pth = '', dictionary_id=None, 
    calibrates non-planar target cube with charuco board on each face. 
    The cube can be generated with the script generate_cube_charuco.py and the resulting files containing the IDs of the tags are used here.
    The calibration is done with the opencv library.

    '''

    # create folder to save calibration results
    if not os.path.exists(calibration_save_pth):
        os.makedirs(calibration_save_pth)

    # all3DIDs_np, allCorners3D_np, boards, number_of_corners_per_face = boards_3D_points(faces, number_horizontally, number_vertically, size_chess, aruco_size, dictionary, charuco_board_save_pth=charuco_board_save_pth)
    if image_pths:
        _, imgPoints, objPoints, image_shape = detect_corners_charuco_cube_images( board, image_pths, waiting_time=0)
    else:
        imgPoints, objPoints = [], [], []
    # calibrate camera with allCorners and allIds (2D) of all cube charuco faces (allCorners3D, all3DIDs)
    if len(imgPoints) > 0:
        # to perform camera calibration with a non-planar target, we need to 
        # provide a rough initial estimate of the camera calibration parameters
        if len(camera_intrinsics_initial_guess) > 0:
            initial_camera_matrix = np.loadtxt(camera_intrinsics_initial_guess)
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
            print("Calibration successful")
            print("Camera matrix:\n", mtx)
            print("Distortion coefficients:\n", dist)
            print("Rotation vectors:\n", rvecs)
            print("Translation vectors:\n", tvecs)
            # Save the camera calibration result in txt files
            np.savetxt(f'{calibration_save_pth}/camera_matrix.txt', mtx, delimiter=',')
            np.savetxt(f'{calibration_save_pth}/dist_coeffs.txt', dist, delimiter=',')


    return mtx, dist



def generate_charuco_board(size_of_checkerboard, return_all_params=False):
    '''
    generate a charuco board with the given size of checkerboards from the boards used in the experiments
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



