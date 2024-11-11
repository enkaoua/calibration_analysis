
import argparse
import glob
import random
import cv2
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from charuco_utils import calibrate_and_evaluate
from main_calibration_analysis import generate_board_table
from record_utils import RealsenseVideoSourceAPI
from rectangle_perspective import project_rect_points_to_image
from utils import extrinsic_vecs_to_matrix, sort_and_filter_matched_corners

def is_well_distributed(rvec, tvec, poses, threshold_angle=10, threshold_distance=0.1):
    """
    Check if the current frame's position (rvec, tvec) is well distributed from the previous ones.
    Parameters:
    - threshold_angle: minimum angle difference between orientations
    - threshold_distance: minimum distance difference between positions
    """
    for prev_rvec, prev_tvec in poses:
        angle_diff = np.linalg.norm(rvec - prev_rvec)
        dist_diff = np.linalg.norm(tvec - prev_tvec)
        if dist_diff < threshold_distance:
            if angle_diff < threshold_angle:
                return False
            return True
    return True





def detect_board_position_and_corners(image, board, percentage_of_corners=0.5, intrinsics=None, distortion=None, cam='endo'):
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
    num_chess_corners = len(charuco_corners_3D)  # number_of_corners_per_face
    charuco_ids_3D = np.arange(0, num_chess_corners)  # all3DIDs_np

    min_num_corners = int(percentage_of_corners * num_chess_corners)

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


def initialise_cameras(endo_port, realsense_save_path):
    # initialising first cam (endoscope)
    endo_cam = cv2.VideoCapture(endo_port)
    
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
    # read realsense
    if realsense_save_path:
        #ret, frames, color_image = read_realsense_image(pipeline)
        ret_rs, right_frame, left_frame, right_image, left_image, rs_img, depth_image = realsense_cam.read_stereo_image()
        name2 = '{}/{:08d}.png'.format(realsense_save_path, count)
        return ret_endo, ret_rs, endo_img, rs_img, name2
    return ret_endo, None, endo_img, None, None

def save_images(name, endo_img, realsense_save_path, realsense_cam, name2, rs_img):
    # save img1
    #print(name)
    cv2.imwrite(f'{name}', endo_img) 

    # save realsense image
    if realsense_save_path:
        # save captured frame from cam 2 (realsense)
        cv2.imwrite(name2, rs_img) 

        # SAVE REALSENSE CAMERA INFO TO BAG
        realsense_cam.save_frameset(f'{realsense_save_path}')

def end_recording(endo_cam, realsense_save_path,realsense_cam ):
    print('stopped recording')

    endo_cam.release()
    cv2.destroyAllWindows()
    
    # Stop streaming
    if realsense_save_path:
        realsense_cam.stop()
    return

def plot_3D_positions(ax, tag2cam_rvec, tag2cam_tvec,  frame_used_for_calib):
    # add 3D plot of 3D positions recorded where positions are added on the go
    if tag2cam_rvec is not None and tag2cam_tvec is not None:

        # Determine whether this frame is for calibration or reprojection
        if frame_used_for_calib:
            ax.scatter(tag2cam_tvec[0], tag2cam_tvec[1], tag2cam_tvec[2], color='green')  # For calibration (green)
            # add arrow in direction of angle
        else:
            ax.scatter(tag2cam_tvec[0], tag2cam_tvec[1], tag2cam_tvec[2], color='red')  # For reprojection (red)
        
        # Update 3D plot
        plt.draw()
        plt.pause(0.001)


def display_rectangle(annotated_endo_img, rects, current_rect_idx):
    # Get the current rectangle
    current_rect = rects[current_rect_idx]
    # Draw rectangle on the image
    cv2.rectangle(annotated_endo_img, current_rect[0], current_rect[1], (255, 0, 0), 2)
    return current_rect

# Helper to check if new frame is far from previous ones
def is_frame_well_distributed(new_frame_pose, previous_frames, min_dist=100, min_angle=10):
    for pose in previous_frames:
        dist = np.linalg.norm(new_frame_pose[1] - pose[1])  # Euclidean distance between translations
        angle_diff = np.linalg.norm(new_frame_pose[0] - pose[0])  # Difference in rotations (angles)
        if dist < min_dist and angle_diff < np.rad2deg(min_angle):
            return False
    return True

def plot_2D_im_positions( endo_img, tag2cam_rvec, tag2cam_tvec, charuco_detected_corners, calib_frames, min_dist=100, min_angle=10):
    
    # find centre of board
    center_point = tuple(charuco_detected_corners.mean(axis=0).astype(int).squeeze())

    if tag2cam_rvec is not None and tag2cam_tvec is not None:
        # Save frame if it is well distributed from previous calibration frames
        new_pose = (tag2cam_rvec, tag2cam_tvec)
        frame_used_for_calib = is_frame_well_distributed(new_pose, calib_frames['poses'], min_dist=min_dist, min_angle=min_angle)
        if frame_used_for_calib:
            # add data to calib_frames
            cv2.putText(endo_img, 'Frame saved for calibration', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(endo_img, center_point, 5, (0, 255, 0), -1)  # Green for calibration
        else:
            #reproj_frames.poses.append(new_pose)
            cv2.putText(endo_img, 'Frame saved for reprojection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(endo_img, center_point, 5, (0, 0, 255), -1)  # Red for reprojection
    return frame_used_for_calib, center_point, new_pose    


def generate_rectangles(grid_size, camera):
    """
    Generate a list of rectangles covering the camera's field of view, 
    based on the specified grid size.
    
    grid_size: (int) Number of divisions in both width and height.
    camera: Camera object to get the image dimensions.
    
    Returns: List of rectangles [(top_left, bottom_right), ...]
    """
    ret, frame = camera.read()  # Read one frame to get dimensions
    h, w = frame.shape[:2]  # Get image dimensions

    step_w = w // grid_size
    step_h = h // grid_size
    rectangles = []

    for i in range(grid_size):
        for j in range(grid_size):
            top_left = (j * step_w, i * step_h)
            bottom_right = ((j + 1) * step_w, (i + 1) * step_h)
            rectangles.append((top_left, bottom_right))

    return rectangles


def is_board_inside_rectangle(board_corners, rectangle, tolerance=10):
    """
    Check if the outer corners of the detected Charuco board are inside the given rectangle.
    
    board_corners: Corners of the detected Charuco board.
    rectangle: Tuple ((x1, y1), (x2, y2)) representing the rectangle's top-left and bottom-right corners.
    tolerance: (int) A tolerance value to account for slight misalignments.
    
    Returns: True if all board corners are inside the rectangle (with tolerance), else False.
    """
    top_left, bottom_right = rectangle
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Check if each of corners is inside the rectangle, considering tolerance
    for corner in board_corners:
        x, y = corner[0][0], corner[0][1]  # Extract x, y coordinates
        if not (x1 - tolerance <= x <= x2 + tolerance and y1 - tolerance <= y <= y2 + tolerance):
            return False

    return True


def generate_rectangles_with_angles_____________old(grid_size, camera):
    """
    Generate a list of rectangles covering the camera's field of view,
    based on the specified grid size, and assign a target rotation angle for each rectangle.
    
    grid_size: (int) Number of divisions in both width and height.
    camera: Camera object to get the image dimensions.
    
    Returns: List of tuples (rectangle [(top_left, bottom_right)], target_angle).
    """
    ret, frame = camera.read()  # Read one frame to get dimensions
    h, w = frame.shape[:2]  # Get image dimensions

    step_w = w // grid_size
    step_h = h // grid_size
    rectangles = []

    for i in range(grid_size):
        for j in range(grid_size):
            top_left = (j * step_w, i * step_h)
            bottom_right = ((j + 1) * step_w, (i + 1) * step_h)
            target_angle = random.choice([0, 45, 90, 135])  # Random or predefined angles
            rectangles.append((top_left, bottom_right, target_angle))

    return rectangles


def generate_perspective_rectangles(grid_size, camera, num_orientations=4):
    """
    Generate a list of perspective rectangles to guide both position and orientation.
    
    grid_size: Number of divisions in both width and height.
    camera: Camera object to get the image dimensions.
    num_orientations: Number of distinct angles to guide the user for.
    
    Returns: List of rectangles with angles [(top_left, bottom_right, angle), ...]
    """
    ret, frame = camera.read()  # Read one frame to get dimensions
    h, w = frame.shape[:2]  # Get image dimensions

    step_w = w // grid_size
    step_h = h // grid_size
    rectangles = []

    for i in range(grid_size):
        for j in range(grid_size):
            top_left = (j * step_w, i * step_h)
            bottom_right = ((j + 1) * step_w, (i + 1) * step_h)

            # Add perspective by suggesting an angle for each rectangle
            angle = random.choice([30]) #(360 / num_orientations) * (i + j) % 360
            rectangles.append((top_left, bottom_right, angle))

    return rectangles



def rotate_point_3d(point, angle, axis):
    """
    Rotate a 3D point around a specified axis.
    
    point: 3D point (x, y, z).
    angle: Rotation angle in degrees.
    axis: Axis to rotate around ('x', 'y', 'z').
    
    Returns: Rotated 3D point.
    """
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    x, y, z = point
    
    if axis == 'x':
        # Rotation around x-axis
        y_new = y * cos_a - z * sin_a
        z_new = y * sin_a + z * cos_a
        return x, y_new, z_new
    elif axis == 'y':
        # Rotation around y-axis
        x_new = x * cos_a + z * sin_a
        z_new = -x * sin_a + z * cos_a
        return x_new, y, z_new
    elif axis == 'z':
        # Rotation around z-axis
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        return x_new, y_new, z

def draw_perspective_rectangle(image, top_left, bottom_right, angle, axis='x'):
    """
    Draw a rotated rectangle on the image by rotating the corners in 3D space.
    
    image: Image on which to draw.
    top_left: Top-left corner of the rectangle in 2D.
    bottom_right: Bottom-right corner of the rectangle in 2D.
    angle: Rotation angle in degrees.
    axis: Axis around which to rotate ('x', 'y', or 'z').
    
    Returns: Image with the rotated rectangle drawn.
    """
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    # Define the 4 corners of the rectangle in 3D (Z=0 initially)
    corners_3d = np.array([
        [top_left[0], top_left[1], 0],  # Top-left corner
        [bottom_right[0], top_left[1], 0],  # Top-right corner
        [bottom_right[0], bottom_right[1], 0],  # Bottom-right corner
        [top_left[0], bottom_right[1], 0]  # Bottom-left corner
    ])

    # Apply rotation to the corners
    rotated_corners_3d = np.array([rotate_point_3d(corner, angle, axis) for corner in corners_3d])

    # Project the rotated corners to 2D by ignoring the Z-axis
    rotated_corners_2d = rotated_corners_3d[:, :2].astype(np.int32)

    # Draw the resulting rectangle by connecting the 4 corners
    for i in range(4):
        cv2.line(image, tuple(rotated_corners_2d[i]), tuple(rotated_corners_2d[(i + 1) % 4]), (255, 0, 0), 2)

    return image


def is_frame_well_distributed_with_angle(new_frame_pose, previous_frames, min_dist=0.1, min_angle=10, min_orientations=4):
    """
    Checks if a new frame is well distributed based on distance and angular difference.
    Also, ensures a minimum number of distinct orientations have been captured.
    
    new_frame_pose: Tuple (rvec, tvec) representing the new frame pose.
    previous_frames: List of previous poses to compare against.
    min_dist: Minimum distance in translation to consider the frame well-distributed.
    min_angle: Minimum angle in degrees for rotation difference.
    min_orientations: Minimum number of distinct orientations to capture.
    """
    num_angles = 0  # To count distinct angles
    for pose in previous_frames:
        dist = np.linalg.norm(new_frame_pose[1] - pose[1])  # Euclidean distance between translations
        angle_diff = np.linalg.norm(new_frame_pose[0] - pose[0])  # Difference in rotations (angles)
        if dist < min_dist and angle_diff < np.deg2rad(min_angle):
            return False
        if angle_diff >= np.deg2rad(min_angle):
            num_angles += 1

    # Ensure at least min_orientations distinct orientations are captured
    if num_angles < min_orientations:
        return False
    
    return True

def is_board_inside_polygon(board_corners, rectangle_points, tolerance=10):
    """
    Check if the Charuco board corners are inside a polygon (which represents a rotated rectangle).
    
    board_corners: Detected corners of the Charuco board (Nx2 array).
    rectangle_points: The 4 corner points of the polygon (rotated rectangle) (4x2 array).
    tolerance: Tolerance to account for slight misalignments.
    
    Returns: True if all board corners are inside the polygon, False otherwise.
    """
    # Convert board corners to numpy array for easy handling
    board_corners_np = np.array([corner[0] for corner in board_corners])
    
    # Create a polygon using the provided rectangle points
    polygon = np.array(rectangle_points[0:2])

    # Loop through each board corner
    for corner in board_corners_np:
        # Check if the corner is inside the polygon using cv2.pointPolygonTest
        inside = cv2.pointPolygonTest(polygon, (corner[0], corner[1]), False)
        if inside < 0:
            # If any corner is outside the polygon, return False
            return False
        
    # All corners are inside the polygon
    return True

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def is_board_inside_projected_rectangle(charuco_detected_corners, projected_pnts, tolerance=10):
    
    pnts_inside = 0
    # Store the first point in the polygon and initialize the second point
    projected_pnts = [Point(p[0], p[1]) for p in projected_pnts]
    
    for pnt in charuco_detected_corners:
        point = Point(pnt[0][0], pnt[0][1])
        # Checking if a point is inside a polygon
        num_vertices = len(projected_pnts)
        x, y = point.x, point.y
        inside = False


        p1 = projected_pnts[0]
    
        # Loop through each edge in the polygon
        for i in range(1, num_vertices + 1):
            # Get the next point in the polygon
            p2 = projected_pnts[i % num_vertices]
    
            # Check if the point is above the minimum y coordinate of the edge    | .   
            if y > min(p1.y, p2.y):
                # Check if the point is below the maximum y coordinate of the edge     . |
                if y <= max(p1.y, p2.y):
                    # Check if the point is to the left of the maximum x coordinate of the edge  . |
                    if x <= max(p1.x, p2.x):
                        # Calculate the x-intersection of the line connecting the point to the edge .-
                        x_intersection = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
    
                        # Check if the point is on the same line as the edge or to the left of the x-intersection
                        if p1.x == p2.x or x <= x_intersection:
                            # Flip the inside flag
                            inside = not inside
    
            # Store the current point as the first point for the next iteration
            p1 = p2

        if inside:
            pnts_inside += 1
    
    # Return the value of the inside flag
    if pnts_inside > len(charuco_detected_corners) * 0.9:
        return True
    else:
        return False
    

def record_board_live(endo_save_path, 
                      realsense_save_path='', 
                      endo_port = 0, board=None, 
                      #calibration_estimates_pth = 'calibration_estimates/intrinsics_endo.txt', 
                      calibration_estimates_pth = 'calibration_estimates/intrinsics_mac.txt', 
                      max_frames=np.inf, 
                      grid_size=2 # number of rectangles to guide user
                      ):
    """
    records video with realsense and endo cams

    Press 'q' or esc to quit.
    parames:
        - endo_save_path (str): path where to save images from endoscope camera
        - realsense_save_path (str, optional): path where to save images for calibration of second camera (if you have a second camera set up)
        - endo_port (int, optional): specify port of first camera for reading opencv images
    """

    # Creating folders where vid and images for first cam will be saved 
    if not os.path.exists(endo_save_path):
        os.makedirs(endo_save_path)

    # ----- CAMERA INIIALISATIONS AND CONFIGURATIONS -----

    endo_cam, realsense_cam = initialise_cameras(endo_port, realsense_save_path)

    count = 0 # count number of frames
    poses = [] # list to store poses of board
    calib_frames = {
        'imgPoints': [],
        'objPoints': [],
        'paths': [],
        'poses': [],
        'num_detected_corners': [],
        'centre_point': []
    } 
    
    reproj_frames = {
        'imgPoints': [],
        'objPoints': [],
        'paths': [],
        'poses': [],
        'num_detected_corners': [],
        'centre_point': []
    } 
    
    current_rect_idx = 0  # Index of the current rectangle
    rects = generate_rectangles(grid_size, endo_cam)  # Generate rectangles based on the camera frame size

    # Create 3D plot for visualizing camera positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # -------------- RECORDING ------------------
    # load `poses to display for user to follow
    poses = np.load('results/user_study/poses/test2/final_poses.npy')
    intrinsics = np.loadtxt(calibration_estimates_pth)
    distortion = np.zeros(5)
    err = np.inf
    #record_calib = False
    while True:
        key = 0xFF & cv2.waitKey(1)
        # -------> READ IMAGES
        ret_endo, ret_rs, endo_img, rs_img, name2 = read_images(endo_cam, realsense_cam, realsense_save_path, count)
        if not ret_endo:
            continue
        # name and show frame from first cam (endoscope)
        endo_frame_name = '{}/{:08d}.png'.format(endo_save_path, count)
        annotated_endo_img = endo_img.copy()

        # -------> DISPLAY CURRENT RECTANGLE (to guide user)
        """ x,y,z,deg_x, deg_y, deg_z = poses[current_rect_idx]
        annotated_endo_img, projected_pnts = project_rect_points_to_image(annotated_endo_img, intrinsics, distortion, x, y, z, deg_x, deg_y, deg_z)
        projected_pnts = projected_pnts.squeeze() """
        if current_rect_idx == len(rects) :
            current_rect_idx = 0
            if len(calib_frames['paths']) < 30:
                print('Not enough frames, continuing')
                continue
            else:
                mtx, dist, err, num_corners_detected = perform_calibration(calib_frames, reproj_frames, calibration_estimates_pth, endo_img)
                intrinsics = mtx
                distortion = dist
                if err < 0.5:
                    end_recording(endo_cam, realsense_save_path,realsense_cam )
                    break
                else:
                    print('Calibration failed after rectangles, continuing')
                    continue
        current_rect = display_rectangle(annotated_endo_img, rects, current_rect_idx)

        # -------> DISPLAY CURRENT PERSPECTIVE RECTANGLE (to guide user)
        #draw_perspective_rectangle(endo_img, current_rect[0], current_rect[1], current_rect[2])
        # display all imgPoints
        for imgPoints in calib_frames['imgPoints']:
            cv2.aruco.drawDetectedCornersCharuco(annotated_endo_img, imgPoints, None, (0, 255, 0))
        for imgPoints in reproj_frames['imgPoints']:
            cv2.aruco.drawDetectedCornersCharuco(annotated_endo_img, imgPoints, None, (0, 0, 255))
        # display all centre points (green for calibration frames, red for reprojection frames)
        """ for centre_point in calib_frames['imgPoints']:
            cv2.circle(annotated_endo_img, centre_point, 5, (0, 255, 0), -1)
        for centre_point in reproj_frames['imgPoints']:
            cv2.circle(annotated_endo_img, centre_point, 5, (0, 0, 255), -1) """

        # -------> DETECT BOARD AND CORNERS
        # detect corners
        annotated_endo_img, tag2cam_rvec, tag2cam_tvec, charuco_detected_corners, allCorners3D_np_sorted_filtered, num_detected_corners, allIDs3D_np_sorted_filtered  = detect_board_position_and_corners(annotated_endo_img, board, percentage_of_corners=0.5, intrinsics=intrinsics, distortion=distortion, cam='endo')
        
        
        # Check if the Charuco board corners are inside the current rectangle
        if charuco_detected_corners is not None:
            ## -------> PLOT 3D POSITIONS
            
            frame_used_for_calib,centre_point, pose  = plot_2D_im_positions( annotated_endo_img, tag2cam_rvec, tag2cam_tvec, charuco_detected_corners, calib_frames, min_dist=10, min_angle=10)            
            """ if not record_calib:
                frame_used_for_calib = False """
            
            if frame_used_for_calib:
                calib_frames['poses'].append(pose)
                # path to frame
                calib_frames['paths'].append(endo_frame_name)
                calib_frames['imgPoints'].append(charuco_detected_corners)
                calib_frames['objPoints'].append(allCorners3D_np_sorted_filtered)
                calib_frames['num_detected_corners'].append(num_detected_corners)
                calib_frames['centre_point'].append(centre_point)
                #calib_frames['frame_used_for_reprojection'].append(frame_used_for_calib)
            else:

                reproj_frames['poses'].append(pose)
                reproj_frames['paths'].append(endo_frame_name)
                reproj_frames['imgPoints'].append(charuco_detected_corners)
                reproj_frames['objPoints'].append(allCorners3D_np_sorted_filtered)
                reproj_frames['num_detected_corners'].append(num_detected_corners)
                reproj_frames['centre_point'].append(centre_point)
                #reproj_frames['frame_used_for_reprojection'].append(frame_used_for_calib)

            plot_3D_positions(ax, tag2cam_rvec, tag2cam_tvec,  frame_used_for_calib)

            # plot all            

            if is_board_inside_rectangle(charuco_detected_corners, current_rect):
            #if is_board_inside_projected_rectangle(charuco_detected_corners, projected_pnts, tolerance=10):
                # If the board is inside the rectangle, move to the next one
                current_rect_idx = (current_rect_idx + 1) #% len(rects)
                cv2.putText(annotated_endo_img, 'Board inside rectangle, moving to next.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # -------> SHOW IMAGES
        # add current reprojection error to image
        cv2.putText(annotated_endo_img, f'error: {round(err,1)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
        cv2.namedWindow(f'endoscope port {endo_port}', cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f'endoscope port {endo_port}', annotated_endo_img)
        
        # ----------- SAVING IMAGES ---------------------
        
        save_images(endo_frame_name, endo_img, realsense_save_path, realsense_cam, name2, rs_img)

        count += 1

        """ if len(reproj_frames['paths']) > 100:
            record_calib = True """




        # when finished recording, save video and release streams
        if key == 27 or key==ord('q') or count == max_frames:
            end_recording(endo_cam, realsense_save_path,realsense_cam )
            mtx, dist, err, num_corners_detected = perform_calibration(calib_frames, reproj_frames, calibration_estimates_pth, endo_img)
            break
        
        elif count%30==0:
            if len(calib_frames['paths']) > 6:
                mtx, dist, err, num_corners_detected = perform_calibration(calib_frames, reproj_frames, calibration_estimates_pth, endo_img)
                intrinsics = mtx
                distortion = dist
            else:
                print('Not enough frames, continuing')
                continue
            if err < 0.5:
                print('Calibration successful')
                print('mtx: ', mtx)
                print('dist: ', dist)
                print('err: ', err)
                print('num_corners_detected: ', num_corners_detected)
                end_recording(endo_cam, realsense_save_path,realsense_cam )
                break
            else:
                print('30 frames unsuccessful, continuing')
                continue
            # end_recording(endo_cam, realsense_save_path,realsense_cam )
    
    return mtx, dist, err, num_corners_detected






def perform_calibration(calib_frames, reproj_frames, calibration_estimates_pth, endo_img, visualise_reprojection_error = False):
    calibration_data_df = pd.DataFrame(calib_frames)
    reproj_data_df = pd.DataFrame(reproj_frames)
    num_images_to_sample_for_calibration = None

    intrinsics_initial_guess_pth = f'{calibration_estimates_pth}'
    image_shape = endo_img.shape[:2]
    waitTime = 1
    

    # perform calibration on selected frames
    args = calibration_data_df, num_images_to_sample_for_calibration, reproj_data_df, intrinsics_initial_guess_pth, image_shape, visualise_reprojection_error, waitTime
    mtx, dist, err, num_corners_detected = calibrate_and_evaluate(args)
    """ print('calibration done')
    print('mtx: ', mtx)
    print('dist: ', dist)
    print('err: ', err)
    print('num_corners_detected: ', num_corners_detected) """
    return mtx, dist, err, num_corners_detected

""" def record_board_from_images(rs_):
    return """

def main(aruco_w=7,
         aruco_h=11,
         size_of_checkerboard=15,
         aruco_size=11,
         endo_save_path='results/user_study/endo_calibration_images'): 

    # 0) initialise params
    aruco_dict = cv2.aruco.DICT_4X4_1000
    board = cv2.aruco.CharucoBoard(
        (aruco_w, aruco_h),  # 7x5
        size_of_checkerboard,
        aruco_size,
        cv2.aruco.getPredefinedDictionary(aruco_dict)
    )

    # 1) record video of board with different poses and angles- tell user if they're too close
    # record images
    record_board_live(endo_save_path, 
                                realsense_save_path='', 
                                endo_port = 0, 
                                board=board)
    
    """ # 2) detect corners and save them
    table_data_pth = endo_save_path
    table_info_pth = endo_save_path
    data_df, _ = generate_board_table(endo_save_path, board, table_data_pth, table_info_pth, 
                         #min_num_corners=1, percentage_of_corners=0.2, 
                         waiting_time=1,
                         intrinsics=None, distortion=None, visualise_corner_detection=False)


    # 2) select best frames and use rest to get reprojection error 
    # filter out any images less than 50% detection rate
    charuco_corners_3D = board.getChessboardCorners()  # allCorners3D_np
    num_chess_corners = len(charuco_corners_3D)  # number_of_corners_per_face
    selected_min_num_corners = int(0.5 * num_chess_corners)
    data_df_filtered = data_df[data_df['num_detected_corners'] > selected_min_num_corners]

    # select 20 frames to perform initial intrinsic calibration

    # create column to group each row of dataframe into position number (1-10) for x,y,z and rotation (1-10) for r,p,y. Each group should contain distances/angles to the nearest 10th of a degree/mm
    # convert T to 

    # sample 50 images evenly distributed to perform calibration

    # 3) perform calibration

    # 4) add more frames if necessary """

    return 


if __name__=='__main__': 

    """ # board
    aruco_h = 7
    aruco_w = 11
    aruco_size = 11
    size_of_checkerboard = 15
    aruco_dict = cv2.aruco.DICT_4X4_1000
    board = cv2.aruco.CharucoBoard(
        (aruco_w, aruco_h),  # 7x5
        size_of_checkerboard,
        aruco_size,
        cv2.aruco.getPredefinedDictionary(aruco_dict)
    )

    pths = glob.glob('/Users/aure/Documents/CARES/data/massive_calibration_data/15_charuco/pose0/acc_15_pos0_deg0_1/raw/he_calibration_images/hand_eye_endo/*.png')
    for pth in pths:
        # read image
        img = cv2.imread(pth)
        
        # detect corners
        image, tag2cam, charuco_detected_corners, allCorners3D_np_sorted_filtered, charuco_detected_corners, allIDs3D_np_sorted_filtered  = detect_board_position_and_corners(img, board, percentage_of_corners=0.5, intrinsics=None, distortion=None)
        # show image
        cv2.imshow('charuco board', image)
        cv2.waitKey(1) """
   

    parser = argparse.ArgumentParser(
        description='user study calibration') 
    
    # adding all necessary args for cl app
    parser.add_argument('--save_path', type=str, default='results/user_study/endo_calibration_images', 
                        help='path to where images uesd for calibration are stored')
    parser.add_argument('--aruco_w', type=int, default=13,
                        help='')
    parser.add_argument('--aruco_h', type=int, default=9,
                    help='')
    parser.add_argument('--size_of_checkerboard', type=int, default=20,
                    help='')
    parser.add_argument('--aruco_size', type=int, default=14,
                    help='')
    # grabbing args selected
    args = parser.parse_args()


    main(aruco_w=int(args.aruco_w),
         aruco_h=int(args.aruco_h),
         size_of_checkerboard=int(args.size_of_checkerboard),
         aruco_size=int(args.aruco_size),
         endo_save_path=args.save_path) 