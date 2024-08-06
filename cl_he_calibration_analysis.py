from main_calibration_analysis import main_intrinsics
import argparse
#import configparser


def add_recording_args_to_parser(parser):
    parser.add_argument('--data_path', type=str, default='/Users/aure/Documents/CARES/data/massive_calibration_data', help='path to where images uesd for calibration are stored')
    parser.add_argument('--img_ext', type=str, default='png', help='extension of images')
    parser.add_argument('--reprojection_sample_size', type=int, default=100, help='number of samples to use for reprojection error')
    parser.add_argument('--min_num_corners', type=str, default=None, help='minimum number of corners to use for calibration')
    parser.add_argument('--percentage_of_corners', type=float, default=0.2, help='percentage of corners to use for calibration')
    parser.add_argument('--visualise_corner_detection', type=bool, default=False, help='if set to true, will visualise corner detection')
    parser.add_argument('--repeats', type=int, default=3, help='number of repeats per number of images analysis')
    parser.add_argument('--num_images_start', type=int, default=5, help='number of images to start analysis')
    parser.add_argument('--num_images_end', type=int, default=60, help='number of images to end analysis')
    parser.add_argument('--num_images_step', type=int, default=1, help='step size for number of images analysis')
    parser.add_argument('--visualise_reprojection_error', type=bool, default=False, help='if set to true, will visualise reprojection error')
    parser.add_argument('--waitTime', type=int, default=1, help='time to wait before capturing next image')
    parser.add_argument('--results_pth', type=str, default='results/intrinsics', help='path to save results')
    parser.add_argument('--chess_sizes', type=list, default=[15, 20, 25, 30], help='sizes of chessboard used for calibration')
    parser.add_argument('--cameras', type=list, default=['endo', 'realsense'], help='cameras used for calibration')

    return parser

def main(): 

    parser = argparse.ArgumentParser(
        description='recording realsense and endoscope (images or video) \
            When in image mode, press "c" to capture image, "q" or esc to quit. \
            You can either add a config path containing a json file with all the \
            arguments or you can enter the arguments directly in the command line. \
                If you do both, the command line arguments will be used. ')   
    add_recording_args_to_parser(parser)
    
    args = parser.parse_args()
    data_path = args.data_path
    img_ext = args.img_ext
    reprojection_sample_size = int(args.reprojection_sample_size)
    min_num_corners = args.min_num_corners
    # convert string to int or none
    if min_num_corners is not None:
        min_num_corners = int(min_num_corners)
    



    percentage_of_corners = float(args.percentage_of_corners)
    visualise_corner_detection = bool(args.visualise_corner_detection)
    repeats = int(args.repeats)
    num_images_start = int(args.num_images_start)
    num_images_end = int(args.num_images_end)
    num_images_step = int(args.num_images_step)
    visualise_reprojection_error = bool(args.visualise_reprojection_error)
    waitTime = int(args.waitTime)
    results_pth = args.results_pth
    chess_sizes = args.chess_sizes
    cameras = args.cameras
 

    main_intrinsics(data_path = data_path,
                    img_ext = img_ext,
                    reprojection_sample_size = reprojection_sample_size,
                    min_num_corners = min_num_corners, # if none selected, the percentage of corners is used (with min 6 corners)
                    percentage_of_corners = percentage_of_corners ,
                    visualise_corner_detection=visualise_corner_detection,
                    # analysis parameters
                    repeats=repeats, # number of repeats per number of images analysis
                    num_images_start=num_images_start,
                    num_images_end=num_images_end,
                    num_images_step=num_images_step,
                    visualise_reprojection_error=visualise_reprojection_error,
                    waitTime = waitTime, 
                    results_pth = results_pth, 
                    chess_sizes = chess_sizes,
                    cameras = cameras )
    
    return 




if __name__=='__main__': 
    
    
    main() 
