import argparse

from main_calibration_analysis import main_intrinsics

def add_recording_args_to_parser(parser):
    parser.add_argument('--data_path', type=str, default='/Users/aure/Documents/CARES/data/massive_calibration_data',
                        help='path to where images uesd for calibration are stored')
    parser.add_argument('--img_ext', type=str, default='png', help='extension of images')
    parser.add_argument('--reprojection_sample_size', type=int, default=None,
                        help='number of samples to use for reprojection error')
    parser.add_argument('--min_num_corners', type=str, default=6.0,
                        help='minimum number of corners to use for calibration')
    parser.add_argument('--percentage_of_corners', type=str, default=0.3,
                        help='percentage of corners to use for calibration')
    parser.add_argument('--visualise_corner_detection', type=bool, default=False,
                        help='if set to true, will visualise corner detection')
    parser.add_argument('--repeats', type=int, default=10, help='number of repeats per number of images analysis')
    parser.add_argument('--num_images_start', type=int, default=5, help='number of images to start analysis')
    parser.add_argument('--num_images_end', type=int, default=50, help='number of images to end analysis')
    parser.add_argument('--num_images_step', type=int, default=5, help='step size for number of images analysis')
    parser.add_argument('--visualise_reprojection_error', type=bool, default=False,
                        help='if set to true, will visualise reprojection error')
    parser.add_argument('--waitTime', type=int, default=0, help='time to wait before capturing next image')
    parser.add_argument('--chess_sizes', type=list, default=[15, 20, 25, 30],
                        help='sizes of chessboard used for calibration')
    parser.add_argument('--cameras', type=list, default=['endo', 'realsense'], help='cameras used for calibration')

    """ parser.add_argument('--results_pth', type=str, default='results/hand_eye', help='path to save results')
    parser.add_argument('--intrinsics_for_he', type=str,
                        default='results/intrinsics/best_intrinsics', #results/intrinsics/best_intrinsics
                        help='path to intrinsics results for he') """
    
    parser.add_argument('--results_pth', type=str, default='results/intrinsics', help='path to save results')
    parser.add_argument('--intrinsics_for_he', type=str,
                        default='', #results/intrinsics/best_intrinsics
                        help='path to intrinsics results for he')
    return parser


def main():
    parser = argparse.ArgumentParser(
        description='intrinsic calibration analysis code \
            Code has several steps: \
            1) goes through images to detect corners as specified and generates large table with information such as corner location, frame number etc\
               Data will be saved in folder raw_corner_data \
            2) Once table is generated, the data is split into reprojection dataset and calibration dataset. \
               Data will be saved in folder split_data \
               The fildered data and original data table will be stored under their respective folders, but within that \
               respective folder, the name will be in the format MC_x_PC_x (MC- min num of corners, PC- percentage of corners)\
            3) Finally, analysis is done. The analysis is done by performing calibration on a set of images between num_images_start and num_images_end at step of num_images_step. Each of these calibrations on N images is performed repeats number of times.\
            The analysis will be saved under calibration_analysis with the format R{reprojection_sample_size}_N{start}_{end}_{step}_repeats_{repeats}\
            \
            ')

    # adding all necessary args for cl app
    add_recording_args_to_parser(parser)
    # grabbing args selected
    args = parser.parse_args()

    data_path = args.data_path
    img_ext = args.img_ext
    reprojection_sample_size = args.reprojection_sample_size
    # convert string to int or none
    if reprojection_sample_size is not None:
        reprojection_sample_size = int(reprojection_sample_size)
    min_num_corners = args.min_num_corners
    # convert string to int or none
    if min_num_corners is not None:
        min_num_corners = int(min_num_corners)

    percentage_of_corners = args.percentage_of_corners
    # convert string to float or none
    if min_num_corners is not None:
        min_num_corners = float(min_num_corners)

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

    # for hand eye, we need intrinsics
    intrinsics_for_he = args.intrinsics_for_he

    main_intrinsics(data_path=data_path,
                    img_ext=img_ext,
                    reprojection_sample_size=reprojection_sample_size,
                    min_num_corners=min_num_corners,
                    # if none selected, the percentage of corners is used (with min 6 corners)
                    percentage_of_corners=percentage_of_corners,
                    visualise_corner_detection=visualise_corner_detection,
                    # analysis parameters
                    repeats=repeats,  # number of repeats per number of images analysis
                    num_images_start=num_images_start,
                    num_images_end=num_images_end,
                    num_images_step=num_images_step,
                    visualise_reprojection_error=visualise_reprojection_error,
                    waitTime=waitTime,
                    results_pth=results_pth,
                    chess_sizes=chess_sizes,
                    cameras=cameras,

                    intrinsics_for_he=intrinsics_for_he)

    return


if __name__ == '__main__':
    main()
