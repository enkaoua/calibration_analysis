import glob

from matplotlib import pyplot as plt
import pandas as pd


def plot_info_as_bar(raw_corner_data, filtered_data_30, filtered_data_50, analysis_data):
    
    # plot results as bar chart
    plt.figure(figsize=(15, 5))
    # subplot 1
    plt.subplot(1, 3, 1)
    plt.title('raw number of images')
    
    # loop through key and value pairs
    for key, value in raw_corner_data.items():
        # get number of images
        num_images = len(value)
        # plot bar
        plt.bar(key, num_images)
        plt.text(key, num_images + 100, str(num_images), ha='center')

    plt.ylabel('number of images')
    plt.xlabel('Board Size')
    plt.ylim(0,10000)
    # plotting each board as separate cluster
    # subplot 2
    plt.subplot(1, 3, 2)
    plt.title('Nº images after 30% corner filtering')
    # loop through key and value pairs
    for key, value in filtered_data_30.items():
        # get number of images
        num_images = len(value)
        # plot bar
        plt.bar(key, num_images)
        plt.text(key, num_images + 100, str(num_images), ha='center')

    #plt.ylabel('number of images')
    plt.xlabel('Board Size')
    plt.ylim(0,10000)

    # number of corners
    """ plt.subplot(1, 3, 3)
    plt.title('number of corners detected after filtering 30% of corners')
    # loop through key and value pairs
    for key, value in filtered_data.items():
        # get number of corners
        num_corners = value['num_detected_corners'].sum()
        # plot bar
        plt.bar(key, num_corners)
    plt.ylabel('number of corners')
    plt.xlabel('Board Size')
    # make all plots same axis in y axis
    # plt.ylim(0,  info_15_rs.data.values[2].max()+100)
    plt.legend()
    plt.show() """

    plt.subplot(1, 3, 3)
    plt.title('Nº images after 50% corner filtering')
    # loop through key and value pairs
    for key, value in filtered_data_50.items():
        # get number of images
        num_images = len(value)
        # plot bar
        plt.bar(key, num_images)
        # plot number on top of the bars
        plt.text(key, num_images + 100, str(num_images), ha='center')

    #plt.ylabel('number of images')
    plt.xlabel('Board Size')
    plt.ylim(0,10000)
    plt.legend()
    # make all image fit with titles not squished
    plt.tight_layout()

    


def main():
    
    for camera in ['endo', 'realsense']:
        # load data 
        rec_filtered_data_1 = f'MC_{min_num_corners}_PC_{percentage_of_corners_1}'
        rec_filtered_data_2 = f'MC_{min_num_corners}_PC_{percentage_of_corners_2}'
        rec_analysis = f'R{R}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}_{rec_filtered_data_1}'

        # load analysis data (num_images_lst, errors_lst, num_corners_detected_lst, hand_eye/intrinsics&distortion, average_error, std_error)
        analysis_data_pths = glob.glob(f'{calibration_pth}/calibration_analysis/{rec_analysis}/*{camera}*.pkl')
        # load raw corner data
        raw_corner_data_pths = glob.glob(f'{calibration_pth}/raw_corner_data/MC_None_PC_None/*{camera}*.pkl')
        # load filtered data
        filter_data_pths_1 = glob.glob(f'{calibration_pth}/filtered_data/{rec_filtered_data_1}/*{camera}*.pkl')
        filter_data_pths_2 = glob.glob(f'{calibration_pth}/filtered_data/{rec_filtered_data_2}/*{camera}*.pkl')

        # read data
        #data = [pd.read_pickle(pth) for pth in data_pths]
        analysis_data = {}
        for pth in analysis_data_pths:
            chess_size = str(pth.split('/')[-1][0:2])
            analysis_data[chess_size] = pd.read_pickle(pth)

        # read raw corner data
        raw_corner_data = {}
        for pth in raw_corner_data_pths:
            chess_size = str(pth.split('/')[-1][0:2])
            raw_corner_data[chess_size] = pd.read_pickle(pth)
        
        # read filtered data
        filter_data_1 = {}
        for pth in filter_data_pths_1:
            chess_size = str(pth.split('/')[-1][0:2])
            filter_data_1[chess_size] = pd.read_pickle(pth)


        filter_data_2 = {}
        for pth in filter_data_pths_2:
            chess_size = str(pth.split('/')[-1][0:2])
            filter_data_2[chess_size] = pd.read_pickle(pth)

        plot_info_as_bar(raw_corner_data, filter_data_1, filter_data_2, analysis_data)
        plt.savefig(f'results/number_images_filtering_{camera}.png')


if __name__ == '__main__':
    hand_eye = False

    if hand_eye == True:
        calibration_pth = 'results/hand_eye'

        min_num_corners = 6.0  # if none selected, the percentage of corners is used (with min 6 corners)
        percentage_of_corners_1 = 0.3
        percentage_of_corners_2 = 0.5
        threshold = 30
        # analysis parameters
        R = None
        repeats = 100  # number of repeats per number of images analysis
        num_images_start = 5
        num_images_end = 60
        num_images_step = 5
        endo = True
        rs = False
        shift = [0.3, 0.1]

    else:
        calibration_pth = 'results/intrinsics'
        min_num_corners = 6.0
        percentage_of_corners = 0.5
        percentage_of_corners_1 = 0.3
        percentage_of_corners_2 = 0.5

        threshold = 2

        # analysis parameters
        R = 100
        num_images_start = 5
        num_images_end = 60
        num_images_step = 5
        repeats = 10  # number of repeats per number of images analysis
        endo = True
        rs = True
        shift = [0.3, 0.1]

    
    main()
    # plot_info_as_bar(info_pth=f'{calibration_pth}/raw_corner_data/{rec_data}')
