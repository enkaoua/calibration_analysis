import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(board_size):
    endo_data = pd.read_pickle(f'results/calibration_error_data/{board_size}_endo_calibration_data.pkl')
    rs_data = pd.read_pickle(f'results/calibration_error_data/{board_size}_realsense_calibration_data.pkl')
    return endo_data, rs_data

def load_info(board_size):
    endo_data = pd.read_csv(f'results/corner_data/{board_size}_endo_corner_info.csv')
    rs_data = pd.read_csv(f'results/corner_data/{board_size}_realsense_corner_info.csv')
    return endo_data, rs_data

def main(): 
    # load all results
    
    endo = True
    rs = False

    #results_df = results_df.sort_values(by=['size_chess', 'camera'])
    endo_data_30, rs_data_30 = load_data(30)
    endo_data_25, rs_data_25 = load_data(25)
    endo_data_20, rs_data_20 = load_data(20)
    endo_data_15, rs_data_15 = load_data(15)

    start_val = 3
    num_images_lst = np.arange(1, 10, 2)
    num_images_lst = num_images_lst[start_val:]
    # plot results
    plt.figure()

    if endo == True:
        #plt.plot(endo_data_30['average_error'][start_val:], label='30x30 endo')
        plt.errorbar(num_images_lst, endo_data_30['average_error'][start_val:], yerr=endo_data_30['std_error'][start_val:], label='30 endo', fmt='o', elinewidth=2)
        
        #plt.plot(endo_data_25['average_error'][start_val:], label='25x25 endo')
        plt.errorbar(num_images_lst, endo_data_25['average_error'][start_val:], yerr=endo_data_25['std_error'][start_val:], label='25 endo', fmt='*', elinewidth=1.5)
        
        #plt.plot(endo_data_20['average_error'][start_val:], label='20x20 endo')
        plt.errorbar(num_images_lst, endo_data_20['average_error'][start_val:], yerr=endo_data_20['std_error'][start_val:], label='20 endo', fmt='^', elinewidth=1)
    
        #plt.plot(endo_data_15['average_error'][start_val:], label='15x15 endo')
        plt.errorbar(num_images_lst, endo_data_15['average_error'][start_val:], yerr=endo_data_15['std_error'][start_val:], label='15 endo', fmt='s', elinewidth=0.5)

    if rs == True:
        #plt.plot(rs_data_30['average_error'][start_val:], label='30x30 realsense')
        plt.errorbar(num_images_lst, rs_data_30['average_error'][start_val:], yerr=rs_data_30['std_error'][start_val:], label='30 realsense', fmt='o')

        #plt.plot(rs_data_25['average_error'][start_val:], label='25x25 realsense')
        plt.errorbar(num_images_lst, rs_data_25['average_error'][start_val:], yerr=rs_data_25['std_error'][start_val:], label='25 realsense', fmt='*')

        #plt.plot(rs_data_20['average_error'][start_val:], label='20x20 realsense')
        plt.errorbar(num_images_lst, rs_data_20['average_error'][start_val:], yerr=rs_data_20['std_error'][start_val:], label='20 realsense', fmt='^')
    
        #plt.plot(rs_data_15['average_error'][start_val:], label='15x15 realsense')
        plt.errorbar(num_images_lst, rs_data_15['average_error'][start_val:], yerr=rs_data_15['std_error'][start_val:], label='15 realsense', fmt='s')

    plt.legend()
    plt.show()

    return 


def plot_info_as_bar():
    # load all info pkl files
    info_15_rs, info_15_endo = load_info(15)
    info_20_rs, info_20_endo = load_info(20)
    info_25_rs, info_25_endo = load_info(25)
    info_30_rs, info_30_endo = load_info(30)

    # plot results as bar chart
    plt.figure()
    # subplot 1
    plt.subplot(1,3,1)
    plt.title(info_15_rs.titles.values[0])
    plt.bar('15', info_15_rs.data.values[0], label=info_30_rs.titles.values[0])   
    plt.bar('20', info_20_rs.data.values[0], label=info_30_rs.titles.values[0])
    plt.bar('25', info_25_rs.data.values[0], label=info_30_rs.titles.values[0])
    plt.bar('30', info_30_rs.data.values[0], label=info_30_rs.titles.values[0])
    plt.xlabel('Board Size')

    
    #plt.bar(info_15_rs.titles.values[3], info_15_rs.data.values[3], label='15 realsense')   

    # plotting each board as separate cluster
    # subplot 2
    plt.subplot(1,3,2)
    plt.title(info_15_rs.titles.values[1])
    plt.bar('15', info_15_rs.data.values[1], label=info_30_rs.titles.values[1])   
    plt.bar('20', info_20_rs.data.values[1], label=info_30_rs.titles.values[1])
    plt.bar('25', info_25_rs.data.values[1], label=info_30_rs.titles.values[1])
    plt.bar('30', info_30_rs.data.values[1], label=info_30_rs.titles.values[1])

    #plt.bar(info_20_rs.titles.values[3], info_20_rs.data.values[3], label='20 realsense')

    plt.subplot(1,3,3)
    plt.title(info_15_rs.titles.values[2])
    plt.bar('15', info_15_rs.data.values[2], label=info_30_rs.titles.values[2])   
    plt.bar('20', info_20_rs.data.values[2], label=info_30_rs.titles.values[2])
    plt.bar('25', info_25_rs.data.values[2], label=info_30_rs.titles.values[2])
    plt.bar('30', info_30_rs.data.values[2], label=info_30_rs.titles.values[2])

    #plt.bar(info_25_rs.titles.values[3], info_25_rs.data.values[3], label='25 realsense')

    #plt.subplot(1,4,4)

    #plt.bar(info_30_rs.titles.values[3], info_30_rs.data.values[3], label='30 realsense')
    #plt.bar(info_15_endo.data.values[0], info_15_endo.data.values[1],info_15_endo.data.values[2], label='15 realsense')

    plt.legend()


if __name__=='__main__': 
    plot_info_as_bar()
    main() 