import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(board_size, analysis_pth='results/calibration_error_data'):
    endo_data = pd.read_pickle(f'{analysis_pth}/{board_size}_endo_calibration_data.pkl')
    try:
        rs_data = pd.read_pickle(f'{analysis_pth}/{board_size}_realsense_calibration_data.pkl')
    except:
        rs_data = None
    return endo_data, rs_data

def load_info(board_size, info_pth='results/corner_data'):
    endo_data = pd.read_csv(f'{info_pth}/{board_size}_endo_corner_info.csv')
    try:
        rs_data = pd.read_csv(f'{info_pth}/{board_size}_realsense_corner_info.csv')
    except:
        rs_data = None
    return endo_data, rs_data

def get_average_std(data, threshold=100):
    avg_lst = []
    std_lst = []
    for errors in data:
        errors_np = np.array(errors)
        avg_lst.append(np.mean(errors_np[errors_np<threshold]))
        std_lst.append(np.std(errors_np[errors_np<threshold]))
    """     data_np = np.array(data)
    avg_lst = np.mean(data_np, axis=1)
    std_lst = np.std(data_np, axis=1) """
    return np.array(avg_lst), np.array(std_lst)
    

def plot_with_shaded_error(num_images_lst, data_dict, label, fmt, alpha=0.3, start_val=0, threshold=50):
    #avg_error = data_dict['average_error'][start_val:]
    #std_error = data_dict['std_error'][start_val:]
    errors = data_dict['errors'][start_val:]
    #avg_error = np.mean(errors, axis=1)
    #std_error = np.std(errors, axis=1)
    avg_error,std_error= get_average_std(errors, threshold=threshold)
    plt.plot(num_images_lst, avg_error, label=label, marker=fmt)
    plt.fill_between(num_images_lst, avg_error - std_error, avg_error + std_error, alpha=alpha)
    plt.plot(num_images_lst, avg_error - std_error, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=1)
    plt.plot(num_images_lst, avg_error + std_error, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=1)

def filter_errors(errors, threshold=100):
    """ 
    Errors are clipped to the threshold value- anything larger than threshold value simply becomes threshold value
    """    
    # count number of all errors larger than threshold in all errors
    errors_np = np.asarray(errors)
    total_count = np.sum(errors_np>threshold)
    # clip errors to threshold
    filtered = errors_np.clip(None, threshold)#[np.clip(err, None, threshold) for err in errors]
    return filtered, total_count


def plot_boxplots(num_images_lst, data_dict, shift, color, threshold=1.5, start_val=0, shift_y = 0):
    all_errors = data_dict['errors']
    for i, pos in enumerate(num_images_lst):
        filtered_errors, num_larger_than_threshold = filter_errors(all_errors[start_val:][i], threshold)

        bp = plt.boxplot(
            filtered_errors, positions=[pos+shift], widths=0.2, patch_artist=True, 
            showfliers=True, flierprops=dict(marker='o',  markeredgecolor=color, markersize=2, alpha=0.5), #markerfacecolor
            boxprops=dict(color=color), 
            capprops=dict(color=color), 
            whiskerprops=dict(color=color),
            medianprops=dict(color=color)
        )
        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color=color)
        
        # plot at the threshold value how many errors are larger than threshold
        plt.text(pos, threshold+shift_y, f'{num_larger_than_threshold}', color=color)
    #return num_larger_than_threshold


def main(analysis_pth = f'results/calibration_analysis/'): 
    # load all results
    
    endo = True
    rs = False


    #results_df = results_df.sort_values(by=['size_chess', 'camera'])
    endo_data_30, rs_data_30 = load_data(30, analysis_pth=analysis_pth)
    endo_data_25, rs_data_25 = load_data(25, analysis_pth=analysis_pth)
    endo_data_20, rs_data_20 = load_data(20, analysis_pth=analysis_pth)
    endo_data_15, rs_data_15 = load_data(15, analysis_pth=analysis_pth)

    start_val = 0
    num_images_start=5
    num_images_end=50
    num_images_step=2

    num_images_lst = np.arange(num_images_start, num_images_end, num_images_step)
    num_images_lst = num_images_lst[start_val:]


    if endo == True:
        threshold = 100
        """ # plot results
        plt.figure()
        #plt.plot(endo_data_30['average_error'][start_val:], label='30x30 endo')
        plt.errorbar(num_images_lst, endo_data_30['average_error'][start_val:], yerr=endo_data_30['std_error'][start_val:], label='30 endo', fmt='o', elinewidth=2)
        
        #plt.plot(endo_data_25['average_error'][start_val:], label='25x25 endo')
        plt.errorbar(num_images_lst, endo_data_25['average_error'][start_val:], yerr=endo_data_25['std_error'][start_val:], label='25 endo', fmt='*', elinewidth=1.5)
        
        #plt.plot(endo_data_20['average_error'][start_val:], label='20x20 endo')
        plt.errorbar(num_images_lst, endo_data_20['average_error'][start_val:], yerr=endo_data_20['std_error'][start_val:], label='20 endo', fmt='^', elinewidth=1)
    
        #plt.plot(endo_data_15['average_error'][start_val:], label='15x15 endo')
        plt.errorbar(num_images_lst, endo_data_15['average_error'][start_val:], yerr=endo_data_15['std_error'][start_val:], label='15 endo', fmt='s', elinewidth=0.5)
        plt.legend()
        plt.show() """
        
        shift = [0.3, 0.1]
        # Line plots with outliers for better visualization
        plt.figure(figsize=(12, 8))
        plot_boxplots(num_images_lst, endo_data_30, shift=-shift[0], color='blue', shift_y=0.01)
        plot_boxplots(num_images_lst, endo_data_25, shift=-shift[1], color='green',shift_y=0.03)
        plot_boxplots(num_images_lst, endo_data_20, shift=shift[1], color='red', shift_y=0.06)
        plot_boxplots(num_images_lst, endo_data_15, shift=shift[0], color='purple', shift_y=0.09)
        # add legends to the colors and add the legend to outside the plot on the bottom
        plt.legend(handles=[
            plt.Line2D([0], [0], color='blue', lw=4),
            plt.Line2D([0], [0], color='green', lw=4),
            plt.Line2D([0], [0], color='red', lw=4),
            plt.Line2D([0], [0], color='purple', lw=4)
        ], labels=['30', '25', '20', '15']
        , loc='lower center', ncol=4, fontsize='large')
        
        plt.xticks(num_images_lst, labels=[str(x) for x in num_images_lst]) 
        plt.title('Reprojection Error Distribution vs Number of Images (Endo)')
        plt.xlabel('Number of Images')
        plt.ylabel('Reprojection Error')
        #plt.ylim(None, 1.65)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 8))
        plot_with_shaded_error(num_images_lst, endo_data_30, '30 endo', 'o', threshold=threshold)
        plot_with_shaded_error(num_images_lst, endo_data_25, '25 endo', '*', threshold=threshold)
        plot_with_shaded_error(num_images_lst, endo_data_20, '20 endo', '^', threshold=threshold)
        plot_with_shaded_error(num_images_lst, endo_data_15, '15 endo', 's', threshold=threshold)
        plt.legend()
        plt.title('Reprojection Error vs Number of Images (Endo)')
        plt.xlabel('Number of Images')
        plt.ylabel('Reprojection Error')
        plt.grid(True)
        plt.show()
    if rs == True:
        """ plt.figure()

        #plt.plot(rs_data_30['average_error'][start_val:], label='30x30 realsense')
        plt.errorbar(num_images_lst, rs_data_30['average_error'][start_val:], yerr=rs_data_30['std_error'][start_val:], label='30 realsense', fmt='o')

        #plt.plot(rs_data_25['average_error'][start_val:], label='25x25 realsense')
        plt.errorbar(num_images_lst, rs_data_25['average_error'][start_val:], yerr=rs_data_25['std_error'][start_val:], label='25 realsense', fmt='*')

        #plt.plot(rs_data_20['average_error'][start_val:], label='20x20 realsense')
        plt.errorbar(num_images_lst, rs_data_20['average_error'][start_val:], yerr=rs_data_20['std_error'][start_val:], label='20 realsense', fmt='^')
    
        #plt.plot(rs_data_15['average_error'][start_val:], label='15x15 realsense')
        plt.errorbar(num_images_lst, rs_data_15['average_error'][start_val:], yerr=rs_data_15['std_error'][start_val:], label='15 realsense', fmt='s')

        plt.legend()
        plt.show() """
        
        shift = [0.3, 0.1]
        threshold = 10
        th_y = -4

        plt.figure(figsize=(12, 8))

        plot_with_shaded_error(num_images_lst, rs_data_30, '30 realsense', 'o', threshold=threshold)
        plot_with_shaded_error(num_images_lst, rs_data_25, '25 realsense', '*', threshold=threshold)
        plot_with_shaded_error(num_images_lst, rs_data_20, '20 realsense', '^', threshold=threshold)
        plot_with_shaded_error(num_images_lst, rs_data_15, '15 realsense', 's', threshold=threshold)
        
        plt.legend()
        plt.title('Reprojection Error vs Number of Images (Realsense)')
        plt.xlabel('Number of Images')
        plt.ylabel('Reprojection Error')
        plt.grid(True)
        plt.show()

        # Line plots with outliers for better visualization
        plt.figure(figsize=(12, 8))
        plot_boxplots(num_images_lst, rs_data_30, shift=-shift[0], color='blue', shift_y=th_y-0, threshold=threshold)
        plot_boxplots(num_images_lst, rs_data_25, shift=-shift[1], color='green',shift_y=th_y-3, threshold=threshold)
        plot_boxplots(num_images_lst, rs_data_20, shift=shift[1], color='red', shift_y=th_y-6, threshold=threshold)
        plot_boxplots(num_images_lst, rs_data_15, shift=shift[0], color='purple', shift_y=th_y-9, threshold=threshold)
        # add legends to the colors and add the legend to outside the plot on the bottom
        plt.legend(handles=[
            plt.Line2D([0], [0], color='blue', lw=4),
            plt.Line2D([0], [0], color='green', lw=4),
            plt.Line2D([0], [0], color='red', lw=4),
            plt.Line2D([0], [0], color='purple', lw=4)
        ], labels=['30', '25', '20', '15']
        , loc='lower center', ncol=4, fontsize='large')
        
        plt.xticks(num_images_lst, labels=[str(x) for x in num_images_lst]) 
        plt.title('Reprojection Error Distribution vs Number of Images (Realsense)')
        plt.xlabel('Number of Images')
        plt.ylabel('Reprojection Error')
        #plt.ylim(None, threshold)
        plt.grid(True)
        plt.show()

    return 


def plot_info_as_bar(info_pth):
    # load all info pkl files
    info_15_rs, info_15_endo = load_info(15, info_pth=info_pth)
    info_20_rs, info_20_endo = load_info(20, info_pth=info_pth)
    info_25_rs, info_25_endo = load_info(25, info_pth=info_pth)
    info_30_rs, info_30_endo = load_info(30, info_pth=info_pth)

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
    plt.bar('15', info_15_rs.data.values[1], label=info_15_rs.titles.values[1])   
    plt.bar('20', info_20_rs.data.values[1], label=info_20_rs.titles.values[1])
    plt.bar('25', info_25_rs.data.values[1], label=info_25_rs.titles.values[1])
    plt.bar('30', info_30_rs.data.values[1], label=info_30_rs.titles.values[1])

    #plt.bar(info_20_rs.titles.values[3], info_20_rs.data.values[3], label='20 realsense')

    # number of corners
    plt.subplot(1,3,3)
    plt.title(info_15_rs.titles.values[2])
    plt.bar('15', info_15_rs.data.values[2], label=info_15_rs.titles.values[2])   
    plt.bar('20', info_20_rs.data.values[2], label=info_20_rs.titles.values[2])
    plt.bar('25', info_25_rs.data.values[2], label=info_25_rs.titles.values[2])
    plt.bar('30', info_30_rs.data.values[2], label=info_30_rs.titles.values[2])

    #plt.bar(info_25_rs.titles.values[3], info_25_rs.data.values[3], label='25 realsense')

    #plt.subplot(1,4,4)

    #plt.bar(info_30_rs.titles.values[3], info_30_rs.data.values[3], label='30 realsense')
    #plt.bar(info_15_endo.data.values[0], info_15_endo.data.values[1],info_15_endo.data.values[2], label='15 realsense')
    
    # make all plots same axis in y axis
    plt.ylim(0,  info_15_rs.data.values[2].max()+100)
    plt.legend()
    plt.show()

if __name__=='__main__': 
    min_num_corners = None # if none selected, the percentage of corners is used (with min 6 corners)
    percentage_of_corners = 0.2

    reprojection_sample_size = 1000

    # analysis parameters
    repeats=50 # number of repeats per number of images analysis
    num_images_start=5
    num_images_end=50 
    num_images_step=2
    
    rec_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    rec_analysis = f'R{reprojection_sample_size}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}'

    main(analysis_pth = f'results/calibration_analysis/{rec_analysis}') 
    plot_info_as_bar(info_pth=f'results/raw_corner_data/{rec_data}')
