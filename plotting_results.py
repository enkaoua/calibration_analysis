import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(board_size, analysis_pth='results/calibration_error_data'):
    try:
        endo_data = pd.read_pickle(f'{analysis_pth}/{board_size}_endo_calibration_data.pkl')
    except:
        endo_data = None
    try:
        rs_data = pd.read_pickle(f'{analysis_pth}/{board_size}_realsense_calibration_data.pkl')
    except:
        rs_data = None
    return endo_data, rs_data

def load_info(board_size, info_pth='results/raw_corner_data'):
    try:
        info_rs = pd.read_csv(f'{info_pth}/{board_size}_realsense_corner_info.csv')
    except:
        info_rs = None
    try:
        info_endo = pd.read_csv(f'{info_pth}/{board_size}_endo_corner_info.csv')
    except:
        info_endo = None
    return info_endo, info_rs



def get_average_std(data, threshold=100):
    avg_lst = []
    std_lst = []
    Q1_lst = []
    Q3_lst = []
    if threshold is None:
        threshold = np.median(np.array(data))
    for errors in data:
        errors_np = np.array(errors)
    
        e = errors_np[errors_np<threshold]
        if e.size == 0:
            avg_lst.append(threshold)
            std_lst.append(threshold)
            Q1_lst.append(threshold)
            Q3_lst.append(threshold)
            continue
        avg_lst.append( np.percentile(e, 50) ) # np.mean(e)
        std_lst.append(np.std(e))
        Q1_lst.append( np.percentile(e, 25) )
        Q3_lst.append( np.percentile(e, 75) )

    return np.array(avg_lst), np.array(std_lst), Q1_lst, Q3_lst
    

def plot_with_shaded_error(num_images_lst, data_dict, label, fmt, alpha=0.3, start_val=0, threshold=50, param_to_plot='errors_lst'):
    #avg_error = data_dict['average_error'][start_val:]
    #std_error = data_dict['std_error'][start_val:]
    if param_to_plot=='errors_lst':
        errors = data_dict[param_to_plot][start_val:]
        
        avg_error,std_error, Q1, Q3= get_average_std(errors, threshold=threshold)
        plt.plot(num_images_lst, avg_error, label=label, marker=fmt)
        plt.fill_between(num_images_lst, Q1, Q3, alpha=alpha)
        plt.plot(num_images_lst, Q1, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=1)
        plt.plot(num_images_lst, Q3, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=1)
    else:
        intrinsics = data_dict['intrinsics'][start_val:]
        # splot intrinsics into fx, fy, cx, cy
        param_lst = []
        for intrinsics_i in intrinsics:
            param_rep_lst = []
            for repeat in intrinsics_i:
                if param_to_plot=='fx':
                    param_rep_lst.append(repeat[0,0])
                elif param_to_plot=='fy':
                    param_rep_lst.append(repeat[1,1])
                elif param_to_plot=='cx':
                    param_rep_lst.append(repeat[0,2])
                elif param_to_plot=='cy':
                    param_rep_lst.append(repeat[1,2])
            param_lst.append(param_rep_lst)

        # plot fx
        #names=['fx', 'fy', 'cx', 'cy']
        #for idx, value_to_plot in enumerate([fx_lst, fy_lst, cx_lst, cy_lst]):
        avg_error,std_error, Q1, Q3= get_average_std(param_lst, threshold=None)
        plt.plot(num_images_lst, avg_error, label=f'{param_to_plot} {label}', marker=fmt)
        plt.fill_between(num_images_lst, Q1, Q3, alpha=alpha)
        plt.plot(num_images_lst, Q1, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=1)
        plt.plot(num_images_lst, Q3, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=1)



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


def plot_boxplots(num_images_lst, data_dict, shift, color, threshold=1.5, start_val=0, shift_y = 0, param_to_plot='errors_lst'):
    all_errors = data_dict[param_to_plot]
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


def plot_all_boxplots(num_images_lst, data_30,data_25,data_20,data_15, shift=[0.3, 0.1],th_y=50, threshold=50, param_to_plot='errors_lst', cam='Realsense'):
    

    plt.figure(figsize=(12, 8))
    shift_percentage = threshold*0.05
    print(shift_percentage)

    plot_boxplots(num_images_lst, data_30, shift=-shift[0], color='blue', shift_y=th_y-0*shift_percentage, threshold=threshold, param_to_plot=param_to_plot)
    plot_boxplots(num_images_lst, data_25, shift=-shift[1], color='green',shift_y=th_y-1*shift_percentage, threshold=threshold, param_to_plot=param_to_plot)
    plot_boxplots(num_images_lst, data_20, shift=shift[1], color='red', shift_y=th_y-2*shift_percentage, threshold=threshold, param_to_plot=param_to_plot)
    plot_boxplots(num_images_lst, data_15, shift=shift[0], color='purple', shift_y=th_y-3*shift_percentage, threshold=threshold, param_to_plot=param_to_plot) 
    # add legends to the colors and add the legend to outside the plot on the bottom
    plt.legend(handles=[
        plt.Line2D([0], [0], color='blue', lw=4),
        plt.Line2D([0], [0], color='green', lw=4),
        plt.Line2D([0], [0], color='red', lw=4),
        plt.Line2D([0], [0], color='purple', lw=4)
    ], labels=['30', '25', '20', '15']
    , loc='lower center', ncol=4, fontsize='large')
    
    plt.xticks(num_images_lst, labels=[str(x) for x in num_images_lst]) 
    plt.title(f'Reprojection Error Distribution vs Number of Images ({cam})')
    plt.xlabel('Number of Images')
    plt.ylabel('Reprojection Error')
    plt.ylim(None, threshold)
    plt.grid(True)
    plt.show()


def plot_all_shaded_plots(num_images_lst,data_30, data_25, data_20, data_15, threshold=50, param_to_plot='error', cam='realsense'):
    plt.figure(figsize=(12, 8))

    plot_with_shaded_error(num_images_lst, data_30, f'30 {cam}', 'o', threshold=threshold, param_to_plot=param_to_plot)
    plot_with_shaded_error(num_images_lst, data_25, f'25 {cam}', '*', threshold=threshold, param_to_plot=param_to_plot)
    plot_with_shaded_error(num_images_lst, data_20, f'20 {cam}', '^', threshold=threshold, param_to_plot=param_to_plot)
    plot_with_shaded_error(num_images_lst, data_15, f'15 {cam}', 's', threshold=threshold, param_to_plot=param_to_plot)
    
    plt.legend()
    if param_to_plot=='errors_lst':
        plt.title(f'Reprojection Error vs Number of Images ({cam})')
        plt.ylabel('Reprojection Error')
    else:
        plt.title(f'{param_to_plot} vs Number of Images ({cam})')
        plt.ylabel(f'{param_to_plot}')
    plt.xlabel('Number of Images')
    plt.grid(True)
    plt.show()




def main(analysis_pth = f'results/calibration_analysis/'): 
    # load all results
    
    #results_df = results_df.sort_values(by=['size_chess', 'camera'])
    endo_data_30, rs_data_30 = load_data(30, analysis_pth=analysis_pth)
    endo_data_25, rs_data_25 = load_data(25, analysis_pth=analysis_pth)
    endo_data_20, rs_data_20 = load_data(20, analysis_pth=analysis_pth)
    endo_data_15, rs_data_15 = load_data(15, analysis_pth=analysis_pth)

    start_val = 0
    th_y = 0-threshold*0.02

    #num_images_lst = np.arange(num_images_start, num_images_end, num_images_step)
    #num_images_lst = num_images_lst[start_val:]
    num_images_lst = endo_data_30['num_images_lst'].values[start_val:]


    if endo == True:
        # plot shaded plots
        plot_all_shaded_plots(num_images_lst, endo_data_30, endo_data_25, endo_data_20, endo_data_15, threshold=threshold, param_to_plot='errors_lst', cam='endo')
        # Line plots with outliers for better visualization
        plot_all_boxplots(num_images_lst, endo_data_30, endo_data_25, endo_data_20, endo_data_15, shift=shift, th_y=th_y, threshold=threshold, param_to_plot='errors_lst', cam='endo')
        # plot intrinsics params in subplots
        if hand_eye != True:
            for param in ['fx', 'fy', 'cx', 'cy']:
                plot_all_shaded_plots(num_images_lst, endo_data_30, endo_data_25, endo_data_20, endo_data_15, threshold=threshold, param_to_plot=param, cam='endo')
        #plot_all_shaded_plots(num_images_lst, endo_data_30, endo_data_25, endo_data_20, endo_data_15, threshold=threshold, param_to_plot='fx', cam='endo')


    if rs == True:

        # plot shaded plots
        plot_all_shaded_plots(num_images_lst,rs_data_30,rs_data_25,rs_data_20,rs_data_15, threshold=threshold, param_to_plot='errors_lst', cam='Realsense')
        # Line plots with outliers for better visualization
        plot_all_boxplots(num_images_lst, rs_data_30,rs_data_25,rs_data_20,rs_data_15, shift=shift,th_y=th_y, threshold=threshold, param_to_plot='errors_lst', cam='Realsense')
        print('----------------------------------')
        print('--->',rs_data_30.columns)
        if hand_eye != True:
            for param in ['fx', 'fy', 'cx', 'cy']:
                plot_all_shaded_plots(num_images_lst,rs_data_30,rs_data_25,rs_data_20,rs_data_15, threshold=threshold, param_to_plot=param, cam='Realsense')
        #plot_all_shaded_plots(num_images_lst,rs_data_30,rs_data_25,rs_data_20,rs_data_15, threshold=threshold, param_to_plot='fx', cam='Realsense' )


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
    plt.bar('15', info_15_rs.data.values[0])#, label=info_30_rs.titles.values[0])   
    plt.bar('20', info_20_rs.data.values[0])#, label=info_30_rs.titles.values[0])
    plt.bar('25', info_25_rs.data.values[0])#, label=info_30_rs.titles.values[0])
    plt.bar('30', info_30_rs.data.values[0])#, label=info_30_rs.titles.values[0])
    plt.xlabel('Board Size')

    # plotting each board as separate cluster
    # subplot 2
    plt.subplot(1,3,2)
    plt.title(info_15_rs.titles.values[1])
    plt.bar('15', info_15_rs.data.values[1])#, label=info_15_rs.titles.values[1])   
    plt.bar('20', info_20_rs.data.values[1])#, label=info_20_rs.titles.values[1])
    plt.bar('25', info_25_rs.data.values[1])#, label=info_25_rs.titles.values[1])
    plt.bar('30', info_30_rs.data.values[1])#, label=info_30_rs.titles.values[1])

    # number of corners
    plt.subplot(1,3,3)
    plt.title(info_15_rs.titles.values[2])
    plt.bar('15', info_15_rs.data.values[2])#, label=info_15_rs.titles.values[2])   
    plt.bar('20', info_20_rs.data.values[2])#, label=info_20_rs.titles.values[2])
    plt.bar('25', info_25_rs.data.values[2])#, label=info_25_rs.titles.values[2])
    plt.bar('30', info_30_rs.data.values[2])#, label=info_30_rs.titles.values[2])

    # make all plots same axis in y axis
    #plt.ylim(0,  info_15_rs.data.values[2].max()+100)
    plt.legend()
    plt.show()



if __name__=='__main__': 
    hand_eye = True

    if hand_eye == True:
        calibration_pth = 'results/hand_eye'
        
        min_num_corners = 6.0 # if none selected, the percentage of corners is used (with min 6 corners)
        percentage_of_corners = 0.2
        threshold = 30
        # analysis parameters
        R = None
        repeats=1000 # number of repeats per number of images analysis
        num_images_start=5
        num_images_end=60 
        num_images_step=5
        endo = True
        rs = False
        shift = [0.3, 0.1]

    else:
        calibration_pth = 'results/intrinsics'
        min_num_corners = 6
        percentage_of_corners = 0.5
        threshold = 2

        # analysis parameters
        R = 100
        num_images_start=5
        num_images_end=60 
        num_images_step=5
        repeats=10 # number of repeats per number of images analysis
        endo = True
        rs = True
        shift = [0.3, 0.1]

    rec_data = f'MC_{min_num_corners}_PC_{percentage_of_corners}'
    rec_analysis = f'R{R}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}_{rec_data}'

    main(analysis_pth = f'{calibration_pth}/calibration_analysis/{rec_analysis}') 
    #plot_info_as_bar(info_pth=f'{calibration_pth}/raw_corner_data/{rec_data}')


