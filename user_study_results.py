

import glob

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def main(): 
    # load reprojection errors
    data_pth = '/Users/aure/Documents/CARES/code/charuco_calibration_analysis/results/user_study/study2'
    
    # initialise dict
    reprojection_errors_endo_dict = {
        'matt': [],
        'joao': [],
        'mobarak': [],
        'aure': [],
    }
    reprojection_errors_rs_dict = {
        'matt': [],
        'joao': [],
        'mobarak': [],
        'aure': [],
    }
    reprojection_errors_he_dict = {
        'matt': [],
        'joao': [],
        'mobarak': [],
        'aure': [],
    }

    for name in reprojection_errors_endo_dict.keys():
        reprojection_errors_endo = glob.glob(f'{data_pth}/{name}/[0-6]/calibration/err_endo.txt')
        reprojection_errors_rs = glob.glob(f'{data_pth}/{name}/[0-6]/calibration/err_rs.txt')
        reprojection_errors_he = glob.glob(f'{data_pth}/{name}/[0-6]/calibration/err_he_calib.txt')

        # load errors
        reprojection_errors_endo = [np.loadtxt(f) for f in reprojection_errors_endo]
        reprojection_errors_rs = [np.loadtxt(f) for f in reprojection_errors_rs]
        reprojection_errors_he = [np.loadtxt(f) for f in reprojection_errors_he]

        # append to dict
        reprojection_errors_endo_dict[name] = reprojection_errors_endo
        reprojection_errors_rs_dict[name] = reprojection_errors_rs
        reprojection_errors_he_dict[name] = reprojection_errors_he
        
    # convert to df where rows are the different participants and columns are the different images
    
    reprojection_errors_endo_df = pd.DataFrame(reprojection_errors_endo_dict)
    reprojection_errors_rs_df = pd.DataFrame(reprojection_errors_rs_dict)
    reprojection_errors_he_df = pd.DataFrame(reprojection_errors_he_dict)

    # convert rows to columns snd columns to rows
    reprojection_errors_endo_df = reprojection_errors_endo_df
    reprojection_errors_rs_df = reprojection_errors_rs_df
    reprojection_errors_he_df = reprojection_errors_he_df

    # plot boxplots with participants as each box
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].boxplot(reprojection_errors_endo_df.values)
    ax[0].set_ylabel('reprojection error (pixels)')

    ax[0].set_title('endoscope intrinsic calibration')
    ax[1].boxplot(reprojection_errors_rs_df.values)
    ax[1].set_title('realsense intrinsic calibration')
    ax[2].boxplot(reprojection_errors_he_df.values)
    ax[2].set_title('eye-eye')
    # x label for all plots is participant (cmmon x lavel)

    plt.show()

    # plot errors as bars with std
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].bar(height=reprojection_errors_endo_df.mean(axis=1),x=reprojection_errors_endo_df.index, label='endo')
    # plot min and max error bars
    y_min = reprojection_errors_endo_df.min(axis=1)
    # convert to list
    y_min = [float(y) for y in y_min]
    y_max = reprojection_errors_endo_df.max(axis=1)
    y_max = [float(y) for y in y_max]
    y_err = [y_min, y_max] #reprojection_errors_endo_df.mean(axis=1)
    ax[0].errorbar(y=y_err, x=reprojection_errors_endo_df.index, yerr=reprojection_errors_endo_df.std(axis=1), fmt='o', label='endo', color='red')
    #ax[0].errorbar(y=y_err, x=reprojection_errors_endo_df.index, yerr=reprojection_errors_endo_df.std(axis=1), fmt='o', label='endo', color='red')

    ax[1].bar(height=reprojection_errors_rs_df.mean(axis=1), x=reprojection_errors_rs_df.index, label='rs')
    y_min = reprojection_errors_rs_df.min(axis=1)
    y_max = reprojection_errors_rs_df.max(axis=1)
    y_err = [y_min, y_max]
    ax[1].errorbar(y=reprojection_errors_rs_df.mean(axis=1), x=reprojection_errors_rs_df.index, yerr=reprojection_errors_rs_df.std(axis=1), fmt='o', label='rs', color='red')

    ax[2].bar(height=reprojection_errors_he_df.mean(axis=1), x=reprojection_errors_he_df.index, label='he')
    y_min = reprojection_errors_he_df.min(axis=1)
    y_max = reprojection_errors_he_df.max(axis=1)
    y_err = [y_min, y_max]
    ax[2].errorbar(y=reprojection_errors_he_df.mean(axis=1), x=reprojection_errors_he_df.index, yerr=reprojection_errors_he_df.std(axis=1), fmt='o', label='he', color='red')


    ax[0].set_title('Mean reprojection error')
    ax[0].set_xlabel('Image')
    ax[0].set_ylabel('Reprojection error')
    ax[0].legend()
    plt.show()
    """ ax[1].plot(reprojection_errors_endo.median(axis=1), label='endo')
    ax[1].plot(reprojection_errors_rs.median(axis=1), label='rs')
    ax[1].plot(reprojection_errors_he.median(axis=1), label='he')
    ax[1].set_title('Median reprojection error')
    ax[1].set_xlabel('Image')
    ax[1].set_ylabel('Reprojection error') """

    
    

    # load calibration txt files



    return 


if __name__=='__main__': 
    main() 