from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import glob


def load_all_analysis_data(camera, analysis_data_pth):

    all_analysis_data_df = pd.DataFrame()
    for data_pth in glob.glob(f'{analysis_data_pth}/*_{camera}_calibration_data.pkl'):
        data_df = pd.read_pickle(data_pth)
        data_df['chess_size'] = int(data_pth.split('/')[-1].split('_')[0])
        all_analysis_data_df = pd.concat([all_analysis_data_df, data_df], ignore_index=True)
    #pd.concat([pd.read_pickle(data_pth)['chess_size']='' for data_pth in glob.glob(f'{analysis_data_pth}/*_{camera}_calibration_data.pkl')], ignore_index=True)
    return all_analysis_data_df

""" 
def plot_with_shaded_error(num_images_lst, data_dict, label, fmt, alpha=0.3, start_val=0, threshold=50, param_to_plot='errors_lst'):
    #avg_error = data_dict['average_error'][start_val:]
    #std_error = data_dict['std_error'][start_val:]
    if param_to_plot=='errors_lst':
        #errors = data_dict[param_to_plot][start_val:]
        
        #avg_error,std_error, Q1, Q3= get_average_std(errors, threshold=threshold)
        avg_error = data_dict
        plt.plot(num_images_lst, avg_error, label=label, marker=fmt)
        plt.fill_between(num_images_lst, Q1, Q3, alpha=alpha)
        plt.plot(num_images_lst, Q1, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=1)
        plt.plot(num_images_lst, Q3, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=1)
    else:
        intrinsics = data_dict[param_to_plot][start_val:]
        # splot intrinsics into fx, fy, cx, cy
        fx_lst = []
        fy_lst = []
        cx_lst = []
        cy_lst = []
        for intrinsics_i in intrinsics:
            fx_lst.append(intrinsics_i[0])
            fy_lst.append(intrinsics_i[1])
            cx_lst.append(intrinsics_i[2])
            cy_lst.append(intrinsics_i[3])

        avg_error,std_error, Q1, Q3= get_average_std(fx_lst, threshold=threshold)
        plt.plot(num_images_lst, avg_error, label=label, marker=fmt)
        plt.fill_between(num_images_lst, Q1, Q3, alpha=alpha)
        plt.plot(num_images_lst, Q1, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=1)
        plt.plot(num_images_lst, Q3, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=1)
 """

def plot_main(cameras=['endo', 'realsense'],
              chess_sizes=[15,20, 25, 30],
              calibration_pth = 'results/intrinsics',
                # analysis parameters,
                R = 100,
                num_images_start=5,
                num_images_end=60 ,
                num_images_step=5,
                repeats=10,
                min_corners=6.0,
                percentage_of_corners=0.2
              ):
    
    rec_analysis = f'R{R}_N{num_images_start}_{num_images_end}_{num_images_step}_repeats_{repeats}_MC_{min_corners}_PC_{percentage_of_corners}'
    analysis_data_pth = f'{calibration_pth}/calibration_analysis/{rec_analysis}'
    for camera in tqdm(cameras, desc='cameras'):
        # merge all filtered_datasets into one large dataset
        all_analysis_data_df = load_all_analysis_data(camera, analysis_data_pth)
            
        plt.figure(figsize=(12, 8))

        for size_chess in tqdm(chess_sizes, desc='chess_sizes', leave=True):
            
            #data_df = pd.read_pickle(f'{filtered_table_pth}/{size_chess}_{camera}_corner_data.pkl')
            data_df = all_analysis_data_df[all_analysis_data_df['chess_size']==size_chess]
            # expand data so that all repeats are in new rows
            data_df = data_df.explode(['errors_lst', 'num_corners_detected_lst', 'intrinsics', 'distortion']).reset_index()
            data_df['num_corners_detected_lst'] = data_df['num_corners_detected_lst'].astype(int)
            data_df['errors_lst'] = data_df['errors_lst'].astype(float)

            # filter corners larger than 1000
            #data_df = data_df[data_df['num_corners_detected_lst']<1000]

            corners = data_df['num_corners_detected_lst'].values
            errors = data_df['errors_lst'].values
            plt.plot(corners, errors, label=f'{size_chess}mm', marker='o', alpha=0.5)
        plt.ylim(0, 2)
        plt.legend()
        plt.title('Reprojection Error vs Number of Images (Realsense)')
        plt.xlabel('Number of Images')
        plt.ylabel('Reprojection Error')
        plt.grid(True)
        plt.show()


if __name__=='__main__': 
    plot_main(
        cameras=['endo', 'realsense'],
              chess_sizes=[15,20, 25, 30],
              calibration_pth = 'results/intrinsics',
                # analysis parameters,
                R = 100,
                num_images_start=5,
                num_images_end=60 ,
                num_images_step=5,
                repeats=10,
                min_corners=6,
                percentage_of_corners=0.5
    ) 