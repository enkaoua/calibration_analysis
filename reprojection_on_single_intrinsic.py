

from charuco_utils import calculate_reprojection_error
import pandas as pd
import numpy as np

def main(reprojection_data_pth='results/intrinsics/split_data/RNone_MC_6.0_PC_0.5',board=15,cam='realsense',visualise_reprojection_error=True, waitTime=0, calibration_pth='results/intrinsics/best_intrinsics'): 
    
    # load calibration data
    reprojection_data = pd.read_pickle(f'{reprojection_data_pth}/{board}_{cam}_corner_data_reprojection_dataset.pkl')
    # load intrinsics
    mtx = np.loadtxt(f'{calibration_pth}/{board}_{cam}_intrinsics.txt')
    dist = np.loadtxt(f'{calibration_pth}/{board}_{cam}_distortion.txt')
    
    # calculate reprojection error with these values
    objPoints_reprojection = reprojection_data.objPoints.values
    imgPoints_reprojection = reprojection_data.imgPoints.values
    if visualise_reprojection_error:
        image_paths = reprojection_data.paths.values
    else:
        image_paths = None
    err = calculate_reprojection_error(mtx, dist, objPoints_reprojection, imgPoints_reprojection,
                                       image_pths=image_paths, waitTime=waitTime)
    print(err)
    return 


if __name__=='__main__': 
    main() 