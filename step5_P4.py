import numpy as np
'''
this file is step 5 (calculate the real curvature radius):
1- calculate the curvature radius
2- calculate the difference from the center of the lane
'''
def curverad(xm_per_pix,ym_per_pix,res_yvals,yvals,midx):
    
    curve_fit_cr=np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,np.array(midx,np.float32)*xm_per_pix,2)
    curverad=((1+(2*curve_fit_cr[0]*yvals[-1]*ym_per_pix+curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0])
    return curverad

def centerDiff(warped,left_fitx,right_fitx,xm_per_pix):
    camera_center=(left_fitx[-1]+right_fitx[-1])/2
    center_diff=(camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos='left'
    if center_diff<=0:
        side_pos='right'
    return center_diff,side_pos
    