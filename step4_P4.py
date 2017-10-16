import numpy as np
import cv2
from tracker import tracker
from step3_P4 import changePerspective
'''
this file is step 4 (extract the lane lines from a warped image):
1- find lane lines
2- draw the lane lines on the video frames
'''


def findLanes(warped,My_ym, My_xm,window_width=25,window_height=80):
    #setup the overall class to do all the tracking
    curve_centers=tracker(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin=25, My_ym=My_ym, My_xm=My_xm, Mysmooth_factor=15)
    
    window_centroids=curve_centers.find_window_centroids(warped)
    
    #points used to find the left and right lanes
    rightx=[]
    leftx=[]
    
    #go through each level and draw the windows
    for level in range(0,len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        
    leftx=np.array(leftx,dtype=np.int)
    rightx=np.array(rightx,dtype=np.int)
    midx=(leftx+rightx)/2
    return leftx,rightx,midx

def drawLanes(img,warped,leftx,rightx,midx,trapizoid_pct,window_width=25,window_height=80):
    #fit the lane boundries to the left, right, and center position found
    '''
    res_yvals to calculate the polynomial coofs
    yvals to make the lanes continuous 
    '''
    yvals=range(0,warped.shape[0])
    
    res_yvals=np.arange(warped.shape[0]-(window_height/2),0,-window_height)
    
    left_fit=np.polyfit(res_yvals,leftx,2)
    left_fitx=left_fit[0]*yvals*yvals+left_fit[1]*yvals+left_fit[2]
    left_fitx=np.array(left_fitx,np.int32)
    
    right_fit=np.polyfit(res_yvals,rightx,2)
    right_fitx=right_fit[0]*yvals*yvals+right_fit[1]*yvals+right_fit[2]
    right_fitx=np.array(right_fitx,np.int32)
    
    mid_fit=np.polyfit(res_yvals,midx,2)
    mid_fitx=mid_fit[0]*yvals*yvals+mid_fit[1]*yvals+mid_fit[2]
    mid_fitx=np.array(mid_fitx,np.int32)
    
    left_lane=np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane=np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    inner_lane=np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)    
    mid_lane=np.array(list(zip(np.concatenate((mid_fitx-window_width/2,mid_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    
    road=np.zeros_like(img)
    road_bkg=np.zeros_like(img)
    cv2.fillPoly(road,[left_lane],color=[255,0,0])
    cv2.fillPoly(road,[right_lane],color=[0,0,255])
    cv2.fillPoly(road,[mid_lane],color=[165,42,42])#brown color
    cv2.fillPoly(road,[inner_lane],color=[0,255,0])
    cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
    cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])
    cv2.fillPoly(road_bkg,[mid_lane],color=[255,255,255])
    
    road_warped=changePerspective(road,trapizoid_pct,0.25,inverse=True)
    road_warped_bkg=changePerspective(road_bkg,trapizoid_pct,0.25,inverse=True)
    
    base=cv2.addWeighted(img,1.0,road_warped_bkg,-1.0,0.0)
    result=cv2.addWeighted(base,1.0,road_warped,0.7,0.0)
    return result,res_yvals,yvals,left_fitx,right_fitx