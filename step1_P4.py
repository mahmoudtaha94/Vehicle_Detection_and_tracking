import numpy as np
import cv2
import matplotlib.image as mpimg
import glob
'''
this file is step 1 (camera calibration and image distortion correction):
1- calibrate the camera
2- undistort the images
'''

def calibrate():
    images=glob.glob('./camera_cal/calibration*.jpg')
    objpoints=[]
    imgpoints=[]
    objp=np.zeros((6*9,3),np.float32)
    objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
    for fname in images:
        img=mpimg.imread(fname)
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,(9,6),None)
        if ret==True:
            imgpoints.append(corners)
            objpoints.append(objp)
    ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
    print('finished calibration')
    return mtx,dist


def undistort(img,mtx,dist):
    undistorted=cv2.undistort(img,mtx,dist,None,mtx)
    return undistorted

