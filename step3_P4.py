from sys import exit
import numpy as np
import cv2
'''
this file is step 3 (change perspective):
1- change perspective
'''
def changePerspective(img,trapizoid,offset_pct,inverse=False):
    '''
    trapizoid_pct is a dictionary of the 4 points of the defined trapizoid:
        1-first : top left point
        2-second: top right point
        3-third : bottom right point
        4-forth : bottom left point
    offset_pct: offset from the edges of the wraped image
    '''
    if len(trapizoid)<4:
        print('Error: changePerspective needs atleast 4 source points')
        exit()
    img_size=(img.shape[1],img.shape[0])
    src=np.float32([[trapizoid['first'][0],trapizoid['first'][1]],
                    [trapizoid['second'][0],trapizoid['second'][1]],
                    [trapizoid['third'][0],trapizoid['third'][1]],
                    [trapizoid['forth'][0],trapizoid['forth'][1]]])
    offset=img_size[0]*offset_pct
    dst=np.float32([[offset,0],
                    [img_size[0]-offset,0],
                    [img_size[0]-offset,img_size[1]],
                    [offset,img_size[1]]])
    
    # Warp the image using OpenCV warpPerspective()
    if inverse:
        M = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size,flags=cv2.INTER_LINEAR)
    return warped