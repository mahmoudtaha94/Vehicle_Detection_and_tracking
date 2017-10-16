from sys import exit
import numpy as np
import cv2
'''
this file is step 2 (color/gradient thresholding):
1-extracts color channels
2-get the desired gradient from a channel
3-make binary image from a channel
4-combine 2 binary images using AND or OR logics
5- stack 2 channel ontop of eachother
'''

def getChannel(img,mode='gray',channel=None): 
    result=np.zeros_like(img)
    if mode=='gray':
        '''
        gray
        '''
        result = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif mode=='RGB':
        '''
        RGB 
        '''
        if channel=='R':
            result= img[:,:,0]
        elif channel=='G':
            result= img[:,:,1]
        elif channel=='B':
            result= img[:,:,2]
    elif mode=='HLS':
        '''
        HLS
        '''
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if channel=='H':
            result= hls[:,:,0]
        elif channel=='L':
            result= hls[:,:,1]
        elif channel=='S':
            result= hls[:,:,2]
    elif mode=='HSV':
        '''
        HSV 
        '''
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if channel=='H':
            result= hsv[:,:,0]
        elif channel=='S':
            result= hsv[:,:,1]
        elif channel=='V':
            result= hsv[:,:,2]
    if result.sum()==0:
        print('Error:getChannel() needs a matching mode and channel')
        exit()
    return result


def getGradient(channel,mode='x',sobel_kernel=3):
    sobelx=cv2.Sobel(channel,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely=cv2.Sobel(channel,cv2.CV_64F,0,1,ksize=sobel_kernel)
    if mode=='x':
        abs_sobelx=np.absolute(sobelx)
        sobel=np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    elif mode=='y':
        abs_sobely=np.absolute(sobely)
        sobel=np.uint8(255*abs_sobely/np.max(abs_sobely))
    elif mode=='mag':
        magnitude=np.sqrt((sobelx)**2+(sobely)**2)
        sobel=np.uint8(255*magnitude/np.max(magnitude))
    elif mode=='dir':
        abs_sobelx=np.absolute(sobelx)
        abs_sobely=np.absolute(sobely)
        sobel=np.arctan2(abs_sobely, abs_sobelx)
    return sobel


def applyBinary(channel,thresh=(0,255)):
    binary=np.zeros_like(channel)
    binary[(channel>=thresh[0])&(channel<=thresh[1])]=1
    return binary

def combineAND(channel1,channel2):
    if (channel1.shape[0]==channel2.shape[0])&(channel1.shape[1]==channel2.shape[1]):
        combined=np.zeros_like(channel1)
        combined[(channel1==1)&(channel2==1)]=1
    else:
        print('Error: combineAND() needs 2 arguments of the same shape')
        exit()
    return combined

def combineOR(channel1,channel2):
    if (channel1.shape[0]==channel2.shape[0])&(channel1.shape[1]==channel2.shape[1]):
        combined=np.zeros_like(channel1)
        combined[(channel1==1)|(channel2==1)]=1
    else:
        print('Error: combineOR() needs 2 arguments of the same shape')
        exit()
    return combined

def stack2Channels(channel1,channel2):
    if (channel1.shape[0]==channel2.shape[0])&(channel1.shape[1]==channel2.shape[1]):
        stacked=np.dstack( (np.zeros_like(channel1),channel1, channel2))
    else:
        print('Error stack2Channels() need 2 arguments of the same shape')
        exit()
    return stacked
