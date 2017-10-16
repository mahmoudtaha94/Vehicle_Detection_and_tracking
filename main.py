import numpy as np 
from scipy.ndimage.measurements import label
from Functions import *
import pickle


from P4 import pipeline
from step1_P4 import calibrate,undistort
'''
the main code
'''
#define feaature parameters
ProjectParameters=pickle.load( open("ProjectParameters.p", "rb" ) )
svc=ProjectParameters['SVCmodel']
X_scaler=ProjectParameters['X_scaler']
ColorSpace=ProjectParameters['ColorSpace']
orient=ProjectParameters['orient']
PixPerCell=ProjectParameters['PixPerCell']
CellPerBlock=ProjectParameters['CellPerBlock']
HogChannel=ProjectParameters['HogChannel']
SpatialSize=ProjectParameters['SpatialSize']
HistBins=ProjectParameters['HistBins']
SpatialFeatures=ProjectParameters['SpatialFeatures']
HistFeatures=ProjectParameters['HistFeatures']
last_frame=[]

'''
calibrate once
'''
mtx,dist=calibrate()
def process_image(img):
	global last_frame
	'''
	Project 4
	'''
	P4_result=pipeline(img,mtx,dist)
	'''
	Project 5
	'''
	ystart=400
	ystop=660
	scale1=1
	scale2=1.5
	scale3=1.7
	scale4=2
	heatmap=find_cars(img, ystart, ystop, scale1, svc, X_scaler, orient, PixPerCell, CellPerBlock, SpatialSize, HistBins)
	heatmap+=find_cars(img, ystart, ystop, scale2, svc, X_scaler, orient, PixPerCell, CellPerBlock, SpatialSize, HistBins)
	heatmap+=find_cars(img, ystart, ystop, scale3, svc, X_scaler, orient, PixPerCell, CellPerBlock, SpatialSize, HistBins)
	heatmap+=find_cars(img, ystart, ystop, scale4, svc, X_scaler, orient, PixPerCell, CellPerBlock, SpatialSize, HistBins)
	heatmap=apply_threshold(heatmap,4)
	labels=label(heatmap)
	#draw bounding box
	draw_img,last_frame=draw_labeled_bboxes(P4_result,labels,last_frame=last_frame)
	return draw_img

from moviepy.editor import VideoFileClip
test_output='test2.mp4'
clip=VideoFileClip('project_video.mp4')
#clip=VideoFileClip('test_video.mp4')
test_clip=clip.fl_image(process_image)
test_clip.write_videofile(test_output,audio=False)











