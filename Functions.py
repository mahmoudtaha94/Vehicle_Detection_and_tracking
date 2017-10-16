import numpy as np 
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg

'''
this file contains all the functions used in project 5
'''

'''
define a function to just convert colors
'''
def ConvertColor(img,conv='RGB2YCrCb'):
	if conv=='RGB2YCrCb':
		return cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
	elif conv=='BGR2YCrCb':
		return cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
	elif conv=='RGB2LUV':
		return cv2.cvtColor(img,cv2.COLOR_RGB2LUV)
	elif conv=='RGB2HSV':
		return cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
	elif conv=='RGB2HLS':
		return cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
	elif conv=='RGB2YUV':
		return cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

def GetHogFeatures(img,orient,PixPerCell,CellPerBlock,vis=False,featureVec=True):
	if vis:
		features,hogImage=hog(img,orientations=orient,pixels_per_cell=(PixPerCell,PixPerCell),cells_per_block=(CellPerBlock,CellPerBlock),transform_sqrt=False,visualise=vis,feature_vector=featureVec)
		return features,hogImage
	else:
		features=hog(img,orientations=orient,pixels_per_cell=(PixPerCell,PixPerCell),cells_per_block=(CellPerBlock,CellPerBlock),transform_sqrt=False,visualise=vis,feature_vector=featureVec)
		return features
'''
define a function to compute binned color features
cv2.resize(img,size).ravel() creats the feature vector
'''
def BinSpatial(img,size=(32,32)):
	color1=cv2.resize(img[:,:,0],size).ravel()
	color2=cv2.resize(img[:,:,1],size).ravel()
	color3=cv2.resize(img[:,:,2],size).ravel()
	return np.hstack((color1,color2,color3))
'''
define a color histogram features
'''
def ColorHist(img,nbins=32):
	channel1Hist=np.histogram(img[:,:,0],bins=nbins)
	channel2Hist=np.histogram(img[:,:,1],bins=nbins)
	channel3Hist=np.histogram(img[:,:,2],bins=nbins)
	#concatenate the histograms into one big feature vector
	HistFeatures=np.concatenate((channel1Hist[0],channel2Hist[0],channel3Hist[0]))
	return HistFeatures

def extractFeatures(imgs, ColorSpace='RGB',SpatialSize=(32,32),HistBins=32,
					orient=9,PixPerCell=8,CellPerBlock=2,HogChannel=0,SpatialFeatures=True,
					HistFeatures=True,HogFeatures=True):
	#creat a list to append the features to
	features=[]
	for file in imgs:
		fileFeatures=[]
		image=mpimg.imread(file)
		#apply color conversions if other than RGB
		if ColorSpace!='RGB':
			featureImage=ConvertColor(image,conv=('RGB2'+ ColorSpace))
		else:
			featureImage=np.copy(image)

		if SpatialFeatures:
			spatialFeat=BinSpatial(featureImage,size=SpatialSize)
			fileFeatures.append(spatialFeat)
		if HistFeatures:
			histFeat=ColorHist(featureImage,nbins=HistBins)
			fileFeatures.append(histFeat)
		if HogFeatures:
			if HogChannel=='ALL':
				hogFeat=[]
				for channel in range(featureImage.shape[2]):
					hogFeat.append(GetHogFeatures(featureImage[:,:,channel],orient,PixPerCell,CellPerBlock,
									vis=False,featureVec=False))
				hogFeat=np.ravel(hogFeat)
			else:
				hogFeat=GetHogFeatures(featureImage[:,:,HogChannel],orient,PixPerCell,CellPerBlock,
								vis=False,featureVec=False)
			fileFeatures.append(hogFeat)
		features.append(np.concatenate(fileFeatures))
	return features



def find_cars(img, yStart, yStop, scale, svc, X_scaler, orient, PixPerCell, CellPerBlock, SpatialSize, HistBins):
	draw_img=np.copy(img)
	#make a heat map of zeros
	heatmap=np.zeros_like(img[:,:,0])
	img=img.astype(np.float32)/255

	img_tosearch= img[yStart:yStop,:,:]
	ctrans_tosearch=ConvertColor(img_tosearch,conv='RGB2YCrCb')
	if scale!=1:
		imshape=ctrans_tosearch.shape
		ctrans_tosearch=cv2.resize(ctrans_tosearch,(np.int(imshape[1]/scale),np.int(imshape[0]/scale)))
		
	ch1=ctrans_tosearch[:,:,0]
	ch2=ctrans_tosearch[:,:,1]
	ch3=ctrans_tosearch[:,:,2]

	#define blocks and steps
	nxblocks=(ch1.shape[1]// PixPerCell)-1
	nyblocks=(ch1.shape[0]// PixPerCell)-1
	window=64
	nblocks_per_window=(window//PixPerCell)-1
	cells_per_step=2 #instead of overlap this is just how many cells per step
	nxsteps=(nxblocks- nblocks_per_window)// cells_per_step
	nysteps=(nyblocks- nblocks_per_window)// cells_per_step

	'''
	compute hog features for the entire image
	'''
	hog1=GetHogFeatures(ch1,orient,PixPerCell,CellPerBlock,vis=False,featureVec=False)
	hog2=GetHogFeatures(ch2,orient,PixPerCell,CellPerBlock,vis=False,featureVec=False)
	hog3=GetHogFeatures(ch3,orient,PixPerCell,CellPerBlock,vis=False,featureVec=False)

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos=yb*cells_per_step	
			xpos=xb*cells_per_step
			#extract HOG for this patch
			hog_feat1=hog1[ypos:ypos+nblocks_per_window,xpos:xpos+nblocks_per_window].ravel()
			hog_feat2=hog2[ypos:ypos+nblocks_per_window,xpos:xpos+nblocks_per_window].ravel()
			hog_feat3=hog3[ypos:ypos+nblocks_per_window,xpos:xpos+nblocks_per_window].ravel()
			hog_features=np.hstack((hog_feat1,hog_feat2,hog_feat3))

			xleft=xpos*PixPerCell
			ytop=ypos*PixPerCell

			#extract the image patch
			subimg=cv2.resize(ctrans_tosearch[ytop:ytop+window,xleft:xleft+window],(64,64))

			#get color featres
			spatial_features=BinSpatial(subimg,size=SpatialSize)
			hist_features=ColorHist(subimg,nbins=HistBins)

			#scale features and make a prediction
			test_features=X_scaler.transform(np.hstack((spatial_features,hist_features,hog_features)))
			test_prediction=svc.predict(test_features)

			if test_prediction==1:
				xbox_left=np.int(xleft*scale)
				ytop_draw=np.int(ytop*scale)
				win_draw=np.int(window*scale)
				cv2.rectangle(draw_img,(xbox_left,ytop_draw+yStart),(xbox_left+win_draw,ytop_draw+win_draw+yStart),(0,0,255),6)
				heatmap[ytop_draw+yStart:ytop_draw+win_draw+yStart,xbox_left:xbox_left+win_draw]+=1

	return heatmap

def apply_threshold(heatmap,threshold):
	heatmap[heatmap<=threshold]=0
	return heatmap

def draw_labeled_bboxes(img,labels,last_frame=[]):
	#iterate through all detected cars
	current_frame=[]

	for car_number in range(1,labels[1]+1):
		flag=False
		minx=None
		maxx=None
		miny=None
		maxy=None
		#find pixels with each car number label value
		nonzero=(labels[0]==car_number).nonzero()
		#indentify x and y values of those pixels
		nonzeroy=np.array(nonzero[0])
		nonzerox=np.array(nonzero[1])
		#define a bounding box based on min/max x and y
		bbox=((np.min(nonzerox),np.min(nonzeroy)),(np.max(nonzerox),np.max(nonzeroy)))
		if(len(last_frame)>0):# and (len(last_frame)>(car_number-1)):
			'''
			all the following code is a try to distinguish and stablize the bounding box of each car
			-here I'm assuming that the first car in the first frame will remain the first car in the rest of the frames
			'''
			i=0
			for frame in range(len(last_frame)-1,0,-1):
				if len(last_frame[frame])>car_number-1:
					if(i==0):
						minx=int(np.min(nonzerox))
						miny=int(np.min(nonzeroy))
						maxx=int(np.max(nonzerox))
						maxy=int(np.max(nonzeroy))
					minx+=int(last_frame[frame][car_number-1][0][0])
					miny+=int(last_frame[frame][car_number-1][0][1])
					maxx+=int(last_frame[frame][car_number-1][1][0])
					maxy+=int(last_frame[frame][car_number-1][1][1])
					flag=True
					i+=1
					if(i==5):
						break;
			if flag:
				minx=int(minx/6)
				miny=int(miny/6)
				maxx=int(maxx/6)
				maxy=int(maxy/6)
				cv2.rectangle(img,(minx,miny),(maxx,maxy),(0,0,255),6)
				current_frame.append(((minx,miny),(maxx,maxy)))
				flag=False
		
		if minx==None and maxx==None and miny==None and maxy==None:
			cv2.rectangle(img,bbox[0],bbox[1],(0,0,255),6)
			current_frame.append(bbox)	
		
		
	if(len(current_frame)>0):
		last_frame.append(current_frame)
	return img,last_frame




